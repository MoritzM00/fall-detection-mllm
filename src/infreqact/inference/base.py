import logging
import time
import warnings

import json_repair
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

logger = logging.getLogger(__name__)


def init_hf_qwen_model(size="2B", attn_implementation="flash_attention_2", cache_dir=".cache"):
    model_path = f"Qwen/Qwen3-VL-{size}-Instruct"

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype="bfloat16",
        device_map="auto",
        output_loading_info=False,
        cache_dir=cache_dir,
        attn_implementation=attn_implementation,
    )

    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def inference(
    video,
    prompt,
    model,
    processor,
    max_new_tokens=2048,
    total_pixels=20480 * 32 * 32,
    min_pixels=64 * 32 * 32,
    max_pixels=256 * 32 * 32,
    max_frames=2048,
    sample_fps=2,
    target_fps=2,
    return_inputs=False,
):
    """
    Perform multimodal inference on input video and text prompt to generate model response.

    Args:
        video (str or list/tuple): Video input, supports two formats:
            - str: Path or URL to a video file. The function will automatically read and sample frames.
            - list/tuple: Pre-sampled list of video frames (PIL.Image or url).
              In this case, `sample_fps` indicates the frame rate at which these frames were sampled from the original video.
        prompt (str): User text prompt to guide the model's generation.
        max_new_tokens (int, optional): Maximum number of tokens to generate. Default is 2048.
        total_pixels (int, optional): Maximum total pixels for video frame resizing (upper bound). Default is 20480*32*32.
        min_pixels (int, optional): Minimum total pixels for video frame resizing (lower bound). Default is 16*32*32.
        sample_fps (int, optional): ONLY effective when `video` is a list/tuple of frames!
            Specifies the original sampling frame rate (FPS) from which the frame list was extracted.
            Used for temporal alignment or normalization in the model. Default is 2.
        target_fps (int, optional): only effective when 'video' is a str
            Specifies the sampling rate for the model. Default is 2.
            `max_frames` can be used to limit the number of frames to override this setting.


    Returns
    -------
        str: Generated text response from the model.

    Notes
    -----
        - When `video` is a string (path/URL), `sample_fps` is ignored and will be overridden by the video reader backend.
        - When `video` is a frame list, `sample_fps` informs the model of the original sampling rate to help understand temporal density.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "video": video,
                    # "total_pixels": total_pixels,
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                    # "max_frames": max_frames,
                    # "fps": target_fps,
                    # "sample_fps": sample_fps,
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    start_time_process_vision = time.perf_counter()
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        [messages],
        return_video_kwargs=True,
        image_patch_size=processor.image_processor.patch_size,
        return_video_metadata=True,
    )
    end_time_process_vision = time.perf_counter()
    print(
        f"Time taken for process_vision_info: {end_time_process_vision - start_time_process_vision:.2f} seconds"
    )

    if video_inputs is not None:
        video_inputs, video_metadatas = zip(*video_inputs)
        video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
        print(f"Video input shape: {video_inputs[0].shape}")
    else:
        video_metadatas = None
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        video_metadata=video_metadatas,
        **video_kwargs,
        do_resize=False,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    start_time_generate = time.perf_counter()
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    end_time_generate = time.perf_counter()
    print(f"Time taken for model.generate: {end_time_generate - start_time_generate:.2f} seconds")

    generated_ids = [
        output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return (output_text[0], inputs) if return_inputs else output_text[0]


def prepare_inputs_for_vllm(messages, processor):
    """
    Prepare inputs for vLLM.

    Args:
        messages: List of messages in standard conversation format
        processor: AutoProcessor instance

    Returns:
        dict: Input format required by vLLM
    """
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # qwen_vl_utils 0.0.14+ required
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    return {"prompt": text, "multi_modal_data": mm_data, "mm_processor_kwargs": video_kwargs}


def parse_llm_outputs(outputs: list[dict], samples: list[dict], verbose: bool | int = False):
    """
    Parse LLM outputs and extract predicted labels.

    Args:
        outputs: List of LLM output objects containing generated text
        samples: List of ground truth samples with labels
        verbose: Controls output printing:
            - False/0: No printing
            - True: Print first 10 samples
            - int > 0: Print first N samples

    Returns:
        tuple: (predictions dict, predicted_labels list, true_labels list)
    """
    # Extract predicted labels from LLM outputs
    predicted_labels = []
    true_labels = []

    # Determine how many samples to print
    if verbose is False or verbose == 0:
        n_print = 0
    elif verbose is True:
        n_print = 10
    else:
        n_print = int(verbose)

    predictions = {}
    for i, (output, sample) in enumerate(zip(outputs, samples)):
        generated_text = output.outputs[0].text

        # Print output if within verbosity limit
        should_print = i < n_print
        if should_print:
            print()
            print("=" * 40)
            print(f"Sample {i + 1}/{len(outputs)}")

        true_labels.append(sample["label_str"])
        try:
            json_obj = json_repair.loads(generated_text)
            predicted_label = json_obj.get("label", "other")  # Default to 'other' if missing

            if should_print:
                print(f"JSON output: {json_obj}")
                print(f"  predicted label: {predicted_label}")
                print(f"  true label: {sample['label_str']}")

            predicted_labels.append(predicted_label)

            prediction = sample.copy()
            prediction["predicted_label"] = predicted_labels[-1]
            prediction["reasoning"] = json_obj.get("reasoning", "")
            predictions[f"sample_{i}"] = prediction
        except Exception as e:
            import traceback

            logger.error(f"Failed to parse JSON for sample {i}: {e}")
            logger.error(f"Generated text: {generated_text}")
            logger.error(f"Traceback: {traceback.format_exc()}")

            # Default to 'other' for failed parses
            predicted_labels.append("other")

    # Print summary if verbosity is enabled but there are more samples
    if n_print > 0 and len(outputs) > n_print:
        print()
        print("=" * 40)
        print(f"... {len(outputs) - n_print} more samples (use verbose={len(outputs)} to see all)")

    return predictions, predicted_labels, true_labels
