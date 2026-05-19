import logging
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
logger = logging.getLogger(__name__)


def prepare_inputs_for_vllm(frames, messages, processor, model_fps=8, needs_video_metadata=True):
    """
    Prepare inputs for vLLM.

    Args:
        frames: Video frames tensor
        messages: List of message dicts (system + user) or single message dict
        processor: AutoProcessor instance
        model_fps: Frame rate to use for video metadata
        needs_video_metadata: Whether to include video metadata in the multi-modal data

    Returns:
        dict: Input format required by vLLM
    """
    # Ensure messages is a list
    if isinstance(messages, dict):
        messages = [messages]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if needs_video_metadata:
        video_meta = dict(
            total_num_frames=frames.shape[0],
            fps=model_fps,
            frames_indices=list(range(frames.shape[0])),
        )
        mm_data = dict(video=(frames, video_meta))
    else:
        mm_data = dict(video=frames)

    video_kwargs = dict(do_sample_frames=False)

    return dict(prompt=text, multi_modal_data=mm_data, mm_processor_kwargs=video_kwargs)
