import textwrap

from PIL import Image


def get_system_prompt(cot=False):
    """
    Generate system prompt for Human Activity Recognition.

    Args:
        cot: If True, include chain-of-thought reasoning in the output.
             If False, only output the label.

    Returns:
        str: Formatted system prompt
    """
    if cot:
        output_format = textwrap.dedent("""
            Output Format:
            Return a JSON object:
            {
              "reason": "Brief analysis of intent, movement type, and objects.",
              "label": "<class_label>"
            }
        """).strip()
    else:
        output_format = textwrap.dedent("""
            Output Format:
            Return a JSON object:
            {
              "label": "<class_label>"
            }
        """).strip()

    prompt = textwrap.dedent(f"""
        Role:
        You are an expert Human Activity Recognition (HAR) specialist. Analyze the video clip to classify the action or posture of the main subject.

        Allowed Labels:
        * Core: walk, fall, fallen, sit_down, sitting, lie_down, lying, stand_up, standing, other
        * Extended (Rare): kneel_down, kneeling, squat_down, squatting, crawl, jump

        Definitions & Constraints:
        * Walk vs. Other: walk includes jogging, drunk walking, and carrying small items. Pushing large objects (chairs, carts) must be labeled other.
        * Fall vs. Lie/Sit:
            * fall: Uncontrolled, rapid descent (accidental).
            * lie_down / sit_down: Intentional, controlled descent.
        * Dynamic vs. Static:
            * Dynamic (Actions): walk, fall, sit_down, lie_down, stand_up. Label starts at first frame of motion.
            * Static (States): fallen, sitting, lying, standing. Label starts when subject comes to complete rest.

        Sequence Rules:
        * Moment of Rest: If a person transitions Sit -> Lie without pausing, use only the destination label.
        * Fall Termination: fall ends when inertia stops. fallen begins only when the person is on the ground in a resting state.

        {output_format}
    """).strip()  # .strip() removes the very first and last newlines caused by the triple quotes

    return prompt


def build_prompts(
    dataset, n_samples=5, cot=False
):  # loads the omnifall test set and creates prompts with video samples
    """
    Build prompts for video classification.

    Args:
        dataset: Dataset to sample from
        n_samples: Number of samples to process
        cot: If True, request chain-of-thought reasoning in outputs

    Returns:
        tuple: (messages, samples)
    """
    prompt = get_system_prompt(cot=cot)
    messages = []
    samples = []
    for i in range(n_samples):
        sample = dataset[i]
        video_sample = sample.pop("video")  # list of frames (numpy arrays)
        samples.append(sample)

        # convert to PIL image
        video_sample = [Image.fromarray(frame) for frame in video_sample]
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_sample,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        )

    return messages, samples


def build_prompt_for_sample(sample, cot=False):
    """
    Build a single prompt message for one sample.

    Args:
        sample: Dictionary containing video frames and metadata
        cot: If True, request chain-of-thought reasoning

    Returns:
        tuple: (message dict, sample metadata dict)
    """
    prompt = get_system_prompt(cot=cot)

    # Extract video without modifying the original sample
    video_sample = sample["video"]  # list of frames (numpy arrays)
    sample_metadata = {k: v for k, v in sample.items() if k != "video"}

    # Convert to PIL images
    video_sample = [Image.fromarray(frame) for frame in video_sample]

    message = {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_sample,
            },
            {"type": "text", "text": prompt},
        ],
    }

    return message, sample_metadata


def collate_fn(batch, cot=False):
    """
    Collate function for DataLoader to process batches of samples.

    Args:
        batch: List of samples from the dataset
        cot: If True, request chain-of-thought reasoning

    Returns:
        tuple: (list of messages, list of sample metadata)
    """
    messages = []
    samples = []

    for sample in batch:
        message, sample_metadata = build_prompt_for_sample(sample, cot=cot)
        messages.append(message)
        samples.append(sample_metadata)

    return messages, samples


def run_inference():
    pass


def evaluate_predictions():
    pass
