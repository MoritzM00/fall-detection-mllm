import csv
import logging
import random
import time

import av
import numpy as np
import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset


class GenericVideoDataset(Dataset):
    def __init__(
        self,
        video_root,
        annotations_file,
        target_fps,
        vid_frame_count,
        data_fps=None,
        path_format="{video_root}/{filename}.mp4",
        max_retries=10,
        mode="train",
        fast=True,
        size=None,
    ):
        self.video_root = video_root
        self.slow_video_file = annotations_file.replace(".csv", "_slow.csv")
        self.target_fps = target_fps
        self.vid_frame_count = vid_frame_count
        self.data_fps = data_fps
        self.path_format = path_format
        self.max_retries = max_retries
        self.mode = mode
        self.load_video = (
            self.load_video_fast if fast and vid_frame_count is not None else self.load_video_slow
        )
        self.annotations = {}
        self.size = size

    def load_annotations(self, annotations_file):
        # call manually after init if needed
        annotations = {}
        with open(annotations_file) as file:
            reader = csv.reader(file)
            for row in reader:
                annotations[row[0]] = int(row[1])
        self.annotations = annotations
        self.paths = list(sorted(self.annotations.keys()))

    def __len__(self):
        return len(self.paths)

    def _id2label(self, idx):
        path = self.paths[idx]
        label = self.annotations[path]
        return path, label

    def load_item(self, idx):
        path, label = self._id2label(idx)

        # Measure video IO time
        video_io_start = time.time()
        video_path = self.path_format.format(video_root=self.video_root, filename=path)
        frames = self.load_video(video_path, idx)
        video_io_end = time.time()

        # Measure video processing time
        video_processing_start = time.time()
        inputs = self.transform_frames(frames)
        video_processing_end = time.time()

        inputs.update(
            {
                "label": label,
                "video_io_time": video_io_end - video_io_start,
                "video_processing_time": video_processing_end - video_processing_start,
            }
        )
        return inputs

    def __getitem__(self, idx):
        retries = 0
        while retries < self.max_retries:
            try:
                return self.load_item(idx)

            except Exception as e:
                retries += 1
                if retries >= self.max_retries:
                    video_path = self.path_format.format(
                        video_root=self.video_root, filename=self.paths[idx]
                    )
                    logging.error(f"Error loading video {video_path} at index {idx}: {str(e)}")
                    raise e
                idx = random.randint(0, len(self.paths) - 1)

        raise RuntimeError(f"Failed to load a valid video after {self.max_retries} attempts ()")

    def transform_frames(self, frames):
        # frames is a list of ndarrays (H, W, C)
        # Stack and convert to tensor: (T, H, W, C) -> (T, C, H, W)
        frames = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()

        if self.size is not None:
            # size is (height, width)
            frames = F.resize(frames, self.size)

        return {"video": frames}

    def get_random_offset(self, length, target_interval, idx, fps, start=0):
        if self.vid_frame_count is None or length < self.vid_frame_count * target_interval:
            return 0
        else:
            return random.randint(0, length - self.vid_frame_count * target_interval)

    def load_video_fast(self, path, idx):
        try:
            with av.open(path) as container:
                vs = next(s for s in container.streams if s.type == "video")

                # robust fps
                rate = vs.average_rate or vs.base_rate
                if not rate or rate.denominator == 0:
                    raise ValueError("Cannot determine FPS")
                fps = float(rate)

                # stream time‑base
                tb = vs.time_base
                if not tb:
                    raise ValueError("Missing time_base")
                tb = float(tb)

                frame_cnt = None if vs.frames in (0, None) else int(vs.frames)

                # random start (in frames) for augmentation
                begin_frame = self.get_random_offset(frame_cnt, 1, idx, fps) if frame_cnt else 0

                # Calculate desired timestamps
                desired_timestamps = [
                    (begin_frame / fps) + n / self.target_fps for n in range(self.vid_frame_count)
                ]
                desired_pts = [int(ts / tb) for ts in desired_timestamps]

                # Try seeking to just before first desired frame
                if desired_pts:
                    try:
                        container.seek(desired_pts[0], any_frame=False, backward=True, stream=vs)
                    except av.error.FFmpegError:
                        # Fallback if no keyframe found
                        container.seek(0, stream=vs)

                frames, want_idx, prev = [], 0, None
                for f in container.decode(vs):
                    if f.pts is None:
                        continue

                    # Collect frames matching desired PTS values
                    while want_idx < len(desired_pts) and f.pts >= desired_pts[want_idx]:
                        if prev and abs(prev.pts - desired_pts[want_idx]) < abs(
                            f.pts - desired_pts[want_idx]
                        ):
                            frames.append(prev.to_ndarray(format="rgb24"))
                        else:
                            frames.append(f.to_ndarray(format="rgb24"))
                        want_idx += 1

                    if want_idx == len(desired_pts):
                        break
                    prev = f

                if not frames:
                    logging.warning(f"{path}: fallback to slow loader")
                    return self.load_video_slow(path, idx)

                # Better frame padding strategy
                if len(frames) < self.vid_frame_count:
                    if len(frames) > 0:
                        # Repeat last frame instead of cycling
                        last_frame = frames[-1]
                        while len(frames) < self.vid_frame_count:
                            frames.append(last_frame)
                    else:
                        raise ValueError("No frames decoded")

                return frames

        except Exception as e:
            logging.error(f"Error reading video {path}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to process video {path}") from e

    def load_video_slow(self, video_path, idx):
        try:
            with av.open(video_path) as container:
                video_stream = next(s for s in container.streams if s.type == "video")

                frame_rate = video_stream.average_rate  # Detect actual frame rate

                fps = (
                    float(frame_rate.numerator / frame_rate.denominator)
                    if frame_rate
                    else self.vid_fps
                )

                target_interval = round(fps / self.target_fps)  # Calculate downsampling interval

                frames = []
                for i, frame in enumerate(container.decode(video_stream)):
                    if i % target_interval == 0:  # Keep only frames at the target interval
                        img = frame.to_ndarray(format="rgb24")
                        frames.append(img)

        except Exception as e:
            logging.error(f"Error reading video {video_path}: {e}")
            raise RuntimeError("Failed to process video")

        if self.vid_frame_count is None:
            # Load full video, no cycling required
            return frames

        if len(frames) < self.vid_frame_count:
            # Handle short videos by cycling frames
            logging.debug(
                f"Video {video_path} is too short. "
                + f"Got {len(frames)} sampled at {self.target_fps} instead of {self.vid_frame_count}. "
                + f"Cycling frames to match {self.vid_frame_count} frames."
            )
            frames = (frames * ((self.vid_frame_count // len(frames)) + 1))[: self.vid_frame_count]
        else:
            # Select a random consecutive sequence of frames
            start_index = self.get_random_offset(len(frames), self.target_fps, idx, fps)
            frames = frames[start_index : start_index + self.vid_frame_count]

        return frames


def main():
    import argparse
    import cProfile
    import pstats
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Debug HMDB51 dataset")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to use (train, val, or test)",
    )
    args = parser.parse_args()

    # Dataset parameters
    dataset_config = {
        "annotations": {
            "ann_type": "single-label-class",
            "ann_name": "action",
            "annf_train": "/lsdf/data/activity/Kinetics/k400_full_labels/train.csv",
            "annf_val": "/lsdf/data/activity/Kinetics/k400_full_labels/val.csv",
            "annf_test": "/lsdf/data/activity/Kinetics/k400_full_labels/test.csv",
            "attributes": "action",
            "num_classes": 400,
        },
        "augmentation": {
            "all": "fourier_phase_only",
            "train": "fourier_phase_only",
            "val": "fourier_phase_only",
            "test": "fourier_phase_only",
        },
        "name": "Kinetics400",
        "dataset_fps": [30],
        "model_fps": 7.5,
        "num_frames": 16,
        "root": "/lsdf/data/activity/Kinetics/",
        "videos": "k400_full",
        "video_root_train": "/lsdf/data/activity/Kinetics/k400_full",
        "video_root_val": "/lsdf/data/activity/Kinetics/k400_full",
        "video_root_test": "/lsdf/data/activity/Kinetics/k400_full",
        "path_format_train": "{video_root}/{filename}",
        "path_format_val": "{video_root}/{filename}",
        "path_format_test": "{video_root}/{filename}",
    }

    # Create dataset instance
    try:
        if args.split == "train":
            annotations_file = dataset_config["annotations"]["annf_train"]
            path_format = dataset_config["path_format_train"]
            video_root = dataset_config["video_root_train"]
        elif args.split == "val":
            annotations_file = dataset_config["annotations"]["annf_val"]
            path_format = dataset_config["path_format_val"]
            video_root = dataset_config["video_root_val"]
        else:  # test
            annotations_file = dataset_config["annotations"]["annf_test"]
            path_format = dataset_config["path_format_test"]
            video_root = dataset_config["video_root_test"]

        # Assuming the main dataset class is named HMDB51Dataset
        # Modify this according to your actual class name
        dataset = GenericVideoDataset(
            annotations_file=annotations_file,
            path_format=path_format,
            video_root=video_root,
            vid_frame_count=dataset_config["num_frames"],
            mode=args.split,
            data_fps=dataset_config["dataset_fps"][0],
            target_fps=dataset_config["model_fps"],
        )

        # IMPORTANT: GPU-specific transforms like PhaseOnly should NOT be applied here
        # using .map() because this happens in the CPU-based DataLoader worker process
        # before accelerate moves the data to the correct GPU.
        # from fourier import PhaseOnly
        # dataset = dataset.map(PhaseOnly(in_arrangement="t c h w")) # <-- REMOVED THIS LINE

        # Define TubeMasker
        class TubeMasker:
            def __init__(self, input_size, mask_ratio):
                self.frames, self.height, self.width = input_size
                self.num_patches_per_frame = self.height * self.width
                self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)

            def __call__(self, x):
                # Determine the device from the input dictionary (assuming 'pixel_values' exists)
                # Fallback to CPU if 'pixel_values' is not present or not a tensor
                input_tensor = x.get("pixel_values")
                device = (
                    input_tensor.device
                    if isinstance(input_tensor, torch.Tensor)
                    else torch.device("cpu")
                )

                n_zeros = self.num_patches_per_frame - self.num_masks_per_frame
                n_ones = self.num_masks_per_frame
                mask_per_frame = np.hstack([np.zeros(n_zeros), np.ones(n_ones)])
                np.random.shuffle(mask_per_frame)
                # Create the mask directly on the target device
                mask = np.tile(mask_per_frame, (self.frames, 1)).flatten()
                x.update({"bool_masked_pos": torch.tensor(mask, dtype=torch.bool, device=device)})
                return x

            def __repr__(self):
                repr_str = (
                    f"Mask: total patches {self.total_patches}, mask patches {self.total_masks}"
                )
                return repr_str

        # Wrap with DataLoader *after* all map operations intended for CPU workers
        # Note: accelerate will further wrap this DataLoader.
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

        print(
            f"Dataset created with {len(dataloader.dataset)} samples (before DataLoader batching)"
        )  # Corrected length reference
        print(f"Iterating through {args.split} split...")
        from pathlib import Path

        from tqdm import tqdm

        profiler = cProfile.Profile()
        profiler.enable()

        # In a real training loop with accelerate:
        # for batch in dataloader:
        #     # batch is now on the correct GPU device assigned by accelerate
        #     # Apply GPU-specific transforms HERE:
        #     batch = phase_only_transform(batch)
        #     # ... rest of your training step (forward pass, loss, backward, etc.) ...

        # --- Profiling loop (simulating iteration) ---
        # This loop won't apply PhaseOnly correctly as it's before accelerate's device placement
        # It's kept here for the original profiling purpose.
        for i, item in tqdm(enumerate(dataloader)):
            if i >= 100:  # Limit to 100 iterations
                break
            # In a real loop, PhaseOnly would be applied to 'item' here (after dataloader yields it)
            # For profiling purposes, we just print the item as loaded by the DataLoader.
            print(f"Item {i} (Batch):")
            for key, value in item.items():
                if hasattr(value, "shape"):
                    print(
                        f"  - {key}: shape {value.shape}, device {value.device}"
                    )  # Added device info
                else:
                    print(f"  - {key}: {value}")

        profiler.disable()

        # Create directory for profile output if it doesn't exist
        profile_dir = Path("profile_output")
        profile_dir.mkdir(exist_ok=True)

        # Write profiling data to a binary file
        profile_path = profile_dir / "profiling_results.prof"
        profiler.dump_stats(str(profile_path))

        print(f"Profile data written to {profile_path}")
        print(f"You can visualize it with: snakeviz {profile_path}")

        # Also create a text report for convenience
        stats_path = profile_dir / "profiling_stats.txt"
        with open(stats_path, "w") as f:
            stats = pstats.Stats(profiler, stream=f)
            stats.strip_dirs()
            stats.sort_stats("cumtime")
            stats.print_stats()

        print(f"Text stats written to {stats_path}")

    except Exception as e:
        print(f"Error creating or iterating through dataset: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
