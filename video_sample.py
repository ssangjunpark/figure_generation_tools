import argparse
from pathlib import Path

import cv2
import numpy as np


def sample_video_frames(
    video_path: str,
    output_dir: str,
    interval_seconds: float = 1.5,
    image_extension: str = "png",
):
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be greater than 0.")

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if total_frames <= 0 or fps <= 0:
            raise RuntimeError("Could not read video metadata.")

        duration_seconds = total_frames / fps
        sample_times = np.arange(0, duration_seconds, interval_seconds, dtype=float)
        if sample_times.size == 0:
            sample_times = np.array([0.0], dtype=float)

        saved_paths = []

        for sample_index, time_seconds in enumerate(sample_times):
            cap.set(cv2.CAP_PROP_POS_MSEC, time_seconds * 1000.0)
            ok, frame = cap.read()
            if not ok:
                print(f"Skipping sample at {time_seconds:.2f}s because the frame could not be read.")
                continue

            frame_index = int(round(time_seconds * fps))
            output_name = (
                f"frame_{sample_index:04d}_t{time_seconds:07.2f}s_f{frame_index:06d}.{image_extension}"
            )
            output_path = output_dir / output_name

            saved = cv2.imwrite(str(output_path), frame)
            if not saved:
                raise RuntimeError(f"Failed to save frame to {output_path}")

            saved_paths.append(output_path)
            print(f"Saved {output_path}")

        print(f"Saved {len(saved_paths)} frame(s) to {output_dir}")
        return saved_paths
    finally:
        cap.release()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample frames from a video at a fixed time interval and save them to a folder."
    )
    parser.add_argument("video_path", help="Path to the input video file.")
    parser.add_argument("output_dir", help="Folder where sampled frames will be saved.")
    parser.add_argument(
        "--interval",
        type=float,
        default=1.5,
        help="Sampling interval in seconds. Default: 1.5",
    )
    parser.add_argument(
        "--ext",
        default="png",
        choices=["png", "jpg", "jpeg", "bmp", "tiff"],
        help="Image format for saved frames. Default: png",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sample_video_frames(
        video_path=args.video_path,
        output_dir=args.output_dir,
        interval_seconds=args.interval,
        image_extension=args.ext,
    )
