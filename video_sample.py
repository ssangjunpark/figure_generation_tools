import argparse
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


def normalize_time_range(time_range_seconds: Sequence[float] | None):
    if time_range_seconds is None:
        return None

    if len(time_range_seconds) != 2:
        raise ValueError("time_range_seconds must contain exactly two values: [start_seconds, end_seconds].")

    start_seconds = float(time_range_seconds[0])
    end_seconds = float(time_range_seconds[1])

    if start_seconds < 0 or end_seconds < 0:
        raise ValueError("time_range_seconds values must be non-negative.")

    if end_seconds <= start_seconds:
        raise ValueError("time_range_seconds must satisfy end_seconds > start_seconds.")

    return start_seconds, end_seconds


def parse_time_range_arg(raw_values: list[str] | None):
    if raw_values is None:
        return None

    parts = " ".join(raw_values).strip().strip("[]").replace(",", " ").split()
    if len(parts) != 2:
        raise ValueError("time range must be provided as two numbers, e.g. [12,13] or 12 13.")

    return normalize_time_range(parts)


def sample_video_frames(
    video_path: str,
    output_dir: str,
    interval_seconds: float = 1.5,
    image_extension: str = "png",
    time_range_seconds: Sequence[float] | None = None,
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
        normalized_time_range = normalize_time_range(time_range_seconds)
        range_start_seconds = 0.0
        range_end_seconds = duration_seconds

        if normalized_time_range is not None:
            range_start_seconds, requested_end_seconds = normalized_time_range
            if range_start_seconds >= duration_seconds:
                raise ValueError(
                    f"time_range_seconds starts at {range_start_seconds:.2f}s, "
                    f"but the video ends at {duration_seconds:.2f}s."
                )
            range_end_seconds = min(requested_end_seconds, duration_seconds)

        sample_times = np.arange(range_start_seconds, range_end_seconds, interval_seconds, dtype=float)
        if normalized_time_range is not None:
            interval_count = round((range_end_seconds - range_start_seconds) / interval_seconds)
            aligned_end_seconds = range_start_seconds + (interval_count * interval_seconds)
            if np.isclose(aligned_end_seconds, range_end_seconds) and (
                sample_times.size == 0 or not np.isclose(sample_times[-1], range_end_seconds)
            ):
                sample_times = np.append(sample_times, range_end_seconds)
        if sample_times.size == 0:
            sample_times = np.array([range_start_seconds], dtype=float)

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
    parser.add_argument(
        "--time-range",
        nargs="+",
        default=None,
        help="Optional sampling window in seconds. Examples: --time-range 12 13 or --time-range [12,13]",
    )

    args = parser.parse_args()
    try:
        args.time_range = parse_time_range_arg(args.time_range)
    except ValueError as exc:
        parser.error(str(exc))
    return args


if __name__ == "__main__":
    args = parse_args()
    sample_video_frames(
        video_path=args.video_path,
        output_dir=args.output_dir,
        interval_seconds=args.interval,
        image_extension=args.ext,
        time_range_seconds=args.time_range,
    )
