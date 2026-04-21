#!/usr/bin/env python3
"""Plot torque CSV files into a PDF.

The expected CSV format is:
  time,<actuator_0>,<actuator_1>,...

Time ranges are half-open: `[start_sec, end_sec)`.
For example, `[0,1.5]` selects the first 1.5 seconds.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import numpy as np

# Leave this empty to plot every joint.
# Add exact CSV joint names here to plot only a subset.
# PLOT_ONLY_JOINTS: list[str] = [
#     "left_hip_pitch_joint",
#     "left_hip_roll_joint",
#     "left_hip_yaw_joint",
#     "left_knee_joint",
#     "left_ankle_pitch_joint",
#     "left_ankle_roll_joint"
# ]

PLOT_ONLY_JOINTS: list[str] = [
    "l_hip_yaw_actuator",
    "l_hip_roll_actuator",
    "l_hip_pitch_actuator",
    "l_knee_pitch_actuator",
    "l_ankle_pitch_actuator",
]

def parse_frame_range(value: str) -> tuple[int, int]:
    """Parse a frame range like `[0,30]`, `(0,30)`, or `0:30`."""
    text = value.strip()
    if ":" in text:
        parts = text.split(":")
    else:
        cleaned = text.replace("[", "").replace("]", "")
        cleaned = cleaned.replace("(", "").replace(")", "")
        parts = [part.strip() for part in cleaned.split(",")]

    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "frame range must have exactly two integers, e.g. [0,30]"
        )

    try:
        start = int(parts[0].strip())
        end = int(parts[1].strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "frame range entries must be integers, e.g. [0,30]"
        ) from exc

    if start < 0 or end < 0:
        raise argparse.ArgumentTypeError("frame range values must be non-negative")
    if end <= start:
        raise argparse.ArgumentTypeError(
            "frame range must be half-open with end > start, e.g. [0,30]"
        )

    return start, end


def parse_time_range(value: str) -> tuple[float, float]:
    """Parse a time range like `[0,1.5]`, `(0,1.5)`, or `0:1.5`."""
    text = value.strip()
    if ":" in text:
        parts = text.split(":")
    else:
        cleaned = text.replace("[", "").replace("]", "")
        cleaned = cleaned.replace("(", "").replace(")", "")
        parts = [part.strip() for part in cleaned.split(",")]

    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "time range must have exactly two numbers, e.g. [0,1.5]"
        )

    try:
        start = float(parts[0].strip())
        end = float(parts[1].strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "time range entries must be numbers, e.g. [0,1.5]"
        ) from exc

    if start < 0.0 or end < 0.0:
        raise argparse.ArgumentTypeError("time range values must be non-negative")
    if end <= start:
        raise argparse.ArgumentTypeError(
            "time range must be half-open with end > start, e.g. [0,1.5]"
        )

    return start, end


def load_torque_csv(csv_path: Path) -> tuple[list[str], np.ndarray]:
    with csv_path.open(newline="") as file:
        reader = csv.reader(file)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{csv_path} is empty") from exc

        if len(header) < 2:
            raise ValueError(
                f"{csv_path} must have at least two columns: time plus one actuator"
            )
        if header[0] != "time":
            raise ValueError(
                f"{csv_path} must start with a 'time' column, found {header[0]!r}"
            )

        rows: list[list[float]] = []
        for line_number, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError(
                    f"{csv_path}:{line_number} has {len(row)} columns; "
                    f"expected {len(header)}"
                )
            try:
                rows.append([float(value) for value in row])
            except ValueError as exc:
                raise ValueError(
                    f"{csv_path}:{line_number} contains a non-numeric value"
                ) from exc

    if not rows:
        raise ValueError(f"{csv_path} has no data rows")

    return header, np.asarray(rows, dtype=np.float64)


def infer_sample_rate(times: np.ndarray) -> float | None:
    if len(times) < 2:
        return None

    deltas = np.diff(times)
    positive_deltas = deltas[deltas > 0.0]
    if len(positive_deltas) == 0:
        return None

    median_dt = float(np.median(positive_deltas))
    if median_dt <= 0.0:
        return None
    return 1.0 / median_dt


def resolve_frame_slice(
    frame_range: tuple[int, int] | None, total_frames: int
) -> tuple[int, int]:
    if frame_range is None:
        return 0, total_frames

    start, end = frame_range
    if end > total_frames:
        raise ValueError(
            f"frame range {frame_range} exceeds available frames [0,{total_frames})"
        )
    return start, end


def resolve_time_slice(
    time_range: tuple[float, float] | None, times: np.ndarray
) -> tuple[int, int]:
    if time_range is None:
        return 0, len(times)

    start_time, end_time = time_range
    available_start = float(times[0])
    available_end = float(times[-1])

    if start_time < available_start or end_time > available_end + 1e-9:
        raise ValueError(
            f"time range {time_range} exceeds available times "
            f"[{available_start:.3f}, {available_end:.3f}]"
        )

    start = int(np.searchsorted(times, start_time, side="left"))
    end = int(np.searchsorted(times, end_time, side="left"))
    if end <= start:
        raise ValueError(
            f"time range {time_range} does not include any samples"
        )
    return start, end


def default_output_path(input_csv: Path, mode: str) -> Path:
    return input_csv.with_name(f"{input_csv.stem}_{mode}.pdf")


def sanitize_filename_part(value: str) -> str:
    cleaned = "".join(character if character.isalnum() else "_" for character in value)
    cleaned = cleaned.strip("_")
    return cleaned or "joint"


def default_png_path(output_pdf: Path) -> Path:
    return output_pdf.with_suffix(".png")


def separate_png_path(output_pdf: Path, joint_name: str, joint_index: int) -> Path:
    joint_label = sanitize_filename_part(joint_name)
    return output_pdf.with_name(
        f"{output_pdf.stem}_{joint_index:02d}_{joint_label}.png"
    )


def range_summary(
    start: int,
    end: int,
    times: np.ndarray,
    sample_rate: float | None,
) -> str:
    start_time = float(times[0])
    end_time = float(times[-1])
    duration = (
        float(end - start) / sample_rate
        if sample_rate is not None
        else end_time - start_time
    )
    rate_text = (
        f"{sample_rate:.3f} Hz inferred"
        if sample_rate is not None
        else "sample rate unknown"
    )
    return (
        f"samples [{start}, {end}) | "
        f"time {start_time:.3f}s to {end_time:.3f}s | "
        f"span {duration:.3f}s | {rate_text}"
    )


def filter_joint_columns(
    joint_names: Sequence[str],
    torques: np.ndarray,
    requested_joint_names: Sequence[str],
) -> tuple[list[str], np.ndarray, list[str]]:
    if not requested_joint_names:
        return list(joint_names), torques, []

    joint_index_by_name = {joint_name: index for index, joint_name in enumerate(joint_names)}
    selected_indices: list[int] = []
    selected_joint_names: list[str] = []
    missing_joint_names: list[str] = []

    for joint_name in requested_joint_names:
        joint_index = joint_index_by_name.get(joint_name)
        if joint_index is None:
            missing_joint_names.append(joint_name)
            continue
        selected_indices.append(joint_index)
        selected_joint_names.append(joint_name)

    if not selected_indices:
        return [], torques[:, :0], missing_joint_names

    return selected_joint_names, torques[:, selected_indices], missing_joint_names


def plot_together(
    output_pdf: Path,
    joint_names: Sequence[str],
    times: np.ndarray,
    torques: np.ndarray,
    summary: str,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 8.5))

    for joint_index, joint_name in enumerate(joint_names):
        ax.plot(times, torques[:, joint_index], linewidth=1.0, label=joint_name)

    ax.set_title(f"Joint Torques ({summary})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Torque")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(output_pdf, format="pdf", bbox_inches="tight")
    output_png = default_png_path(output_pdf)
    fig.savefig(output_png, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_png


def plot_separate(
    output_pdf: Path,
    joint_names: Sequence[str],
    times: np.ndarray,
    torques: np.ndarray,
    summary: str,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    output_pngs: list[Path] = []
    with PdfPages(output_pdf) as pdf:
        for joint_index, joint_name in enumerate(joint_names):
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.plot(times, torques[:, joint_index], linewidth=1.2)
            ax.set_title(f"{joint_name} Torque ({summary})")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Torque")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            pdf.savefig(fig)
            output_png = separate_png_path(output_pdf, joint_name, joint_index)
            fig.savefig(output_png, format="png", dpi=200, bbox_inches="tight")
            output_pngs.append(output_png)
            plt.close(fig)
    return output_pngs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a PDF plot from a torque CSV. "
            "Time ranges are half-open, so [0,1.5] selects the first 1.5 seconds."
        )
    )
    parser.add_argument("input_csv", type=Path, help="Path to a torque CSV file")
    parser.add_argument(
        "output_pdf",
        type=Path,
        nargs="?",
        help="Output PDF path (default: <input_stem>_<mode>.pdf)",
    )
    parser.add_argument(
        "--mode",
        choices=("together", "separate"),
        default="together",
        help="Plot all joints on one figure or one joint per PDF page",
    )
    parser.add_argument(
        "--time-range",
        type=parse_time_range,
        default=None,
        help="Half-open time range in seconds like [0,1.5] or 0:1.5",
    )
    parser.add_argument(
        "--frame-range",
        type=parse_frame_range,
        default=None,
        help="Legacy half-open frame range like [0,30] or 0:30",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_csv = args.input_csv
    if not input_csv.exists():
        raise SystemExit(f"Input CSV does not exist: {input_csv}")

    output_pdf = args.output_pdf or default_output_path(input_csv, args.mode)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    header, data = load_torque_csv(input_csv)
    times = data[:, 0]
    joint_names = header[1:]

    if args.time_range is not None and args.frame_range is not None:
        raise SystemExit("Use only one of --time-range or --frame-range")

    if args.time_range is not None:
        start, end = resolve_time_slice(args.time_range, times)
    else:
        start, end = resolve_frame_slice(args.frame_range, len(times))

    selected_times = times[start:end]
    selected_torques = data[start:end, 1:]
    plotted_joint_names, selected_torques, missing_joint_names = filter_joint_columns(
        joint_names, selected_torques, PLOT_ONLY_JOINTS
    )

    for joint_name in missing_joint_names:
        print(f"Skipping missing joint: {joint_name}")

    if not plotted_joint_names:
        if PLOT_ONLY_JOINTS:
            raise SystemExit("No requested joints were found in the CSV, so nothing was plotted.")
        raise SystemExit("No joints were available to plot.")

    sample_rate = infer_sample_rate(times)
    summary = range_summary(start, end, selected_times, sample_rate)

    try:
        if args.mode == "together":
            output_pngs = [
                plot_together(
                    output_pdf,
                    plotted_joint_names,
                    selected_times,
                    selected_torques,
                    summary,
                )
            ]
        else:
            output_pngs = plot_separate(
                output_pdf,
                plotted_joint_names,
                selected_times,
                selected_torques,
                summary,
            )
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required to generate PDF plots in this environment."
        ) from exc

    print(f"Loaded {len(times)} samples and {len(joint_names)} joints from {input_csv}")
    print(
        f"Selected {end - start} samples: [{start}, {end}) "
        f"-> {selected_times[0]:.3f}s to {selected_times[-1]:.3f}s"
    )
    print(f"Plotting {len(plotted_joint_names)} joint(s)")
    if sample_rate is not None:
        print(f"Inferred sample rate: {sample_rate:.3f} Hz")
    print(f"Wrote {args.mode} plot PDF to: {output_pdf}")
    if args.mode == "together":
        print(f"Wrote PNG plot to: {output_pngs[0]}")
    else:
        print(f"Wrote {len(output_pngs)} PNG plot(s) with base stem: {output_pdf.stem}")


if __name__ == "__main__":
    main()
