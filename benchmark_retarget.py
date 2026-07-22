#!/usr/bin/env python3
"""Generate the LAFAN1 retargeting speed benchmark bar chart."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


EXPECTED_METHODS = (
    "FRDV No Refine (20)",
    "FRDV Refine (20)",
    "FRDV (1)",
    "ProtoMotion (Pyroki)",
    "GMR",
    "PHC",
    "Holosoma",
)
EXPECTED_MOTION_COUNT = 77

METHOD_COLORS = (
    "#2ecc71",
    "#82e0aa",
    "#27ae60",
    "#3498db",
    "#e67e22",
    "#9b59b6",
    "#e74c3c",
)


@dataclass(frozen=True)
class BenchmarkData:
    method_names: tuple[str, ...]
    motion_names: tuple[str, ...]
    fps: np.ndarray


@dataclass(frozen=True)
class MethodStatistics:
    median: np.ndarray
    first_quartile: np.ndarray
    third_quartile: np.ndarray
    slowdown: np.ndarray


def load_benchmark_csv(csv_path: Path) -> BenchmarkData:
    """Load motion rows and validate, but never return the blank summary row."""
    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{csv_path} is empty") from exc

        if not header or header[0].strip() != "Motion":
            raise ValueError(f"{csv_path} must start with a Motion column")

        method_names = tuple(cell.strip() for cell in header[1:])
        if method_names != EXPECTED_METHODS:
            raise ValueError(
                "Unexpected benchmark methods. Expected: "
                + ", ".join(EXPECTED_METHODS)
            )

        motion_names: list[str] = []
        motion_values: list[list[float]] = []
        summary_values: list[list[float]] = []

        for line_number, row in enumerate(reader, start=2):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) != len(header):
                raise ValueError(
                    f"{csv_path}:{line_number} has {len(row)} columns; "
                    f"expected {len(header)}"
                )

            try:
                values = [float(value) for value in row[1:]]
            except ValueError as exc:
                raise ValueError(
                    f"{csv_path}:{line_number} contains a non-numeric FPS value"
                ) from exc

            motion_name = row[0].strip()
            if motion_name:
                motion_names.append(motion_name)
                motion_values.append(values)
            else:
                summary_values.append(values)

    if len(motion_names) != EXPECTED_MOTION_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_MOTION_COUNT} motion rows after excluding summaries; "
            f"found {len(motion_names)}"
        )
    if len(summary_values) != 1:
        raise ValueError(
            f"Expected exactly one blank-Motion summary row; found {len(summary_values)}"
        )

    fps = np.asarray(motion_values, dtype=np.float64)
    summary = np.asarray(summary_values[0], dtype=np.float64)
    if not np.all(np.isfinite(fps)) or np.any(fps <= 0.0):
        raise ValueError("All motion FPS measurements must be finite and positive")
    if not np.allclose(summary, fps.mean(axis=0), rtol=0.0, atol=1e-6):
        raise ValueError("The blank-Motion summary row does not equal the 77-row mean")

    return BenchmarkData(method_names, tuple(motion_names), fps)


def compute_method_statistics(fps: np.ndarray) -> MethodStatistics:
    first_quartile, median, third_quartile = np.percentile(
        fps, (25.0, 50.0, 75.0), axis=0
    )
    fastest_median = float(np.max(median))
    slowdown = fastest_median / median
    return MethodStatistics(median, first_quartile, third_quartile, slowdown)


def format_fps(value: float) -> str:
    if value >= 100.0:
        return f"{value:,.0f}"
    if value >= 10.0:
        return f"{value:.1f}"
    return f"{value:.2f}"


def configure_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["cmr10"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "font.size": 11.0,
            "axes.titlesize": 11.0,
            "axes.titleweight": "normal",
            "axes.labelsize": 11.0,
            "axes.labelweight": "normal",
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "figure.dpi": 150,
            "text.usetex": False,
            "pdf.fonttype": 42,
        }
    )


def plot_method_panel(ax, data: BenchmarkData, stats: MethodStatistics) -> None:
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    y_positions = np.arange(len(data.method_names), dtype=float)
    widths = stats.median - 1.0
    asymmetric_iqr = np.vstack(
        (
            stats.median - stats.first_quartile,
            stats.third_quartile - stats.median,
        )
    )

    bars = ax.barh(
        y_positions,
        widths,
        left=1.0,
        height=0.6,
        color=METHOD_COLORS,
        edgecolor="white",
        linewidth=0.8,
        xerr=asymmetric_iqr,
        error_kw={
            "ecolor": "black",
            "elinewidth": 1.4,
            "capsize": 4.0,
            "capthick": 1.4,
        },
        zorder=3,
    )
    for bar in bars[:3]:
        bar.set_edgecolor("#145a32")
        bar.set_linewidth(2.0)

    fastest_index = int(np.argmax(stats.median))
    for index, (median, third_quartile, slowdown) in enumerate(
        zip(stats.median, stats.third_quartile, stats.slowdown, strict=True)
    ):
        comparison = (
            "(fastest)"
            if index == fastest_index
            else f"({slowdown:,.0f}$\\times$ slower)"
        )
        ax.text(
            third_quartile * 1.15,
            float(index),
            f"{format_fps(float(median))} FPS\n{comparison}",
            ha="left",
            va="center",
            fontsize=8.5,
            fontweight="normal",
            linespacing=1.1,
            color="black",
            zorder=5,
        )

    ax.set_xscale("log")
    ax.set_xlim(1.0, 65_000.0)
    ax.set_yticks(y_positions, data.method_names)
    ax.invert_yaxis()
    ax.set_xlabel("Frames per Second (FPS, log scale)", labelpad=7.0)

    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: f"{value:,.0f}")
    )
    ax.xaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2.0, 10.0) * 0.1)
    )
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(
        axis="x",
        which="both",
        linestyle="--",
        linewidth=0.8,
        alpha=0.4,
        zorder=0,
    )
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)


def create_figure(data: BenchmarkData, output_pdf: Path) -> MethodStatistics:
    configure_matplotlib()
    from matplotlib import pyplot as plt

    stats = compute_method_statistics(data.fps)
    figure, method_ax = plt.subplots(figsize=(7.25, 3.9), layout="constrained")
    plot_method_panel(method_ax, data, stats)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_pdf,
        format="pdf",
        metadata={
            "Title": "LAFAN1 Retargeting Speed Benchmark",
            "Subject": "Median retargeting FPS across 77 motions",
        },
    )
    plt.close(figure)
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the LAFAN1 retargeting speed benchmark bar-chart PDF."
    )
    parser.add_argument("input_csv", type=Path, help="Benchmark CSV input")
    parser.add_argument("output_pdf", type=Path, help="Output PDF path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output_pdf.suffix.lower() != ".pdf":
        raise SystemExit("output_pdf must use a .pdf extension")

    try:
        data = load_benchmark_csv(args.input_csv)
        stats = create_figure(data, args.output_pdf)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Loaded {len(data.motion_names)} motion rows from {args.input_csv}")
    print("Excluded and validated 1 blank-Motion mean summary row")
    for method_name, median, slowdown in zip(
        data.method_names, stats.median, stats.slowdown, strict=True
    ):
        comparison = "fastest" if np.isclose(slowdown, 1.0) else f"{slowdown:,.0f}x slower"
        print(f"{method_name}: median {median:.2f} FPS ({comparison})")
    print(f"Wrote figure to {args.output_pdf}")


if __name__ == "__main__":
    main()
