#!/usr/bin/env python3
"""Create a simple horizontal LAFAN1 speed benchmark chart."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import fmean, quantiles


DISPLAY_NAMES = {
    "FRDV No Refine (20)": "FRDV (20 workers)\n(No RefineBoundary)",
    "FRDV Refine (20)": "FRDV (20 workers)\n(RefineBoundary)",
    "FRDV (1)": "FRDV (1 worker)",
}


def load_fps_statistics(
    csv_path: Path,
) -> tuple[list[str], list[float], list[float], list[float], int]:
    """Return mean and IQR statistics, excluding the blank summary row."""
    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{csv_path} is empty") from exc

        if len(header) < 2 or header[0].strip() != "Motion":
            raise ValueError("The first CSV column must be named 'Motion'")

        methods = [name.strip() for name in header[1:]]
        measurements: list[list[float]] = []
        summary: list[float] | None = None

        for line_number, row in enumerate(reader, start=2):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) != len(header):
                raise ValueError(
                    f"Line {line_number} has {len(row)} columns; expected {len(header)}"
                )

            try:
                values = [float(value) for value in row[1:]]
            except ValueError as exc:
                raise ValueError(f"Line {line_number} contains a non-numeric value") from exc
            if any(value <= 0 for value in values):
                raise ValueError(f"Line {line_number} contains a non-positive FPS value")

            if row[0].strip():
                measurements.append(values)
            elif summary is None:
                summary = values
            else:
                raise ValueError("The CSV contains more than one blank summary row")

    if not measurements:
        raise ValueError("The CSV contains no motion measurements")

    columns = [list(column) for column in zip(*measurements, strict=True)]
    mean_fps = [fmean(column) for column in columns]
    if summary is not None:
        for method, calculated, provided in zip(methods, mean_fps, summary, strict=True):
            tolerance = max(1e-6, abs(calculated) * 1e-9)
            if abs(calculated - provided) > tolerance:
                raise ValueError(
                    f"Summary value for {method!r} does not match the motion-row mean"
                )

    first_quartiles: list[float] = []
    third_quartiles: list[float] = []
    for column in columns:
        first_quartile, _median_value, third_quartile = quantiles(
            column, n=4, method="inclusive"
        )
        first_quartiles.append(first_quartile)
        third_quartiles.append(third_quartile)

    return methods, mean_fps, first_quartiles, third_quartiles, len(measurements)


def format_fps(value: float) -> str:
    if value >= 100:
        return f"{value:,.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def create_chart(
    methods: list[str],
    mean_fps: list[float],
    first_quartiles: list[float],
    third_quartiles: list[float],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.titleweight": "normal",
            "axes.labelsize": 12,
            "axes.labelweight": "normal",
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 11.5,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
        }
    )

    rows = sorted(
        zip(methods, mean_fps, first_quartiles, third_quartiles, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    sorted_methods = [DISPLAY_NAMES.get(row[0], row[0]) for row in rows]
    sorted_fps = [row[1] for row in rows]
    sorted_first_quartiles = [row[2] for row in rows]
    sorted_third_quartiles = [row[3] for row in rows]
    fastest = sorted_fps[0]
    positions = list(range(len(rows)))

    color_map = LinearSegmentedColormap.from_list(
        "speed_gradient",
        ["#278C82", "#D09A57", "#B45F5F"],
    )
    colors = [color_map(index / (len(rows) - 1)) for index in positions]

    figure, axis = plt.subplots(figsize=(9.2, 4.8), facecolor="white")
    axis.set_facecolor("white")
    axis.barh(
        positions,
        [value - 1 for value in sorted_fps],
        left=1,
        height=0.58,
        color=colors,
        edgecolor="none",
        zorder=3,
    )

    for position, first_quartile, third_quartile in zip(
        positions,
        sorted_first_quartiles,
        sorted_third_quartiles,
        strict=True,
    ):
        axis.hlines(
            position,
            first_quartile,
            third_quartile,
            color="#202428",
            linewidth=1.5,
            zorder=5,
        )
        axis.vlines(
            [first_quartile, third_quartile],
            position - 0.11,
            position + 0.11,
            color="#202428",
            linewidth=1.5,
            zorder=5,
        )

    for position, value, third_quartile in zip(
        positions, sorted_fps, sorted_third_quartiles, strict=True
    ):
        slowdown = fastest / value
        comparison = "fastest" if slowdown < 1.0005 else f"{slowdown:,.1f}× slower"
        axis.text(
            max(value, third_quartile) * 1.16,
            position,
            f"{format_fps(value)} FPS  ·  {comparison}",
            va="center",
            ha="left",
            fontsize=10.5,
            color="#263238",
            fontweight="normal",
        )

    axis.set_xscale("log")
    axis.set_xlim(1, fastest * 5.0)
    axis.set_yticks(positions, sorted_methods)
    for tick_label, method in zip(axis.get_yticklabels(), sorted_methods, strict=True):
        if "\n" in method:
            tick_label.set_fontsize(10.5)
            tick_label.set_linespacing(1.0)
            tick_label.set_multialignment("center")
    axis.invert_yaxis()
    axis.set_xlabel("Mean FPS (log scale)", labelpad=10)

    axis.xaxis.set_major_locator(LogLocator(base=10))
    axis.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: f"{value:,.0f}")
    )
    axis.xaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.grid(axis="x", which="major", color="#D9DEE3", linewidth=0.8, zorder=0)
    axis.tick_params(axis="y", length=0, pad=9)
    axis.tick_params(axis="x", which="major", length=4, color="#A7AFB7")
    axis.tick_params(axis="x", which="minor", length=0)

    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color("#B8C0C7")
    axis.spines["bottom"].set_linewidth(0.8)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight", pad_inches=0.12, dpi=220)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.suffix.lower() not in {".pdf", ".png"}:
        raise SystemExit("Output must use a .pdf or .png extension")

    try:
        methods, mean_fps, first_quartiles, third_quartiles, motion_count = (
            load_fps_statistics(args.input_csv)
        )
        create_chart(
            methods,
            mean_fps,
            first_quartiles,
            third_quartiles,
            args.output,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    fastest = max(mean_fps)
    print(f"Loaded {motion_count} motion rows; excluded the blank summary row")
    for method, value in sorted(
        zip(methods, mean_fps, strict=True), key=lambda item: item[1], reverse=True
    ):
        slowdown = fastest / value
        comparison = "fastest" if slowdown < 1.0005 else f"{slowdown:.1f}x slower"
        print(f"{method}: mean {format_fps(value)} FPS ({comparison})")
    print("Error bars show the 25th--75th percentile range")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
