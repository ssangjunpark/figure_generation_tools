#!/usr/bin/env python3
"""Plot per-motion FPS grouped by LAFAN1 action category."""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
from pathlib import Path
from statistics import fmean, median


DEFAULT_METHOD = "FRDV No Refine (20)"

CATEGORY_COLORS = (
    "#4D8FB8",
    "#A8C4E2",
    "#E58B43",
    "#52A860",
    "#9DCB8D",
    "#F09A91",
    "#8161A3",
    "#9C6A5B",
    "#C28C7E",
    "#B55496",
    "#92999D",
    "#BEB5AF",
    "#D3CF78",
    "#3FAFBA",
    "#96D1DB",
)


def motion_category(motion_name: str) -> str:
    """Convert, for example, fallAndGetUp2_subject3 to Fall And Get Up."""
    match = re.fullmatch(r"([A-Za-z]+)\d+_subject\d+", motion_name)
    if match is None:
        raise ValueError(f"Cannot determine an action category from {motion_name!r}")

    words = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", match.group(1))
    return words.title()


def load_category_fps(
    csv_path: Path, method_name: str
) -> tuple[list[str], list[list[float]], int]:
    """Load one method's real motion rows and group them by action category."""
    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{csv_path} is empty") from exc

        if len(header) < 2 or header[0].strip() != "Motion":
            raise ValueError("The first CSV column must be named 'Motion'")
        try:
            method_index = [cell.strip() for cell in header].index(method_name)
        except ValueError as exc:
            raise ValueError(f"Method {method_name!r} is not present in the CSV") from exc

        grouped_values: dict[str, list[float]] = {}
        all_values: list[float] = []
        summary_value: float | None = None

        for line_number, row in enumerate(reader, start=2):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) != len(header):
                raise ValueError(
                    f"Line {line_number} has {len(row)} columns; expected {len(header)}"
                )

            try:
                value = float(row[method_index])
            except ValueError as exc:
                raise ValueError(f"Line {line_number} contains a non-numeric FPS value") from exc
            if value <= 0:
                raise ValueError(f"Line {line_number} contains a non-positive FPS value")

            motion_name = row[0].strip()
            if motion_name:
                category = motion_category(motion_name)
                grouped_values.setdefault(category, []).append(value)
                all_values.append(value)
            elif summary_value is None:
                summary_value = value
            else:
                raise ValueError("The CSV contains more than one blank summary row")

    if not all_values:
        raise ValueError("The CSV contains no motion measurements")
    if summary_value is not None:
        calculated_mean = fmean(all_values)
        tolerance = max(1e-6, abs(calculated_mean) * 1e-9)
        if abs(calculated_mean - summary_value) > tolerance:
            raise ValueError("The blank summary value does not match the motion-row mean")

    return list(grouped_values), list(grouped_values.values()), len(all_values)


def create_chart(
    categories: list[str],
    grouped_values: list[list[float]],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.labelweight": "normal",
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 10.5,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
        }
    )

    positions = list(range(len(categories)))
    figure, axis = plt.subplots(figsize=(13.2, 4.8), facecolor="white")
    axis.set_facecolor("white")
    random_generator = random.Random(20260722)

    for position, values, color in zip(
        positions,
        grouped_values,
        (CATEGORY_COLORS[index % len(CATEGORY_COLORS)] for index in positions),
        strict=True,
    ):
        jittered_positions = [
            position + random_generator.uniform(-0.13, 0.13) for _ in values
        ]
        axis.scatter(
            jittered_positions,
            values,
            s=34,
            color=color,
            alpha=0.88,
            edgecolors="white",
            linewidths=0.45,
            zorder=3,
        )
        axis.hlines(
            median(values),
            position - 0.22,
            position + 0.22,
            color="#202428",
            linewidth=2.2,
            zorder=5,
        )

    flattened_values = [value for values in grouped_values for value in values]
    lower_limit = math.floor(min(flattened_values) / 500) * 500 - 250
    upper_limit = math.ceil(max(flattened_values) / 500) * 500 + 250

    axis.set_xlim(-0.65, len(categories) - 0.35)
    axis.set_ylim(lower_limit, upper_limit)
    axis.set_xticks(
        positions,
        categories,
        rotation=27,
        ha="right",
        rotation_mode="anchor",
    )
    axis.set_ylabel("FPS", labelpad=10)

    axis.yaxis.set_major_locator(MultipleLocator(1_000))
    axis.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: f"{value:,.0f}")
    )
    axis.grid(
        axis="y",
        which="major",
        color="#D9DEE3",
        linestyle="--",
        linewidth=0.8,
        zorder=0,
    )
    axis.tick_params(axis="x", length=0, pad=8)
    axis.tick_params(axis="y", length=4, color="#A7AFB7")

    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color("#B8C0C7")
        axis.spines[side].set_linewidth(0.8)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight", pad_inches=0.12, dpi=220)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--method",
        default=DEFAULT_METHOD,
        help=f"CSV method column to plot (default: {DEFAULT_METHOD!r})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.suffix.lower() not in {".pdf", ".png"}:
        raise SystemExit("Output must use a .pdf or .png extension")

    try:
        categories, grouped_values, motion_count = load_category_fps(
            args.input_csv, args.method
        )
        create_chart(categories, grouped_values, args.output)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(
        f"Plotted {motion_count} motions from {args.method!r} "
        f"across {len(categories)} action categories"
    )
    print("Black horizontal markers show category medians")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
