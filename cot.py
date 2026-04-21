from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


G = 9.81


@dataclass(frozen=True)
class TrialConfig:
    label: str
    path: Path
    walk_start_s: float


TRIALS = (
    TrialConfig(
        label="toe_walk",
        path=Path(
            "/home/sangjunpark/Documents/figure_generation_tools/bench/toe_walk/toe_walk.npz"
        ),
        walk_start_s=3.0,
    ),
    TrialConfig(
        label="toe_walk_pure",
        path=Path(
            "/home/sangjunpark/Documents/figure_generation_tools/bench/toe_walk_pure/toe_walk_pure.npz"
        ),
        walk_start_s=0.0,
    ),
)


def _mass_kg(data: np.lib.npyio.NpzFile) -> float:
    if "redirection_loss_mass" in data.files:
        return float(data["redirection_loss_mass"])
    raise KeyError("NPZ file does not contain redirection_loss_mass")


def _interp(time: np.ndarray, values: np.ndarray, query_time: float) -> float:
    return float(np.interp(query_time, time, values))


def calculate_cot(
    trial: TrialConfig,
    duration_s: float | None = None,
    distance_m: float | None = None,
) -> dict[str, float | str]:
    if duration_s is not None and distance_m is not None:
        raise ValueError("Use duration_s or distance_m, not both")

    data = np.load(trial.path, allow_pickle=True)

    time = data["cot_time"]
    distance = data["cot_dist_m"]
    positive_joint_work = data["cot_joint_positive_work_j"]
    mass = _mass_kg(data)

    start_time = trial.walk_start_s
    if start_time < float(time[0]) or start_time >= float(time[-1]):
        raise ValueError(f"{trial.label}: start time {start_time} is outside the data")

    start_distance = _interp(time, distance, start_time)
    start_work = _interp(time, positive_joint_work, start_time)

    if duration_s is not None:
        end_time = start_time + duration_s
        if end_time > float(time[-1]):
            raise ValueError(f"{trial.label}: requested duration extends past the data")
    elif distance_m is not None:
        target_distance = start_distance + distance_m
        if target_distance > float(distance[-1]):
            raise ValueError(f"{trial.label}: requested distance extends past the data")
        end_time = float(np.interp(target_distance, distance, time))
    else:
        end_time = float(time[-1])

    end_distance = _interp(time, distance, end_time)
    end_work = _interp(time, positive_joint_work, end_time)

    delta_time = end_time - start_time
    delta_distance = end_distance - start_distance
    delta_work = end_work - start_work

    if delta_distance <= 0.0:
        raise ValueError(
            f"{trial.label}: walking window has no forward distance "
            f"({delta_distance:.6g} m)"
        )

    cot = delta_work / (mass * G * delta_distance)
    average_speed = delta_distance / delta_time if delta_time > 0.0 else float("nan")

    return {
        "label": trial.label,
        "path": str(trial.path),
        "start_s": start_time,
        "end_s": end_time,
        "duration_s": delta_time,
        "distance_m": delta_distance,
        "average_speed_m_s": average_speed,
        "positive_joint_work_j": delta_work,
        "mass_kg": mass,
        "mechanical_cot": cot,
    }


def print_result(title: str, result: dict[str, float | str]) -> None:
    print(title)
    print(f"  trial:                  {result['label']}")
    print(f"  window:                 {result['start_s']:.3f}s to {result['end_s']:.3f}s")
    print(f"  duration:               {result['duration_s']:.3f} s")
    print(f"  distance:               {result['distance_m']:.3f} m")
    print(f"  average speed:          {result['average_speed_m_s']:.3f} m/s")
    print(f"  positive joint work:    {result['positive_joint_work_j']:.3f} J")
    print(f"  mechanical COT:         {result['mechanical_cot']:.3f}")
    print()


def main() -> None:
    print("Mechanical COT = positive joint work / (mass * 9.81 * distance)")
    print()

    print("Full available walking windows")
    full_results = [calculate_cot(trial) for trial in TRIALS]
    for result in full_results:
        print_result("Walking-phase COT", result)

    # Fairness check:
    # COT is work divided by body-weight distance, so matching by distance is the
    # cleanest comparison when the recordings have different walking durations.
    available_distances = [float(result["distance_m"]) for result in full_results]
    matched_distance = min(available_distances)

    print(f"Matched-distance walking windows ({matched_distance:.3f} m each)")
    for trial in TRIALS:
        result = calculate_cot(trial, distance_m=matched_distance)
        print_result("Matched-distance COT", result)


if __name__ == "__main__":
    main()
