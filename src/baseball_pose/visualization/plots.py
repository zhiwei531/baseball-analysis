"""Plotting helpers for report figures."""

from __future__ import annotations

import csv
from pathlib import Path


def figure_name(clip_id: str, condition_id: str, figure_type: str) -> str:
    return f"{clip_id}__{condition_id}__{figure_type}.png"


def plot_wrist_trajectories(
    feature_paths: dict[str, str | Path],
    output_path: str | Path,
    title: str,
) -> None:
    """Plot left/right wrist trajectories for one clip across conditions."""

    import matplotlib.pyplot as plt

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False)
    for condition_id, source_path in sorted(feature_paths.items()):
        rows = _read_rows(source_path)
        for axis, side in zip(axes, ("left", "right")):
            xs = [_optional_float(row[f"{side}_wrist_x"]) for row in rows]
            ys = [_optional_float(row[f"{side}_wrist_y"]) for row in rows]
            points = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
            if not points:
                continue
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                linewidth=1.3,
                label=condition_id,
            )
            axis.scatter(points[0][0], points[0][1], s=18)
            axis.scatter(points[-1][0], points[-1][1], s=18, marker="x")

    for axis, side in zip(axes, ("Left wrist", "Right wrist")):
        axis.set_title(side)
        axis.set_xlabel("x position")
        axis.set_ylabel("y position")
        axis.invert_yaxis()
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(target, dpi=180)
    plt.close(fig)


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _optional_float(value: str) -> float | None:
    if value == "":
        return None
    return float(value)
