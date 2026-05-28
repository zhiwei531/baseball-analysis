"""Default placeholder backend for future temporal 3D lifting."""

from __future__ import annotations

from baseball_pose.pose.schema import PoseRecord
from baseball_pose.pose3d.base import Pose3DLifter
from baseball_pose.pose3d.schema import Pose3DRecord


class TemporalLifterStub(Pose3DLifter):
    """Non-executable backend that documents the intended 3D handoff point."""

    backend_name = "temporal_lifter_stub"

    def lift_sequence(
        self,
        pose_records: list[PoseRecord],
        clip_id: str,
        condition_id: str,
    ) -> list[Pose3DRecord]:
        raise NotImplementedError(
            "3D lifting is planned to consume smoothed 2D pose CSV files, but no"
            " temporal 3D backend is wired in yet. See docs/3d_extension_plan.md."
        )
