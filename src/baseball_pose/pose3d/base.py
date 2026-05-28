"""Interface for 3D lifting backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from baseball_pose.pose.schema import PoseRecord
from baseball_pose.pose3d.schema import Pose3DRecord


class Pose3DLifter(ABC):
    backend_name: str

    @abstractmethod
    def lift_sequence(
        self,
        pose_records: list[PoseRecord],
        clip_id: str,
        condition_id: str,
    ) -> list[Pose3DRecord]:
        """Lift one cleaned 2D pose sequence into a relative 3D joint sequence."""
