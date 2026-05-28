"""Planned 3D baseball feature extraction surface."""

from __future__ import annotations

from baseball_pose.pose3d.schema import Pose3DRecord


def summarize_pose3d_feature_inventory(records: list[Pose3DRecord]) -> list[str]:
    """Return the planned first-wave 3D feature names for documentation and CLI previews."""

    del records
    return [
        "pelvis_rotation_3d_deg",
        "shoulder_rotation_3d_deg",
        "hip_shoulder_separation_3d_deg",
        "trunk_tilt_3d_deg",
        "lead_knee_lift_3d",
        "hand_depth_excursion",
        "center_of_mass_z_proxy",
    ]
