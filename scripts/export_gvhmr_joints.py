"""Export GVHMR demo predictions to the baseball-pose 3D joint CSV contract.

Run this from an environment where GVHMR and its model files are installed.
The script intentionally lives in this repo so the handoff format stays under
version control, while GVHMR itself can remain an external research dependency.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


SMPL_JOINT_NAMES = (
    "hip",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
)


def main() -> None:
    args = _parse_args()
    gvhmr_root = args.gvhmr_root.resolve()
    sys.path.insert(0, str(gvhmr_root))

    import torch
    from einops import einsum
    from hmr4d.utils.smplx_utils import make_smplx
    from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    pred = torch.load(args.result, map_location=device)
    smpl_params = pred.get(args.smpl_params_key)
    if smpl_params is None:
        raise KeyError(f"Missing {args.smpl_params_key!r} in {args.result}")

    smplx = make_smplx("supermotion").to(device)
    smplx2smpl_path = gvhmr_root / "hmr4d" / "utils" / "body_model" / "smplx2smpl_sparse.pt"
    regressor_path = gvhmr_root / "hmr4d" / "utils" / "body_model" / "smpl_neutral_J_regressor.pt"
    smplx2smpl = torch.load(smplx2smpl_path, map_location=device)
    joint_regressor = torch.load(regressor_path, map_location=device)

    with torch.no_grad():
        smplx_out = smplx(**_to_device(smpl_params, device))
        verts = torch.stack([torch.matmul(smplx2smpl, vertices) for vertices in smplx_out.vertices])
        if args.face_z:
            verts = _move_to_start_point_face_z(verts, joint_regressor, einsum, apply_T_on_points, compute_T_ayfz2ay)
        joints = einsum(joint_regressor, verts, "j v, l v i -> l j i").detach().cpu()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "clip_id",
                "frame_index",
                "timestamp_sec",
                "joint_name",
                "x_3d",
                "y_3d",
                "z_3d",
                "confidence",
                "scale_mode",
                "lift_backend",
            ),
        )
        writer.writeheader()
        for frame_index in range(joints.shape[0]):
            timestamp_sec = frame_index / args.fps
            for joint_index, joint_name in enumerate(SMPL_JOINT_NAMES[: joints.shape[1]]):
                writer.writerow(
                    {
                        "clip_id": args.clip_id,
                        "frame_index": frame_index,
                        "timestamp_sec": timestamp_sec,
                        "joint_name": joint_name,
                        "x_3d": float(joints[frame_index, joint_index, 0]),
                        "y_3d": float(joints[frame_index, joint_index, 1]),
                        "z_3d": float(joints[frame_index, joint_index, 2]),
                        "confidence": "",
                        "scale_mode": "gvhmr_smpl_world",
                        "lift_backend": "gvhmr",
                    }
                )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gvhmr-root", type=Path, default=Path("external/GVHMR"))
    parser.add_argument("--result", type=Path, required=True, help="Path to GVHMR hmr4d_results.pt.")
    parser.add_argument("--output", type=Path, required=True, help="Output 3D joint CSV path.")
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--smpl-params-key", default="smpl_params_global")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--face-z", action="store_true", help="Normalize the first frame to face +Z, as in GVHMR demo rendering.")
    return parser.parse_args()


def _to_device(value, device: str):
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if hasattr(value, "to"):
        return value.to(device)
    return value


def _move_to_start_point_face_z(verts, joint_regressor, einsum, apply_T_on_points, compute_T_ayfz2ay):
    offset = einsum(joint_regressor, verts[0], "j v, v i -> j i")[0]
    offset[1] = verts[:, :, [1]].min()
    verts = verts - offset
    transform = compute_T_ayfz2ay(
        einsum(joint_regressor, verts[[0]], "j v, l v i -> l j i"),
        inverse=True,
    )
    return apply_T_on_points(verts, transform)


if __name__ == "__main__":
    main()
