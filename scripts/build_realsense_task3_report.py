"""Build Task 3 RealSense D435 lab-capture comparison report."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/baseball_mpl_cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Polygon, Rectangle


OUTPUT_DIR = Path("reports")
FIGURE_DIR = OUTPUT_DIR / "figures"


METRIC_ROWS = [
    ("Estimated Bat Speed", "batting", "possible if bat is reconstructed/tracked in 3D; otherwise RGB/object proxy", "2D proxy only without calibration", "D435 can recover metric scale only when bat endpoints are segmented/tracked in calibrated 3D."),
    ("Swing Speed", "batting", "3D bat endpoint speed or wrist/hand speed proxy", "2D pixel/normalized proxy", "D435 is better for physical speed; phone needs scale/camera calibration."),
    ("Hip Rotation", "batting", "reliable 3D pelvis yaw range", "view-dependent 2D/monocular 3D proxy", "D435 reduces perspective ambiguity."),
    ("Hip-Shoulder Sep", "both", "reliable 3D hip/shoulder yaw difference", "usable but orientation drift/view dependent", "Clear geometry for both, stronger with depth."),
    ("Weight Transfer", "both", "pelvis/COM trajectory proxy; better if fused with force plate", "not reliable from monocular video", "D435 gives metric 3D displacement but not true force transfer."),
    ("Lead Knee Angle", "both", "reliable 3D anatomical angle", "usable when view is favorable; poor with side occlusion", "Depth resolves side/front projection ambiguity."),
    ("Trunk Tilt / Lean", "both", "reliable 3D torso-vs-vertical angle", "usable but camera-pitch dependent", "Needs gravity/vertical definition in both systems."),
    ("Contact Time", "batting", "not reliable unless ball and bat contact are visible/tracked at high fps", "not reliable", "D435 frame rate may still be too low for short bat-ball contact."),
    ("Attack Angle", "batting", "3D bat-axis angle if bat endpoints are tracked", "often image-plane angle only", "D435 helps only after robust bat segmentation/tracking."),
    ("Head Stability", "both", "3D head drift relative to pelvis/stride line", "2D/root-relative proxy", "D435 provides metric displacement but still needs phase definition."),
    ("Elbow Bend", "pitching", "reliable 3D joint angle", "usable but occlusion/view dependent", "Clear three-point joint angle."),
    ("Arm Abduction", "pitching", "reliable 3D upper-arm vs torso angle", "often inaccurate from side/back views", "Depth strongly improves arm-slot interpretation."),
    ("Stride Angle", "pitching", "3D foot/hip geometry at landing", "2D proxy, sensitive to camera angle", "D435 needs landing-frame detector."),
    ("Stride Length", "pitching", "metric foot displacement or height-normalized length", "height-normalized 2D proxy", "D435 can output real distance in mm if calibrated."),
    ("Foot Direction", "pitching", "possible if foot/toe orientation is visible in RGB/depth", "often poor without toe keypoints", "D435 still needs a toe/foot orientation model."),
    ("Wrist Snap", "pitching", "wrist-hand angle change proxy; fingertip model preferred", "wrist/hand proxy only", "D435 depth helps but does not create missing fingertip landmarks."),
    ("Arm Speed", "pitching", "3D wrist/hand speed", "monocular unit speed or pixel speed", "D435 provides metric speed after synchronization/calibration."),
    ("Fingertip Speed", "pitching", "possible with fingertip keypoints or hand model", "usually unavailable/proxy", "D435 RGB/depth alone needs a hand-pose model."),
    ("Ball Speed", "pitching", "possible but difficult: small fast ball needs high fps and robust detection", "2D px/s proxy without calibration", "D435 is not a dedicated high-speed ball-tracking system."),
]


COMPARISON_ROWS = [
    ("Deployment complexity", "High: fixed tripods, calibration target, sync wiring or timestamp alignment", "Low: one phone on tripod or handheld"),
    ("Environment robustness", "Good in controlled indoor lab; must manage IR interference and reflective surfaces", "Strongly depends on light, background, motion blur, and occlusion"),
    ("Spatial information", "True metric 3D depth/point cloud after calibration", "2D projection or monocular 3D estimate with depth ambiguity"),
    ("Time synchronization", "Needs hardware sync or software timestamp alignment across cameras", "Single stream, no multi-view sync issue"),
    ("Processing pipeline", "Depth/RGB capture, calibration, point-cloud fusion, pose fitting, coordinate transforms", "Video decode, 2D pose/object detection, optional monocular 3D lifting"),
    ("Angle/distance accuracy", "Higher for joint angles and distances when depth is valid", "Lower, especially for out-of-plane motion"),
    ("Occlusion handling", "Better with three views, still fails under self-occlusion or depth holes", "Single-view occlusion is a major limitation"),
    ("Fast baseball motions", "Better 3D geometry; still limited by frame rate and exposure for ball/bat contact", "Easier setup but motion blur and 2D ambiguity are severe"),
    ("Cost", "Higher: 3 cameras, mounts, sync/USB host, calibration effort", "Lower: phone plus tripod"),
    ("Best use case", "Controlled lab measurement, repeatable athlete testing, 3D joint/segment metrics", "Field-friendly screening, quick visual feedback, simple qualitative comparison"),
]


SOURCES = [
    ("Intel RealSense D435 product page", "https://www.intelrealsense.com/depth-camera-d435/"),
    ("Intel RealSense D400 series datasheet", "https://dev.intelrealsense.com/docs/intel-realsense-d400-series-product-family-datasheet"),
    ("Intel RealSense D400 external synchronization guide", "https://dev.intelrealsense.com/docs/external-synchronization-of-intel-realsense-depth-cameras"),
    ("Intel RealSense SDK 2.0 projection documentation", "https://dev.intelrealsense.com/docs/projection-in-intel-realsense-sdk-20"),
    ("Intel RealSense librealsense examples", "https://github.com/IntelRealSense/librealsense/tree/master/examples"),
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    draw_lab_layout(FIGURE_DIR / "realsense_3cam_lab_layout.png")
    draw_depth_pointcloud_example(FIGURE_DIR / "realsense_depth_pointcloud_example.png")
    write_metric_csv(OUTPUT_DIR / "realsense_slymask_metric_comparison.csv")
    write_report(OUTPUT_DIR / "realsense_d435_lab_comparison.md")
    print(OUTPUT_DIR / "realsense_d435_lab_comparison.md")
    print(OUTPUT_DIR / "realsense_slymask_metric_comparison.csv")


def draw_lab_layout(path: Path) -> None:
    fig, (top_ax, side_ax) = plt.subplots(1, 2, figsize=(14, 6), dpi=160)
    for ax in (top_ax, side_ax):
        ax.set_aspect("equal")
        ax.axis("off")

    top_ax.set_title("Top View: 3 x D435 Lab Capture")
    top_ax.add_patch(Rectangle((-3, -2), 6, 4, fill=False, linewidth=1.5))
    top_ax.add_patch(Rectangle((-1.2, -0.55), 2.4, 1.1, facecolor="#f3f3f3", edgecolor="black", label="hitting / pitching zone"))
    top_ax.text(0, 0, "athlete\nactivity zone", ha="center", va="center", fontsize=9)

    cameras = [(-2.5, -1.5, 35, "D435-L"), (0, 1.8, -90, "D435-C"), (2.5, -1.5, 145, "D435-R")]
    for x, y, angle, label in cameras:
        top_ax.add_patch(Circle((x, y), 0.12, color="#1f77b4"))
        top_ax.text(x, y + 0.22, label, ha="center", fontsize=9)
        wedge = Polygon([(x, y), (x * 0.25, y * 0.25), (x * -0.15, y * -0.15)], closed=True, alpha=0.14, color="#1f77b4")
        top_ax.add_patch(wedge)
        top_ax.add_patch(Arc((x, y), 1.0, 1.0, angle=angle, theta1=-30, theta2=30, color="#1f77b4"))
    top_ax.annotate("baseline / calibration board positions", xy=(-1.2, -2.15), xytext=(-1.2, -2.15), fontsize=8)
    top_ax.set_xlim(-3.4, 3.4)
    top_ax.set_ylim(-2.4, 2.4)

    side_ax.set_title("Side View: Height and Coverage")
    side_ax.plot([-3, 3], [0, 0], color="black", linewidth=1)
    side_ax.add_patch(Rectangle((-0.7, 0), 1.4, 1.8, facecolor="#f2f2f2", edgecolor="black"))
    side_ax.text(0, 0.95, "athlete", ha="center", va="center", fontsize=9)
    side_ax.add_patch(Rectangle((-2.6, 1.1), 0.35, 0.18, color="#1f77b4"))
    side_ax.add_patch(Rectangle((2.25, 1.1), 0.35, 0.18, color="#1f77b4"))
    side_ax.text(-2.42, 1.45, "side D435\n~1.0-1.5 m high", ha="center", fontsize=8)
    side_ax.text(2.42, 1.45, "side D435\n~1.0-1.5 m high", ha="center", fontsize=8)
    side_ax.plot([-2.25, -0.2], [1.2, 1.4], color="#1f77b4", alpha=0.6)
    side_ax.plot([2.25, 0.2], [1.2, 1.4], color="#1f77b4", alpha=0.6)
    side_ax.text(0, -0.25, "floor plane / shared world coordinate system", ha="center", fontsize=8)
    side_ax.set_xlim(-3.2, 3.2)
    side_ax.set_ylim(-0.4, 2.2)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def draw_depth_pointcloud_example(path: Path) -> None:
    x = np.linspace(-1.2, 1.2, 160)
    y = np.linspace(0, 2.0, 180)
    xx, yy = np.meshgrid(x, y)
    depth = 2.2 + 0.25 * yy + 0.25 * np.exp(-((xx / 0.45) ** 2 + ((yy - 1.0) / 0.7) ** 2))

    skeleton = np.array(
        [
            [0.0, 1.75],
            [0.0, 1.45],
            [-0.35, 1.25],
            [-0.55, 0.85],
            [-0.45, 0.15],
            [0.35, 1.25],
            [0.65, 0.95],
            [0.9, 1.25],
            [-0.22, 1.0],
            [-0.35, 0.55],
            [-0.30, 0.05],
            [0.22, 1.0],
            [0.35, 0.55],
            [0.30, 0.05],
        ]
    )
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (8, 11)]

    fig, (depth_ax, cloud_ax) = plt.subplots(1, 2, figsize=(13, 5), dpi=160)
    im = depth_ax.imshow(depth, cmap="viridis", origin="lower", extent=[-1.2, 1.2, 0, 2.0])
    depth_ax.set_title("Depth Map Example (schematic)")
    depth_ax.set_xlabel("image x")
    depth_ax.set_ylabel("image y")
    fig.colorbar(im, ax=depth_ax, fraction=0.046, pad=0.04, label="depth (m)")
    for a, b in edges:
        depth_ax.plot([skeleton[a, 0], skeleton[b, 0]], [skeleton[a, 1], skeleton[b, 1]], color="white", linewidth=2)
    depth_ax.scatter(skeleton[:, 0], skeleton[:, 1], color="red", s=12)

    cloud_ax = fig.add_subplot(1, 2, 2, projection="3d")
    sample = np.linspace(0, len(x) * len(y) - 1, 1500, dtype=int)
    flat_x = xx.ravel()[sample]
    flat_y = yy.ravel()[sample]
    flat_z = depth.ravel()[sample]
    cloud_ax.scatter(flat_x, flat_y, flat_z, s=1, alpha=0.18, color="#1f77b4")
    joints_3d = np.column_stack([skeleton[:, 0], skeleton[:, 1], 2.4 + 0.12 * skeleton[:, 1]])
    for a, b in edges:
        cloud_ax.plot(*zip(joints_3d[a], joints_3d[b]), color="red", linewidth=2)
    cloud_ax.scatter(joints_3d[:, 0], joints_3d[:, 1], joints_3d[:, 2], color="black", s=12)
    cloud_ax.set_title("Point Cloud + 3D Skeleton Example")
    cloud_ax.set_xlabel("X (m)")
    cloud_ax.set_ylabel("Y (m)")
    cloud_ax.set_zlabel("Z depth (m)")
    cloud_ax.view_init(elev=22, azim=-55)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_metric_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["metric", "task_context", "three_d435_capability", "phone_cv_capability", "note"])
        writer.writerows(METRIC_ROWS)


def write_report(path: Path) -> None:
    lines = [
        "# Task 3: Intel RealSense D435 Lab Capture vs Phone Video CV",
        "",
        "Scope: evaluate a 3-camera Intel RealSense D435 laboratory setup for baseball motion analysis and compare it with single-phone video plus CV/pose estimation.",
        "",
        "Important device note: `D435` provides RGB, stereo infrared/depth, and point-cloud data through the RealSense SDK. It does not include an IMU; that is the `D435i` variant.",
        "",
        "## 1. Three-D435 Lab Capture Plan",
        "",
        "### Hardware Deployment",
        "",
        "![3-camera D435 lab layout](figures/realsense_3cam_lab_layout.png)",
        "",
        "D435 hardware summary for planning:",
        "",
        "| item | practical meaning for this project |",
        "|---|---|",
        "| Active infrared stereo depth | provides metric depth and point cloud, but multiple cameras can interfere through projected IR patterns |",
        "| Global-shutter stereo imagers | better for fast motion than rolling-shutter stereo, but exposure and frame rate still matter |",
        "| Depth / IR / RGB streams | body pose can use RGB, depth can map 2D joints into 3D, and IR/depth can help segmentation |",
        "| No IMU on D435 | acceleration/gyro are unavailable unless using D435i; do not claim IMU-based motion metrics from D435 |",
        "| USB host bandwidth | three cameras need careful stream-resolution selection and USB-controller planning |",
        "",
        "- Use three fixed cameras around the athlete: left-front/side, center/front or back, and right-front/side. The exact angles should be tuned to the batting or pitching direction.",
        "- Keep all cameras on rigid mounts. Put the athlete action zone inside the overlapping depth field of view, not merely the RGB field of view.",
        "- Start with camera height around the athlete's torso level, then adjust to keep hands, bat, stride foot, and release/contact region inside the common volume.",
        "- Use a calibration board or AprilTag/Charuco board at multiple positions across the activity volume for extrinsic calibration.",
        "",
        "### Synchronization and Calibration",
        "",
        "- Preferred synchronization: hardware sync wiring across the D400 cameras, with one master and two slaves, then validate frame timestamps in software.",
        "- Fallback synchronization: software timestamp alignment. This is easier but less safe for high-speed baseball motions because one-frame timing error changes speed and release/contact metrics.",
        "- Intrinsics: use factory intrinsics from RealSense SDK as a starting point, then verify RGB-depth alignment and depth scale.",
        "- Extrinsics: estimate each camera pose in a shared lab coordinate system using a calibration target. Transform depth points and 3D joints into that shared coordinate system.",
        "- Multi-camera depth: RealSense depth uses active infrared stereo; overlapping projectors can interfere. In practice, test emitter on/off, emitter power, camera angles, and temporal staggering if depth noise appears.",
        "",
        "### Best Deployment Guidelines",
        "",
        "- Use an indoor controlled-light lab; avoid direct sunlight and reflective/transparent surfaces.",
        "- Leave overlap between all cameras but avoid placing cameras so close that one camera's IR pattern saturates another.",
        "- Mark the athlete action zone on the floor and keep the calibration board coverage larger than that zone.",
        "- Capture a short calibration/quality clip before each session: static athlete pose, slow arm swing, and a known-distance object.",
        "- For bat/ball metrics, D435 is not enough by itself: you still need robust object detection/segmentation and may need higher-speed cameras for ball-contact timing.",
        "",
        "## 2. Data Outputs and Metric Extraction",
        "",
        "D435 raw outputs and derived data:",
        "",
        "| data type | available from D435? | use in pose/baseball analysis | limitation |",
        "|---|---|---|---|",
        "| RGB image | yes | 2D pose, bat/ball detection, visual QA | motion blur and lighting still matter |",
        "| Stereo infrared images | yes | depth computation, low-texture support | IR interference across cameras can reduce quality |",
        "| Depth map | yes | metric 3D position, subject segmentation, occlusion reasoning | holes/noise on reflective, distant, or fast-moving regions |",
        "| Point cloud | yes, derived from depth | 3D skeleton fitting, body/bat spatial reconstruction | needs extrinsic calibration for multi-view fusion |",
        "| IMU | no for D435 | unavailable unless using D435i | do not assume acceleration/gyro data from D435 |",
        "| 3D joints | derived, not native | OpenPose/RTMPose + depth lifting, RGB-D pose models, SMPL fitting | joint quality depends on pose model and depth association |",
        "",
        "![Depth and point cloud example](figures/realsense_depth_pointcloud_example.png)",
        "",
        "### SlyMask Metrics Alignment",
        "",
        "| metric | context | 3 x D435 capability | phone CV capability | note |",
        "|---|---|---|---|---|",
    ]
    for metric, context, d435, phone, note in METRIC_ROWS:
        lines.append(f"| {metric} | {context} | {d435} | {phone} | {note} |")
    lines.extend(
        [
            "",
            "## 3. D435 vs Phone Video CV",
            "",
            "| comparison dimension | 3 x D435 | single phone + CV |",
            "|---|---|---|",
        ]
    )
    for dimension, d435, phone in COMPARISON_ROWS:
        lines.append(f"| {dimension} | {d435} | {phone} |")
    lines.extend(
        [
            "",
            "## 4. Recommendation",
            "",
            "- Use the 3-D435 lab setup when the goal is repeatable measurement: real 3D joint angles, metric distances, stride length, head/pelvis displacement, and controlled comparison across athletes/sessions.",
            "- Use phone video CV when the goal is low-friction field feedback: quick overlay review, coarse posture metrics, and product demo workflows.",
            "- For the current SlyMask-style metric set, D435 is most valuable for geometry-heavy body metrics: hip-shoulder separation, trunk lean, lead knee, elbow bend, arm abduction, stride length, and head stability.",
            "- D435 alone does not solve every baseball metric. Bat speed, ball speed, attack angle, and contact time still need reliable bat/ball tracking, calibration, and possibly higher frame-rate cameras.",
            "- A pragmatic lab protocol is: D435 for calibrated body kinematics, synchronized RGB for visual QA, and an optional high-speed side camera for bat-ball contact and ball velocity validation.",
            "",
            "## Sources",
            "",
        ]
    )
    for label, url in SOURCES:
        lines.append(f"- [{label}]({url})")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
