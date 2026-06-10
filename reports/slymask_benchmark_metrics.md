# SlyMask Benchmark Metrics

Source decision: body kinematics use GVHMR/SMPL24 3D because hip/shoulder rotation, trunk tilt, stride direction, and COM shift are angle/view dependent in 2D. Bat/ball metrics use the existing 2D object pipeline because there is no 3D bat/ball reconstruction.

Capability boundary: SlyMask-style percentile and reliability scores require a proprietary or population reference distribution. This report outputs raw values or proxy values and marks unsupported outputs explicitly.

## Preliminary Conclusions

### Trustworthy enough for first-pass comparison

- Hip-Shoulder Sep: geometric definition is clear in 3D, using projected hip and shoulder lines. It is suitable for relative comparison between clips, but still depends on GVHMR orientation stability.
- Lead Knee Angle / Elbow Bend / Arm Abduction / Trunk Tilt: these are direct joint or torso angles from SMPL24. They have clear geometry and are the most defensible body metrics in the current pipeline.
- Hip Rotation: usable as a 3D pelvis-yaw range, especially for within-clip or same-camera comparisons.
- Stride Length: usable as a height-normalized foot-separation proxy. It is not exactly SlyMask's proprietary definition but the biomechanical meaning is clear.
- Ball Speed in pitching: usable only as 2D px/s for tracking QA and relative comparison inside the same video setup. It is not a physical speed.

### Usable only as proxy

- Weight Transfer / Head Stability: these rely on inferred stride direction and a simple hip-center/head-drift model. They are useful for automation experiments, but should not be treated as validated coaching scores yet.
- Stride Angle / Foot Direction: the outputs are geometric, but landing-frame and toe-direction inference are approximate. SMPL24 has a foot marker, not a real toe orientation model.
- Swing Speed / Estimated Bat Speed / Attack Angle: current values come from the 2D object tracker. They are useful for debugging bat tracking, but without calibration they cannot be reported as km/h or true 3D attack angle.
- Wrist Snap / Fingertip Speed: SMPL24 has wrist/hand joints but no fingertip joints, so these are hand/wrist proxies only.

### Clearly unreasonable or not actionable yet

- SlyMask-style percentiles and reliability percentages cannot be reproduced from our pipeline alone because we do not have their reference population or reliability model.
- Contact Time is unavailable for the current batting benchmark because there is no ball track and no bat-ball impact event detector.
- Values pinned near 0% or 100% for Weight Transfer should be treated as a warning sign, not as a coaching result. The current COM proxy can saturate when stride direction or event timing is imperfect.
- `benchmark_pitch_vertical_09` has landing-frame 0, so its stride/lead-knee/foot-direction landing metrics are likely not meaningful; the clip starts too late or the automatic landing detector lacks enough pre-landing frames.
- `benchmark_hit_horizontal_06` reports Wrist/Hand Speed near zero at the bat peak-speed frame, which means bat peak and body wrist-speed event are not aligned. That metric is not reliable for this clip without better contact/release event logic.

### Concrete suspicious outputs in this run

- benchmark_pitch_vertical_09: Lead Knee Angle uses frame 0 as landing frame, so this landing-phase metric is weak.
- benchmark_pitch_vertical_09: Stride Angle uses frame 0 as landing frame, so this landing-phase metric is weak.
- benchmark_pitch_vertical_09: Stride Length uses frame 0 as landing frame, so this landing-phase metric is weak.
- benchmark_pitch_vertical_09: Foot Direction uses frame 0 as landing frame, so this landing-phase metric is weak.
- benchmark_hit_vertical_02: Attack Angle is -57.185 deg from image-plane bat tracking; this is not a credible true attack angle.
- benchmark_hit_horizontal_06: Weight Transfer is 0.000%, likely saturated by the current COM/stride proxy.
- benchmark_hit_horizontal_06: Wrist/Hand Speed is 0.130 3d_unit/s at bat peak-speed frame, indicating event mismatch.

## benchmark_pitch_vertical_10

| metric | value | unit | status | source | frame | reason |
|---|---:|---|---|---|---:|---|
| Hip-Shoulder Sep | 28.738 | deg | available | 3d_pose | 27 | SMPL24 hip/shoulder lines projected to horizontal plane. |
| Lead Knee Angle | 120.714 | deg | available | 3d_pose | 18 | Lead side inferred as left; value is anatomical knee angle, not flexion-only label. |
| Trunk Tilt | 30.171 | deg | available | 3d_pose | 27 | Torso vector relative to reconstructed vertical axis. |
| Weight Transfer | 84.423 | % | proxy | 3d_pose | 27 | COM proxy is hip center shift along inferred stride direction. |
| Head Stability | 62.472 | % | proxy | 3d_pose | 27 | Score from head drift perpendicular to stride line; no SlyMask reference scale. |
| Dominant Side | right |  | proxy | 3d_pose | 27 | Inferred from larger hand peak speed. |
| Lead Side | left |  | proxy | 3d_pose | 18 | Inferred from foot position along stride direction. |
| Elbow Bend | 78.177 | deg | available | 3d_pose | 27 | Throwing side inferred by peak hand speed. |
| Arm Abduction | 144.806 | deg | available | 3d_pose | 27 | Upper arm angle relative to torso axis. |
| Stride Angle | 3.118 | deg | proxy | 3d_pose | 18 | Angle between foot line and hip line at inferred landing frame. |
| Stride Length | 0.594 | height_ratio | proxy | 3d_pose | 18 | Foot separation normalized by reconstructed body-height proxy. |
| Foot Direction | 54.868 | deg | proxy | 3d_pose | 18 | SMPL24 has foot marker but no toe; this approximates toe direction from ankle-to-foot vector. |
| Wrist Snap | 8.018 | deg | proxy | 3d_pose | 27 | Uses elbow-wrist-hand angle change; no fingertip joint is available. |
| Arm Speed | 3.339 | 3d_unit/s | proxy | 3d_pose | 27 | Raw wrist speed at release proxy; percentile needs a normative database. |
| Fingertip Speed | 3.629 | 3d_unit/s | proxy | 3d_pose | 27 | SMPL24 hand joint is used because fingertip joints are unavailable. |
| Ball Speed | 1619.559 | px/s | proxy | object_2d | 27 | 2D ball speed without camera calibration; not physical mph/km/h. |

## benchmark_pitch_vertical_09

| metric | value | unit | status | source | frame | reason |
|---|---:|---|---|---|---:|---|
| Hip-Shoulder Sep | 54.699 | deg | available | 3d_pose | 19 | SMPL24 hip/shoulder lines projected to horizontal plane. |
| Lead Knee Angle | 160.739 | deg | available | 3d_pose | 0 | Lead side inferred as left; value is anatomical knee angle, not flexion-only label. |
| Trunk Tilt | 39.922 | deg | available | 3d_pose | 19 | Torso vector relative to reconstructed vertical axis. |
| Weight Transfer | 54.660 | % | proxy | 3d_pose | 19 | COM proxy is hip center shift along inferred stride direction. |
| Head Stability | 38.573 | % | proxy | 3d_pose | 19 | Score from head drift perpendicular to stride line; no SlyMask reference scale. |
| Dominant Side | right |  | proxy | 3d_pose | 19 | Inferred from larger hand peak speed. |
| Lead Side | left |  | proxy | 3d_pose | 0 | Inferred from foot position along stride direction. |
| Elbow Bend | 122.727 | deg | available | 3d_pose | 19 | Throwing side inferred by peak hand speed. |
| Arm Abduction | 113.470 | deg | available | 3d_pose | 19 | Upper arm angle relative to torso axis. |
| Stride Angle | 18.956 | deg | proxy | 3d_pose | 0 | Angle between foot line and hip line at inferred landing frame. |
| Stride Length | 0.573 | height_ratio | proxy | 3d_pose | 0 | Foot separation normalized by reconstructed body-height proxy. |
| Foot Direction | 23.024 | deg | proxy | 3d_pose | 0 | SMPL24 has foot marker but no toe; this approximates toe direction from ankle-to-foot vector. |
| Wrist Snap | 10.437 | deg | proxy | 3d_pose | 19 | Uses elbow-wrist-hand angle change; no fingertip joint is available. |
| Arm Speed | 2.477 | 3d_unit/s | proxy | 3d_pose | 19 | Raw wrist speed at release proxy; percentile needs a normative database. |
| Fingertip Speed | 2.672 | 3d_unit/s | proxy | 3d_pose | 19 | SMPL24 hand joint is used because fingertip joints are unavailable. |
| Ball Speed | 2190.378 | px/s | proxy | object_2d | 19 | 2D ball speed without camera calibration; not physical mph/km/h. |

## benchmark_hit_vertical_02

| metric | value | unit | status | source | frame | reason |
|---|---:|---|---|---|---:|---|
| Hip-Shoulder Sep | 51.075 | deg | available | 3d_pose | 106 | SMPL24 hip/shoulder lines projected to horizontal plane. |
| Lead Knee Angle | 132.274 | deg | available | 3d_pose | 73 | Lead side inferred as left; value is anatomical knee angle, not flexion-only label. |
| Trunk Tilt | 22.455 | deg | available | 3d_pose | 106 | Torso vector relative to reconstructed vertical axis. |
| Weight Transfer | 19.102 | % | proxy | 3d_pose | 106 | COM proxy is hip center shift along inferred stride direction. |
| Head Stability | 86.205 | % | proxy | 3d_pose | 106 | Score from head drift perpendicular to stride line; no SlyMask reference scale. |
| Dominant Side | right |  | proxy | 3d_pose | 106 | Inferred from larger hand peak speed. |
| Lead Side | left |  | proxy | 3d_pose | 73 | Inferred from foot position along stride direction. |
| Swing Speed | 8.041 | norm/s | proxy | object_2d | 106 | SlyMask percentile is unavailable; this is normalized 2D bat speed. |
| Estimated Bat Speed | 5920.921 | px/s | proxy | object_2d | 106 | No camera calibration/bat scale, so km/h cannot be recovered. |
| Hip Rotation | 83.922 | deg | available | 3d_pose | 106 | Range of pelvis yaw over the clip. |
| Attack Angle | -57.185 | deg | proxy | object_2d | 106 | Image-plane bat angle at peak bat speed; not true 3D attack angle. |
| Wrist/Hand Speed | 1.460 | 3d_unit/s | proxy | 3d_pose | 106 | Useful internal body-speed proxy; SlyMask swing percentile needs a reference database. |
| Contact Time | N/A |  | unavailable | none | N/A | No ball track in batting benchmark and no bat-ball impact event detector; cannot determine physical contact duration. |

## benchmark_hit_horizontal_06

| metric | value | unit | status | source | frame | reason |
|---|---:|---|---|---|---:|---|
| Hip-Shoulder Sep | 2.436 | deg | available | 3d_pose | 110 | SMPL24 hip/shoulder lines projected to horizontal plane. |
| Lead Knee Angle | 157.620 | deg | available | 3d_pose | 71 | Lead side inferred as left; value is anatomical knee angle, not flexion-only label. |
| Trunk Tilt | 3.052 | deg | available | 3d_pose | 110 | Torso vector relative to reconstructed vertical axis. |
| Weight Transfer | 0.000 | % | proxy | 3d_pose | 110 | COM proxy is hip center shift along inferred stride direction. |
| Head Stability | 50.273 | % | proxy | 3d_pose | 110 | Score from head drift perpendicular to stride line; no SlyMask reference scale. |
| Dominant Side | right |  | proxy | 3d_pose | 110 | Inferred from larger hand peak speed. |
| Lead Side | left |  | proxy | 3d_pose | 71 | Inferred from foot position along stride direction. |
| Swing Speed | 8.064 | norm/s | proxy | object_2d | 110 | SlyMask percentile is unavailable; this is normalized 2D bat speed. |
| Estimated Bat Speed | 7309.942 | px/s | proxy | object_2d | 110 | No camera calibration/bat scale, so km/h cannot be recovered. |
| Hip Rotation | 146.795 | deg | available | 3d_pose | 110 | Range of pelvis yaw over the clip. |
| Attack Angle | -4.272 | deg | proxy | object_2d | 110 | Image-plane bat angle at peak bat speed; not true 3D attack angle. |
| Wrist/Hand Speed | 0.130 | 3d_unit/s | proxy | 3d_pose | 110 | Useful internal body-speed proxy; SlyMask swing percentile needs a reference database. |
| Contact Time | N/A |  | unavailable | none | N/A | No ball track in batting benchmark and no bat-ball impact event detector; cannot determine physical contact duration. |
