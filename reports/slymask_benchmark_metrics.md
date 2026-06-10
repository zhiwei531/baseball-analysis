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

- Head Stability: uses root-relative head drift against an inferred stride direction. It is useful for automation experiments, but should not be treated as a validated coaching score yet.
- Stride Angle / Foot Direction: the outputs are geometric, but landing-frame and toe-direction inference are approximate. SMPL24 has a foot marker, not a real toe orientation model.
- Swing Speed / Estimated Bat Speed / Attack Angle: current values come from the 2D object tracker. They are useful for debugging bat tracking, but without calibration they cannot be reported as km/h or true 3D attack angle.
- Wrist Snap / Fingertip Speed: SMPL24 has wrist/hand joints but no fingertip joints, so these are hand/wrist proxies only.

### Clearly unreasonable or not actionable yet

- SlyMask-style percentiles and reliability percentages cannot be reproduced from our pipeline alone because we do not have their reference population or reliability model.
- Contact Time is unavailable for the current batting benchmark because there is no ball track and no bat-ball impact event detector.
- Weight Transfer is now marked unavailable. The GVHMR global hip/root track is not a calibrated field-coordinate COM trajectory, so previous 0%/100% values were a calculation-definition problem, not a trustworthy biomechanics finding.
- `benchmark_hit_horizontal_06` reports Wrist/Hand Speed near zero at the bat peak-speed frame, which means bat peak and body wrist-speed event are not aligned. That metric is not reliable for this clip without better contact/release event logic.

### Motion-phase handling caveat

- The current script does not perform full phase segmentation. It uses event proxies: pitching release is dominant-hand peak speed; pitching landing is the maximum ankle-separation frame before release; batting contact is bat peak-speed frame when a bat track exists; batting landing is the first frame before the event where ankle separation reaches 90% of its pre-event maximum.
- That means preparation or ending frames can still leak into metrics when the clip starts late, ends late, or the object/body peak-speed proxy does not match the real biomechanical event.
- Phase-dependent metrics should be upgraded with explicit phase classifiers before being used as coaching-grade outputs: front-foot landing, max external rotation/acceleration, release/contact, and follow-through.

### Concrete suspicious outputs in this run

- benchmark_hit_vertical_02: Attack Angle is -57.185 deg from image-plane bat tracking; this is not a credible true attack angle.
- benchmark_hit_horizontal_06: Wrist/Hand Speed is 0.130 3d_unit/s at bat peak-speed frame, indicating event mismatch.

## benchmark_pitch_vertical_10

| metric | value | unit | status | source | frame | reason |
|---|---:|---|---|---|---:|---|
| Hip-Shoulder Sep | 8.543 | deg | available | 3d_pose | 24 | SMPL24 hip/shoulder lines projected to horizontal plane. |
| Lead Knee Angle | 126.939 | deg | available | 3d_pose | 22 | Lead side inferred as left; value is anatomical knee angle, not flexion-only label. |
| Trunk Tilt | 18.941 | deg | available | 3d_pose | 24 | Torso vector relative to reconstructed vertical axis. |
| Weight Transfer | N/A |  | unavailable | none | N/A | Current GVHMR output is not calibrated to field/world translation; hip/root drift should not be interpreted as COM transfer. |
| Head Stability | 55.572 | % | proxy | 3d_pose | 24 | Root-relative head drift score; no SlyMask reference scale. |
| Dominant Side | right |  | proxy | 3d_pose | 24 | Inferred from larger hand peak speed. |
| Lead Side | left |  | proxy | 3d_pose | 22 | Inferred from foot position along stride direction. |
| Elbow Bend | 68.069 | deg | available | 3d_pose | 24 | Throwing side inferred by peak hand speed. |
| Arm Abduction | 143.617 | deg | available | 3d_pose | 24 | Upper arm angle relative to torso axis. |
| Stride Angle | 29.352 | deg | proxy | 3d_pose | 22 | Angle between foot line and hip line at inferred landing frame. |
| Stride Length | 0.635 | height_ratio | proxy | 3d_pose | 22 | Foot separation normalized by reconstructed body-height proxy. |
| Foot Direction | 30.874 | deg | proxy | 3d_pose | 22 | SMPL24 has foot marker but no toe; this approximates toe direction from ankle-to-foot vector. |
| Wrist Snap | 13.127 | deg | proxy | 3d_pose | 24 | Uses elbow-wrist-hand angle change; no fingertip joint is available. |
| Arm Speed | 4.115 | 3d_unit/s | proxy | 3d_pose | 24 | Raw wrist speed at release proxy; percentile needs a normative database. |
| Fingertip Speed | 4.258 | 3d_unit/s | proxy | 3d_pose | 24 | SMPL24 hand joint is used because fingertip joints are unavailable. |
| Ball Speed | 1619.559 | px/s | proxy | object_2d | 24 | 2D ball speed without camera calibration; not physical mph/km/h. |

## benchmark_pitch_vertical_09

| metric | value | unit | status | source | frame | reason |
|---|---:|---|---|---|---:|---|
| Hip-Shoulder Sep | 7.241 | deg | available | 3d_pose | 10 | SMPL24 hip/shoulder lines projected to horizontal plane. |
| Lead Knee Angle | 139.372 | deg | available | 3d_pose | 5 | Lead side inferred as left; value is anatomical knee angle, not flexion-only label. |
| Trunk Tilt | 13.435 | deg | available | 3d_pose | 10 | Torso vector relative to reconstructed vertical axis. |
| Weight Transfer | N/A |  | unavailable | none | N/A | Current GVHMR output is not calibrated to field/world translation; hip/root drift should not be interpreted as COM transfer. |
| Head Stability | 30.155 | % | proxy | 3d_pose | 10 | Root-relative head drift score; no SlyMask reference scale. |
| Dominant Side | right |  | proxy | 3d_pose | 10 | Inferred from larger hand peak speed. |
| Lead Side | left |  | proxy | 3d_pose | 5 | Inferred from foot position along stride direction. |
| Elbow Bend | 92.035 | deg | available | 3d_pose | 10 | Throwing side inferred by peak hand speed. |
| Arm Abduction | 122.099 | deg | available | 3d_pose | 10 | Upper arm angle relative to torso axis. |
| Stride Angle | 29.349 | deg | proxy | 3d_pose | 5 | Angle between foot line and hip line at inferred landing frame. |
| Stride Length | 0.595 | height_ratio | proxy | 3d_pose | 5 | Foot separation normalized by reconstructed body-height proxy. |
| Foot Direction | 8.117 | deg | proxy | 3d_pose | 5 | SMPL24 has foot marker but no toe; this approximates toe direction from ankle-to-foot vector. |
| Wrist Snap | 24.405 | deg | proxy | 3d_pose | 10 | Uses elbow-wrist-hand angle change; no fingertip joint is available. |
| Arm Speed | 5.261 | 3d_unit/s | proxy | 3d_pose | 10 | Raw wrist speed at release proxy; percentile needs a normative database. |
| Fingertip Speed | 5.401 | 3d_unit/s | proxy | 3d_pose | 10 | SMPL24 hand joint is used because fingertip joints are unavailable. |
| Ball Speed | 2190.378 | px/s | proxy | object_2d | 10 | 2D ball speed without camera calibration; not physical mph/km/h. |

## benchmark_hit_vertical_02

| metric | value | unit | status | source | frame | reason |
|---|---:|---|---|---|---:|---|
| Hip-Shoulder Sep | 51.075 | deg | available | 3d_pose | 106 | SMPL24 hip/shoulder lines projected to horizontal plane. |
| Lead Knee Angle | 132.274 | deg | available | 3d_pose | 73 | Lead side inferred as left; value is anatomical knee angle, not flexion-only label. |
| Trunk Tilt | 22.455 | deg | available | 3d_pose | 106 | Torso vector relative to reconstructed vertical axis. |
| Weight Transfer | N/A |  | unavailable | none | N/A | Current GVHMR output is not calibrated to field/world translation; hip/root drift should not be interpreted as COM transfer. |
| Head Stability | 89.882 | % | proxy | 3d_pose | 106 | Root-relative head drift score; no SlyMask reference scale. |
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
| Weight Transfer | N/A |  | unavailable | none | N/A | Current GVHMR output is not calibrated to field/world translation; hip/root drift should not be interpreted as COM transfer. |
| Head Stability | 64.386 | % | proxy | 3d_pose | 110 | Root-relative head drift score; no SlyMask reference scale. |
| Dominant Side | right |  | proxy | 3d_pose | 110 | Inferred from larger hand peak speed. |
| Lead Side | left |  | proxy | 3d_pose | 71 | Inferred from foot position along stride direction. |
| Swing Speed | 8.064 | norm/s | proxy | object_2d | 110 | SlyMask percentile is unavailable; this is normalized 2D bat speed. |
| Estimated Bat Speed | 7309.942 | px/s | proxy | object_2d | 110 | No camera calibration/bat scale, so km/h cannot be recovered. |
| Hip Rotation | 146.795 | deg | available | 3d_pose | 110 | Range of pelvis yaw over the clip. |
| Attack Angle | -4.272 | deg | proxy | object_2d | 110 | Image-plane bat angle at peak bat speed; not true 3D attack angle. |
| Wrist/Hand Speed | 0.130 | 3d_unit/s | proxy | 3d_pose | 110 | Useful internal body-speed proxy; SlyMask swing percentile needs a reference database. |
| Contact Time | N/A |  | unavailable | none | N/A | No ball track in batting benchmark and no bat-ball impact event detector; cannot determine physical contact duration. |
