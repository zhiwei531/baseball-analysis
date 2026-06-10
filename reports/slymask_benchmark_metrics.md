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

### Motion-phase handling caveat

- The current script does not use object tracker outputs for phase selection. Pitching release is selected from dominant-hand 3D speed. Batting event is selected from a body-motion composite: hand/wrist speed plus hip and shoulder yaw velocity inside the middle 35%-75% of the clip.
- Range-style body metrics now use the selected body-motion phase window instead of the full clip, so non-batting ending motion such as the running segment in `benchmark_hit_horizontal_06` is not included in Hip Rotation or Head Stability.
- That means preparation or ending frames can still leak into metrics when the clip starts late, ends late, or the object/body peak-speed proxy does not match the real biomechanical event.
- Phase-dependent metrics should be upgraded with explicit phase classifiers before being used as coaching-grade outputs: front-foot landing, max external rotation/acceleration, release/contact, and follow-through.

### Concrete suspicious outputs in this run

- benchmark_hit_horizontal_06: Attack Angle is -174.757 deg from image-plane bat tracking; this is not a credible true attack angle.

## benchmark_pitch_vertical_10

| metric | value | unit | status | source | frame | reason |
|---|---:|---|---|---|---:|---|
| Motion Phase Start | 0.733 | s | proxy | 3d_pose | 22 | Body-motion only: release proxy is dominant-hand 3D speed peak; object tracker is not used for phase selection. |
| Motion Phase Event | 0.800 | s | proxy | 3d_pose | 24 | Body-motion only: release proxy is dominant-hand 3D speed peak; object tracker is not used for phase selection. |
| Motion Phase End | 1.100 | s | proxy | 3d_pose | 33 | Body-motion only: release proxy is dominant-hand 3D speed peak; object tracker is not used for phase selection. |
| Hip-Shoulder Sep | 8.543 | deg | available | 3d_pose | 24 | SMPL24 hip/shoulder lines projected to horizontal plane. |
| Lead Knee Angle | 126.939 | deg | available | 3d_pose | 22 | Lead side inferred as left; value is anatomical knee angle, not flexion-only label. |
| Trunk Tilt | 18.941 | deg | available | 3d_pose | 24 | Torso vector relative to reconstructed vertical axis. |
| Weight Transfer | N/A |  | unavailable | none | N/A | Current GVHMR output is not calibrated to field/world translation; hip/root drift should not be interpreted as COM transfer. |
| Head Stability | 92.928 | % | proxy | 3d_pose | 24 | Root-relative head drift score; no SlyMask reference scale. |
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
| Motion Phase Start | 0.167 | s | proxy | 3d_pose | 5 | Body-motion only: release proxy is dominant-hand 3D speed peak; object tracker is not used for phase selection. |
| Motion Phase Event | 0.333 | s | proxy | 3d_pose | 10 | Body-motion only: release proxy is dominant-hand 3D speed peak; object tracker is not used for phase selection. |
| Motion Phase End | 0.700 | s | proxy | 3d_pose | 21 | Body-motion only: release proxy is dominant-hand 3D speed peak; object tracker is not used for phase selection. |
| Hip-Shoulder Sep | 7.241 | deg | available | 3d_pose | 10 | SMPL24 hip/shoulder lines projected to horizontal plane. |
| Lead Knee Angle | 139.372 | deg | available | 3d_pose | 5 | Lead side inferred as left; value is anatomical knee angle, not flexion-only label. |
| Trunk Tilt | 13.435 | deg | available | 3d_pose | 10 | Torso vector relative to reconstructed vertical axis. |
| Weight Transfer | N/A |  | unavailable | none | N/A | Current GVHMR output is not calibrated to field/world translation; hip/root drift should not be interpreted as COM transfer. |
| Head Stability | 73.196 | % | proxy | 3d_pose | 10 | Root-relative head drift score; no SlyMask reference scale. |
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
| Motion Phase Start | 2.033 | s | proxy | 3d_pose | 61 | Body-motion composite only: max of dominant hand/wrist speed plus hip/shoulder yaw velocity inside the middle 35%-75% clip window; object tracker is not used for phase selection. |
| Motion Phase Event | 2.467 | s | proxy | 3d_pose | 74 | Body-motion composite only: max of dominant hand/wrist speed plus hip/shoulder yaw velocity inside the middle 35%-75% clip window; object tracker is not used for phase selection. |
| Motion Phase End | 2.900 | s | proxy | 3d_pose | 87 | Body-motion composite only: max of dominant hand/wrist speed plus hip/shoulder yaw velocity inside the middle 35%-75% clip window; object tracker is not used for phase selection. |
| Hip-Shoulder Sep | 4.238 | deg | available | 3d_pose | 74 | SMPL24 hip/shoulder lines projected to horizontal plane. |
| Lead Knee Angle | 129.599 | deg | available | 3d_pose | 69 | Lead side inferred as left; value is anatomical knee angle, not flexion-only label. |
| Trunk Tilt | 15.498 | deg | available | 3d_pose | 74 | Torso vector relative to reconstructed vertical axis. |
| Weight Transfer | N/A |  | unavailable | none | N/A | Current GVHMR output is not calibrated to field/world translation; hip/root drift should not be interpreted as COM transfer. |
| Head Stability | 96.063 | % | proxy | 3d_pose | 74 | Root-relative head drift score; no SlyMask reference scale. |
| Dominant Side | right |  | proxy | 3d_pose | 74 | Inferred from larger hand peak speed. |
| Lead Side | left |  | proxy | 3d_pose | 69 | Inferred from foot position along stride direction. |
| Swing Speed | 2.338 | norm/s | proxy | object_2d | 74 | SlyMask percentile is unavailable; this is normalized 2D bat speed. |
| Estimated Bat Speed | 1692.456 | px/s | proxy | object_2d | 74 | No camera calibration/bat scale, so km/h cannot be recovered. |
| Hip Rotation | 51.308 | deg | available | 3d_pose | 74 | Range of pelvis yaw over the body-motion phase window. |
| Attack Angle | 43.452 | deg | proxy | object_2d | 74 | Image-plane bat angle at body-event frame; not true 3D attack angle. |
| Wrist/Hand Speed | 2.515 | 3d_unit/s | proxy | 3d_pose | 74 | Useful internal body-speed proxy; SlyMask swing percentile needs a reference database. |
| Contact Time | N/A |  | unavailable | none | N/A | No ball track in batting benchmark and no bat-ball impact event detector; cannot determine physical contact duration. |

## benchmark_hit_horizontal_06

| metric | value | unit | status | source | frame | reason |
|---|---:|---|---|---|---:|---|
| Motion Phase Start | 4.200 | s | proxy | 3d_pose | 126 | Body-motion composite only: max of dominant hand/wrist speed plus hip/shoulder yaw velocity inside the middle 35%-75% clip window; object tracker is not used for phase selection. |
| Motion Phase Event | 4.467 | s | proxy | 3d_pose | 134 | Body-motion composite only: max of dominant hand/wrist speed plus hip/shoulder yaw velocity inside the middle 35%-75% clip window; object tracker is not used for phase selection. |
| Motion Phase End | 5.500 | s | proxy | 3d_pose | 165 | Body-motion composite only: max of dominant hand/wrist speed plus hip/shoulder yaw velocity inside the middle 35%-75% clip window; object tracker is not used for phase selection. |
| Hip-Shoulder Sep | 20.263 | deg | available | 3d_pose | 134 | SMPL24 hip/shoulder lines projected to horizontal plane. |
| Lead Knee Angle | 131.790 | deg | available | 3d_pose | 130 | Lead side inferred as left; value is anatomical knee angle, not flexion-only label. |
| Trunk Tilt | 13.221 | deg | available | 3d_pose | 134 | Torso vector relative to reconstructed vertical axis. |
| Weight Transfer | N/A |  | unavailable | none | N/A | Current GVHMR output is not calibrated to field/world translation; hip/root drift should not be interpreted as COM transfer. |
| Head Stability | 99.148 | % | proxy | 3d_pose | 134 | Root-relative head drift score; no SlyMask reference scale. |
| Dominant Side | right |  | proxy | 3d_pose | 134 | Inferred from larger hand peak speed. |
| Lead Side | left |  | proxy | 3d_pose | 130 | Inferred from foot position along stride direction. |
| Swing Speed | 2.297 | norm/s | proxy | object_2d | 134 | SlyMask percentile is unavailable; this is normalized 2D bat speed. |
| Estimated Bat Speed | 2856.713 | px/s | proxy | object_2d | 134 | No camera calibration/bat scale, so km/h cannot be recovered. |
| Hip Rotation | 75.822 | deg | available | 3d_pose | 134 | Range of pelvis yaw over the body-motion phase window. |
| Attack Angle | -174.757 | deg | proxy | object_2d | 134 | Image-plane bat angle at body-event frame; not true 3D attack angle. |
| Wrist/Hand Speed | 3.064 | 3d_unit/s | proxy | 3d_pose | 134 | Useful internal body-speed proxy; SlyMask swing percentile needs a reference database. |
| Contact Time | N/A |  | unavailable | none | N/A | No ball track in batting benchmark and no bat-ball impact event detector; cannot determine physical contact duration. |
