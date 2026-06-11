# Vicon Swing Metrics

Input: `../Vicon_Wave_250506(1)` Vicon trajectory CSV files.

Assumptions: coordinates are in millimeters; sample rate comes from row 2 (`100 Hz`); `Z` is vertical; the bat axis is `bat4 -> bat1`. Bat angle is `atan2(delta_Z, horizontal_distance)` in degrees.

Swing event is selected as the frame where the bat-axis angle is closest to `-27 deg`, matching the provided reference. Swing duration is the contiguous high-speed window around that event where `bat4` speed remains above `40%` of its trial peak. This threshold gives the Coach reference trial a `0.16 s` window.

## CSV Structure

- Row 1 identifies the Vicon section as `Trajectories`.
- Row 2 contains the sample rate, `100 Hz` for both files.
- Row 3 contains marker names. Each marker then has three coordinate columns.
- Row 4 contains `Frame`, `Sub Frame`, then repeated `X,Y,Z` coordinate fields.
- Row 5 states coordinate units as `mm`.
- Timestamps are reconstructed as `(Frame - 1) / sample_rate_hz`.
- Body markers include head (`LFHD/RFHD/LBHD/RBHD`), shoulders, elbows, wrists, pelvis (`LASI/RASI/LPSI/RPSI`), knees, ankles, heels, toes, and finger markers.
- Bat markers are `bat1`, `bat2`, `bat3`, `bat4`; the longest and most stable axis is `bat1-bat4` at about `293 mm`.

## Required Parameters

| trial | event time (s) | bat angle (deg) | bat1 speed (km/h) | bat4 speed (km/h) | best hand marker | hand speed (km/h) | swing time (s) |
|---|---:|---:|---:|---:|---|---:|---:|
| 0506Coach_wave | 0.990 | -27.1 | 131.7 | 87.9 | LFIN | 39.4 | 0.160 |
| Julian_wave02 | 2.560 | -26.3 | 74.0 | 45.3 | RFIN | 19.0 | 0.320 |

Reference comparison: `0506Coach_wave` matches the requested bat angle (`-27.1 deg`). Its fastest endpoint `bat1` is `131.7 km/h`; the opposite endpoint `bat4` is `87.9 km/h`, closer to the requested `95 km/h`. The best hand/finger marker at the same frame is `LFIN = 39.4 km/h`, matching the requested wrist/hand-speed reference better than wrist markers.

## SlyMask Metric Coverage

| metric | computable from Vicon? | method / limitation |
|---|---|---|
| Estimated Bat Speed | yes | Direct 3D marker speed; report both `bat1` and `bat4` because marker-to-barrel definition is ambiguous. |
| Swing Speed | yes, raw only | Raw km/h available; SlyMask percentile requires a reference population. |
| Hip Rotation | yes | Pelvis yaw range from LASI/RASI/LPSI/RPSI over swing window. |
| Hip-Shoulder Sep | yes | Absolute yaw difference between pelvis line and shoulder line at event frame. |
| Lead Knee Angle | yes | Left lead-knee anatomical angle proxy from LASI-LKNE-LANK. |
| Trunk Tilt | yes | Torso vector relative to vertical Z axis. |
| Head Stability | proxy | Head midpoint drift relative to pelvis midpoint over swing window. |
| Attack Angle | proxy | Bat axis angle relative to horizontal plane; true ball-contact attack angle still needs contact definition. |
| Contact Time | no | No ball-contact labels or ball marker. |
| Weight Transfer | partial/proxy | Pelvis translation exists, but a validated COM model is not implemented here. |
| Pitching-only metrics | not applicable | Dataset is batting swing, not pitching. |
| Wrist Snap / Fingertip Speed | partial | Finger markers exist; true wrist-snap amplitude needs a formal wrist/hand segment definition. |
| Elbow Bend | technically yes | Elbow angle can be computed from shoulder-elbow-wrist markers, but the SlyMask text defines it for pitching acceleration; this dataset is batting. |
| Arm Abduction | technically yes | Upper-arm angle relative to torso can be computed, but pitching arm-slot interpretation is not applicable. |
| Arm Speed | yes, raw only | Wrist/hand marker speed is available; SlyMask percentile requires a reference population. |
| Stride Angle / Stride Length / Foot Direction | partial | Feet and pelvis markers exist; definitions are pitching-specific and require a validated phase/event definition. |

## Body Metrics at Event

| trial | hip rotation (deg) | hip-shoulder sep (deg) | lead knee (deg) | trunk tilt (deg) | head stability (%) |
|---|---:|---:|---:|---:|---:|
| 0506Coach_wave | 34.523 | 36.178 | 136.435 | 24.262 | 49.174 |
| Julian_wave02 | 99.432 | 9.657 | 164.733 | 41.206 | 48.623 |
