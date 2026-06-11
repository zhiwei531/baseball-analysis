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

## Complete SlyMask Metric Table

Status meanings: `direct` is a direct 3D marker geometry calculation; `raw_only` means the physical value exists but SlyMask's percentile/rating needs a reference population; `proxy` means the value is computable but the app's exact event or target definition is missing; `direct_batting_context` means the geometry is direct, but SlyMask describes the metric for pitching while these Vicon files are batting swings.

The SlyMask categorical labels (`Good`, `Attention`, `Deviate`) and reliability percentages are not reproduced here because they require proprietary thresholds, a reference population, and the app's internal confidence model. This report focuses on whether the underlying physical metric can be extracted automatically from the Vicon CSV.

| trial | family | metric | value | unit | status | method / reason |
|---|---|---|---:|---|---|---|
| 0506Coach_wave | Swing Analysis | Estimated Bat Speed | 131.717 | km/h | direct | 3D speed of bat1 at selected swing event; bat4 is also reported in the summary because barrel marker identity is ambiguous. |
| 0506Coach_wave | Swing Analysis | Swing Speed | 131.717 | km/h | raw_only | Raw bat endpoint speed is available; SlyMask-style percentile needs a reference population. |
| 0506Coach_wave | Swing Analysis | Hip Rotation | 34.523 | deg | direct | Pelvis yaw range over the selected high-speed swing window. |
| 0506Coach_wave | Swing Analysis | Hip-Shoulder Sep | 36.178 | deg | direct | Absolute yaw difference between pelvis axis and shoulder axis at the event frame. |
| 0506Coach_wave | Swing Analysis | Weight Transfer | 1.475 | % | proxy | Pelvis midpoint translation from phase start to event, normalized by ankle stance width; not a full COM model. |
| 0506Coach_wave | Swing Analysis | Lead Knee Angle | 136.435 | deg | direct | Left lead-knee angle from LASI-LKNE-LANK. |
| 0506Coach_wave | Swing Analysis | Trunk Tilt | 24.262 | deg | direct | Torso midpoint vector relative to vertical Z axis. |
| 0506Coach_wave | Swing Analysis | Contact Time |  | ms | unavailable | No ball-contact label, bat-ball impact event, or ball marker exists in this CSV. |
| 0506Coach_wave | Swing Analysis | Attack Angle | -27.105 | deg | proxy | Bat axis angle relative to the horizontal plane at selected event; true attack angle needs a contact-frame definition. |
| 0506Coach_wave | Swing Analysis | Head Stability | 49.174 | % | proxy | Head midpoint drift relative to pelvis midpoint over the selected swing window. |
| 0506Coach_wave | Motion Metrics | Elbow Bend | 138.147 | deg | direct_batting_context | Right elbow angle from RSHO-RELB-RWRA; SlyMask's pitching acceleration interpretation is not directly applicable to this batting dataset. |
| 0506Coach_wave | Motion Metrics | Arm Abduction | 105.572 | deg | direct_batting_context | Right upper-arm angle relative to torso axis; pitching arm-slot interpretation is not directly applicable. |
| 0506Coach_wave | Motion Metrics | Trunk Lean | 24.262 | deg | direct_batting_context | Same torso-vs-vertical geometry as Trunk Tilt, measured at swing event instead of pitching release. |
| 0506Coach_wave | Motion Metrics | Stride Angle | 57.488 | deg | proxy | Planar angle between ankle stance line and pelvis axis at event; pitching front-foot-landing definition is unavailable. |
| 0506Coach_wave | Motion Metrics | Lead Knee | 136.435 | deg | direct_batting_context | Left lead-knee angle from LASI-LKNE-LANK; measured at swing event. |
| 0506Coach_wave | Motion Metrics | Hip-Shoulder Sep | 36.178 | deg | direct_batting_context | Same pelvis-shoulder yaw separation as batting metric, measured at swing event. |
| 0506Coach_wave | Motion Metrics | Arm Speed | 30.425 | km/h | raw_only | Fastest wrist marker at event is LWRA; percentile needs a reference population. |
| 0506Coach_wave | Motion Metrics | Stride Length | 0.620 | body heights | proxy | Ankle stance distance divided by head-to-foot height proxy; pitching stride event definition is unavailable. |
| 0506Coach_wave | Motion Metrics | Weight Transfer | 1.475 | % | proxy | Same pelvis-shift proxy as batting Weight Transfer; not a validated COM transfer metric. |
| 0506Coach_wave | Motion Metrics | Head Stability | 49.174 | % | proxy | Head drift relative to pelvis over the swing window; SlyMask stride-line definition is unavailable. |
| 0506Coach_wave | Motion Metrics | Foot Direction | 79.280 | deg | proxy | Left heel-to-toe direction relative to stance line at event; home-plate target direction is unavailable. |
| 0506Coach_wave | Motion Metrics | Wrist Snap | 57.287 | deg | proxy | Change in elbow-wrist-finger angle from phase start to event using the fastest hand side. |
| 0506Coach_wave | Motion Metrics | Fingertip Speed | 39.432 | km/h | raw_only | Fastest hand/finger marker at event is LFIN; percentile needs a reference population. |
| Julian_wave02 | Swing Analysis | Estimated Bat Speed | 74.019 | km/h | direct | 3D speed of bat1 at selected swing event; bat4 is also reported in the summary because barrel marker identity is ambiguous. |
| Julian_wave02 | Swing Analysis | Swing Speed | 74.019 | km/h | raw_only | Raw bat endpoint speed is available; SlyMask-style percentile needs a reference population. |
| Julian_wave02 | Swing Analysis | Hip Rotation | 99.432 | deg | direct | Pelvis yaw range over the selected high-speed swing window. |
| Julian_wave02 | Swing Analysis | Hip-Shoulder Sep | 9.657 | deg | direct | Absolute yaw difference between pelvis axis and shoulder axis at the event frame. |
| Julian_wave02 | Swing Analysis | Weight Transfer | 3.851 | % | proxy | Pelvis midpoint translation from phase start to event, normalized by ankle stance width; not a full COM model. |
| Julian_wave02 | Swing Analysis | Lead Knee Angle | 164.733 | deg | direct | Left lead-knee angle from LASI-LKNE-LANK. |
| Julian_wave02 | Swing Analysis | Trunk Tilt | 41.206 | deg | direct | Torso midpoint vector relative to vertical Z axis. |
| Julian_wave02 | Swing Analysis | Contact Time |  | ms | unavailable | No ball-contact label, bat-ball impact event, or ball marker exists in this CSV. |
| Julian_wave02 | Swing Analysis | Attack Angle | -26.255 | deg | proxy | Bat axis angle relative to the horizontal plane at selected event; true attack angle needs a contact-frame definition. |
| Julian_wave02 | Swing Analysis | Head Stability | 48.623 | % | proxy | Head midpoint drift relative to pelvis midpoint over the selected swing window. |
| Julian_wave02 | Motion Metrics | Elbow Bend | 135.717 | deg | direct_batting_context | Right elbow angle from RSHO-RELB-RWRA; SlyMask's pitching acceleration interpretation is not directly applicable to this batting dataset. |
| Julian_wave02 | Motion Metrics | Arm Abduction | 122.050 | deg | direct_batting_context | Right upper-arm angle relative to torso axis; pitching arm-slot interpretation is not directly applicable. |
| Julian_wave02 | Motion Metrics | Trunk Lean | 41.206 | deg | direct_batting_context | Same torso-vs-vertical geometry as Trunk Tilt, measured at swing event instead of pitching release. |
| Julian_wave02 | Motion Metrics | Stride Angle | 14.478 | deg | proxy | Planar angle between ankle stance line and pelvis axis at event; pitching front-foot-landing definition is unavailable. |
| Julian_wave02 | Motion Metrics | Lead Knee | 164.733 | deg | direct_batting_context | Left lead-knee angle from LASI-LKNE-LANK; measured at swing event. |
| Julian_wave02 | Motion Metrics | Hip-Shoulder Sep | 9.657 | deg | direct_batting_context | Same pelvis-shoulder yaw separation as batting metric, measured at swing event. |
| Julian_wave02 | Motion Metrics | Arm Speed | 15.011 | km/h | raw_only | Fastest wrist marker at event is RWRA; percentile needs a reference population. |
| Julian_wave02 | Motion Metrics | Stride Length | 0.611 | body heights | proxy | Ankle stance distance divided by head-to-foot height proxy; pitching stride event definition is unavailable. |
| Julian_wave02 | Motion Metrics | Weight Transfer | 3.851 | % | proxy | Same pelvis-shift proxy as batting Weight Transfer; not a validated COM transfer metric. |
| Julian_wave02 | Motion Metrics | Head Stability | 48.623 | % | proxy | Head drift relative to pelvis over the swing window; SlyMask stride-line definition is unavailable. |
| Julian_wave02 | Motion Metrics | Foot Direction | 78.326 | deg | proxy | Left heel-to-toe direction relative to stance line at event; home-plate target direction is unavailable. |
| Julian_wave02 | Motion Metrics | Wrist Snap | 47.194 | deg | proxy | Change in elbow-wrist-finger angle from phase start to event using the fastest hand side. |
| Julian_wave02 | Motion Metrics | Fingertip Speed | 18.954 | km/h | raw_only | Fastest hand/finger marker at event is RFIN; percentile needs a reference population. |

## Body Metrics at Event

| trial | hip rotation (deg) | hip-shoulder sep (deg) | lead knee (deg) | trunk tilt (deg) | head stability (%) |
|---|---:|---:|---:|---:|---:|
| 0506Coach_wave | 34.523 | 36.178 | 136.435 | 24.262 | 49.174 |
| Julian_wave02 | 99.432 | 9.657 | 164.733 | 41.206 | 48.623 |
