# Vicon Benchmark Report README

这份文档说明当前中文棒球动作体检报告的完整构建流程。报告入口是
`scripts/build_benchmark_report_html.py`，设计约束来自 `DESIGN.md`，最终产物是
`report.html`。

## 当前数据原则

当前报告统一 raw data source：身体运动学、逐帧曲线、指标来源表和 C3D 动图都来自
`../vicon_2026` 文件夹下的 Vicon C3D。

- 主分析人：`bryan`
- 教练模块对照：`green`
- 被试名来源：`../vicon_2026/{subfolder}`，子文件夹名就是被试名。
- 临时 coach 参考：投球教练参考暂时沿用既有 `data_full/coach_pose3d/gvhmr/pitch_horizontal_coach.csv` 和相关 coach 指标表，只作为黑色参考线或评分参考，不作为 raw source。

报告不再把 benchmark 视频 CV/GVHMR 身体数据和 Vicon 身体数据混在一起。旧的
`reports/slymask_benchmark_metrics.csv`、RTMPose/GVHMR benchmark pose CSV 和截图缩略图不再作为当前报告主体数据来源。

## 一键构建顺序

在 `baseball-analysis/` 目录运行：

```bash
.venv312/bin/python scripts/build_vicon_2026_metrics.py
MPLCONFIGDIR=/private/tmp/baseball_mpl_cache .venv312/bin/python scripts/render_vicon_reconstruction_images.py
.venv312/bin/python scripts/build_benchmark_report_html.py
```

三步分别完成：

1. `build_vicon_2026_metrics.py` 从 `../vicon_2026/*/*.c3d` 读取 C3D，生成 `reports/vicon_2026_metrics.csv` 和 `reports/vicon_2026_point_summary.csv`。
2. `render_vicon_reconstruction_images.py` 从 C3D 关键动作窗口渲染 PNG/GIF。
3. `build_benchmark_report_html.py` 读取 Vicon 派生表和原始 C3D，生成 `report.html`。

注意：第三步需要 `numpy`，请使用 `.venv312/bin/python`，不要用系统 `python3`。

## 当前纳入的 C3D

| 被试 | 动作 | C3D | 报告用途 |
|---|---|---|---|
| `bryan` | pitching | `vicon_2026/bryan/001 Cal 04 Pitch 05.c3d` | 主体投球分析 |
| `bryan` | batting | `vicon_2026/bryan/001 Cal 04 Bat 05.c3d` | 主体打击分析 |
| `green` | pitching | `vicon_2026/green/006 Pitch 09.c3d` | 教练模块投球对照 |
| `green` | batting | `vicon_2026/green/006 Bat 04.c3d` | 教练模块打击对照 |

## 输入与输出

HTML 报告当前直接读取：

```text
reports/vicon_2026_metrics.csv
reports/vicon_2026_point_summary.csv
reports/assets/vicon_reconstruction/*.png
reports/assets/vicon_reconstruction/*.gif
../vicon_2026/bryan/*.c3d
../vicon_2026/green/*.c3d
output/data/benchmark_pitch_vertical_09_motion_metrics_full.csv
output/data/benchmark_pitch_vertical_09_vs_pitch_horizontal_coach_metrics.csv
data_full/coach_pose3d/gvhmr/pitch_horizontal_coach.csv
```

其中最后三项只用于临时 coach 参考，报告正文会明确标注。主体 raw data source 仍是 Vicon C3D。

构建产物：

```text
report.html
reports/vicon_2026_metrics.csv
reports/vicon_2026_point_summary.csv
reports/assets/vicon_reconstruction/*.png
reports/assets/vicon_reconstruction/*.gif
```

## 指标计算

`build_vicon_2026_metrics.py` 解析 C3D 后输出 trial 级指标：

- 髋肩分离：髋部轴与肩部轴的 planar yaw 差。
- 前腿膝角：髋、膝、踝 marker 的三点夹角。
- 躯干倾斜：髋部中心到颈/躯干 marker 的向量相对竖直方向的倾角。
- 手部速度：手腕 marker 三维坐标逐帧差分后的峰值速度。
- 髋部/躯干速度：对应 marker 中心的三维速度峰值。
- 球棒速度：Bat1/Bat5 相关 marker 三维速度峰值。
- 挥棒窗口：球棒速度超过该 trial 峰值一定比例的连续高速度窗口。
- 数据质量：C3D 有效点比例。

`build_benchmark_report_html.py` 会继续复用已有评分函数：

- `score_close`：用主体值与参考值的接近程度给分。
- `score_ratio`：用主体值与对照/参考值的比例给分。

但这些分数的主体输入已换成 bryan 的 Vicon 指标。green 只作为教练模块对照。coach 参考只用于投球部分的临时参考线和部分评分目标。

## 逐帧曲线

HTML 构建脚本会直接读取 bryan/green 的原始 C3D，把 Vicon marker 映射到报告内部关节名，然后生成逐帧曲线：

- 投球角度时间曲线：前腿膝角、肘角、躯干倾斜、髋肩分离。
- 投球速度时间曲线：髋部中心、躯干中心、手部末端速度。
- 打击角度时间曲线：前腿膝角、肘角、躯干倾斜、髋肩分离。
- 打击速度时间曲线：髋部中心、躯干中心、手部末端速度。
- 数据质量图：bryan/green 投球与打击的帧数、关节完整率和有效情况。

## C3D 动图

Vicon 动图由 `scripts/render_vicon_reconstruction_images.py` 预渲染到：

```text
reports/assets/vicon_reconstruction/{trial_id}.png
reports/assets/vicon_reconstruction/{trial_id}.gif
```

关键动作帧由 `build_vicon_2026_metrics.py` 自动选择：

- batting：球棒速度峰值帧，规则名 `bat_speed_peak`。
- pitching：主导手/出手侧手部速度峰值帧，规则名 `right_hand_speed_peak` 或 `left_hand_speed_peak`。
- 若速度不可用，退回 `mid_frame_fallback`。

GIF 默认展示关键帧前约 `0.6 s`、后约 `0.4 s` 的窗口，最多 `72` 帧。渲染时使用固定相机视角和固定坐标范围，只画真实身体 marker、`CentreOfMass` 和打击时的 `Bat1-Bat5`。头部 `LFHD/RFHD/LBHD/RBHD` 四点连接为闭合立体面；左右脚的踝、跟、趾三点连接为脚部三角面。

## 报告模块

- 球员模块：只展示 bryan 的投球和打击诊断、评分、关键指标卡、C3D 动图和训练建议。
- 教练模块：展示 bryan vs green 的同源 Vicon 对比；投球保留临时 coach 参考线。
- 研究者模块：展示 C3D 逐帧曲线、C3D 来源表、事件点、数据质量和指标来源表。

## 已知限制

- 当前 coach 参考还没有统一到 Vicon C3D，报告已标为临时参考。
- Vicon 打击的 `swing_time_sec` 是高速挥棒窗口，不是 bat-ball contact time。
- `valid_point_pct` 只表示 C3D 点有效率，不等同真实头部稳定评分。
- 攻击角目前是球棒轴相对水平面的角度，严格 attack angle 仍需要真实击球点/接触帧。
- 训练建议用于训练参考，不是医学诊断。
