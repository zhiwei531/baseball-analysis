import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const ROOT = "/Volumes/T7/DKU/Course/CS 207/final-project/baseball-analysis";
const ASSET_DIR = path.join(ROOT, "outputs_full/benchmark_rtmpose_test/report_assets/benchmark_pitch_vertical_09");
const OUT = path.join(ROOT, "output/pptx/benchmark_pitch_vertical_09_report_zh.pptx");
const PREVIEW_DIR = "/private/tmp/codex-presentations/baseball_pitch_report/tmp/preview";
const QA_DIR = "/private/tmp/codex-presentations/baseball_pitch_report/tmp/qa";

const C = {
  bg: "#f5f7fb",
  ink: "#101828",
  body: "#344054",
  muted: "#667085",
  blue: "#2563eb",
  lightBlue: "#eff6ff",
  green: "#16a34a",
  orange: "#f97316",
  red: "#ef4444",
  border: "#d0d5dd",
  white: "#ffffff",
  dark: "#101828",
};

async function bytes(file) {
  return new Uint8Array(await fs.readFile(file));
}

async function textFile(file) {
  return await fs.readFile(file, "utf8");
}

function addText(slide, text, x, y, w, h, size = 18, opts = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    fontSize: Math.max(size, 14),
    bold: !!opts.bold,
    color: opts.color || C.body,
    fontFace: opts.fontFace || "Microsoft YaHei",
  };
  return shape;
}

function addCard(slide, x, y, w, h, fill = C.white, line = C.border) {
  return slide.shapes.add({
    geometry: "roundRect",
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { style: "solid", fill: line, width: 1 },
    borderRadius: "rounded-2xl",
  });
}

async function addImage(slide, file, x, y, w, h, fit = "contain", alt = "") {
  const ext = path.extname(file).toLowerCase();
  const contentType = ext === ".gif" ? "image/gif" : ext === ".jpg" || ext === ".jpeg" ? "image/jpeg" : "image/png";
  return slide.images.add({
    blob: await bytes(file),
    contentType,
    alt,
    fit,
    geometry: "roundRect",
    borderRadius: "rounded-xl",
    position: { left: x, top: y, width: w, height: h },
  });
}

function header(slide, title, subtitle, part = "") {
  slide.background.fill = C.bg;
  addText(slide, "棒球动作实验室", 54, 30, 220, 28, 15, { bold: true, color: C.blue });
  addText(slide, part, 1010, 30, 210, 28, 15, { bold: true, color: C.muted });
  addText(slide, title, 54, 74, 760, 48, 34, { bold: true, color: C.ink });
  if (subtitle) addText(slide, subtitle, 56, 124, 880, 30, 19, { color: C.muted });
}

function footer(slide, page) {
  addText(slide, "3D视频动作分析报告，仅用于训练参考", 54, 686, 400, 22, 14, { color: "#98a2b3" });
  addText(slide, String(page).padStart(2, "0"), 1180, 686, 50, 22, 14, { color: "#98a2b3" });
}

function splitText(text, maxChars) {
  const chunks = [];
  let rest = text.trim();
  while (rest.length > maxChars) {
    let idx = rest.lastIndexOf("\n", maxChars);
    if (idx < maxChars * 0.55) idx = maxChars;
    chunks.push(rest.slice(0, idx).trim());
    rest = rest.slice(idx).trim();
  }
  if (rest) chunks.push(rest);
  return chunks;
}

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const cells = line.split(",");
    return Object.fromEntries(headers.map((h, i) => [h, cells[i] ?? ""]));
  });
}

function fmt(v, unit) {
  const n = Number(v);
  if (!Number.isFinite(n)) return "暂无";
  if (unit === "deg") return `${n.toFixed(0)}°`;
  if (unit === "%" || unit === "%height" || unit === "%stride") return `${n.toFixed(0)}%`;
  return `${n.toFixed(1)} ${unit}`;
}

async function main() {
  await fs.mkdir(path.dirname(OUT), { recursive: true });
  await fs.mkdir(PREVIEW_DIR, { recursive: true });
  const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });

  const metrics = parseCsv(await textFile(path.join(ROOT, "output/data/benchmark_pitch_vertical_09_motion_metrics_full.csv")));
  const prompt = await textFile(path.join(ROOT, "output/data/benchmark_pitch_vertical_09_parent_prompt.txt"));
  const report = await textFile(path.join(ROOT, "output/data/benchmark_pitch_vertical_09_parent_guidance.md"));
  const promptParts = splitText(prompt, 1350);
  const reportParts = splitText(report, 1450);

  let page = 1;
  {
    const slide = presentation.slides.add();
    slide.background.fill = C.bg;
    addCard(slide, 54, 82, 1172, 480, C.dark, C.dark);
    addText(slide, "青少年棒球投球动作体检报告", 96, 126, 620, 64, 42, { bold: true, color: C.white });
    addText(slide, "孩子和教练差在哪里、差多少、怎么改", 98, 204, 620, 36, 24, { color: "#dbeafe" });
    addText(slide, "PDF 与 PPTX 同一套视觉风格；PPTX 增加 3D动态纠正展示。", 98, 260, 620, 34, 19, { color: "#cbd5e1" });
    await addImage(slide, path.join(ASSET_DIR, "thumb_2d_overlay.png"), 770, 126, 365, 320, "cover", "孩子投球视频截图");
    addText(slide, "中文报告", 104, 360, 110, 28, 16, { bold: true, color: C.blue });
    addText(slide, "可量化诊断", 230, 360, 140, 28, 16, { bold: true, color: C.green });
    addText(slide, "训练建议", 392, 360, 110, 28, 16, { bold: true, color: C.orange });
    footer(slide, page++);
  }

  {
    const slide = presentation.slides.add();
    header(slide, "Part 1 原始数据展示", "CV/视频参数与 3D/Vicon近似参数表", "Part 1");
    const rows = [
      ["指标", "孩子", "教练", "说明"],
      ...metrics.slice(0, 10).map((m) => [m.label_cn, fmt(m.child_value, m.unit), fmt(m.coach_value, m.unit), m.method.includes("proxy") ? "3D估算" : "3D计算"]),
    ];
    const table = slide.tables.add({ rows: rows.length, columns: 4, left: 60, top: 180, width: 760, height: 430, values: rows });
    table.styleOptions = { headerRow: true, bandedRows: true };
    table.borders.assign({ style: "solid", fill: C.border, width: 1 });
    for (let c = 0; c < 4; c++) table.getCell(0, c).fill = "#eaf2ff";
    addCard(slide, 860, 180, 330, 430, C.white);
    addText(slide, "本页保留全部可用指标", 885, 210, 280, 32, 22, { bold: true, color: C.ink });
    addText(slide, "当前可用 CV/视频参数包括球速估算、2D叠加截图和原始视频相位；3D/Vicon近似参数来自 GVHMR/3D骨架，包括关节角、速度、跨步、身体中心位移和稳定性评分。", 885, 260, 270, 180, 18, { color: C.body });
    addText(slide, "正式采集时可替换为真实 Vicon、雷达枪和力板数据。", 885, 480, 270, 80, 18, { color: C.orange });
    footer(slide, page++);
  }

  {
    const slide = presentation.slides.add();
    header(slide, "关键相位截图对照", "孩子与正常速度教练的抬腿、出手、随挥对照", "Part 1");
    const items = [
      ["phase_1_stride.png", "coach_phase_1_lift.png", "抬腿/跨步"],
      ["phase_2_release.png", "coach_phase_2_release.png", "出手附近"],
      ["phase_3_follow.png", "coach_phase_3_follow.png", "随挥稳定"],
    ];
    for (let i = 0; i < items.length; i++) {
      const x = 60 + i * 400;
      addCard(slide, x, 175, 350, 410, C.white);
      addText(slide, items[i][2], x + 22, 195, 200, 28, 22, { bold: true, color: C.ink });
      await addImage(slide, path.join(ASSET_DIR, items[i][0]), x + 24, 235, 300, 145, "cover", "孩子相位截图");
      await addImage(slide, path.join(ASSET_DIR, items[i][1]), x + 24, 410, 300, 145, "cover", "教练相位截图");
    }
    footer(slide, page++);
  }

  {
    const slide = presentation.slides.add();
    header(slide, "原始曲线图", "角度-时间、速度-时间与身体中心轨迹", "Part 1");
    await addImage(slide, path.join(ASSET_DIR, "angle_chart.png"), 60, 170, 560, 240, "contain", "角度曲线");
    await addImage(slide, path.join(ASSET_DIR, "speed_chart.png"), 660, 170, 560, 240, "contain", "速度曲线");
    await addImage(slide, path.join(ASSET_DIR, "com_chart.png"), 60, 445, 560, 200, "contain", "身体中心轨迹");
    addCard(slide, 660, 445, 560, 200, C.white);
    addText(slide, "读图重点", 690, 475, 160, 30, 24, { bold: true, color: C.blue });
    addText(slide, "曲线用于观察动作顺序和峰值出现时间。速度类和身体中心位移适合同设备、同机位复测看趋势，不单独作为医学或真实重心结论。", 690, 520, 480, 90, 19, { color: C.body });
    footer(slide, page++);
  }

  {
    const slide = presentation.slides.add();
    header(slide, "推荐展示图表", "比纯曲线更适合家长和教练快速判断", "Part 2");
    await addImage(slide, path.join(ASSET_DIR, "kinematic_dashboard.png"), 45, 155, 390, 250, "contain", "核心运动学仪表盘");
    await addImage(slide, path.join(ASSET_DIR, "pitch_phase_timeline.png"), 455, 155, 770, 250, "contain", "投球阶段时间轴");
    await addImage(slide, path.join(ASSET_DIR, "kinetic_chain_flow.png"), 45, 430, 780, 220, "contain", "动力链流");
    addCard(slide, 865, 430, 330, 220, C.white);
    addText(slide, "推荐保留", 895, 455, 180, 30, 24, { bold: true, color: C.blue });
    addText(slide, "1. 核心运动学仪表盘\n2. 投球阶段时间轴\n3. 动力链流\n4. 左右平衡/风险图\n\n纯角度曲线保留在原始数据页，不作为家长最终页主图。", 895, 500, 260, 120, 18, { color: C.body });
    footer(slide, page++);
  }

  for (let i = 0; i < Math.min(2, promptParts.length); i++) {
    const slide = presentation.slides.add();
    header(slide, `Part 2 输入 Prompt（${i + 1}/${Math.min(2, promptParts.length)}）`, "完整展示输入给模型的原始指标数据", "Part 2");
    addCard(slide, 60, 168, 1160, 475, C.dark, C.dark);
    addText(slide, promptParts[i], 86, 190, 1110, 420, 14, { color: "#e5e7eb" });
    footer(slide, page++);
  }

  {
    const slide = presentation.slides.add();
    header(slide, "大模型输出报告", "完整中文报告引用", "Part 2");
    addCard(slide, 60, 170, 1160, 470, C.white);
    addText(slide, reportParts.join("\n\n").slice(0, 1900), 90, 200, 1100, 400, 16, { color: C.body });
    footer(slide, page++);
  }

  {
    const slide = presentation.slides.add();
    header(slide, "核心结论可视化", "雷达图、热力图、左右平衡图", "Part 2");
    await addImage(slide, path.join(ASSET_DIR, "radar_chart.png"), 55, 165, 365, 255, "contain", "六维雷达图");
    await addImage(slide, path.join(ASSET_DIR, "balance_chart.png"), 455, 165, 365, 255, "contain", "左右平衡");
    await addImage(slide, path.join(ASSET_DIR, "deviation_heatmap.png"), 55, 445, 765, 210, "contain", "动作偏差热力图");
    addCard(slide, 850, 165, 350, 490, C.white);
    addText(slide, "提炼结论", 880, 195, 180, 30, 24, { bold: true, color: C.blue });
    addText(slide, "1. 出手侧速度、跨步长度和前脚方向是本次优先短板。\n2. 身体中心前移不足会限制下肢力量向上肢传递。\n3. 训练应先稳落脚和发力顺序，再追求手速。", 880, 245, 285, 220, 19, { color: C.body });
    addText(slide, "可视化数量：雷达图、热力图、左右平衡、曲线图、截图对照、标准姿态纠正图。", 880, 515, 285, 90, 17, { color: C.muted });
    footer(slide, page++);
  }

  {
    const slide = presentation.slides.add();
    header(slide, "儿童 vs 儿童对比", "Julian 与 Youyou 的同口径 3D 动作指标", "Part 2");
    await addImage(slide, path.join(ASSET_DIR, "child_compare_chart.png"), 55, 150, 790, 500, "contain", "Julian 与 Youyou 儿童对比图");
    addCard(slide, 880, 170, 310, 430, C.white);
    addText(slide, "阅读方式", 910, 200, 160, 30, 24, { bold: true, color: C.blue });
    addText(slide, "蓝点是 Julian，橙点是 Youyou，黑线是正常速度教练参考。\n\n这页不做儿童排名，只用来回答：孩子和同龄样本差在哪里、差多少。", 910, 250, 250, 150, 19, { color: C.body });
    addText(slide, "当前可比较：躯干速度、骨盆速度、髋肩分离、跨步长度、手臂速度、前脚方向。", 910, 445, 250, 90, 18, { color: C.muted });
    footer(slide, page++);
  }

  {
    const slide = presentation.slides.add();
    header(slide, "3D模型动态纠正展示", "教练标准按孩子身材缩放后叠加到孩子动作", "Part 2");
    await addImage(slide, path.join(ASSET_DIR, "standard_pose_overlay.gif"), 70, 165, 560, 420, "contain", "标准姿态动态纠正 GIF");
    await addImage(slide, path.join(ASSET_DIR, "standard_pose_overlay.png"), 670, 165, 500, 310, "contain", "标准姿态静态纠正图");
    addCard(slide, 670, 500, 500, 105, C.lightBlue, "#bfdbfe");
    addText(slide, "浅蓝虚线=孩子原始动作连线；绿色=按孩子身材缩放后的教练标准；红色=偏差较大的原始骨段。PPT 中左侧 GIF 可动态播放，适合课堂或家长沟通展示。", 695, 520, 450, 76, 18, { color: C.body });
    footer(slide, page++);
  }

  {
    const slide = presentation.slides.add();
    header(slide, "Part 3 针对性训练建议", "每条建议对应一个数据短板", "Part 3");
    const cards = [
      ["跨步落点练习", "每周 3 次，每次 3 组，每组 6 次", "解决：跨步长度偏短、身体前移不足"],
      ["前脚方向控制", "每周 3 次，每次 3 组，每组 8 次", "解决：前脚方向偏差、落脚后身体提前打开"],
      ["髋先转肩延迟", "每周 2-3 次，每次 3 组，每组 5 次", "解决：发力顺序和髋肩分离控制"],
      ["轻药球侧抛", "每周 2 次，每次 3 组，每组 6 次", "解决：下肢到躯干再到手臂的力量传递"],
    ];
    for (let i = 0; i < cards.length; i++) {
      const x = 60 + (i % 2) * 590;
      const y = 180 + Math.floor(i / 2) * 225;
      addCard(slide, x, y, 530, 180, i % 2 ? "#f0fdf4" : "#eff6ff");
      addText(slide, cards[i][0], x + 26, y + 24, 260, 30, 24, { bold: true, color: C.ink });
      addText(slide, cards[i][1], x + 26, y + 72, 420, 28, 18, { color: C.blue });
      addText(slide, cards[i][2], x + 26, y + 112, 430, 40, 18, { color: C.body });
    }
    footer(slide, page++);
  }

  {
    const slide = presentation.slides.add();
    header(slide, "复测与年度成长档案", "把训练后的变化量记录下来", "Part 3");
    const rows = [
      ["复测指标", "本次结果", "下次希望看到的变化"],
      ["跨步长度", "88.4 cm", "逐步接近教练水平，落脚后身体能跟上"],
      ["前脚方向", "33°", "脚尖更接近目标方向"],
      ["出手侧手速", "5.0 m/s", "动作顺序稳定后逐步提高"],
      ["髋肩分离", "-45°", "保持可训练范围，不追求越大越好"],
    ];
    const table = slide.tables.add({ rows: rows.length, columns: 3, left: 70, top: 175, width: 720, height: 280, values: rows });
    table.styleOptions = { headerRow: true, bandedRows: true };
    table.borders.assign({ style: "solid", fill: C.border, width: 1 });
    for (let c = 0; c < 3; c++) table.getCell(0, c).fill = "#eaf2ff";
    addCard(slide, 835, 175, 340, 280, C.white);
    addText(slide, "年度档案", 865, 205, 160, 30, 24, { bold: true, color: C.blue });
    addText(slide, "每 4 周复测一次，保留同一相位截图、关键指标、训练内容和身体反应。动态展示可放入 GIF 或 3D模型视频。", 865, 255, 280, 130, 18, { color: C.body });
    addText(slide, "动态视频：coach 3D标准模型与孩子纠正 GIF 已随 PPTX 资产嵌入。", 865, 400, 280, 50, 16, { color: C.muted });
    footer(slide, page++);
  }

  {
    const slide = presentation.slides.add();
    header(slide, "任务二：实验室采集复盘", "Vicon录制与数据处理工作总结", "Task 2");
    const sections = [
      ["我的职责", "主要负责操作实验室电脑，控制 Vicon 系统完成采集录制。录制过程中关注采集状态、动作开始结束时机和数据是否正常保存。", C.lightBlue],
      ["已完成工作", "实验结束后处理了一位小朋友的数据，包括 static、ROM、4 条 pitch 和 2 条 bat；完成补点、补 gap 等基础清理，为后续分析提供可用数据。", "#f0fdf4"],
      ["整体感受", "上次录制中，我负责的电脑操作和数据处理部分整体进展比较顺利，没有出现明显流程中断。", "#fff7ed"],
    ];
    for (let i = 0; i < sections.length; i++) {
      const [title, body, fill] = sections[i];
      const x = 70 + i * 390;
      addCard(slide, x, 180, 350, 330, fill);
      addText(slide, title, x + 26, 210, 220, 34, 25, { bold: true, color: C.ink });
      addText(slide, body, x + 26, 270, 295, 180, 19, { color: C.body });
    }
    addCard(slide, 70, 545, 1110, 85, C.white);
    addText(slide, "复盘重点", 100, 565, 130, 28, 22, { bold: true, color: C.blue });
    addText(slide, "采集本身较顺利，但儿童站位会直接影响顶部摄像机能否拍到 marker 点，因此下一次需要把“站位中心区域”前置为采集检查项。", 245, 565, 850, 44, 18, { color: C.body });
    footer(slide, page++);
  }

  {
    const slide = presentation.slides.add();
    header(slide, "本周采集优化建议", "把站位和可见性检查前置，减少后期补点压力", "Task 2");
    const items = [
      ["遇到的问题", "小朋友站位偏离中心区域时，顶部摄像机容易拍不到 marker 点，后期更容易出现缺点和 gap。", "#fff7ed"],
      ["优化建议", "提前在地板上用胶带标出最佳站位范围，让小朋友在范围内完成 static、ROM、pitch 和 bat 录制。", "#eff6ff"],
      ["采集前检查", "开始录制前先确认顶部摄像机视角中关键 marker 可见，必要时让被试微调站位后再开始。", "#f0fdf4"],
      ["待确认事项", "确认胶带标记范围、顶部摄像机可见区域、每位小朋友的动作顺序和文件命名规则。", "#ffffff"],
    ];
    for (let i = 0; i < items.length; i++) {
      const [title, body, fill] = items[i];
      const x = 70 + (i % 2) * 390;
      const y = 175 + Math.floor(i / 2) * 205;
      addCard(slide, x, y, 350, 165, fill);
      addText(slide, title, x + 24, y + 22, 200, 28, 22, { bold: true, color: C.ink });
      addText(slide, body, x + 24, y + 62, 290, 78, 17, { color: C.body });
    }
    addCard(slide, 865, 175, 330, 375, "#eff6ff", "#bfdbfe");
    addText(slide, "执行清单", 915, 205, 160, 30, 24, { bold: true, color: C.blue });
    addText(slide, "1. 地板贴胶带标中心范围\n2. static 前检查顶部视角\n3. 每条动作开始前确认站位\n4. 录制后立即抽查缺点情况\n5. 统一命名并记录异常", 915, 255, 245, 175, 19, { color: C.body });
    addText(slide, "目标：减少顶部摄像机漏点，降低后期补点和补 gap 的工作量。", 915, 470, 245, 50, 18, { color: C.orange });
    footer(slide, page++);
  }

  for (const [idx, slide] of presentation.slides.items.entries()) {
    const stem = `slide-${String(idx + 1).padStart(2, "0")}`;
    const png = await presentation.export({ slide, format: "png", scale: 1 });
    await fs.writeFile(path.join(PREVIEW_DIR, `${stem}.png`), new Uint8Array(await png.arrayBuffer()));
  }
  const montage = await presentation.export({ format: "webp", montage: true, scale: 1 });
  await fs.writeFile(path.join(PREVIEW_DIR, "deck-montage.webp"), new Uint8Array(await montage.arrayBuffer()));
  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(OUT);
  await fs.writeFile(path.join(QA_DIR, "visual-qa.txt"), "已渲染所有幻灯片并导出 PPTX；后续由脚本运行后的预览图进行人工检查。\n");
  console.log(OUT);
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
