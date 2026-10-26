// Build the progress check-in deck from the generated data in ../data.
// usage: NODE_PATH=<dir with pptxgenjs, sharp, lucide> node build_slides.js
// The map and every number on the slides are read from the data files, so
// regenerating the data and rebuilding keeps the deck honest.
const fs = require("fs"), path = require("path");
const pptxgen = require("pptxgenjs");
const sharp = require("sharp");
const lucide = require("lucide");

const ROOT = path.join(__dirname, "..");
const SKILL = process.env.PPTX_SKILL_DIR;
const { applyTheme } = require(path.join(SKILL, "scripts", "apply_theme.js"));

const load = (scn, f) => JSON.parse(fs.readFileSync(path.join(ROOT, "data", scn, f)));
const SCN = ["normal", "surge", "constrained"];
const SUM = Object.fromEntries(SCN.map(s => [s, load(s, "summary.json")]));
const REPORT = JSON.parse(fs.readFileSync(path.join(ROOT, "data", "validation_report.json")));
const SURGE = {
  hospitals: load("surge", "hospitals.json"),
  stations: load("surge", "stations.json"),
  incidents: load("surge", "incidents.json"),
  ambulances: load("surge", "ambulances.json"),
};
const checks = Object.values(REPORT).flat();
const passed = checks.filter(c => c.passed).length;
const cluster = SURGE.incidents.filter(i => i.mci_cluster);
const MCI = SUM.surge.mci_cluster; // patients excludes BLACK (expectant) victims

// Dispatch-board palette: ink and slate carry the deck, signal red is the one
// accent, and the START triage colours are reserved for patients on the map.
const THEME = {
  name: "Dispatch Board",
  headFontFace: "Arial",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B1F24", lt1: "FFFFFF", dk2: "2B3440", lt2: "EEF1F4",
    accent1: "D7263D", // RED triage, signal accent
    accent2: "E8A317", // YELLOW triage
    accent3: "2E9E5B", // GREEN triage, done
    accent4: "3C6E91", // agents and system parts
    accent5: "5F6B78", // muted text on light
    accent6: "A3ADB8", // muted text on dark
    hlink: "3C6E91", folHlink: "5F6B78",
  },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.author = "Shubham Bagwe";
pres.title = "Emergency Response Coordination: simulation design and data progress";
pres.subject = "CSYE7380 mid-term project progress check-in";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const TRIAGE = { RED: C.accent1, YELLOW: C.accent2, GREEN: C.accent3, BLACK: C.text1 };

async function icon(name, hex = "FFFFFF") {
  const inner = lucide[name].map(([tag, attrs]) =>
    `<${tag} ${Object.entries(attrs).map(([k, v]) => `${k}="${v}"`).join(" ")}/>`).join("");
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 24 24" fill="none" stroke="#${hex}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${inner}</svg>`;
  const buf = await sharp(Buffer.from(svg)).resize(256, 256).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

// ---------- layouts ----------
pres.defineSlideMaster({
  title: "Title Dark",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 1.85, w: 7.4, h: 1.4,
      fontFace: THEME.headFontFace, fontSize: 40, bold: true, color: C.background1, align: "left", valign: "top", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "subtitle", type: "body", x: 0.7, y: 3.35, w: 7.4, h: 0.5,
      fontFace: THEME.bodyFontFace, fontSize: 20, color: C.background2, align: "left", valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "Content",
  background: { color: C.background1 },
  margin: [0.4, 0.6, 0.5, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.38, w: 12.13, h: 0.68,
      fontFace: THEME.headFontFace, fontSize: 34, bold: true, color: C.text1, align: "left", valign: "top", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "subtitle", type: "body", x: 0.6, y: 1.06, w: 12.13, h: 0.42,
      fontFace: THEME.bodyFontFace, fontSize: 16, color: C.accent5, align: "left", valign: "top", margin: 0 }, text: "" } },
    { text: { text: "CSYE7380 · Shubham Bagwe · Mid-term progress check-in",
      options: { x: 0.6, y: 7.05, w: 8, h: 0.25, fontFace: THEME.bodyFontFace, fontSize: 10, color: C.accent5, margin: 0 } } },
  ],
  slideNumber: { x: 12.33, y: 7.05, w: 0.4, h: 0.25, fontFace: THEME.bodyFontFace, fontSize: 10, color: C.accent5, align: "right" },
});

// ---------- schematic map (shared by title and data slides) ----------
function drawMap(slide, x0, y0, size, { dark = false, labels = true } = {}) {
  const s = size / 20; // inches per km
  const px = x => x0 + x * s, py = y => y0 + (20 - y) * s; // north is up
  slide.addShape(pres.shapes.RECTANGLE, { x: x0, y: y0, w: size, h: size,
    fill: { color: dark ? C.text2 : C.background2 }, line: { type: "none" }, objectName: "map-ground" });
  for (let k = 2; k < 20; k += 2) {
    const gl = { color: dark ? C.text1 : C.background1, width: 0.75 };
    slide.addShape(pres.shapes.LINE, { x: px(k), y: y0, w: 0, h: size, line: { ...gl }, objectName: `grid-x${k}` });
    slide.addShape(pres.shapes.LINE, { x: x0, y: py(k), w: size, h: 0, line: { ...gl }, objectName: `grid-y${k}` });
  }
  // downtown congestion zone, 6 km radius around (10, 10)
  slide.addShape(pres.shapes.OVAL, { x: px(4), y: py(16), w: 12 * s, h: 12 * s,
    fill: { color: C.accent2, transparency: dark ? 88 : 84 }, line: { type: "none" }, objectName: "downtown" });
  if (labels && cluster.length) {
    const xs = cluster.map(i => i.x), ys = cluster.map(i => i.y), pad = 1.3;
    const cx0 = Math.min(...xs) - pad, cx1 = Math.max(...xs) + pad;
    const cy0 = Math.min(...ys) - pad, cy1 = Math.max(...ys) + pad;
    slide.addShape(pres.shapes.OVAL, { x: px(cx0), y: py(cy1), w: (cx1 - cx0) * s, h: (cy1 - cy0) * s,
      fill: { color: C.accent1, transparency: 92 }, line: { color: C.accent1, width: 1.25, dashType: "dash" }, objectName: "mci-ring" });
    slide.addText([
      { text: "Mass-casualty cluster", options: { bold: true, color: C.accent1, breakLine: true } },
      { text: `${MCI.patients} patients (${MCI.red} RED) in ${MCI.span_min} min vs ${MCI.seats} seats, ${MCI.als_units} ALS units`, options: { color: C.text1 } },
    ], { x: px(cx1) + 0.05, y: y0 + 0.08, w: 2.3, h: 0.5, fontSize: 11, margin: 0, valign: "top",
      fill: { color: C.background1, transparency: 15 },
      isTextBox: true, objectName: "mci-label" });
  }
  for (const st of SURGE.stations) {
    slide.addShape(pres.shapes.ISOSCELES_TRIANGLE, { x: px(st.x) - 0.1, y: py(st.y) - 0.09, w: 0.2, h: 0.18,
      fill: { color: dark ? C.accent6 : C.accent5 }, line: { type: "none" }, objectName: `station-${st.id}` });
  }
  for (const inc of SURGE.incidents) {
    const d = 0.13 + 0.075 * Math.sqrt(inc.true_victims);
    slide.addShape(pres.shapes.OVAL, { x: px(inc.x) - d / 2, y: py(inc.y) - d / 2, w: d, h: d,
      fill: { color: TRIAGE[inc.worst_triage] }, line: { color: dark ? C.text2 : C.background1, width: 1 },
      objectName: `incident-${inc.incident_id}` });
  }
  for (const h of SURGE.hospitals) {
    const w = 0.34;
    slide.addText(h.id, { shape: pres.shapes.RECTANGLE, x: px(h.x) - w / 2, y: py(h.y) - w / 2, w, h: w,
      fill: { color: dark ? C.background1 : C.text1 }, color: dark ? C.text1 : C.background1,
      fontSize: 10, bold: true, align: "center", valign: "middle", margin: 0, objectName: `hospital-${h.id}` });
  }
}

function legend(slide, x, y, dark = false) {
  const ink = dark ? C.background2 : C.text1;
  const items = [["RED", "RED"], ["YELLOW", "YELLOW"], ["GREEN", "GREEN"]];
  let cx = x;
  for (const [lbl, key] of items) {
    slide.addShape(pres.shapes.OVAL, { x: cx, y: y + 0.05, w: 0.16, h: 0.16, fill: { color: TRIAGE[key] }, line: { type: "none" }, objectName: `legend-${key}` });
    slide.addText(lbl, { x: cx + 0.22, y, w: 0.7, h: 0.26, fontSize: 11, color: ink, margin: 0, valign: "middle", isTextBox: true });
    cx += 0.4 + 0.095 * lbl.length;
  }
  slide.addShape(pres.shapes.RECTANGLE, { x: cx, y: y + 0.04, w: 0.18, h: 0.18, fill: { color: dark ? C.background1 : C.text1 }, line: { type: "none" }, objectName: "legend-hospital" });
  slide.addText("Hospital", { x: cx + 0.25, y, w: 0.7, h: 0.26, fontSize: 11, color: ink, margin: 0, valign: "middle", isTextBox: true });
  cx += 0.95;
  slide.addShape(pres.shapes.ISOSCELES_TRIANGLE, { x: cx, y: y + 0.04, w: 0.18, h: 0.17, fill: { color: dark ? C.accent6 : C.accent5 }, line: { type: "none" }, objectName: "legend-station" });
  slide.addText("Station", { x: cx + 0.25, y, w: 0.7, h: 0.26, fontSize: 11, color: ink, margin: 0, valign: "middle", isTextBox: true });
}

async function build() {
  const ICONS = {
    data: await icon("Database"), engine: await icon("Cpu"), dash: await icon("LayoutDashboard"),
    assess: await icon("ClipboardList"), dispatch: await icon("Ambulance"),
    hospital: await icon("Hospital"), alloc: await icon("Scale"), check: await icon("Check"),
    log: await icon("ScrollText"),
  };

  // ---------- 1. title ----------
  pres.addSection({ title: "Title" });
  const s1 = pres.addSlide({ masterName: "Title Dark", sectionTitle: "Title" });
  s1.addText("MID-TERM PROJECT · PROGRESS CHECK-IN · 21 OCT 2026", { x: 0.7, y: 1.45, w: 7.4, h: 0.3,
    fontSize: 12, bold: true, color: C.accent2, charSpacing: 2, margin: 0, isTextBox: true });
  s1.addText("Emergency Response Coordination", { placeholder: "title" });
  s1.addText("Simulation design and synthetic data", { placeholder: "subtitle" });
  const item = (label, value, x, y, w, h) => s1.addText([
    { text: label.toUpperCase(), options: { fontSize: 11, bold: true, color: C.accent6, charSpacing: 2, breakLine: true } },
    { text: value, options: { fontSize: 16, color: C.background1 } },
  ], { x, y, w, h, margin: 0, valign: "top", paraSpaceAfter: 4, isTextBox: true, objectName: `label-${label}` });
  item("Topic", "Intelligent Agents and Multi-Agent Systems for Emergency Response Coordination", 0.7, 4.2, 7.4, 0.85);
  item("Course", "CSYE7380 Theory & Prac App AI Gen Model", 0.7, 5.3, 2.6, 1.0);
  item("Name", "Shubham Bagwe", 3.5, 5.3, 2.1, 1.0);
  item("Instructor", "Prof. Subrata Das", 5.8, 5.3, 2.3, 1.0);
  drawMap(s1, 8.75, 1.45, 3.9, { dark: true, labels: false });
  legend(s1, 8.75, 5.45, true);
  s1.addText(`Surge scenario from the generated data: ${SUM.surge.incidents} incidents, ${SUM.surge.victims} victims, 3 hospitals`,
    { x: 8.75, y: 5.85, w: 3.9, h: 0.5, fontSize: 11, color: C.accent6, margin: 0, valign: "top", isTextBox: true });
  s1.addNotes("Progress check-in for the mid-term project. Two things today: the plan for the simulation and agents, and the synthetic data, which is already generated and validated. The map on the right is the real surge scenario the agents will work on.");

  // ---------- 2. design plan ----------
  pres.addSection({ title: "Design plan" });
  const s2 = pres.addSlide({ masterName: "Content", sectionTitle: "Design plan" });
  s2.addText("Simulation design plan", { placeholder: "title" });
  s2.addText("One rule-checked engine owns the world; four LLM agents propose, negotiate and commit through typed messages",
    { placeholder: "subtitle" });

  // row A: data -> engine -> dashboard
  const boxA = (x, w, ico, title, cap, darkFill) => {
    s2.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.7, w, h: 1.05, rectRadius: 0.08,
      fill: { color: darkFill ? C.text2 : C.background2 }, line: { type: "none" }, objectName: `flow-${title}` });
    // icons are white, so light boxes get a slate disc behind them
    if (!darkFill) {
      s2.addShape(pres.shapes.OVAL, { x: x + 0.1, y: 1.8, w: 0.4, h: 0.4, fill: { color: C.accent4 }, line: { type: "none" }, objectName: `disc-${ico}` });
    }
    s2.addImage({ data: ICONS[ico], x: x + 0.15, y: 1.85, w: 0.3, h: 0.3 });
    s2.addText([
      { text: title, options: { fontSize: 14, bold: true, color: darkFill ? C.background1 : C.text1, breakLine: true } },
      { text: cap, options: { fontSize: 12, color: darkFill ? C.background2 : C.accent5 } },
    ], { x: x + 0.55, y: 1.78, w: w - 0.65, h: 0.92, margin: 0, valign: "top", isTextBox: true });
  };
  boxA(0.6, 2.0, "data", "Synthetic data", "Seeded JSON + CSV, 3 scenarios", false);
  boxA(2.9, 2.35, "engine", "Simulation engine", "1-min ticks; rejects any action that breaks a rule", true);
  boxA(5.55, 2.45, "dash", "Streamlit dashboard", "Map, resources, agent activity, negotiation history, KPIs", false);
  for (const x of [2.62, 5.27]) {
    s2.addShape(pres.shapes.LINE, { x, y: 2.225, w: 0.26, h: 0, line: { color: C.accent5, width: 1.5, endArrowType: "triangle" }, objectName: `arrow-${x}` });
  }
  s2.addShape(pres.shapes.LINE, { x: 4.075, y: 2.78, w: 0, h: 0.34,
    line: { color: C.accent5, width: 1.5, beginArrowType: "triangle", endArrowType: "triangle" }, objectName: "arrow-tools" });
  s2.addText("tools: read state · propose · commit", { x: 4.22, y: 2.8, w: 3.5, h: 0.3, fontSize: 11, italic: true,
    color: C.accent5, margin: 0, valign: "middle", isTextBox: true });

  s2.addText([
    { text: "Bus: PRIORITY_REPORT · PROPOSE · ACCEPT · REJECT · COUNTER · ESCALATE · COMMIT", options: { fontSize: 11, bold: true, breakLine: true } },
    { text: "every message: sender · recipient · type · incident_id · proposed_action · rationale", options: { fontSize: 11, italic: true } },
  ], { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.08, x: 0.6, y: 3.15, w: 7.4, h: 0.66,
    fill: { color: C.accent4 }, color: C.background1, align: "center", valign: "middle", margin: 0,
    objectName: "message-bus" });

  const agents = [
    ["assess", "Incident Assessment", "Reads caller reports, scores urgency 0–100, estimates the triage mix", "sickest incidents first"],
    ["dispatch", "Dispatch", "Assigns ambulances and responder teams by urgency, ETA and availability", "send help now"],
    ["hospital", "Hospital Coordination ×3", "Accepts, rejects or counters each admission; tracks beds and staff", "keep beds free for RED"],
    ["alloc", "Resource Allocation", "Settles clashes; lends float doctors and nurses between hospitals", "most lives saved overall"],
  ];
  agents.forEach(([ico, name, role, wants], k) => {
    const x = 0.6 + (k % 2) * 3.8, y = 3.95 + Math.floor(k / 2) * 1.5;
    s2.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 3.6, h: 1.35, rectRadius: 0.08,
      fill: { color: C.background2 }, line: { type: "none" }, objectName: `agent-card-${ico}` });
    s2.addShape(pres.shapes.OVAL, { x: x + 0.15, y: y + 0.15, w: 0.55, h: 0.55, fill: { color: C.accent4 }, line: { type: "none" }, objectName: `agent-disc-${ico}` });
    s2.addImage({ data: ICONS[ico], x: x + 0.27, y: y + 0.27, w: 0.31, h: 0.31 });
    s2.addText([
      { text: name, options: { fontSize: 15, bold: true, color: C.text1, breakLine: true } },
      { text: role, options: { fontSize: 13, color: C.text2, breakLine: true } },
      { text: "Wants: ", options: { fontSize: 12, bold: true, color: C.accent1 } },
      { text: wants, options: { fontSize: 12, italic: true, color: C.text2 } },
    ], { x: x + 0.85, y: y + 0.12, w: 2.62, h: 1.2, margin: 0, valign: "top", paraSpaceAfter: 2, isTextBox: true, objectName: `agent-text-${ico}` });
  });

  // right column: protocol and evaluation
  s2.addText("How a conflict is resolved", { x: 8.45, y: 1.8, w: 4.28, h: 0.35, fontSize: 16, bold: true, color: C.text1, margin: 0, isTextBox: true });
  const steps = [
    ["Assessment posts an urgency score", 0.45],
    ["Dispatch proposes unit + hospital", 0.45],
    ["Hospital accepts, rejects or counters", 0.45],
    ["Clashes escalate; after 3 rounds Resource Allocation decides by a survival-weighted score and commits", 0.95],
  ];
  let sy = 2.28;
  steps.forEach(([txt, h], k) => {
    s2.addText(String(k + 1), { shape: pres.shapes.OVAL, x: 8.45, y: sy + 0.02, w: 0.32, h: 0.32,
      fill: { color: C.text1 }, color: C.background1, fontSize: 12, bold: true, align: "center", valign: "middle", margin: 0, objectName: `step-${k + 1}` });
    s2.addText(txt, { x: 8.95, y: sy, w: 3.78, h, fontSize: 14, color: C.text1, margin: 0, valign: "top", isTextBox: true });
    sy += h;
  });
  s2.addText("How we will prove it helps", { x: 8.45, y: 4.65, w: 4.28, h: 0.35, fontSize: 16, bold: true, color: C.text1, margin: 0, isTextBox: true });
  [["B0", "Greedy rules, no messages"], ["B1", "Agents, no negotiation"], ["Full", "Agents + negotiation"]].forEach(([tag, txt], k) => {
    const y = 5.1 + k * 0.42;
    s2.addText(tag, { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.06, x: 8.45, y, w: 0.6, h: 0.3,
      fill: { color: k === 2 ? C.accent1 : C.background2 }, color: k === 2 ? C.background1 : C.text1,
      fontSize: 12, bold: true, align: "center", valign: "middle", margin: 0, objectName: `cond-${tag}` });
    s2.addText(txt, { x: 9.2, y, w: 3.5, h: 0.3, fontSize: 14, color: C.text1, margin: 0, valign: "middle", isTextBox: true });
  });
  s2.addText("Same scenarios for all three; LLM runs repeated over several seeds (mean ± spread). Metrics: response time, RED in golden window, victims served, unserved critical, utilisation, conflicts.",
    { x: 8.45, y: 6.3, w: 4.28, h: 0.72, fontSize: 12, color: C.accent5, margin: 0, valign: "top", isTextBox: true });
  s2.addNotes("The engine is deterministic and is the only thing that changes state. Agents are called on events (a new incident, a freed unit, a rejection), not on every tick, which keeps LLM cost and latency down. Agents built with the OpenAI Agents SDK read state and propose actions through function tools; the engine rejects anything invalid, so an LLM mistake cannot break the simulation. The four agents have conflicting goals on purpose: hospitals want to hold beds for critical cases, dispatch wants to place patients now. Conflicts go through a contract-net style protocol, capped at three rounds, then Resource Allocation decides. To show negotiation actually helps, the same seeded scenarios run under a greedy baseline, agents without negotiation, and the full system.");

  // ---------- 3. data progress ----------
  pres.addSection({ title: "Data progress" });
  const s3 = pres.addSlide({ masterName: "Content", sectionTitle: "Data progress" });
  s3.addText("Synthetic data: generated and validated", { placeholder: "title" });
  s3.addText("Seeded, reproducible scenarios with contention built in, ready for the agents to work on", { placeholder: "subtitle" });
  drawMap(s3, 0.6, 1.75, 4.4);
  legend(s3, 0.6, 6.25);
  s3.addText(`Surge scenario, seed 7380, 20 × 20 km grid. Marker size = victims, colour = worst triage. Shaded circle = downtown, travel ×1.5.`,
    { x: 0.6, y: 6.55, w: 4.4, h: 0.4, fontSize: 11, color: C.accent5, margin: 0, valign: "top", isTextBox: true });

  const stats = [[String(SCN.length), "scenarios that share incidents, so results compare"],
    [String(SUM.surge.incidents), `incidents in surge (${SUM.normal.incidents} base + ${MCI.incidents} cluster)`],
    [String(SUM.surge.victims), "surge victims with START triage"], [`${passed}/${checks.length}`, "validation checks pass"]];
  stats.forEach(([big, lbl], k) => {
    const x = 5.55 + k * 1.83;
    s3.addText([
      { text: big, options: { fontSize: 32, bold: true, color: k === 3 ? C.accent3 : C.text1, breakLine: true } },
      { text: lbl, options: { fontSize: 12, color: C.accent5 } },
    ], { x, y: 1.75, w: 1.68, h: 1.05, margin: 0, valign: "top", isTextBox: true, objectName: `stat-${k}` });
  });

  s3.addText("Data pipeline", { x: 5.55, y: 3.0, w: 7.18, h: 0.35, fontSize: 16, bold: true, color: C.text1, margin: 0, isTextBox: true });
  const pipe = [
    ["Schema: 7 entity types, fixed city", true], ["Caller report vs hidden truth", true],
    ["Poisson arrivals, START triage", true], ["Travel-time matrix", true],
    ["Mass-casualty cluster for contention", true], ["Validation + SHA-256 manifest", true],
    ["Controlled scenarios: same incidents", true], ["Next: LLM-written caller reports", false],
  ];
  pipe.forEach(([txt, done], k) => {
    const x = 5.55 + (k % 2) * 3.65, y = 3.42 + Math.floor(k / 2) * 0.42;
    if (done) {
      s3.addShape(pres.shapes.OVAL, { x, y: y + 0.03, w: 0.26, h: 0.26, fill: { color: C.accent3 }, line: { type: "none" }, objectName: `pipe-done-${k}` });
      s3.addImage({ data: ICONS.check, x: x + 0.05, y: y + 0.08, w: 0.16, h: 0.16 });
    } else {
      s3.addShape(pres.shapes.OVAL, { x, y: y + 0.03, w: 0.26, h: 0.26, fill: { color: C.background1 }, line: { color: C.accent5, width: 1.25, dashType: "dash" }, objectName: `pipe-next-${k}` });
    }
    s3.addText(txt, { x: x + 0.38, y, w: 3.2, h: 0.32, fontSize: 14, color: done ? C.text1 : C.accent5, italic: !done,
      margin: 0, valign: "middle", isTextBox: true });
  });

  s3.addText("Plan to submission", { x: 5.55, y: 5.25, w: 7.18, h: 0.35, fontSize: 16, bold: true, color: C.text1, margin: 0, isTextBox: true });
  const tl = [["Today, 21 Oct", "Design + data", true], ["22–25 Oct", "Engine + baseline", false], ["26–29 Oct", "Agents + negotiation", false],
    ["30 Oct–1 Nov", "Streamlit dashboard", false], ["2–3 Nov", "Evaluation + report", false], ["4 Nov", "Submit 18:00", false]];
  const tx0 = 6.2, tdx = 1.17, ty = 5.95;
  s3.addShape(pres.shapes.LINE, { x: tx0, y: ty, w: tdx * 5, h: 0, line: { color: C.accent6, width: 1.5 }, objectName: "timeline" });
  s3.addShape(pres.shapes.LINE, { x: tx0, y: ty, w: tdx * 0.5, h: 0, line: { color: C.accent3, width: 3 }, objectName: "timeline-progress" });
  tl.forEach(([date, task, done], k) => {
    const cx = tx0 + k * tdx;
    if (done) {
      s3.addShape(pres.shapes.OVAL, { x: cx - 0.15, y: ty - 0.15, w: 0.3, h: 0.3, fill: { color: C.accent3 }, line: { type: "none" }, objectName: `tl-node-${k}` });
      s3.addImage({ data: ICONS.check, x: cx - 0.09, y: ty - 0.09, w: 0.18, h: 0.18 });
    } else {
      s3.addShape(pres.shapes.OVAL, { x: cx - 0.12, y: ty - 0.12, w: 0.24, h: 0.24, fill: { color: C.background1 },
        line: { color: k === tl.length - 1 ? C.accent1 : C.accent6, width: 1.5 }, objectName: `tl-node-${k}` });
    }
    s3.addText([
      { text: date, options: { fontSize: 12, bold: true, color: done ? C.accent3 : C.text1, breakLine: true } },
      { text: task, options: { fontSize: 12, color: C.text2 } },
    ], { x: cx - 0.57, y: ty + 0.22, w: 1.14, h: 0.65, align: "center", margin: 0, valign: "top", isTextBox: true, objectName: `tl-label-${k}` });
  });

  const sv = SUM.surge, cn = SUM.constrained, nm = SUM.normal;
  s3.addNotes(`The data generator is done: python -m emergency_sim.datagen writes three scenarios from seed 7380, and python -m emergency_sim.validate regenerates each one and compares SHA-256 hashes, so the data is byte-identical on the same Python version (tested on 3.14). ` +
    `normal: ${nm.incidents} incidents, ${nm.victims} victims, ${nm.er_beds_free_t0}/${nm.er_beds_total} ER beds free. ` +
    `surge: ${sv.incidents} incidents, ${sv.victims} victims, ${sv.er_beds_free_t0}/${sv.er_beds_total} ER beds free, and a mass-casualty cluster of ${MCI.patients} patients (${MCI.red} RED) within ${MCI.span_min} minutes, more than the ${MCI.seats} ambulance seats. ` +
    `constrained: exactly the surge incidents, but only ${cn.er_beds_free_t0}/${cn.er_beds_total} ER beds free, one ambulance out of service and fewer staff. Because scenarios share their random incident streams, differences in results come from resources, not from different incidents. ` +
    `Agents only see the caller report and a noisy victim count; the true victims are kept for scoring. Next step for the data is an LLM pass that rewrites caller reports in varied wording, cached to a file so reruns stay reproducible.`);

  // ---------- 4. system flow ----------
  pres.addSection({ title: "System flow" });
  const s4 = pres.addSlide({ masterName: "Content", sectionTitle: "System flow" });
  s4.addText("System flow: data to decision", { placeholder: "title" });
  s4.addText("Components across the top, the six agents in the middle, the negotiation protocol that resolves conflicts below",
    { placeholder: "subtitle" });

  // top pipeline: data -> engine -> log -> dashboard
  const PY = 1.65, PH = 0.95;
  const flowBox = (x, w, ico, title, cap, darkFill) => {
    s4.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: PY, w, h: PH, rectRadius: 0.08,
      fill: { color: darkFill ? C.text2 : C.background2 }, line: { type: "none" }, objectName: `s4-flow-${title}` });
    if (!darkFill) {
      s4.addShape(pres.shapes.OVAL, { x: x + 0.1, y: PY + 0.1, w: 0.36, h: 0.36, fill: { color: C.accent4 }, line: { type: "none" }, objectName: `s4-disc-${title}` });
    }
    s4.addImage({ data: ICONS[ico], x: x + 0.14, y: PY + 0.14, w: 0.28, h: 0.28 });
    s4.addText([
      { text: title, options: { fontSize: 13, bold: true, color: darkFill ? C.background1 : C.text1, breakLine: true } },
      { text: cap, options: { fontSize: 10.5, color: darkFill ? C.background2 : C.accent5 } },
    ], { x: x + 0.5, y: PY + 0.08, w: w - 0.6, h: PH - 0.16, margin: 0, valign: "top", isTextBox: true });
  };
  const px1 = 0.6, pw = 2.8, pgap = 0.25;
  flowBox(px1, pw, "data", "Synthetic data", "Seeded scenarios, 3 conditions", false);
  flowBox(px1 + (pw + pgap), pw, "engine", "Simulation engine", "Owns state; rejects invalid actions", true);
  flowBox(px1 + 2 * (pw + pgap), pw, "log", "Event log", "Every message, JSONL", false);
  flowBox(px1 + 3 * (pw + pgap), pw, "dash", "Streamlit dashboard", "Map, KPIs, negotiation history", false);
  for (let k = 0; k < 3; k++) {
    const gx = px1 + k * (pw + pgap) + pw;
    s4.addShape(pres.shapes.LINE, { x: gx, y: PY + PH / 2, w: pgap, h: 0,
      line: { color: C.accent5, width: 1.5, endArrowType: "triangle" }, objectName: `s4-pipe-arrow-${k}` });
  }

  // vertical link: engine <-> agents
  const engCx = px1 + (pw + pgap) + pw / 2;
  s4.addShape(pres.shapes.LINE, { x: engCx, y: PY + PH + 0.03, w: 0, h: 0.34,
    line: { color: C.accent5, width: 1.5, beginArrowType: "triangle", endArrowType: "triangle" }, objectName: "s4-arrow-tools" });
  s4.addText("tools: read state · propose · commit", { x: engCx + 0.15, y: PY + PH + 0.04, w: 4.2, h: 0.32,
    fontSize: 11, italic: true, color: C.accent5, margin: 0, valign: "middle", isTextBox: true });

  // middle: six agents
  const AY = PY + PH + 0.42, AW = 1.89, AH = 1.0, agap = 0.15;
  const agentRow = [
    ["assess", "Incident Assessment", "urgency 0–100"],
    ["dispatch", "Dispatch", "assigns units"],
    ["hospital", "Hospital H1", "trauma + burns"],
    ["hospital", "Hospital H2", "general"],
    ["hospital", "Hospital H3", "paediatric"],
    ["alloc", "Resource Alloc.", "resolves clashes"],
  ];
  agentRow.forEach(([ico, title, cap], k) => {
    const x = 0.6 + k * (AW + agap);
    s4.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: AY, w: AW, h: AH, rectRadius: 0.08,
      fill: { color: C.background2 }, line: { type: "none" }, objectName: `s4-agent-${k}` });
    s4.addShape(pres.shapes.OVAL, { x: x + (AW - 0.34) / 2, y: AY + 0.1, w: 0.34, h: 0.34,
      fill: { color: C.accent4 }, line: { type: "none" }, objectName: `s4-agent-disc-${k}` });
    s4.addImage({ data: ICONS[ico], x: x + (AW - 0.2) / 2, y: AY + 0.17, w: 0.2, h: 0.2 });
    s4.addText(title, { x: x + 0.06, y: AY + 0.5, w: AW - 0.12, h: 0.26, fontSize: 11.5, bold: true,
      color: C.text1, align: "center", margin: 0, valign: "top", isTextBox: true });
    s4.addText(cap, { x: x + 0.06, y: AY + 0.76, w: AW - 0.12, h: 0.22, fontSize: 9.5, italic: true,
      color: C.accent5, align: "center", margin: 0, valign: "top", isTextBox: true });
  });

  // bottom: negotiation protocol, five steps
  const SY = AY + AH + 0.28;
  s4.addText("How a conflict is resolved", { x: 0.6, y: SY, w: 6, h: 0.3, fontSize: 15, bold: true, color: C.text1, margin: 0, isTextBox: true });
  const s4ty = SY + 0.75, s4tx0 = 1.5, s4tdx = 2.45;
  s4.addShape(pres.shapes.LINE, { x: s4tx0, y: s4ty, w: s4tdx * 4, h: 0, line: { color: C.accent6, width: 1.5 }, objectName: "s4-steps-line" });
  const negSteps = [
    ["Priority report", "urgency, triage mix"],
    ["Propose", "unit + hospital"],
    ["Respond", "accept / reject / counter"],
    ["Escalate", "on any clash"],
    ["Commit", "≤3 rounds, survival-weighted"],
  ];
  negSteps.forEach(([title, cap], k) => {
    const cx = s4tx0 + k * s4tdx;
    s4.addText(String(k + 1), { shape: pres.shapes.OVAL, x: cx - 0.15, y: s4ty - 0.15, w: 0.3, h: 0.3,
      fill: { color: k === 4 ? C.accent1 : C.text1 }, color: C.background1, fontSize: 12, bold: true,
      align: "center", valign: "middle", margin: 0, objectName: `s4-step-${k}` });
    s4.addText([
      { text: title, options: { fontSize: 12.5, bold: true, color: C.text1, breakLine: true } },
      { text: cap, options: { fontSize: 10.5, color: C.accent5 } },
    ], { x: cx - 1.1, y: s4ty + 0.22, w: 2.2, h: 0.6, align: "center", margin: 0, valign: "top", isTextBox: true, objectName: `s4-step-label-${k}` });
  });
  s4.addText("The engine applies only the final COMMIT; every PROPOSE, ACCEPT, REJECT, COUNTER and ESCALATE is logged and replayed in the dashboard.",
    { x: 0.6, y: s4ty + 1.0, w: 12.13, h: 0.4, fontSize: 11, italic: true, color: C.accent5, margin: 0, valign: "top", isTextBox: true });
  s4.addNotes("This slide is the system flow end to end. Across the top: seeded synthetic data feeds the deterministic engine, which is the only thing that changes state; every message is written to an event log, which the Streamlit dashboard replays. In the middle are the six agent instances: Incident Assessment, Dispatch, one Hospital Coordination agent per hospital, each with different capabilities, and Resource Allocation. Agents only read state and propose actions through tools; the engine enforces every rule. Below that is the negotiation protocol itself: a priority report triggers a proposal, hospitals respond, clashes escalate, and Resource Allocation commits within three rounds using a survival-weighted score.");

  const out = path.join(__dirname, "Shubham_Bagwe_Emergency_Response_Progress.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("wrote", out);
}

build().catch(e => { console.error(e); process.exit(1); });
