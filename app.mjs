import { auditDataset, buildEvaluationReport, classificationMetrics, cohenKappa, confusionMatrix, normalizeArabicText, parseDataset, tokenize } from "./core.mjs";

const $ = (selector) => document.querySelector(selector);
const state = { rows: [], filename: "Synthetic demo · 24 records" };
const esc = (value) => String(value ?? "").replace(/[&<>"']/gu, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]);
const fmt = (value, digits = 2) => Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : "—";
const score = (value) => Number.isFinite(value) ? value.toFixed(3) : "—";
const labelTag = (value) => `<span class="tag ${esc(String(value).toLowerCase())}">${esc(value || "(empty)")}</span>`;
let toastTimer;

function toast(message) {
  const node = $("#toast");
  node.textContent = message;
  node.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { node.hidden = true; }, 3200);
}

function empty(message) {
  return `<div class="empty-state">${esc(message)}</div>`;
}

function renderOverview(report) {
  const audit = report.quality;
  $("#kpi-records").textContent = String(report.recordCount);
  $("#kpi-labeled").textContent = `${audit.labeledCount} / ${report.recordCount}`;
  $("#kpi-f1").textContent = report.model.compared ? fmt(report.model.macro.f1) : "—";
  $("#kpi-kappa").textContent = report.agreement.value === null ? "—" : score(report.agreement.value);

  const dist = Object.entries(audit.labelCounts).sort(([a], [b]) => a.localeCompare(b));
  const highest = Math.max(1, ...dist.map(([, count]) => count));
  $("#distribution").innerHTML = dist.length ? dist.map(([label, count]) => `
    <div class="dist-row"><span class="dist-label">${labelTag(label)}</span><div class="track" aria-label="${count} records"><div class="fill" style="width:${Math.round((count / highest) * 100)}%"></div></div><span class="dist-count">${count}</span></div>`).join("") : empty("No gold labels in this dataset.");

  const issues = [
    ["Empty text", audit.emptyText.length],
    ["Missing gold label", audit.missingGold.length],
    ["Missing prediction", audit.missingPrediction.length],
    ["Repeated IDs", audit.duplicateIds.length],
    ["Repeated normalized text", audit.duplicateTexts.length],
  ];
  $("#quality-list").innerHTML = issues.map(([name, count]) => `<div class="issue"><span>${esc(name)}</span>${count ? `<span class="issue-count">${count}</span>` : `<span class="good">Clear</span>`}</div>`).join("")
    + `<div class="issue"><span>Text length · min / median / max</span><span class="muted">${audit.textLength.min} / ${Math.round(audit.textLength.median)} / ${audit.textLength.max}</span></div>`;
  renderRecords();
}

function renderRecords() {
  const query = $("#record-search").value.trim().toLowerCase();
  const rows = state.rows.filter((row) => !query || Object.values(row).some((value) => String(value ?? "").toLowerCase().includes(query))).slice(0, 100);
  if (!rows.length) {
    $("#records-table").innerHTML = empty(state.rows.length ? "No rows match this filter." : "Import a dataset to inspect records.");
    return;
  }
  const headers = ["id", "text", "gold", "predicted", "annotatorA", "annotatorB", "split"];
  const thead = `<thead><tr>${headers.map((key) => `<th scope="col">${esc(key)}</th>`).join("")}</tr></thead>`;
  const tbody = `<tbody>${rows.map((row) => `<tr><td>${esc(row.id ?? "")}</td><td class="arabic" lang="ar" dir="rtl">${esc(row.text ?? "")}</td><td>${labelTag(row.gold)}</td><td>${labelTag(row.predicted)}</td><td>${labelTag(row.annotatorA)}</td><td>${labelTag(row.annotatorB)}</td><td>${esc(row.split ?? "")}</td></tr>`).join("")}</tbody>`;
  $("#records-table").innerHTML = `<table>${thead}${tbody}</table>`;
}

function renderModel(report) {
  const metrics = report.model;
  const cards = [
    ["Compared rows", String(metrics.compared), "Both gold and prediction present"],
    ["Accuracy", fmt(metrics.accuracy), "Exact matches / compared rows"],
    ["Macro precision", fmt(metrics.macro.precision), "Unweighted mean across observed labels"],
    ["Macro recall", fmt(metrics.macro.recall), "Unweighted mean across observed labels"],
    ["Macro F1", fmt(metrics.macro.f1), "Unweighted mean across observed labels"],
    ["Weighted F1", fmt(metrics.weighted.f1), "Weighted by gold-label support"],
  ];
  $("#model-metrics").innerHTML = cards.map(([title, value, hint]) => `<article class="card"><div class="kpi-label">${esc(title)}</div><div class="metric-number">${esc(value)}</div><div class="metric-caption">${esc(hint)}</div></article>`).join("");
  const result = confusionMatrix(state.rows);
  if (!result.compared) {
    $("#confusion-table").innerHTML = empty("No rows have both a gold label and a prediction.");
  } else {
    const head = `<thead><tr><th scope="col">Gold ↓ · Predicted →</th>${result.labels.map((label) => `<th scope="col">${esc(label)}</th>`).join("")}</tr></thead>`;
    const body = `<tbody>${result.labels.map((actual) => `<tr><th scope="row">${esc(actual)}</th>${result.labels.map((predicted) => {
      const value = result.matrix[actual][predicted];
      return `<td class="${actual === predicted ? "diagonal" : value ? "" : "zero"}">${value}</td>`;
    }).join("")}</tr>`).join("")}</tbody>`;
    $("#confusion-table").innerHTML = `<table class="matrix">${head}${body}</table>`;
  }
  $("#class-scores").innerHTML = metrics.perClass.length ? metrics.perClass.map((item) => `<div><div class="class-row"><span>${labelTag(item.label)}</span><div class="class-score"><i style="width:${Math.round(item.f1 * 100)}%"></i></div><span>F1 ${fmt(item.f1)}</span></div><div class="footnote">P ${fmt(item.precision)} · R ${fmt(item.recall)} · support ${item.support}</div></div>`).join("") : empty("No labels could be scored.");
}

function renderAgreement(report) {
  const agreement = report.agreement;
  const kappa = agreement.value;
  const angle = kappa === null ? 0 : Math.max(0, Math.min(1, (kappa + 1) / 2)) * 360;
  $("#kappa-ring").style.setProperty("--angle", `${angle}deg`);
  $("#kappa-large").textContent = kappa === null ? "—" : score(kappa);
  $("#kappa-summary").textContent = agreement.compared ? `${agreement.compared} paired rows` : "Waiting for paired labels";
  $("#kappa-detail").textContent = agreement.compared ? "Agreement beyond chance, using observed label frequencies." : "Import rows with annotatorA and annotatorB columns.";
  const items = [
    ["Complete pairs", agreement.compared],
    ["Observed agreement", fmt(agreement.observed)],
    ["Expected agreement", fmt(agreement.expected)],
    ["Cohen's κ", kappa === null ? "Not available" : score(kappa)],
  ];
  $("#agreement-components").innerHTML = items.map(([name, value]) => `<div class="issue"><span>${esc(name)}</span><strong>${esc(value)}</strong></div>`).join("");
  const conflicts = state.rows.filter((row) => String(row.annotatorA ?? "").trim() && String(row.annotatorB ?? "").trim() && String(row.annotatorA).trim() !== String(row.annotatorB).trim()).slice(0, 30);
  if (!conflicts.length) $("#disagreements").innerHTML = empty(agreement.compared ? "No disagreements in the paired rows." : "No complete annotator pairs to compare.");
  else {
    const head = "<thead><tr><th scope=\"col\">ID</th><th scope=\"col\">Text</th><th scope=\"col\">Annotator A</th><th scope=\"col\">Annotator B</th></tr></thead>";
    const body = `<tbody>${conflicts.map((row) => `<tr><td>${esc(row.id ?? "")}</td><td class="arabic" lang="ar" dir="rtl">${esc(row.text ?? "")}</td><td>${labelTag(row.annotatorA)}</td><td>${labelTag(row.annotatorB)}</td></tr>`).join("")}</tbody>`;
    $("#disagreements").innerHTML = `<table>${head}${body}</table>`;
  }
}

function render() {
  const report = buildEvaluationReport(state.rows);
  $("#dataset-name").textContent = state.filename;
  renderOverview(report);
  renderModel(report);
  renderAgreement(report);
}

function download(name, content, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

$("#file-input").addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  try {
    const ext = file.name.split(".").pop().toLowerCase();
    const format = ext === "jsonl" ? "jsonl" : ext === "json" ? "json" : "csv";
    const rows = parseDataset(await file.text(), format);
    if (!rows.length) throw new Error("No records found. Check the file and its headers.");
    if (rows.some((row) => !row || typeof row !== "object" || Array.isArray(row))) throw new Error("Each record must be an object with named fields.");
    state.rows = rows;
    state.filename = `${file.name} · ${rows.length} records`;
    $("#status").textContent = "Loaded locally.";
    render();
    toast(`Loaded ${rows.length} records. The file was analyzed in this browser.`);
  } catch (error) {
    $("#status").textContent = "Could not load file.";
    toast(error.message);
  } finally {
    event.target.value = "";
  }
});

$("#download-report").addEventListener("click", () => {
  const report = buildEvaluationReport(state.rows);
  download("masri-bench-report.json", `${JSON.stringify(report, null, 2)}\n`, "application/json");
});
$("#export-jsonl").addEventListener("click", () => {
  const text = state.rows.map((row) => JSON.stringify(row)).join("\n");
  download("masri-bench-rows.jsonl", `${text}\n`, "application/x-ndjson");
});
$("#record-search").addEventListener("input", renderRecords);
$("#normalize-input").addEventListener("input", (event) => {
  const normalized = normalizeArabicText(event.target.value);
  $("#normalize-output").textContent = normalized || "اكتب نصًا لعرض النتيجة";
  $("#token-count").textContent = `${tokenize(event.target.value).length} tokens after normalization`;
});
for (const button of document.querySelectorAll(".nav-btn")) {
  button.addEventListener("click", () => {
    const target = button.dataset.view;
    for (const nav of document.querySelectorAll(".nav-btn")) nav.removeAttribute("aria-current");
    button.setAttribute("aria-current", "page");
    for (const panel of document.querySelectorAll(".panel")) panel.hidden = panel.dataset.panel !== target;
  });
}

try {
  const response = await fetch("./demo.jsonl");
  if (!response.ok) throw new Error(`Demo data request failed (${response.status}).`);
  state.rows = parseDataset(await response.text(), "jsonl");
  render();
  $("#status").textContent = "Synthetic demo loaded.";
} catch (error) {
  state.rows = [];
  state.filename = "No dataset loaded";
  render();
  $("#status").textContent = "Import a file to begin.";
                                               }
