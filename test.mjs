import test from "node:test";
import assert from "node:assert/strict";
import { auditDataset, buildEvaluationReport, classificationMetrics, cohenKappa, confusionMatrix, normalizeArabicText, parseCsv, parseDataset, tokenize } from "./core.mjs";

test("Arabic normalization removes diacritics, tatweel and common Alef variants", () => {
  assert.equal(normalizeArabicText("إِحْمَـد ١٢٣"), "احمد 123");
  assert.deepEqual(tokenize("المنتج جميل!"), ["المنتج", "جميل"]);
});

test("CSV parser supports commas and escaped quotes inside quoted cells", () => {
  const rows = parseCsv('id,text,gold\n1,"قال: ""حلو"", جدا",positive\n2,"A, B",neutral');
  assert.equal(rows.length, 2);
  assert.equal(rows[0].text, 'قال: "حلو", جدا');
  assert.equal(rows[1].text, "A, B");
});

test("dataset parser reads JSONL and rejects malformed lines with a line number", () => {
  assert.deepEqual(parseDataset('{"id":"1"}\n{"id":"2"}', "jsonl"), [{ id: "1" }, { id: "2" }]);
  assert.throws(() => parseDataset('{"ok":true}\nnope', "jsonl"), /line 2/u);
});

test("quality audit identifies empty and repeated rows without changing the input", () => {
  const rows = [
    { id: "x", text: "ممتاز", gold: "positive", predicted: "positive" },
    { id: "x", text: "مُمْتَاز", gold: "positive", predicted: "positive" },
    { id: "y", text: " ", gold: "", predicted: "negative" },
  ];
  const audit = auditDataset(rows);
  assert.equal(audit.recordCount, 3);
  assert.deepEqual(audit.emptyText, [3]);
  assert.equal(audit.duplicateIds.length, 1);
  assert.equal(audit.duplicateTexts.length, 1);
  assert.deepEqual(audit.missingGold, [3]);
  assert.equal(rows[1].text, "مُمْتَاز");
});

test("quality audit safely handles label names that resemble object properties", () => {
  const audit = auditDataset([{ text: "sample", gold: "__proto__" }]);
  assert.equal(audit.labelCounts["__proto__"], 1);
});

test("classification metrics use the supplied labels and guard zero denominators", () => {
  const rows = [
    { gold: "positive", predicted: "positive" },
    { gold: "positive", predicted: "negative" },
    { gold: "negative", predicted: "negative" },
    { gold: "neutral", predicted: "neutral" },
  ];
  const metrics = classificationMetrics(rows);
  assert.equal(metrics.compared, 4);
  assert.equal(metrics.accuracy, 0.75);
  assert.equal(metrics.perClass.find((item) => item.label === "positive").recall, 0.5);
  assert.equal(metrics.perClass.find((item) => item.label === "neutral").precision, 1);
  assert.equal(confusionMatrix(rows).matrix.positive.negative, 1);
});

test("Cohen's kappa is computed over complete pairs only", () => {
  const pairs = [
    { annotatorA: "a", annotatorB: "a" },
    { annotatorA: "b", annotatorB: "b" },
    { annotatorA: "a", annotatorB: "b" },
    { annotatorA: "b", annotatorB: "b" },
    { annotatorA: "a", annotatorB: "" },
  ];
  const result = cohenKappa(pairs);
  assert.equal(result.compared, 4);
  assert.equal(result.observed, 0.75);
  assert.equal(result.expected, 0.5);
  assert.equal(result.value, 0.5);
});


test("Arabic digit normalization handles all ten digits and zero", () => {
  assert.equal(normalizeArabicText("٠١٢٣٤٥٦٧٨٩"), "0123456789");
});

test("dataset parser selects CSV for auto mode and keeps an empty dataset safe", () => {
  assert.deepEqual(parseDataset("id,text,gold\n1,مرحبا,positive"), [{ id: "1", text: "مرحبا", gold: "positive" }]);
  assert.deepEqual(parseDataset("\ufeff  ", "auto"), []);
});

test("evaluation report keeps quality, model, and agreement counts tied to supplied rows", () => {
  const report = buildEvaluationReport([{ text: "ممتاز", gold: "positive", predicted: "positive", annotatorA: "a", annotatorB: "a" }]);
  assert.equal(report.recordCount, 1);
  assert.equal(report.model.compared, 1);
  assert.equal(report.agreement.compared, 1);
  assert.match(report.note, /supplied rows/u);
});

test("Arabic digit normalization handles all ten digits and zero", () => {
  assert.equal(normalizeArabicText("٠١٢٣٤٥٦٧٨٩"), "0123456789");
});

test("dataset parser auto-detects CSV and treats whitespace-only input as empty", () => {
  assert.deepEqual(parseDataset("id,text,gold\n1,مرحبا,positive"), [{ id: "1", text: "مرحبا", gold: "positive" }]);
  assert.deepEqual(parseDataset("  \ufeff  ", "auto"), []);
});

test("evaluation report derives quality, metrics, and agreement from supplied rows", () => {
  const report = buildEvaluationReport([{ text: "ممتاز", gold: "positive", predicted: "positive", annotatorA: "a", annotatorB: "a" }]);
  assert.equal(report.recordCount, 1);
  assert.equal(report.model.compared, 1);
  assert.equal(report.agreement.compared, 1);
  assert.match(report.note, /supplied rows/u);
});
