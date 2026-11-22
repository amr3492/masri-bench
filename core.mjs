export const DEFAULT_LABELS = ["positive", "negative", "neutral"];

export function normalizeArabicText(value) {
  return String(value ?? "")
    .normalize("NFKC")
    .toLowerCase()
    .replace(/[\u064B-\u065F\u0670\u06D6-\u06ED]/gu, "")
    .replace(/\u0640/gu, "")
    .replace(/[أإآٱ]/gu, "ا")
    .replace(/[ىئ]/gu, "ي")
    .replace(/ة/gu, "ه")
    .replace(/[٠-٩]/gu, (digit) => String("٠١٢٣٤٥٦٧٨٩".indexOf(digit)))
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim()
    .replace(/\s+/gu, " ");
}

export function tokenize(value) {
  const normalized = normalizeArabicText(value);
  return normalized ? normalized.split(" ").filter(Boolean) : [];
}

export function parseCsv(text, delimiter = ",") {
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;
  const source = String(text ?? "").replace(/^\uFEFF/u, "");

  for (let index = 0; index < source.length; index += 1) {
    const char = source[index];
    const next = source[index + 1];
    if (char === '"') {
      if (quoted && next === '"') {
        cell += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }
    } else if (char === delimiter && !quoted) {
      row.push(cell);
      cell = "";
    } else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && next === "\n") index += 1;
      row.push(cell);
      if (row.some((item) => item.trim() !== "")) rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += char;
    }
  }
  if (quoted) throw new Error("CSV contains an unclosed quoted field.");
  row.push(cell);
  if (row.some((item) => item.trim() !== "")) rows.push(row);
  if (rows.length < 2) return [];

  const headers = rows[0].map((header, index) => {
    const cleaned = header.trim();
    return cleaned || `column_${index + 1}`;
  });
  return rows.slice(1).map((values) => Object.fromEntries(
    headers.map((header, index) => [header, (values[index] ?? "").trim()]),
  ));
}

export function parseDataset(text, format = "auto") {
  const source = String(text ?? "").trim();
  if (!source) return [];
  if (format === "json" || (format === "auto" && source.startsWith("["))) {
    const parsed = JSON.parse(source);
    if (!Array.isArray(parsed)) throw new Error("JSON dataset must be an array of records.");
    return parsed;
  }
  if (format === "jsonl" || (format === "auto" && source.startsWith("{"))) {
    return source.split(/\r?\n/u).filter((line) => line.trim()).map((line, index) => {
      try {
        const row = JSON.parse(line);
        if (!row || typeof row !== "object" || Array.isArray(row)) {
          throw new Error("Each JSONL line must be an object.");
        }
        return row;
      } catch (error) {
        throw new Error(`Invalid JSONL on line ${index + 1}: ${error.message}`);
      }
    });
  }
  return parseCsv(source);
}

export function discoverLabels(rows, keys = ["gold", "predicted", "annotatorA", "annotatorB"]) {
  const labels = new Set();
  for (const row of rows) {
    for (const key of keys) {
      const value = String(row?.[key] ?? "").trim();
      if (value) labels.add(value);
    }
  }
  return [...labels].sort((a, b) => a.localeCompare(b));
}

function median(values) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

export function auditDataset(rows) {
  const records = Array.isArray(rows) ? rows : [];
  const ids = new Map();
  const texts = new Map();
  const labelCounts = {};
  const emptyText = [];
  const missingGold = [];
  const missingPrediction = [];
  const duplicateIds = [];
  const duplicateTexts = [];
  const lengths = [];

  records.forEach((row, index) => {
    const id = String(row?.id ?? "").trim();
    const text = String(row?.text ?? "").trim();
    const gold = String(row?.gold ?? "").trim();
    const predicted = String(row?.predicted ?? "").trim();
    const rowNumber = index + 1;
    if (!text) emptyText.push(rowNumber);
    else lengths.push([...text].length);
    if (!gold) missingGold.push(rowNumber);
    else labelCounts[gold] = (labelCounts[gold] ?? 0) + 1;
    if (!predicted) missingPrediction.push(rowNumber);
    if (id) {
      if (ids.has(id)) duplicateIds.push({ id, rows: [ids.get(id), rowNumber] });
      else ids.set(id, rowNumber);
    }
    const normalizedText = normalizeArabicText(text);
    if (normalizedText) {
      if (texts.has(normalizedText)) duplicateTexts.push({ rows: [texts.get(normalizedText), rowNumber] });
      else texts.set(normalizedText, rowNumber);
    }
  });

  return {
    recordCount: records.length,
    labeledCount: records.length - missingGold.length,
    predictionCount: records.length - missingPrediction.length,
    emptyText,
    missingGold,
    missingPrediction,
    duplicateIds,
    duplicateTexts,
    labelCounts,
    textLength: {
      min: lengths.length ? lengths.reduce((minimum, length) => Math.min(minimum, length), Infinity) : 0,
      median: median(lengths),
      max: lengths.length ? lengths.reduce((maximum, length) => Math.max(maximum, length), -Infinity) : 0,
      average: lengths.length ? lengths.reduce((sum, length) => sum + length, 0) / lengths.length : 0,
    },
  };
}

export function confusionMatrix(rows, actualKey = "gold", predictedKey = "predicted", labels = discoverLabels(rows, [actualKey, predictedKey])) {
  const matrix = Object.fromEntries(labels.map((label) => [label, Object.fromEntries(labels.map((other) => [other, 0]))]));
  let compared = 0;
  for (const row of rows) {
    const actual = String(row?.[actualKey] ?? "").trim();
    const predicted = String(row?.[predictedKey] ?? "").trim();
    if (!actual || !predicted || !matrix[actual] || matrix[actual][predicted] === undefined) continue;
    matrix[actual][predicted] += 1;
    compared += 1;
  }
  return { labels, matrix, compared };
}

export function classificationMetrics(rows, actualKey = "gold", predictedKey = "predicted") {
  const labels = discoverLabels(rows, [actualKey, predictedKey]);
  const { matrix, compared } = confusionMatrix(rows, actualKey, predictedKey, labels);
  const perClass = labels.map((label) => {
    const truePositive = matrix[label]?.[label] ?? 0;
    const actualCount = labels.reduce((sum, other) => sum + (matrix[label]?.[other] ?? 0), 0);
    const predictedCount = labels.reduce((sum, actual) => sum + (matrix[actual]?.[label] ?? 0), 0);
    const precision = predictedCount ? truePositive / predictedCount : 0;
    const recall = actualCount ? truePositive / actualCount : 0;
    const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
    return { label, support: actualCount, predicted: predictedCount, precision, recall, f1 };
  });
  const correct = perClass.reduce((sum, row) => sum + (matrix[row.label]?.[row.label] ?? 0), 0);
  const macro = (key) => perClass.length ? perClass.reduce((sum, item) => sum + item[key], 0) / perClass.length : 0;
  const totalSupport = perClass.reduce((sum, item) => sum + item.support, 0);
  const weighted = (key) => totalSupport
    ? perClass.reduce((sum, item) => sum + item[key] * item.support, 0) / totalSupport
    : 0;
  return {
    compared,
    accuracy: compared ? correct / compared : 0,
    macro: { precision: macro("precision"), recall: macro("recall"), f1: macro("f1") },
    weighted: { precision: weighted("precision"), recall: weighted("recall"), f1: weighted("f1") },
    perClass,
    confusion: matrix,
  };
}

export function cohenKappa(rows, firstKey = "annotatorA", secondKey = "annotatorB") {
  const pairs = rows
    .map((row) => [String(row?.[firstKey] ?? "").trim(), String(row?.[secondKey] ?? "").trim()])
    .filter(([first, second]) => first && second);
  if (!pairs.length) return { value: null, compared: 0, observed: 0, expected: 0 };
  const countsA = new Map();
  const countsB = new Map();
  let same = 0;
  for (const [first, second] of pairs) {
    countsA.set(first, (countsA.get(first) ?? 0) + 1);
    countsB.set(second, (countsB.get(second) ?? 0) + 1);
    if (first === second) same += 1;
  }
  const observed = same / pairs.length;
  const labels = new Set([...countsA.keys(), ...countsB.keys()]);
  const expected = [...labels].reduce((sum, label) => sum + ((countsA.get(label) ?? 0) / pairs.length) * ((countsB.get(label) ?? 0) / pairs.length), 0);
  const value = expected === 1 ? (observed === 1 ? 1 : 0) : (observed - expected) / (1 - expected);
  return { value, compared: pairs.length, observed, expected };
}

export function buildEvaluationReport(rows) {
  const metrics = classificationMetrics(rows);
  const agreement = cohenKappa(rows);
  return {
    generatedAt: new Date().toISOString(),
    recordCount: rows.length,
    quality: auditDataset(rows),
    model: metrics,
    agreement,
    note: "Metrics describe only the supplied rows. They do not establish real-world model quality or label validity.",
  };
}
