#!/usr/bin/env node
import { readFile, writeFile } from "node:fs/promises";
import { basename, resolve } from "node:path";
import { buildEvaluationReport, parseDataset } from "./core.mjs";

function usage() {
  console.log(`MasriBench · Egyptian Arabic dataset and model evaluation\n\nUsage:\n  node cli.mjs <dataset.csv|dataset.jsonl|dataset.json> [--out report.json]\n\nInput fields used when present: text, gold, predicted, annotatorA, annotatorB, split.\nCSV headers or JSON/JSONL records are supported. No data is uploaded.`);
}

const args = process.argv.slice(2);
if (!args.length || args.includes("--help") || args.includes("-h")) {
  usage();
  process.exit(args.length ? 0 : 2);
}
const inputPath = args[0];
const outputIndex = args.indexOf("--out");
const outputPath = outputIndex >= 0 ? args[outputIndex + 1] : null;
if (outputIndex >= 0 && (!outputPath || outputPath.startsWith("--"))) {
  console.error("--out needs a file path.");
  process.exit(2);
}

try {
  const raw = await readFile(resolve(inputPath), "utf8");
  const format = inputPath.endsWith(".jsonl") ? "jsonl" : inputPath.endsWith(".json") ? "json" : "csv";
  const rows = parseDataset(raw, format);
  if (!rows.length) throw new Error("The file has no data records.");
  const report = buildEvaluationReport(rows);
  const json = `${JSON.stringify(report, null, 2)}\n`;
  if (outputPath) await writeFile(resolve(outputPath), json, "utf8");
  const summary = {
    file: basename(inputPath),
    records: report.recordCount,
    labeled: report.quality.labeledCount,
    predictions: report.quality.predictionCount,
    accuracy: report.model.accuracy,
    macroF1: report.model.macro.f1,
    annotatorAgreement: report.agreement.value,
    duplicateTexts: report.quality.duplicateTexts.length,
    reportFile: outputPath ?? null,
  };
  console.log(JSON.stringify(summary, null, 2));
} catch (error) {
  console.error(`MasriBench: ${error.message}`);
  process.exit(1);
}
