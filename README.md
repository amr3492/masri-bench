# MasriBench

**A local-first quality and evaluation workspace for Egyptian Arabic text datasets.**

MasriBench helps an NLP practitioner inspect labeled text, compare classification predictions with supplied gold labels, and review agreement between two annotators. It includes a browser dashboard and a dependency-free Node.js CLI. It does not train or call a model.

> The included 24-row Egyptian Arabic example is synthetic and written to demonstrate the workflow. It is not a representative dialect corpus, an external benchmark, or evidence of model quality.

## What it does

- Reads CSV, JSONL, or JSON records with named fields.
- Audits blank text, missing labels or predictions, repeated IDs, and repeated text after normalization.
- Reports text-length summaries and gold-label counts.
- Calculates accuracy, per-label precision/recall/F1, macro and weighted averages, and a confusion matrix.
- Calculates Cohen's kappa and observed agreement for complete pairs of two annotators.
- Provides a transparent Arabic normalization preview and exportable JSON report.
- Runs in the browser without a backend, an API key, external fonts, analytics, or model calls. Imported files stay in the browser unless you choose to export a report or rows.

## Try the dashboard

Open `index.html` from a static web server so the browser can load the bundled example file. From the repository root, for example:

```sh
python -m http.server 8000
```

Then visit `http://localhost:8000`. The dashboard starts with synthetic example data. Use **Import CSV / JSONL / JSON** to load your own file. The page reads the file in the browser and does not send it to a server.

The static app can also be hosted on GitHub Pages. There is no build step.

## Run the CLI

Requires Node.js 20 or later. No npm install is needed.

```sh
node cli.mjs demo.jsonl --out report.json
node --test test.mjs
```

The CLI prints a short summary and writes the full report when `--out` is supplied. Supported input formats are selected by filename extension: `.csv`, `.jsonl`, or `.json`.

## Data format

Column names are case-sensitive. The app uses these optional fields:

| Field | Meaning |
| --- | --- |
| `id` | Stable row identifier, used for duplicate-ID checks |
| `text` | Egyptian Arabic text to inspect |
| `gold` | Reference label for evaluation and distribution summaries |
| `predicted` | Model output to compare with `gold` |
| `annotatorA` | First annotator's categorical label |
| `annotatorB` | Second annotator's categorical label |
| `split` | Optional dataset split shown in the preview |

At minimum, provide the fields needed for the analysis you want. CSV must have a header row. JSON should be an array of objects; JSONL should have one object per non-empty line. A small example:

```json
{"id":"ex-1","text":"الخدمة كانت سريعة","gold":"positive","predicted":"positive","annotatorA":"positive","annotatorB":"positive","split":"test"}
```

## Metric details

- Evaluation compares only rows with both a non-empty `gold` and `predicted` value. Matching is exact after trimming whitespace. Labels are inferred from the values in those two fields.
- Precision, recall, and F1 are computed for each observed label. Macro scores give each observed label equal weight; weighted scores use each label's gold support. A zero denominator yields zero for the affected score.
- Cohen's kappa compares only rows with both `annotatorA` and `annotatorB` labels. It treats labels as categories, not as ordered values.
- Missing fields are not imputed. These metrics describe the rows supplied, not the real-world performance of a model, the validity of the labels, or the quality of a dataset outside this file.

## Normalization behavior and limits

The helper applies Unicode NFKC, lowercases text, removes Arabic diacritics and tatweel, maps common Alef forms to Alef, maps Alef Maqsura and hamza-on-Ya to Ya, maps Ta Marbuta to Ha, converts Arabic-Indic digits to Western digits, and replaces punctuation or spacing runs with one space. It is used only for token preview and duplicate-text checks. Some of these transformations collapse distinctions that may matter for a task. The original text is retained; review normalized matches before removing or merging records.

This is an evaluation utility, not a tokenizer benchmark, spelling corrector, dialect classifier, annotation policy, model-training pipeline, or source of gold labels. It does not include a real-world corpus.

## Project structure

```text
index.html    Responsive dashboard
app.mjs       Browser UI and local import/export
core.mjs      Parsing, normalization, audits, and metrics
cli.mjs       Command-line report runner
demo.jsonl    Synthetic example rows
test.mjs      Node built-in test suite
```

## Tests

```sh
node --test test.mjs
```

The tests cover Arabic text normalization, quoted CSV cells, JSONL errors, dataset quality checks, classification metrics, and Cohen's kappa.
