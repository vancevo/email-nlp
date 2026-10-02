import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const source = path.join(root, "source-data");

function parseCsv(file) {
  const text = fs.readFileSync(path.join(source, file), "utf8").trim();
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;

  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    if (char === '"') {
      if (quoted && text[index + 1] === '"') {
        cell += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }
    } else if (char === "," && !quoted) {
      row.push(cell);
      cell = "";
    } else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && text[index + 1] === "\n") index += 1;
      row.push(cell);
      if (row.some((value) => value.length)) rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += char;
    }
  }
  row.push(cell);
  if (row.some((value) => value.length)) rows.push(row);

  const [headers, ...body] = rows;
  return body.map((values) => Object.fromEntries(headers.map((key, i) => [key, values[i] ?? ""])));
}

const metrics = parseCsv("metrics.csv").map((row) => ({
  model: row.model,
  accuracy: Number(row.accuracy),
  precision: Number(row.precision_spam),
  recall: Number(row.recall_spam),
  f1: Number(row.f1_spam),
  auc: Number(row.roc_auc),
}));

const history = parseCsv("lstm_history.csv").map((row) =>
  Object.fromEntries(Object.entries(row).map(([key, value]) => [key, key === "epoch" ? Number(value) : Number(value)])),
);

const predictions = parseCsv("test_predictions.csv").map((row, index) => ({
  id: index + 1,
  subject: row.Subject,
  label: row.label_text,
  truth: Number(row.true_label),
  models: {
    tfidf: { probability: Number(row.prob_tfidf), prediction: Number(row.pred_tfidf) },
    avgW2v: { probability: Number(row.prob_avg_w2v), prediction: Number(row.pred_avg_w2v) },
    lstm: { probability: Number(row.prob_lstm), prediction: Number(row.pred_lstm) },
  },
}));

const confusion = {};
for (const [key, label] of [["tfidf", "TF-IDF + Logistic Regression"], ["avgW2v", "Average Word2Vec + Logistic Regression"], ["lstm", "Word2Vec + LSTM"]]) {
  const matrix = { tn: 0, fp: 0, fn: 0, tp: 0 };
  for (const row of predictions) {
    const predicted = row.models[key].prediction;
    if (row.truth === 0 && predicted === 0) matrix.tn += 1;
    if (row.truth === 0 && predicted === 1) matrix.fp += 1;
    if (row.truth === 1 && predicted === 0) matrix.fn += 1;
    if (row.truth === 1 && predicted === 1) matrix.tp += 1;
  }
  confusion[key] = { label, ...matrix };
}

const output = {
  generatedFrom: "output_nlp.zip",
  state: JSON.parse(fs.readFileSync(path.join(source, "state.json"), "utf8")),
  metrics,
  history,
  confusion,
  predictions,
};

fs.mkdirSync(path.join(root, "dist", "data"), { recursive: true });
fs.writeFileSync(path.join(root, "dist", "data", "results.json"), JSON.stringify(output));
console.log(`Wrote ${predictions.length} predictions to dist/data/results.json`);
