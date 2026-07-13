import * as XLSX from "xlsx";

import { exportLabeledOnServer } from "../services/api";

function escapeCsv(value) {
  const text = String(value ?? "");
  if (text.includes(",") || text.includes('"') || text.includes("\n")) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
}

function rowsToCsv(rows, headers) {
  const lines = [headers.join(",")];
  rows.forEach((row) => {
    lines.push(headers.map((h) => escapeCsv(row[h])).join(","));
  });
  return lines.join("\n");
}

function triggerDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function lookupPrediction(predById, row, index) {
  const rowId = row.id ?? index;
  return (
    predById.get(rowId) ??
    predById.get(Number(rowId)) ??
    predById.get(index) ??
    predById.get(String(index))
  );
}

function exportLabel(pred, { labelKey = "final_condition" } = {}) {
  const raw = pred?.[labelKey] ?? pred?.predicted_condition ?? pred?.final_condition ?? "";
  return String(raw).trim().toUpperCase();
}

/**
 * Original upload rows plus a single new column (prediction in ALL CAPS).
 */
export function buildMainExportRows(sourceRows, predictions, { columnName, labelKey = "final_condition" }) {
  const predById = new Map(predictions.map((p) => [p.row_id, p]));
  const col = String(columnName || "FAILURE_MECHANISM").trim() || "FAILURE_MECHANISM";

  const rows = sourceRows.map((row, index) => {
    const pred = lookupPrediction(predById, row, index);
    return {
      ...row,
      [col]: exportLabel(pred, { labelKey }),
    };
  });

  return { columnName: col, rows };
}

/**
 * Separate detail rows: confidence, tiers, explainability, model agreement, etc.
 */
export function buildFindingsRows(predictions) {
  return (predictions || []).map((pred, index) => {
    const simple = pred?.xai?.simple || {};
    const confidence = pred?.confidence;
    return {
      row_id: pred.row_id ?? index,
      predicted_condition: pred.predicted_condition ?? "",
      final_condition: pred.final_condition ?? pred.predicted_condition ?? "",
      confidence_percent:
        confidence != null ? Math.round(Number(confidence) * 1000) / 10 : "",
      review_tier: pred.confidence_tier ?? simple.tier ?? "",
      models_agree: pred.models_agree ?? simple.models_agree ?? "",
      models_total: pred.models_total ?? simple.models_total ?? "",
      runner_up: pred.runner_up ?? simple.runner_up ?? "",
      explanation: simple.one_liner ?? pred.xai?.explanation ?? "",
      keywords: Array.isArray(simple.keywords) ? simple.keywords.join(", ") : "",
      actual_label: pred.actual_label ?? "",
      model: pred.model ?? "",
    };
  });
}

function downloadXlsxWorkbook({ mainRows, findingsRows, filename }) {
  const book = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(book, XLSX.utils.json_to_sheet(mainRows), "Dataset");
  XLSX.utils.book_append_sheet(book, XLSX.utils.json_to_sheet(findingsRows), "Findings");
  XLSX.writeFile(book, filename);
}

/** @deprecated Use buildMainExportRows */
export function buildLabeledRows(sourceRows, predictions, options) {
  return buildMainExportRows(sourceRows, predictions, options);
}

export async function downloadLabeledDataset({
  sourceRows,
  uploadId,
  predictions,
  columnName,
  format = "csv",
}) {
  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
  const baseName = `labeled_dataset_${stamp}`;

  if (uploadId || !sourceRows?.length) {
    const blob = await exportLabeledOnServer({
      upload_id: uploadId,
      rows: sourceRows?.length ? sourceRows : undefined,
      predictions,
      column_name: columnName,
      format,
    });
    const ext = format === "xlsx" ? "xlsx" : "csv";
    triggerDownload(blob, `${baseName}.${ext}`);
    if (format === "csv") {
      const findingsCsv = rowsToCsv(
        buildFindingsRows(predictions),
        Object.keys(buildFindingsRows(predictions)[0] || {}),
      );
      triggerDownload(
        new Blob([findingsCsv], { type: "text/csv;charset=utf-8;" }),
        `${baseName}_findings.csv`,
      );
    }
    return;
  }

  const { rows: mainRows, columnName: col } = buildMainExportRows(sourceRows, predictions, { columnName });
  const findingsRows = buildFindingsRows(predictions);
  const headers = mainRows.length ? Object.keys(mainRows[0]) : [];

  if (format === "xlsx") {
    downloadXlsxWorkbook({
      mainRows,
      findingsRows,
      filename: `${baseName}.xlsx`,
    });
    return;
  }

  const csv = rowsToCsv(mainRows, headers);
  triggerDownload(new Blob([csv], { type: "text/csv;charset=utf-8;" }), `${baseName}.csv`);

  const findingsHeaders = findingsRows.length ? Object.keys(findingsRows[0]) : [];
  triggerDownload(
    new Blob([rowsToCsv(findingsRows, findingsHeaders)], { type: "text/csv;charset=utf-8;" }),
    `${baseName}_findings.csv`,
  );
}
