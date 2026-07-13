function escapeCsv(value) {
  const text = String(value ?? "");
  if (text.includes(",") || text.includes('"') || text.includes("\n")) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
}

export function downloadPredictionCsv(prediction) {
  const { results_by_model: byModel = {}, source = [] } = prediction;
  const modelNames = Object.keys(byModel);
  if (!modelNames.length) return;

  const sourceById = new Map(source.map((row, i) => [row.id ?? i, row]));
  const rowIds = new Set();
  modelNames.forEach((model) => {
    (byModel[model] || []).forEach((p) => rowIds.add(p.row_id));
  });

  const headers = [
    "row_id",
    "discrepancy",
    ...modelNames.flatMap((model) => [
      `${model}_predicted`,
      `${model}_confidence`,
      `${model}_review_tier`,
      `${model}_why`,
    ]),
  ];

  const lines = [headers.join(",")];
  Array.from(rowIds)
    .sort((a, b) => Number(a) - Number(b))
    .forEach((rowId) => {
      const first = byModel[modelNames[0]]?.find((p) => p.row_id === rowId);
      const discrepancy = first?.discrepancy ?? "";
      const cells = [rowId, escapeCsv(discrepancy)];
      modelNames.forEach((model) => {
        const p = (byModel[model] || []).find((r) => r.row_id === rowId);
        cells.push(
          escapeCsv(p?.predicted_condition ?? ""),
          p?.confidence != null ? String(Math.round(p.confidence * 100) / 100) : "",
          escapeCsv(p?.confidence_tier ?? p?.xai?.simple?.tier ?? ""),
          escapeCsv(p?.xai?.simple?.one_liner ?? p?.xai?.explanation ?? ""),
        );
      });
      lines.push(cells.join(","));
    });

  const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `predictions_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.csv`;
  link.click();
  URL.revokeObjectURL(url);
}
