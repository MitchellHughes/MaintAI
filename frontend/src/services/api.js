// Single place for all backend communication (per Web Module rules).

const API_BASE = import.meta.env.VITE_API_BASE || "/api";

async function asJson(response) {
  let data = null;
  try {
    data = await response.json();
  } catch {
    data = null;
  }
  if (!response.ok) {
    const message =
      (data && (data.error || (data.errors && data.errors.join(", ")))) ||
      `Request failed (${response.status})`;
    throw new Error(message);
  }
  return data;
}

async function postJson(path, body, { longRunning = false } = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    if (longRunning) {
      throw new Error(
        "Request timed out or connection was lost. Large files (100k+ rows) can take 30–90 minutes. " +
          "Keep this tab open, ensure Docker is running, and retry. The server may still be working.",
      );
    }
    throw new Error(
      "Cannot reach the API — connection lost or timed out. " +
        "Ensure the backend is running (docker compose up).",
    );
  }
  return asJson(response);
}

// Health check.
export async function checkBackend() {
  const response = await fetch(`${API_BASE}/health`);
  return response.json();
}

// Upload a CSV or Excel file for authoritative server-side parsing/validation.
export async function uploadFile(file) {
  const form = new FormData();
  form.append("file", file);
  let response;
  try {
    response = await fetch(`${API_BASE}/upload`, {
      method: "POST",
      body: form,
    });
  } catch {
    throw new Error(
      "Upload failed — cannot reach the API. Is the backend running? (docker compose up, then http://localhost:5050/api/health)",
    );
  }
  return asJson(response);
}

export async function fetchAvailableModels() {
  const response = await fetch(`${API_BASE}/models`);
  return asJson(response);
}

export async function fetchAnalysisSettings() {
  const response = await fetch(`${API_BASE}/settings`);
  return asJson(response);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function fetchPredictJob(jobId, { includeResult = false } = {}) {
  const query = includeResult ? "?result=1" : "";
  const response = await fetch(`${API_BASE}/predict/jobs/${jobId}${query}`);
  return asJson(response);
}

export async function pollPredictJob(jobId, { onProgress, intervalMs = 2000 } = {}) {
  for (;;) {
    const status = await fetchPredictJob(jobId, { includeResult: false });
    onProgress?.(status);
    if (status.status === "done") {
      if (status.result) return status.result;
      const withResult = await fetchPredictJob(jobId, { includeResult: true });
      if (withResult.result) return withResult.result;
      return withResult;
    }
    if (status.status === "failed") {
      throw new Error(status.error || "Classification failed.");
    }
    await sleep(intervalMs);
  }
}

export async function runPrediction({
  rows,
  upload_id,
  source_filename,
  text_column,
  label_column,
  part_column,
  models,
  analysis_config,
  use_saved_models = false,
  onProgress,
}) {
  const body = {
    text_column,
    label_column,
    part_column,
    models,
    analysis_config,
    use_saved_models,
  };
  if (upload_id) {
    body.upload_id = upload_id;
  } else {
    body.rows = rows;
  }
  if (source_filename) {
    body.source_filename = source_filename;
  }

  let response;
  try {
    response = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    throw new Error(
      "Request timed out or connection was lost. For large files the job may still be running on the server — check docker compose logs -f backend.",
    );
  }

  const data = await response.json().catch(() => ({}));
  if (response.status === 202 && data.async && data.job_id) {
    onProgress?.({ progress: data.message || "Processing in background…", status: "running" });
    return pollPredictJob(data.job_id, { onProgress });
  }
  if (!response.ok) {
    throw new Error(data.error || `Request failed (${response.status})`);
  }
  return data;
}

export async function runTraining(payload = {}) {
  return postJson("/train", payload);
}

export async function submitFeedback(records, before = [], user = "anonymous") {
  return postJson("/feedback", { records, before, user });
}

export async function getHistory() {
  const response = await fetch(`${API_BASE}/history`);
  return asJson(response);
}

export async function getHistoryItem(id) {
  const response = await fetch(`${API_BASE}/history/${id}`);
  return asJson(response);
}

export async function saveRunSnapshot(snapshot, user = "anonymous") {
  return postJson("/history/snapshot", { snapshot, user }, { longRunning: true });
}

export async function fetchActiveLearning({ edits, custom_categories, apply = false }) {
  return postJson("/feedback/active-learning", {
    edits,
    custom_categories,
    apply,
  });
}

export async function getMetrics() {
  const response = await fetch(`${API_BASE}/metrics`);
  return asJson(response);
}

export async function fetchClassificationReport({
  predictions,
  custom_categories,
  pred_key = "predicted_condition",
  actual_key = "actual_label",
  label_groups,
  top_k = 3,
}) {
  return postJson("/evaluation/report", {
    predictions,
    custom_categories,
    pred_key,
    actual_key,
    label_groups,
    top_k,
  });
}

export async function suggestLabelGroups({
  predictions,
  custom_categories,
  pred_key = "predicted_condition",
  actual_key = "actual_label",
  label_groups,
}) {
  return postJson("/evaluation/suggest-label-groups", {
    predictions,
    custom_categories,
    pred_key,
    actual_key,
    label_groups,
  });
}

export async function applyLabelGroups({
  custom_categories,
  label_groups,
  merge_categories = false,
}) {
  return postJson("/evaluation/apply-label-groups", {
    custom_categories,
    label_groups,
    merge_categories,
  });
}

export async function generateTokens(labels, corpus = [], options = {}) {
  return postJson("/generate_tokens", {
    labels,
    corpus,
    rows: options.rows,
    text_column: options.textColumn,
    label_column: options.labelColumn,
    part_column: options.partColumn,
    custom_categories: options.customCategories,
  });
}

export async function exportLabeledOnServer({ upload_id, rows, predictions, column_name, format }) {
  const response = await fetch(`${API_BASE}/export/labeled`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      upload_id,
      rows,
      predictions,
      column_name,
      format,
    }),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.error || "Export failed");
  }
  return response.blob();
}
