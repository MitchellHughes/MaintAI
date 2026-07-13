/** Browser session persistence (survives refresh, not cross-tab forever). */

const STORAGE_KEY = "maintai_session_v1";
const MAX_ROWS_IN_SESSION = 500;

export function loadSession() {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

export function saveSession(payload) {
  try {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify({ version: 1, ...payload }));
  } catch {
    /* quota exceeded — drop row preview */
    try {
      const slim = { ...payload, uploadedRows: [], editedRows: [] };
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify({ version: 1, ...slim }));
    } catch {
      /* ignore */
    }
  }
}

export function buildSessionPayload(state) {
  const rowCount = state.rowCount || state.uploadedRows?.length || 0;
  const includeRows = rowCount > 0 && rowCount <= MAX_ROWS_IN_SESSION && !state.previewOnly;

  return {
    lastRunId: state.lastRunId || "",
    runConfig: state.runConfig,
    analysisConfig: state.analysisConfig,
    uploadId: state.uploadId || "",
    sourceFilename: state.sourceFilename || "",
    rowCount: state.rowCount || 0,
    previewOnly: Boolean(state.previewOnly),
    columns: state.columns || [],
    uploadedRows: includeRows ? state.uploadedRows : [],
    editedRows: includeRows ? state.editedRows : [],
    statusMessage: state.statusMessage || "",
  };
}

export function applySessionToState(session, setters) {
  if (!session) return;
  if (session.runConfig) setters.setRunConfig(session.runConfig);
  if (session.analysisConfig) setters.setAnalysisConfig(session.analysisConfig);
  if (session.uploadId) setters.setUploadId(session.uploadId);
  if (session.sourceFilename) setters.setSourceFilename(session.sourceFilename);
  if (session.rowCount != null) setters.setRowCount(session.rowCount);
  if (session.previewOnly != null) setters.setPreviewOnly(session.previewOnly);
  if (session.columns?.length) setters.setColumns(session.columns);
  if (session.uploadedRows?.length) {
    setters.setUploadedRows(session.uploadedRows);
    setters.setEditedRows(session.editedRows?.length ? session.editedRows : session.uploadedRows);
  }
  if (session.lastRunId) setters.setLastRunId(session.lastRunId);
}

export function restoreFromSnapshot(snapshot, setters) {
  if (!snapshot) return;
  const {
    prediction,
    runConfig,
    analysisConfig,
    columns,
    uploadId,
    rowCount,
    previewOnly,
    sourceFilename,
  } = snapshot;
  if (prediction) setters.setPrediction(prediction);
  const name =
    sourceFilename || prediction?.source_filename || "";
  if (name) setters.setSourceFilename(name);
  if (runConfig) setters.setRunConfig(runConfig);
  if (analysisConfig) setters.setAnalysisConfig(analysisConfig);
  if (columns?.length) setters.setColumns(columns);
  if (uploadId) setters.setUploadId(uploadId);
  if (rowCount != null) setters.setRowCount(rowCount);
  if (previewOnly != null) setters.setPreviewOnly(previewOnly);
}

export function clearSession() {
  try {
    sessionStorage.removeItem(STORAGE_KEY);
  } catch {
    /* ignore */
  }
}

export { MAX_ROWS_IN_SESSION };
