import { useEffect, useState, useCallback, useRef, createContext, useContext, useMemo } from "react";

import { getHistoryItem } from "../services/api";
import {
  applySessionToState,
  buildSessionPayload,
  clearSession,
  loadSession,
  restoreFromSnapshot,
  saveSession,
} from "../utils/sessionPersistence";

const AppContext = createContext(null);

export const DEFAULT_RUN_CONFIG = {
  text_column: "",
  label_column: "",
  part_column: "",
  models: ["UMECClassifier"],
};

export function AppProvider({ children }) {
  const [hydrated, setHydrated] = useState(false);
  const [uploadedRows, setUploadedRows] = useState([]);
  const [columns, setColumns] = useState([]);
  const [editedRows, setEditedRows] = useState([]);
  const [uploadId, setUploadId] = useState("");
  const [sourceFilename, setSourceFilename] = useState("");
  const [rowCount, setRowCount] = useState(0);
  const [previewOnly, setPreviewOnly] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [lastRunId, setLastRunId] = useState("");
  const [runConfig, setRunConfig] = useState(DEFAULT_RUN_CONFIG);
  const [analysisConfig, setAnalysisConfig] = useState({
    user_settings: {},
    custom_categories: [],
    label_groups: [],
    xai_top_k: 3,
  });
  const [statusMessage, setStatusMessage] = useState("");

  const settersRef = useRef({});
  settersRef.current = {
    setUploadedRows,
    setColumns,
    setEditedRows,
    setUploadId,
    setSourceFilename,
    setRowCount,
    setPreviewOnly,
    setPrediction,
    setLastRunId,
    setRunConfig,
    setAnalysisConfig,
  };

  const resetWorkspace = useCallback(() => {
    clearSession();
    setUploadedRows([]);
    setColumns([]);
    setEditedRows([]);
    setUploadId("");
    setSourceFilename("");
    setRowCount(0);
    setPreviewOnly(false);
    setPrediction(null);
    setLastRunId("");
    setRunConfig(DEFAULT_RUN_CONFIG);
    setAnalysisConfig({
      user_settings: {},
      custom_categories: [],
      label_groups: [],
      xai_top_k: 3,
    });
    setStatusMessage("");
  }, []);

  const restoreRunById = useCallback(async (recordId) => {
    const item = await getHistoryItem(recordId);
    if (!item?.snapshot) {
      throw new Error("This history record cannot be reopened (no snapshot).");
    }
    restoreFromSnapshot(item.snapshot, settersRef.current);
    setLastRunId(recordId);
    return item;
  }, []);

  useEffect(() => {
    const session = loadSession();
    if (session) {
      applySessionToState(session, settersRef.current);
      if (session.lastRunId) {
        getHistoryItem(session.lastRunId)
          .then((item) => {
            if (item?.snapshot?.prediction) {
              restoreFromSnapshot(item.snapshot, settersRef.current);
            }
          })
          .catch(() => {});
      }
    }
    setHydrated(true);
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    saveSession(
      buildSessionPayload({
        lastRunId,
        runConfig,
        analysisConfig,
        uploadId,
        sourceFilename,
        rowCount,
        previewOnly,
        columns,
        uploadedRows,
        editedRows,
        statusMessage,
      }),
    );
  }, [
    hydrated,
    lastRunId,
    runConfig,
    analysisConfig,
    uploadId,
    sourceFilename,
    rowCount,
    previewOnly,
    columns,
    uploadedRows,
    editedRows,
    statusMessage,
  ]);

  const value = useMemo(
    () => ({
      hydrated,
      uploadedRows,
      setUploadedRows,
      columns,
      setColumns,
      editedRows,
      setEditedRows,
      uploadId,
      setUploadId,
      sourceFilename,
      setSourceFilename,
      rowCount,
      setRowCount,
      previewOnly,
      setPreviewOnly,
      prediction,
      setPrediction,
      lastRunId,
      setLastRunId,
      runConfig,
      setRunConfig,
      analysisConfig,
      setAnalysisConfig,
      statusMessage,
      setStatusMessage,
      restoreRunById,
      resetWorkspace,
    }),
    [
      hydrated,
      uploadedRows,
      columns,
      editedRows,
      uploadId,
      sourceFilename,
      rowCount,
      previewOnly,
      prediction,
      lastRunId,
      runConfig,
      analysisConfig,
      statusMessage,
      restoreRunById,
      resetWorkspace,
    ],
  );

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useApp() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useApp must be used within an AppProvider");
  return ctx;
}
