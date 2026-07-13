import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Database, Layers, PlayCircle, Tags } from "lucide-react";

import { useApp } from "../context/AppContext";
import { uploadFile, runPrediction, fetchAvailableModels, runTraining, saveRunSnapshot } from "../services/api";
import ModelSelector from "../components/ModelSelector";
import FileUpload from "../components/FileUpload";
import EditableGrid from "../components/EditableGrid";
import TokenGenerator from "../components/TokenGenerator";
import CategoryPresets from "../components/CategoryPresets";
import LoadingOverlay from "../components/LoadingOverlay";
import AnalysisSettings from "../components/AnalysisSettings";
import PageHeader from "../components/PageHeader";
import PageShell from "../components/PageShell";
import StatCard from "../components/StatCard";
import DashboardWelcome from "../components/DashboardWelcome";
import SetupChecklist from "../components/SetupChecklist";
import StepNav from "../components/StepNav";
import DatasetSummaryCard from "../components/DatasetSummaryCard";
import CategoryPreviewCard from "../components/CategoryPreviewCard";
import WorkflowJourney from "../components/WorkflowJourney";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

function Dashboard() {
  const navigate = useNavigate();

  const {
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
    setPrediction,
    setLastRunId,
    runConfig,
    setRunConfig,
    analysisConfig,
    setAnalysisConfig,
    resetWorkspace,
  } = useApp();

  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [busyTitle, setBusyTitle] = useState("");
  const [busyMessage, setBusyMessage] = useState("");
  const [busyDetail, setBusyDetail] = useState("");
  const [chunkProgress, setChunkProgress] = useState(null);
  const [activeStep, setActiveStep] = useState(1);

  const corpusTexts = useMemo(() => {
    if (!runConfig.text_column) return [];
    return editedRows
      .map((row) => row[runConfig.text_column])
      .filter((value) => value != null && String(value).trim());
  }, [editedRows, runConfig.text_column]);

  const categoriesCount = useMemo(
    () =>
      (analysisConfig.custom_categories || []).filter((row) => String(row.label || "").trim())
        .length,
    [analysisConfig.custom_categories],
  );

  const effectiveRowCount = rowCount || uploadedRows.length;
  const stepUpload = effectiveRowCount > 0;
  const stepCategories = categoriesCount > 0;
  const hasTextColumn = Boolean(runConfig.text_column);
  const hasModels = Boolean(runConfig.models?.length);
  const stepReady =
    stepUpload && stepCategories && hasTextColumn && hasModels;

  const stepColumns = hasTextColumn;

  const workflowSteps = useMemo(
    () => [
      { id: 1, label: "Upload", state: stepUpload ? "done" : "active" },
      {
        id: 2,
        label: "Map columns",
        state: stepColumns ? "done" : stepUpload ? "active" : "pending",
      },
      {
        id: 3,
        label: "Categories",
        state: stepCategories ? "done" : stepColumns ? "active" : "pending",
      },
      {
        id: 4,
        label: "Run",
        state: stepReady ? "done" : stepCategories ? "active" : "pending",
      },
    ],
    [stepUpload, stepColumns, stepCategories, stepReady],
  );

  const workflowPhase = useMemo(() => {
    if (activeStep === 1) return "upload";
    if (activeStep === 2) return "upload";
    if (activeStep === 3) return "categories";
    return "classify";
  }, [activeStep]);

  useEffect(() => {
    if (!stepUpload) {
      setActiveStep(1);
    } else if (!stepColumns && activeStep > 2) {
      setActiveStep(2);
    } else if (!stepCategories && activeStep > 3) {
      setActiveStep(3);
    }
  }, [stepUpload, stepColumns, stepCategories, activeStep]);

  function handleStartNewAnalysis() {
    if (
      !window.confirm(
        "Start a new analysis? This clears the current upload, categories, and in-progress run from this browser session.",
      )
    ) {
      return;
    }
    resetWorkspace();
    setActiveStep(1);
    setStatus("");
    setError("");
  }

  function normalizeCategories(rows) {
    return (rows || [])
      .map((row) => ({
        label: String(row.label || "").trim(),
        keywords: String(row.keywords || "")
          .split(",")
          .map((item) => item.trim())
          .filter(Boolean),
      }))
      .filter((row) => row.label);
  }

  function handleJobProgress(status) {
    if (typeof status === "string") {
      setBusyMessage(status);
      return;
    }
    if (status?.progress) setBusyMessage(status.progress);
    if (status?.chunk_progress) setChunkProgress(status.chunk_progress);
    const cp = status?.chunk_progress;
    const elapsed = status?.timing?.total_display;
    if (cp?.row_start != null) {
      const range = `Rows ${Number(cp.row_start).toLocaleString()}–${Number(cp.row_end || cp.row_start).toLocaleString()}`;
      setBusyDetail(elapsed ? `${range} · Elapsed: ${elapsed}` : range);
    } else if (elapsed) {
      setBusyDetail(`Elapsed: ${elapsed}`);
    }
  }

  async function handleLoaded(file) {
    setError("");
    setStatus("Uploading and validating file…");

    try {
      const res = await uploadFile(file);
      const rows = res.rows || [];
      const cols = res.columns || [];
      const total = res.row_count ?? rows.length;
      setUploadId(res.upload_id || "");
      setSourceFilename(res.filename || file?.name || "");
      setRowCount(total);
      setPreviewOnly(Boolean(res.preview_only));
      setUploadedRows(rows);
      setColumns(cols);
      setEditedRows(rows);
      setRunConfig((prev) => ({
        ...prev,
        text_column: cols.includes(prev.text_column) ? prev.text_column : "",
        label_column: cols.includes(prev.label_column) ? prev.label_column : "",
        part_column: cols.includes(prev.part_column) ? prev.part_column : "",
      }));
      setActiveStep(2);
      if (res.preview_only) {
        setStatus(
          `Loaded ${total.toLocaleString()} records (showing first ${rows.length.toLocaleString()}). Full file kept on server.`,
        );
      } else {
        setStatus(`Loaded ${total.toLocaleString()} records. Configure categories, then run analysis.`);
      }
    } catch (e) {
      setUploadedRows([]);
      setColumns([]);
      setEditedRows([]);
      setUploadId("");
      setSourceFilename("");
      setRowCount(0);
      setPreviewOnly(false);
      setStatus("");
      setError(e.message);
    }
  }

  async function handlePredict() {
    setError("");
    setBusy(true);
    setChunkProgress(null);
    setBusyTitle("Running analysis");
    const onlyToken =
      runConfig.models?.length === 1 && runConfig.models[0] === "TokenMatchingClassifier";
    const hasUmec = runConfig.models?.includes("UMECClassifier");
    setBusyMessage(
      onlyToken
        ? "Training token model on your dataset and scoring records…"
        : hasUmec
          ? "Training base models and ensemble classifier…"
          : "Training selected models on your dataset…",
    );
    const totalRecords = rowCount || editedRows.length;
    setBusyDetail(
      `${totalRecords.toLocaleString()} records · ${runConfig.models?.length || 0} model(s). ` +
        (totalRecords > 5000
          ? "Large file — runs in background. Keep this tab open until complete."
          : onlyToken
            ? "Typically under one minute."
            : hasUmec
              ? "Ensemble mode includes semantic embeddings and may take several minutes."
              : "Semantic similarity is the slowest step."),
    );

    try {
      if (!runConfig.text_column) {
        setError("Select the text column that contains discrepancy descriptions.");
        setActiveStep(3);
        return;
      }
      if (!runConfig.models?.length) {
        setError("Select at least one classification model.");
        setActiveStep(3);
        return;
      }
      const categories = normalizeCategories(analysisConfig.custom_categories);
      if (!categories.length) {
        setError("Define at least one failure-mechanism category before running analysis.");
        setActiveStep(2);
        return;
      }

      const result = await runPrediction({
        upload_id: uploadId || undefined,
        source_filename: sourceFilename || undefined,
        rows: uploadId ? undefined : editedRows,
        text_column: runConfig.text_column,
        label_column: runConfig.label_column || undefined,
        part_column: runConfig.part_column || undefined,
        models: runConfig.models,
        analysis_config: {
          custom_categories: categories,
          user_settings: analysisConfig.user_settings,
          label_groups: analysisConfig.label_groups || [],
          label_column: runConfig.label_column || undefined,
          xai_top_k: analysisConfig.xai_top_k ?? 3,
        },
        onProgress: handleJobProgress,
      });

      const predictionPayload = {
        ...result,
        source_filename: result.source_filename || sourceFilename || undefined,
        source: previewOnly ? [] : editedRows,
        upload_id: uploadId || result.upload_id,
        row_count: totalRecords,
        config: runConfig,
        custom_categories: categories,
        label_groups: analysisConfig.label_groups || [],
      };

      setPrediction(predictionPayload);

      saveRunSnapshot({
        prediction: predictionPayload,
        runConfig,
        analysisConfig: {
          ...analysisConfig,
          custom_categories: analysisConfig.custom_categories,
        },
        columns,
        uploadId,
        sourceFilename: result.source_filename || sourceFilename || "",
        rowCount: totalRecords,
        previewOnly,
      })
        .then((saved) => setLastRunId(saved.id))
        .catch(() => {});

      navigate("/results");
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
      setBusyTitle("");
      setBusyMessage("");
      setBusyDetail("");
      setChunkProgress(null);
    }
  }

  async function handleTrainSettings() {
    setError("");
    setBusy(true);
    setBusyTitle("Saving models");
    setBusyMessage("Training on your dataset and writing model files to disk…");
    setBusyDetail("Optional. Prediction already trains in memory; use this to persist models for reuse.");
    try {
      if (!runConfig.text_column) {
        setError("Select the text column before saving models.");
        return;
      }
      if (!editedRows.length) {
        setError("Upload a dataset first.");
        return;
      }
      const categories = normalizeCategories(analysisConfig.custom_categories);
      const payload = {
        dataset_meta: {
          rows: editedRows,
          text_column: runConfig.text_column,
          user_settings: analysisConfig.user_settings,
          custom_categories: categories,
        },
      };
      const res = await runTraining(payload);
      setStatus(
        `Models saved (${editedRows.length} records) · ${res.model_version} · ${new Date(res.trained_at).toLocaleString()}.`,
      );
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
      setBusyTitle("");
      setBusyMessage("");
      setBusyDetail("");
    }
  }

  return (
    <PageShell>
      <LoadingOverlay
        open={busy}
        title={busyTitle}
        message={busyMessage}
        detail={busyDetail}
        chunkProgress={chunkProgress}
      />

      <PageHeader
        title="Analysis workspace"
        description="Upload data, map columns, define categories and keywords, then run classification."
        meta={
          sourceFilename ? (
            <Badge variant="outline" className="max-w-full truncate font-mono font-normal" title={sourceFilename}>
              {sourceFilename}
            </Badge>
          ) : null
        }
        actions={
          <Button variant="outline" size="sm" type="button" onClick={handleStartNewAnalysis}>
            Start new analysis
          </Button>
        }
      />

      <WorkflowJourney currentPhase={workflowPhase} />

      <div className="grid min-w-0 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          label="Records loaded"
          value={effectiveRowCount > 0 ? effectiveRowCount.toLocaleString() : "—"}
          hint={previewOnly ? "Large file — full dataset on server" : "From current upload"}
          icon={Database}
          tone={stepUpload ? "success" : "default"}
        />
        <StatCard
          label="Categories"
          value={categoriesCount || "—"}
          hint="Failure-mechanism labels defined"
          icon={Tags}
          tone={stepCategories ? "success" : stepUpload ? "warning" : "default"}
        />
        <StatCard
          label="Models selected"
          value={runConfig.models?.length || "—"}
          hint={runConfig.text_column ? `Text: ${runConfig.text_column}` : "Map text column in step 2"}
          icon={Layers}
          tone={runConfig.models?.length ? "info" : "default"}
        />
        <StatCard
          label="Run readiness"
          value={stepReady ? "Ready" : "Incomplete"}
          hint={stepReady ? "All prerequisites met" : "Finish upload, categories, and config"}
          icon={PlayCircle}
          tone={stepReady ? "success" : "warning"}
        />
      </div>

      {!stepUpload && <DashboardWelcome />}

      <div className="grid min-w-0 items-start gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(260px,320px)]">
        <div className="min-w-0 space-y-6">
          {stepUpload && (
            <StepNav steps={workflowSteps} activeStep={activeStep} onStepChange={setActiveStep} />
          )}

          {(activeStep === 1 || !stepUpload) && (
            <Card className="min-w-0 shadow-sm">
              <CardHeader>
                <Badge variant="secondary" className="w-fit">
                  Step 1
                </Badge>
                <CardTitle>Upload dataset</CardTitle>
                <CardDescription>
                  CSV or Excel file containing discrepancy narrative text.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <FileUpload onLoaded={handleLoaded} />
                {status && (
                  <Alert>
                    <AlertDescription>{status}</AlertDescription>
                  </Alert>
                )}
                {error && activeStep === 1 && (
                  <Alert variant="destructive">
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                )}
              </CardContent>
            </Card>
          )}

          {stepUpload && activeStep === 2 && (
            <Card className="min-w-0 shadow-sm">
              <CardHeader>
                <Badge variant="secondary" className="w-fit">
                  Step 2
                </Badge>
                <CardTitle>Map columns</CardTitle>
                <CardDescription>
                  Select discrepancy text first — keyword auto-fill and classification use this column.
                  Part/asset column is merged into text for equipment context.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <ModelSelector
                  columns={columns}
                  value={runConfig}
                  onChange={setRunConfig}
                  fetchModels={fetchAvailableModels}
                  showColumns
                  showModels={false}
                />
                <div className="flex flex-wrap gap-2">
                  <Button type="button" disabled={!hasTextColumn} onClick={() => setActiveStep(3)}>
                    Continue to categories →
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}

          {stepUpload && activeStep === 3 && (
            <Card className="min-w-0 shadow-sm">
              <CardHeader>
                <Badge variant="secondary" className="w-fit">
                  Step 3
                </Badge>
                <CardTitle>Failure-mechanism categories</CardTitle>
                <CardDescription>
                  Labels for this upload only (e.g. leaking, corroded, cracked). Auto-fill mines keywords
                  from <strong>{runConfig.text_column || "discrepancy text"}</strong>
                  {runConfig.label_column ? ` and reference column ${runConfig.label_column}` : ""}.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <CategoryPresets
                  categories={analysisConfig.custom_categories}
                  onApply={(custom_categories) =>
                    setAnalysisConfig((prev) => ({ ...prev, custom_categories }))
                  }
                />
                <TokenGenerator
                  value={analysisConfig.custom_categories}
                  corpus={corpusTexts}
                  uploadRows={editedRows}
                  textColumn={runConfig.text_column}
                  labelColumn={runConfig.label_column}
                  partColumn={runConfig.part_column || undefined}
                  onChange={(custom_categories) =>
                    setAnalysisConfig((prev) => ({ ...prev, custom_categories }))
                  }
                  embedded
                />
                <div className="flex flex-wrap gap-2">
                  <Button
                    type="button"
                    disabled={!stepCategories}
                    onClick={() => setActiveStep(4)}
                  >
                    Continue to run →
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}

          {stepUpload && activeStep === 4 && (
            <Card className="min-w-0 shadow-sm">
              <CardHeader>
                <Badge variant="secondary" className="w-fit">
                  Step 4
                </Badge>
                <CardTitle>Configure &amp; run</CardTitle>
                <CardDescription>
                  {previewOnly
                    ? `Full dataset: ${effectiveRowCount.toLocaleString()} rows (preview below). Map columns and run — data stays on server.`
                    : `Review records (${effectiveRowCount.toLocaleString()}), map columns, select models, then run.`}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {previewOnly && (
                  <Alert>
                    <AlertDescription>
                      Large-file mode: only the first {editedRows.length.toLocaleString()} rows are
                      shown. Classification runs on all {effectiveRowCount.toLocaleString()} rows.
                    </AlertDescription>
                  </Alert>
                )}

                <EditableGrid
                  columns={columns}
                  rows={editedRows}
                  onRowsChange={setEditedRows}
                  readOnly={previewOnly}
                />

                <ModelSelector
                  columns={columns}
                  value={runConfig}
                  onChange={setRunConfig}
                  fetchModels={fetchAvailableModels}
                  showColumns={false}
                  showModels
                />

                <div className="grid max-w-xs gap-2">
                  <label className="text-sm font-medium" htmlFor="xai-top-k">
                    Explainability top-K (ranked classes in review)
                  </label>
                  <Input
                    id="xai-top-k"
                    type="number"
                    min={1}
                    max={10}
                    value={analysisConfig.xai_top_k ?? 3}
                    onChange={(e) =>
                      setAnalysisConfig((prev) => ({
                        ...prev,
                        xai_top_k: Math.min(10, Math.max(1, Number(e.target.value) || 3)),
                      }))
                    }
                  />
                </div>

                <Card className="min-w-0 border-dashed bg-muted/20">
                  <CardHeader>
                    <CardTitle className="text-base">Advanced settings</CardTitle>
                    <CardDescription>
                      Evidence rules, ensemble behaviour, and training parameters apply on the next
                      run.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <AnalysisSettings
                      selectedModels={runConfig.models}
                      userSettings={analysisConfig.user_settings}
                      onChange={(user_settings) =>
                        setAnalysisConfig((prev) => ({ ...prev, user_settings }))
                      }
                    />
                    <Button variant="outline" type="button" onClick={handleTrainSettings} disabled={busy}>
                      {busy ? "Saving…" : "Save models to disk (optional)"}
                    </Button>
                  </CardContent>
                </Card>

                {error && (
                  <Alert variant="destructive">
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                )}

                <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
                  <Button size="lg" type="button" onClick={handlePredict} disabled={busy || !stepReady}>
                    {busy ? "Running analysis…" : "Run classification →"}
                  </Button>
                  {!stepReady && (
                    <p className="text-sm text-muted-foreground">
                      Complete upload, at least one category, text column, and model selection.
                    </p>
                  )}
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        <aside className="min-w-0 space-y-4 xl:sticky xl:top-6 xl:self-start">
          <SetupChecklist
            stepUpload={stepUpload}
            stepCategories={stepCategories}
            hasTextColumn={hasTextColumn}
            hasModels={hasModels}
            stepReady={stepReady}
            sourceFilename={sourceFilename}
            rowCount={effectiveRowCount}
            categoriesCount={categoriesCount}
          />

          {stepUpload && (
            <DatasetSummaryCard
              sourceFilename={sourceFilename}
              rowCount={effectiveRowCount}
              previewOnly={previewOnly}
              previewRows={editedRows.length}
              columns={columns}
              textColumn={runConfig.text_column}
            />
          )}

          <CategoryPreviewCard categories={analysisConfig.custom_categories} />

          {stepReady && (
            <Card className="border-primary/30 bg-primary/5 shadow-sm">
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Ready to classify</CardTitle>
                <CardDescription>
                  {effectiveRowCount.toLocaleString()} rows · {categoriesCount} categories ·{" "}
                  {runConfig.models?.length} model(s)
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Button className="w-full" size="lg" onClick={handlePredict} disabled={busy}>
                  {busy ? "Running…" : "Run classification"}
                </Button>
              </CardContent>
            </Card>
          )}
        </aside>
      </div>
    </PageShell>
  );
}

export default Dashboard;
