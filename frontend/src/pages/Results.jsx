import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Download, FileSpreadsheet } from "lucide-react";

import { useApp } from "../context/AppContext";
import { saveRunSnapshot } from "../services/api";
import PredictionTable from "../components/PredictionTable";
import LabelMergePanel from "../components/LabelMergePanel";
import ClassificationReportPanel from "../components/ClassificationReportPanel";
import ExportDatasetModal from "../components/ExportDatasetModal";
import RunTimingPanel from "../components/RunTimingPanel";
import ReviewBulkActions from "../components/ReviewBulkActions";
import ActiveLearningPanel from "../components/ActiveLearningPanel";
import ThresholdAnalyticsPanel from "../components/ThresholdAnalyticsPanel";
import WorkflowJourney from "../components/WorkflowJourney";
import PageHeader from "../components/PageHeader";
import PageShell from "../components/PageShell";
import { DEFAULT_AUTO_ACCEPT_THRESHOLD } from "../utils/engineerReview";
import { EMPTY_LABEL_GROUPS, serializeLabelGroups } from "../utils/labelGroups";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

const MODEL_LABELS = {
  TokenMatchingClassifier: "Token matching",
  EquipmentBasedClassifier: "Equipment based",
  SemanticSimilarityClassifier: "Semantic similarity",
  UMECClassifier: "Ensemble (recommended)",
};

function normalizeResultsByModel(prediction) {
  if (!prediction) return {};
  if (prediction.results_by_model && typeof prediction.results_by_model === "object") {
    return prediction.results_by_model;
  }
  if (Array.isArray(prediction.predictions) && prediction.predictions.length) {
    return { UMECClassifier: prediction.predictions };
  }
  return {};
}

function Results() {
  const navigate = useNavigate();
  const { prediction, sourceFilename, lastRunId, restoreRunById, setAnalysisConfig, analysisConfig, setPrediction, runConfig, columns, uploadId, rowCount, previewOnly } = useApp();

  const savedLabelGroups = useMemo(() => {
    const groups = analysisConfig.label_groups;
    return groups?.length ? groups : EMPTY_LABEL_GROUPS;
  }, [analysisConfig.label_groups]);

  const savedGroupsKey = useMemo(() => serializeLabelGroups(savedLabelGroups), [savedLabelGroups]);
  const predictionKey = prediction?.generated_at ?? "";
  const [reportLabelGroups, setReportLabelGroups] = useState(() => {
    const fromPrediction = prediction?.label_groups;
    return fromPrediction?.length ? fromPrediction : EMPTY_LABEL_GROUPS;
  });
  const [evaluationRefreshToken, setEvaluationRefreshToken] = useState(0);

  useEffect(() => {
    const fromPrediction = prediction?.label_groups;
    const source = fromPrediction?.length ? fromPrediction : savedLabelGroups;
    if (!source.length) return;
    setReportLabelGroups((prev) => {
      if (serializeLabelGroups(prev) === serializeLabelGroups(source)) return prev;
      return source;
    });
  }, [predictionKey, savedGroupsKey, savedLabelGroups, prediction?.label_groups]);

  const bumpEvaluationReport = useCallback(() => {
    setEvaluationRefreshToken((token) => token + 1);
  }, []);

  const handleApplyMergePreview = useCallback(
    (draftGroups) => {
      const next = draftGroups.length ? draftGroups : savedLabelGroups;
      setReportLabelGroups(next);
      bumpEvaluationReport();
    },
    [bumpEvaluationReport, savedLabelGroups],
  );

  const handleSaveLabelGroups = useCallback(
    ({ label_groups, custom_categories, merge_categories }) => {
      const nextGroups = label_groups?.length ? label_groups : EMPTY_LABEL_GROUPS;
      const nextCategories = merge_categories
        ? custom_categories
        : analysisConfig.custom_categories;
      const nextConfig = {
        ...analysisConfig,
        label_groups: nextGroups,
        custom_categories: nextCategories,
      };
      setAnalysisConfig(nextConfig);
      setReportLabelGroups(nextGroups);
      bumpEvaluationReport();

      if (prediction) {
        setPrediction((prev) =>
          prev
            ? {
                ...prev,
                label_groups: nextGroups,
                ...(merge_categories ? { custom_categories: nextCategories } : {}),
              }
            : prev,
        );
      }

      if (lastRunId && prediction) {
        saveRunSnapshot({
          prediction: {
            ...prediction,
            label_groups: nextGroups,
            ...(merge_categories ? { custom_categories: nextCategories } : {}),
          },
          runConfig,
          analysisConfig: nextConfig,
          columns,
          uploadId,
          sourceFilename: sourceFilename || prediction.source_filename || "",
          rowCount,
          previewOnly,
        }).catch(() => {});
      }
    },
    [
      analysisConfig,
      columns,
      lastRunId,
      prediction,
      previewOnly,
      rowCount,
      runConfig,
      setAnalysisConfig,
      setPrediction,
      sourceFilename,
      uploadId,
      bumpEvaluationReport,
    ],
  );
  const [error, setError] = useState("");
  const [restoring, setRestoring] = useState(false);
  const [editedPredictions, setEditedPredictions] = useState([]);
  const [undoStack, setUndoStack] = useState([]);
  const [bulkMessage, setBulkMessage] = useState("");
  const [autoAcceptThreshold, setAutoAcceptThreshold] = useState(DEFAULT_AUTO_ACCEPT_THRESHOLD);
  const [exportOpen, setExportOpen] = useState(false);
  const resultsByModel = useMemo(() => normalizeResultsByModel(prediction), [prediction]);

  const modelKeys = useMemo(() => {
    if (prediction?.models?.length) return prediction.models;
    return Object.keys(resultsByModel);
  }, [prediction?.models, resultsByModel]);

  const preferredModel = modelKeys.includes("UMECClassifier")
    ? "UMECClassifier"
    : modelKeys[0] || "";

  const [activeModel, setActiveModel] = useState("");

  useEffect(() => {
    if (!modelKeys.length) {
      setActiveModel("");
      return;
    }
    setActiveModel((current) =>
      current && modelKeys.includes(current) ? current : preferredModel,
    );
  }, [predictionKey, modelKeys, preferredModel]);

  useEffect(() => {
    const rows = resultsByModel[activeModel] ?? [];
    setEditedPredictions(
      rows.map((p) => ({
        ...p,
        final_condition: p.final_condition || p.predicted_condition,
      })),
    );
    setUndoStack([]);
    setBulkMessage("");
  }, [predictionKey, activeModel, resultsByModel]);

  async function handleReopenLastRun() {
    if (!lastRunId) return;
    setRestoring(true);
    setError("");
    try {
      await restoreRunById(lastRunId);
      navigate("/results");
    } catch (e) {
      setError(e.message);
    } finally {
      setRestoring(false);
    }
  }

  function pushUndo() {
    setUndoStack((prev) => [...prev.slice(-4), editedPredictions]);
  }

  function handleBulkApply(updater, description) {
    setEditedPredictions(updater(editedPredictions));
    if (description) setBulkMessage(description);
  }

  function handleUndo() {
    setUndoStack((prev) => {
      if (!prev.length) return prev;
      const next = [...prev];
      const last = next.pop();
      setEditedPredictions(last);
      setBulkMessage("Undid last bulk action");
      return next;
    });
  }

  if (!prediction) {
    return (
      <PageShell>
        <PageHeader
          title="Results & review"
          description="Classification output and engineer review queue."
        />
        <WorkflowJourney currentPhase="upload" />
        <Card className="shadow-sm">
          <CardContent className="flex flex-col items-start gap-4 pt-6">
            <p className="text-muted-foreground">
              No classification run yet. Start in the analysis workspace: upload data, define
              categories, run classification — then you&apos;ll land here to review and export.
            </p>
            <div className="flex flex-wrap gap-2">
              <Button asChild>
                <Link to="/">Open analysis workspace</Link>
              </Button>
              {lastRunId && (
                <Button variant="outline" disabled={restoring} onClick={handleReopenLastRun}>
                  {restoring ? "Restoring…" : "Reopen last run"}
                </Button>
              )}
              <Button variant="ghost" asChild>
                <Link to="/history">Browse run history</Link>
              </Button>
            </div>
            {error && (
              <Alert variant="destructive" className="w-full">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>
      </PageShell>
    );
  }

  const { model_version, generated_at, source = [], source_filename } = prediction;
  const displayFilename = source_filename || sourceFilename || "";
  const totalRows = editedPredictions.length;

  function handleEdit(rowId, value) {
    setEditedPredictions((prev) =>
      prev.map((p) => (p.row_id === rowId ? { ...p, final_condition: value } : p)),
    );
  }

  return (
    <PageShell>
      <ExportDatasetModal
        open={exportOpen}
        onClose={() => setExportOpen(false)}
        sourceRows={source}
        uploadId={prediction.upload_id}
        predictions={editedPredictions}
      />

      <PageHeader
        title="Results & review"
        description="Adjust the confidence threshold, review flagged rows, then export — that completes the workflow."
        meta={
          <>
            {displayFilename && (
              <Badge
                variant="outline"
                className="max-w-full gap-1.5 truncate font-mono font-normal"
                title={displayFilename}
              >
                <FileSpreadsheet className="size-3.5 shrink-0" />
                <span className="truncate">{displayFilename}</span>
              </Badge>
            )}
            <Badge variant="secondary">
              {totalRows.toLocaleString()} rows · {model_version}
            </Badge>
          </>
        }
        actions={
          <>
            <Button size="sm" type="button" onClick={() => setExportOpen(true)}>
              <Download className="mr-1.5 size-4" />
              Export labeled dataset
            </Button>
            <Button variant="outline" size="sm" asChild>
              <Link to="/">Back to workspace</Link>
            </Button>
          </>
        }
      />

      <WorkflowJourney currentPhase="review" />

      {modelKeys.length > 1 && (
        <Card className="min-w-0 shadow-sm">
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Compare models</CardTitle>
            <CardDescription>Charts and review queue follow the selected model.</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-wrap gap-2">
            {modelKeys.map((key) => (
              <Button
                key={key}
                type="button"
                variant={key === activeModel ? "default" : "outline"}
                size="sm"
                onClick={() => setActiveModel(key)}
              >
                {MODEL_LABELS[key] || key}
              </Button>
            ))}
          </CardContent>
        </Card>
      )}

      <ThresholdAnalyticsPanel
        predictions={editedPredictions}
        autoAcceptThreshold={autoAcceptThreshold}
        onThresholdChange={setAutoAcceptThreshold}
        activeModel={activeModel}
        totalRows={totalRows}
      />

      <Card className="border-chart-2/30 bg-chart-2/5 shadow-sm">
        <CardContent className="flex flex-col gap-3 pt-6 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="font-medium">Final step: export your labeled dataset</p>
            <p className="text-sm text-muted-foreground">
              When review is complete, export CSV or Excel with a new failure-mechanism column.
            </p>
          </div>
          <Button onClick={() => setExportOpen(true)}>
            <Download className="mr-1.5 size-4" />
            Export & finish
          </Button>
        </CardContent>
      </Card>

      <RunTimingPanel timing={prediction.timing} />

      {prediction.label_column && (
        <>
          <LabelMergePanel
            predictions={editedPredictions}
            customCategories={prediction.custom_categories}
            savedLabelGroups={savedLabelGroups}
            reportLabelGroups={reportLabelGroups}
            onApplyPreview={handleApplyMergePreview}
            onSave={handleSaveLabelGroups}
          />
          <ClassificationReportPanel
            predictions={editedPredictions}
            customCategories={prediction.custom_categories}
            labelColumn={prediction.label_column}
            labelGroups={reportLabelGroups}
            topK={analysisConfig.xai_top_k ?? 3}
            refreshToken={evaluationRefreshToken}
          />
        </>
      )}

      <ActiveLearningPanel
        predictions={editedPredictions}
        customCategories={prediction.custom_categories}
        onApplyCategories={(categories) =>
          setAnalysisConfig((prev) => ({ ...prev, custom_categories: categories }))
        }
      />

      <Card className="min-w-0 shadow-sm">
        <CardHeader>
          <CardTitle>Review queue</CardTitle>
          <CardDescription>
            Filter flagged rows, edit incorrect labels, and use bulk actions. Threshold above
            controls auto-accept for medium-confidence rows. Re-run classification after changing
            explainability top-K. Generated {new Date(generated_at).toLocaleString()}
            {lastRunId ? ` · run ${lastRunId.slice(0, 8)}` : ""}.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <ReviewBulkActions
            predictions={editedPredictions}
            autoAcceptThreshold={autoAcceptThreshold}
            onPushUndo={pushUndo}
            onApply={handleBulkApply}
            canUndo={undoStack.length > 0}
            onUndo={handleUndo}
          />

          {bulkMessage && (
            <Alert>
              <AlertDescription>{bulkMessage}</AlertDescription>
            </Alert>
          )}

          <PredictionTable
            predictions={editedPredictions}
            onEdit={handleEdit}
            showActualColumn={Boolean(prediction?.label_column)}
            showComponentColumn={Boolean(runConfig.part_column)}
            topKLimit={analysisConfig.xai_top_k ?? 3}
            autoAcceptThreshold={autoAcceptThreshold}
            defaultTierFilter="all"
            categories={prediction?.custom_categories || analysisConfig.custom_categories}
            labelGroups={reportLabelGroups}
          />

          {error && (
            <Alert variant="destructive">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>
    </PageShell>
  );
}

export default Results;
