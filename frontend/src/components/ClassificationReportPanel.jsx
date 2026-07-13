import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { BarChart3, RefreshCw } from "lucide-react";

import { fetchClassificationReport } from "../services/api";
import { EMPTY_LABEL_GROUPS, serializeLabelGroups } from "../utils/labelGroups";
import ConfusionMatrixHeatmap from "./ConfusionMatrixHeatmap";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

function pct(value) {
  if (value == null || value === undefined || Number.isNaN(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function pctOrZero(value) {
  if (value == null || value === undefined || Number.isNaN(Number(value))) return "0.0%";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

export default function ClassificationReportPanel({
  predictions,
  customCategories,
  labelColumn,
  labelGroups = EMPTY_LABEL_GROUPS,
  predKey = "predicted_condition",
  topK = 3,
  refreshToken = 0,
}) {
  const [report, setReport] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [lastRunKey, setLastRunKey] = useState("");
  const requestIdRef = useRef(0);

  const labelGroupsKey = useMemo(() => serializeLabelGroups(labelGroups), [labelGroups]);
  const inputKey = useMemo(
    () =>
      [
        predictions?.length ?? 0,
        labelColumn,
        predKey,
        labelGroupsKey,
        topK,
        serializeLabelGroups(customCategories),
      ].join("|"),
    [predictions?.length, labelColumn, predKey, labelGroupsKey, topK, customCategories],
  );
  const isStale = Boolean(report && lastRunKey && lastRunKey !== inputKey);

  const runReport = useCallback(async () => {
    if (!labelColumn || !customCategories?.length || !predictions?.length) {
      setReport(null);
      setError("");
      return;
    }

    const requestId = ++requestIdRef.current;
    setLoading(true);
    setError("");

    const groups = labelGroups?.length ? labelGroups : undefined;

    try {
      const data = await fetchClassificationReport({
        predictions,
        custom_categories: customCategories,
        pred_key: predKey,
        label_groups: groups,
        top_k: topK,
      });
      if (requestId !== requestIdRef.current) return;
      if (data.error) {
        setError(data.error);
        setReport(null);
      } else {
        setReport(data);
        setLastRunKey(inputKey);
      }
    } catch (e) {
      if (requestId === requestIdRef.current) setError(e.message);
    } finally {
      if (requestId === requestIdRef.current) setLoading(false);
    }
  }, [
    predictions,
    customCategories,
    labelColumn,
    predKey,
    labelGroups,
    topK,
    inputKey,
  ]);

  useEffect(() => {
    if (refreshToken > 0) {
      runReport();
    }
  }, [refreshToken, runReport]);

  if (!labelColumn) return null;

  return (
    <Card className="min-w-0 shadow-sm">
      <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div className="space-y-1">
          <CardTitle className="text-lg">Classification report</CardTitle>
        <CardDescription>
          Evaluation vs <code className="text-foreground">{labelColumn}</code> — final-label accuracy
          can exceed top-K model recall when you corrected rows in review. Re-run classification to
          improve model ranks.
          {labelGroups?.length > 0 && (
            <>
              {" "}
              · {labelGroups.length} label group{labelGroups.length === 1 ? "" : "s"} applied
            </>
          )}
        </CardDescription>
        </div>
        <Button
          type="button"
          size="sm"
          variant={report && !isStale ? "outline" : "default"}
          disabled={loading || !predictions?.length}
          onClick={runReport}
        >
          {loading ? (
            "Computing…"
          ) : report && !isStale ? (
            <>
              <RefreshCw className="mr-1.5 size-4" />
              Refresh report
            </>
          ) : (
            <>
              <BarChart3 className="mr-1.5 size-4" />
              Run evaluation report
            </>
          )}
        </Button>
      </CardHeader>
      <CardContent className="space-y-4">
        {!report && !loading && !error && (
          <p className="text-sm text-muted-foreground">
            Click <strong>Run evaluation report</strong> to score predictions against your reference
            column. Large datasets may take a few seconds.
          </p>
        )}
        {isStale && !loading && (
          <Alert>
            <AlertDescription>
              Predictions or label groups changed since this report was generated. Click{" "}
              <strong>Refresh report</strong> to update scores.
            </AlertDescription>
          </Alert>
        )}
        {error && (
          <Alert variant="destructive">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
        {report && !error && (
          <>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-7">
              <div className="rounded-lg border border-border bg-muted/20 p-3">
                <p className="text-xs text-muted-foreground">Macro F1 (top-1)</p>
                <p className="text-2xl font-semibold tabular-nums">{pct(report.macro_f1)}</p>
              </div>
              {report.strong_classes?.top_n > 0 && (
                <div className="rounded-lg border border-primary/30 bg-primary/5 p-3">
                  <p className="text-xs text-muted-foreground">
                    Macro F1 (top {report.strong_classes.top_n} by support)
                  </p>
                  <p className="text-2xl font-semibold tabular-nums">
                    {pct(report.strong_classes.macro_f1)}
                  </p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    {report.strong_classes.labels?.slice(0, 4).join(", ")}
                    {(report.strong_classes.labels?.length || 0) > 4 ? "…" : ""}
                  </p>
                </div>
              )}
              <div className="rounded-lg border border-border bg-muted/20 p-3">
                <p className="text-xs text-muted-foreground">Accuracy (top-1 model)</p>
                <p className="text-2xl font-semibold tabular-nums">{pct(report.accuracy)}</p>
                <p className="mt-1 text-xs text-muted-foreground">
                  Model&apos;s rank-1 prediction vs reference (not reviewer edits)
                </p>
              </div>
              {report.top_k_evaluated_rows > 0 && (
                <>
                  <div className="rounded-lg border border-primary/30 bg-primary/5 p-3">
                    <p className="text-xs text-muted-foreground">
                      Top-{report.top_k ?? topK} model recall
                    </p>
                    <p className="text-2xl font-semibold tabular-nums">
                      {pct(report.top_k_accuracy)}
                    </p>
                    <p className="mt-1 text-xs text-muted-foreground">
                      Reference appears in model top-{report.top_k ?? topK} (not your edits)
                    </p>
                  </div>
                  <div className="rounded-lg border border-primary/30 bg-primary/5 p-3">
                    <p className="text-xs text-muted-foreground">
                      Top-{report.top_k ?? topK} macro F1
                    </p>
                    <p className="text-2xl font-semibold tabular-nums">
                      {pctOrZero(report.top_k_macro_f1)}
                    </p>
                    <p className="mt-1 text-xs text-muted-foreground">
                      Relaxed F1 if reviewer picks the right class from top-
                      {report.top_k ?? topK}
                    </p>
                  </div>
                </>
              )}
              <div className="rounded-lg border border-border bg-muted/20 p-3">
                <p className="text-xs text-muted-foreground">Rows evaluated</p>
                <p className="text-2xl font-semibold tabular-nums">
                  {report.evaluated_rows?.toLocaleString()}
                </p>
              </div>
              <div className="rounded-lg border border-border bg-muted/20 p-3">
                <p className="text-xs text-muted-foreground">Skipped (unmapped)</p>
                <p className="text-2xl font-semibold tabular-nums">
                  {report.skipped_rows?.toLocaleString()}
                </p>
              </div>
            </div>

            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Class</TableHead>
                  <TableHead className="text-right">Precision</TableHead>
                  <TableHead className="text-right">Recall</TableHead>
                  <TableHead className="text-right">Top-1 F1</TableHead>
                  <TableHead className="text-right">Top-K F1</TableHead>
                  <TableHead className="text-right">Top-K recall</TableHead>
                  <TableHead className="text-right">Support</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {(report.per_class || []).map((row) => (
                  <TableRow key={row.label}>
                    <TableCell className="font-medium capitalize">{row.label}</TableCell>
                    <TableCell className="text-right tabular-nums">{pct(row.precision)}</TableCell>
                    <TableCell className="text-right tabular-nums">{pct(row.recall)}</TableCell>
                    <TableCell className="text-right tabular-nums">{pct(row.f1_score)}</TableCell>
                    <TableCell className="text-right tabular-nums">
                      {pctOrZero(row.top_k_f1)}
                    </TableCell>
                    <TableCell className="text-right tabular-nums">
                      {pctOrZero(row.top_k_recall)}
                    </TableCell>
                    <TableCell className="text-right tabular-nums">{row.support}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>

            {report.labels?.length > 0 && report.confusion_matrix?.length > 0 && (
              <div className="space-y-2 border-t border-border pt-6">
                <h3 className="text-base font-semibold">Confusion matrix</h3>
                <ConfusionMatrixHeatmap
                  labels={report.labels}
                  matrix={report.confusion_matrix}
                />
              </div>
            )}

            <p className="text-xs text-muted-foreground">
              Scoring column: <strong>{predKey}</strong> · {report.target_classes} categories ·
              macro precision {pct(report.macro_precision)} · macro recall {pct(report.macro_recall)}
            </p>
          </>
        )}
      </CardContent>
    </Card>
  );
}
