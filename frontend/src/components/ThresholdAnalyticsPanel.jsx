import { useMemo } from "react";
import { SlidersHorizontal } from "lucide-react";

import { useTheme } from "../context/ThemeContext";
import ChartSummaryTable from "./ChartSummaryTable";
import VegaChart from "./VegaChart";
import {
  buildAcceptCounts,
  buildConfidenceSplit,
  buildLabelVolume,
  buildStackedSummary,
  buildStatusByLabel,
} from "../utils/reviewAnalytics";
import {
  buildThresholdSensitivity,
  confidenceSplitSpec,
  getVegaTheme,
  labelVolumeSpec,
  reviewByLabelSpec,
  thresholdSensitivitySpec,
} from "../utils/vegaTheme";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { cn } from "@/lib/utils";

const MODEL_LABELS = {
  TokenMatchingClassifier: "Token matching",
  EquipmentBasedClassifier: "Equipment based",
  SemanticSimilarityClassifier: "Semantic similarity",
  UMECClassifier: "Ensemble",
};

function pct(part, total) {
  if (!total) return "0%";
  return `${Math.round((part / total) * 100)}%`;
}

export default function ThresholdAnalyticsPanel({
  predictions,
  autoAcceptThreshold,
  onThresholdChange,
  activeModel,
  totalRows,
}) {
  const { isDark } = useTheme();
  const theme = useMemo(() => getVegaTheme(isDark), [isDark]);
  const thresholdPct = Math.round(autoAcceptThreshold * 100);

  const stackSummary = useMemo(
    () => buildStackedSummary(predictions, autoAcceptThreshold),
    [predictions, autoAcceptThreshold],
  );

  const acceptCounts = useMemo(
    () => buildAcceptCounts(predictions, autoAcceptThreshold),
    [predictions, autoAcceptThreshold],
  );

  const sensitivityData = useMemo(
    () => buildThresholdSensitivity(predictions),
    [predictions],
  );

  const chartData = useMemo(
    () => ({
      confidence: buildConfidenceSplit(acceptCounts),
      volume: buildLabelVolume(stackSummary, 10),
      byLabel: buildStatusByLabel(stackSummary.slice(0, 12)),
      sensitivity: sensitivityData,
    }),
    [acceptCounts, stackSummary, sensitivityData],
  );

  const modelLabel = MODEL_LABELS[activeModel] || activeModel || "model";

  const specs = useMemo(
    () => ({
      confidence: confidenceSplitSpec(theme, thresholdPct),
      volume: labelVolumeSpec(theme),
      byLabel: reviewByLabelSpec(theme, modelLabel),
      sensitivity: thresholdSensitivitySpec(theme, thresholdPct),
    }),
    [theme, thresholdPct, modelLabel],
  );

  return (
    <Card id="threshold-analytics" className="min-w-0 overflow-hidden shadow-sm">
      <CardHeader className="border-b border-border/60 bg-muted/20 pb-4">
        <CardTitle className="flex items-center gap-2 text-lg">
          <SlidersHorizontal className="size-5 text-chart-1" />
          Confidence threshold &amp; analytics
        </CardTitle>
        <CardDescription>
          Adjust the slider — all charts and the summary table below update live. High-confidence
          rows are always trusted; the slider only affects medium-tier rows.
        </CardDescription>
      </CardHeader>

      <div className="sticky top-0 z-20 border-b border-border/60 bg-card/95 px-6 py-4 backdrop-blur supports-[backdrop-filter]:bg-card/80">
        <div className="grid gap-4 lg:grid-cols-[1fr_auto] lg:items-end">
          <div className="space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <Label htmlFor="threshold-slider" className="text-sm font-medium">
                Medium-tier trust threshold
              </Label>
              <span className="rounded-md bg-primary/10 px-2.5 py-0.5 text-sm font-semibold tabular-nums text-primary">
                {thresholdPct}%
              </span>
            </div>
            <Slider
              id="threshold-slider"
              min={50}
              max={95}
              step={1}
              value={[thresholdPct]}
              onValueChange={([v]) => onThresholdChange(v / 100)}
            />
          </div>

          <div className="grid grid-cols-3 gap-2 sm:gap-3">
            {[
              {
                label: "Trusted",
                value: acceptCounts.auto.toLocaleString(),
                sub: pct(acceptCounts.auto, totalRows),
                tone: "text-emerald-600 dark:text-emerald-400",
              },
              {
                label: "Review",
                value: acceptCounts.review.toLocaleString(),
                sub: pct(acceptCounts.review, totalRows),
                tone: "text-amber-600 dark:text-amber-400",
              },
              {
                label: "Labels",
                value: String(stackSummary.length),
                sub: `${totalRows.toLocaleString()} rows`,
                tone: "text-sky-600 dark:text-sky-400",
              },
            ].map((item) => (
              <div
                key={item.label}
                className="rounded-lg border border-border/70 bg-background px-3 py-2 text-center"
              >
                <p className="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
                  {item.label}
                </p>
                <p className={cn("text-xl font-semibold tabular-nums", item.tone)}>{item.value}</p>
                <p className="text-[10px] text-muted-foreground">{item.sub}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      <CardContent className="space-y-6 pt-6">
        <div className="grid min-w-0 gap-4 lg:grid-cols-2">
          <div className="min-w-0 w-full rounded-xl border border-border/60 bg-muted/10 p-3">
            <VegaChart spec={specs.confidence} data={{ confidence: chartData.confidence }} />
          </div>
          <div className="min-w-0 w-full rounded-xl border border-border/60 bg-muted/10 p-3">
            <VegaChart spec={specs.volume} data={{ volume: chartData.volume }} />
          </div>
        </div>

        <div className="min-w-0 w-full rounded-xl border border-border/60 bg-muted/10 p-3">
          <VegaChart spec={specs.byLabel} data={{ byLabel: chartData.byLabel }} minHeight={300} />
        </div>

        <div className="min-w-0 w-full rounded-xl border border-border/60 bg-muted/10 p-3">
          <VegaChart
            spec={specs.sensitivity}
            data={{ sensitivity: chartData.sensitivity }}
            minHeight={180}
          />
          <p className="mt-1 text-xs text-muted-foreground">
            Current setting: {thresholdPct}% — move the slider above to see the marker shift along
            this curve.
          </p>
        </div>

        <div className="space-y-2">
          <h3 className="text-sm font-semibold">Summary by label</h3>
          <ChartSummaryTable summaryRows={stackSummary} threshold={autoAcceptThreshold} />
        </div>
      </CardContent>
    </Card>
  );
}
