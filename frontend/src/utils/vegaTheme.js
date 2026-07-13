import { isAutoAcceptRow } from "./engineerReview";

/** Vega-Lite theme tokens from CSS variables (light/dark aware). */

function readCssVar(name, fallback) {
  if (typeof window === "undefined") return fallback;
  const raw = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return raw ? `hsl(${raw})` : fallback;
}

export function getVegaTheme(isDark) {
  return {
    background: "transparent",
    trusted: readCssVar("--chart-2", isDark ? "hsl(160 60% 45%)" : "hsl(160 84% 39%)"),
    review: readCssVar("--chart-3", isDark ? "hsl(30 80% 55%)" : "hsl(32 95% 44%)"),
    accent: readCssVar("--chart-1", isDark ? "hsl(220 70% 50%)" : "hsl(221 83% 53%)"),
    palette: [
      readCssVar("--chart-1", "hsl(221 83% 53%)"),
      readCssVar("--chart-2", "hsl(160 84% 39%)"),
      readCssVar("--chart-3", "hsl(32 95% 44%)"),
      readCssVar("--chart-4", "hsl(280 65% 60%)"),
      readCssVar("--chart-5", "hsl(340 75% 55%)"),
    ],
    foreground: readCssVar("--foreground", isDark ? "hsl(0 0% 98%)" : "hsl(240 10% 3.9%)"),
    muted: readCssVar("--muted-foreground", isDark ? "hsl(240 5% 64.9%)" : "hsl(240 3.8% 46.1%)"),
    border: readCssVar("--border", isDark ? "hsl(240 3.7% 15.9%)" : "hsl(240 5.9% 90%)"),
    card: readCssVar("--card", isDark ? "hsl(240 10% 3.9%)" : "hsl(0 0% 100%)"),
  };
}

export function baseVegaConfig(theme) {
  return {
    background: theme.background,
    view: { stroke: null },
    font: "Inter, sans-serif",
    axis: {
      labelColor: theme.muted,
      titleColor: theme.foreground,
      gridColor: theme.border,
      domainColor: theme.border,
      labelFontSize: 11,
      titleFontSize: 12,
    },
    legend: {
      labelColor: theme.muted,
      titleColor: theme.foreground,
      labelFontSize: 11,
    },
    title: {
      color: theme.foreground,
      fontSize: 14,
      fontWeight: 600,
    },
    range: {
      category: theme.palette,
    },
  };
}

export function confidenceSplitSpec(theme, thresholdPct) {
  return {
    $schema: "https://vega.github.io/schema/vega-lite/v5.json",
    title: `Confidence split · threshold ${thresholdPct}%`,
    height: 220,
    config: baseVegaConfig(theme),
    data: { name: "confidence" },
    mark: { type: "arc", innerRadius: 62, outerRadius: 96, padAngle: 0.02 },
    encoding: {
      theta: { field: "count", type: "quantitative", stack: true },
      color: {
        field: "status",
        type: "nominal",
        scale: {
          domain: ["Trusted", "Needs review"],
          range: [theme.trusted, theme.review],
        },
        legend: { orient: "bottom", direction: "horizontal" },
      },
      tooltip: [
        { field: "status", type: "nominal", title: "Status" },
        { field: "count", type: "quantitative", title: "Rows" },
      ],
    },
  };
}

export function labelVolumeSpec(theme) {
  return {
    $schema: "https://vega.github.io/schema/vega-lite/v5.json",
    title: "Top labels by volume",
    height: 220,
    config: baseVegaConfig(theme),
    data: { name: "volume" },
    mark: { type: "bar", cornerRadiusEnd: 4 },
    encoding: {
      y: {
        field: "label",
        type: "nominal",
        sort: "-x",
        axis: { title: null, labelLimit: 120 },
      },
      x: {
        field: "count",
        type: "quantitative",
        axis: { title: "Rows", grid: true },
      },
      color: {
        field: "label",
        type: "nominal",
        legend: null,
        scale: { range: theme.palette },
      },
      tooltip: [
        { field: "label", type: "nominal", title: "Label" },
        { field: "count", type: "quantitative", title: "Rows" },
      ],
    },
  };
}

export function reviewByLabelSpec(theme, modelLabel) {
  return {
    $schema: "https://vega.github.io/schema/vega-lite/v5.json",
    title: `Review workload by label · ${modelLabel}`,
    height: 280,
    config: baseVegaConfig(theme),
    data: { name: "byLabel" },
    mark: { type: "bar", cornerRadiusEnd: 3 },
    encoding: {
      x: {
        field: "label",
        type: "nominal",
        sort: "-y",
        axis: { title: null, labelAngle: -35, labelLimit: 100 },
      },
      y: {
        field: "count",
        type: "quantitative",
        stack: "zero",
        axis: { title: "Rows" },
      },
      color: {
        field: "status",
        type: "nominal",
        scale: {
          domain: ["Trusted", "Needs review"],
          range: [theme.trusted, theme.review],
        },
        legend: { orient: "top", direction: "horizontal" },
      },
      tooltip: [
        { field: "label", type: "nominal", title: "Label" },
        { field: "status", type: "nominal", title: "Status" },
        { field: "count", type: "quantitative", title: "Rows" },
      ],
    },
  };
}

export function thresholdSensitivitySpec(theme, currentPct) {
  return {
    $schema: "https://vega.github.io/schema/vega-lite/v5.json",
    title: "Threshold sensitivity (medium-tier rows)",
    height: 160,
    config: baseVegaConfig(theme),
    layer: [
      {
        data: { name: "sensitivity" },
        mark: { type: "line", point: { filled: true, size: 60 }, strokeWidth: 2.5 },
        encoding: {
          x: {
            field: "threshold",
            type: "quantitative",
            scale: { domain: [50, 95] },
            axis: { title: "Threshold (%)", format: "d" },
          },
          y: {
            field: "trusted",
            type: "quantitative",
            axis: { title: "Trusted rows" },
          },
          color: { value: theme.accent },
          tooltip: [
            { field: "threshold", type: "quantitative", title: "Threshold %", format: "d" },
            { field: "trusted", type: "quantitative", title: "Trusted" },
            { field: "review", type: "quantitative", title: "Needs review" },
          ],
        },
      },
      {
        data: { name: "sensitivity" },
        transform: [{ filter: `datum.threshold === ${currentPct}` }],
        mark: { type: "rule", color: theme.review, strokeWidth: 2, strokeDash: [4, 4] },
        encoding: {
          x: { field: "threshold", type: "quantitative" },
        },
      },
    ],
  };
}

/** Pre-compute trusted/review counts at each slider step. */
export function buildThresholdSensitivity(rows) {
  const points = [];
  for (let pct = 50; pct <= 95; pct += 1) {
    let auto = 0;
    rows.forEach((row) => {
      if (isAutoAcceptRow(row, pct / 100)) auto += 1;
    });
    points.push({
      threshold: pct,
      trusted: auto,
      review: Math.max(0, rows.length - auto),
    });
  }
  return points;
}
