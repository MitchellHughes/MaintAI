import { isAutoAcceptRow } from "./engineerReview";

/** Aggregate trusted vs review counts per predicted/final label. */
export function buildStackedSummary(rows, threshold) {
  const bucket = new Map();

  rows.forEach((row) => {
    const label = String(row.final_condition || row.predicted_condition || "Unknown");
    const entry = bucket.get(label) || { label, high: 0, low: 0 };
    if (isAutoAcceptRow(row, threshold)) {
      entry.high += 1;
    } else {
      entry.low += 1;
    }
    bucket.set(label, entry);
  });

  return Array.from(bucket.values()).sort((a, b) => b.high + b.low - (a.high + a.low));
}

export function buildAcceptCounts(rows, threshold) {
  let auto = 0;
  rows.forEach((row) => {
    if (isAutoAcceptRow(row, threshold)) auto += 1;
  });
  return { auto, review: Math.max(0, rows.length - auto) };
}

/** Flat rows for Vega stacked bar / summary transforms. */
export function buildStatusByLabel(stackSummary) {
  return stackSummary.flatMap((row) => [
    { label: row.label, status: "Trusted", count: row.high },
    { label: row.label, status: "Needs review", count: row.low },
  ]);
}

export function buildConfidenceSplit(acceptCounts) {
  return [
    { status: "Trusted", count: acceptCounts.auto },
    { status: "Needs review", count: acceptCounts.review },
  ];
}

export function buildLabelVolume(stackSummary, limit = 10) {
  return stackSummary.slice(0, limit).map((row) => ({
    label: row.label,
    count: row.high + row.low,
  }));
}
