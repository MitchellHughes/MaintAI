/** Shared triage rules for results tables and charts. */

import { normalizeLabelGroups } from "./labelGroups";

export const DEFAULT_AUTO_ACCEPT_THRESHOLD = 0.85;

export function tierForRow(row) {
  return row?.confidence_tier || row?.xai?.simple?.tier || "review";
}

/**
 * Conservative auto-accept: trust backend "high" tier first; medium only above slider.
 * Reduces false auto-accepts when confidence is unstable across datasets.
 */
export function isAutoAcceptRow(row, threshold = DEFAULT_AUTO_ACCEPT_THRESHOLD) {
  const tier = tierForRow(row);
  if (tier === "high") return true;
  if (tier === "medium") return (row.confidence ?? 0) >= threshold;
  return false;
}

function normalizeLabel(value) {
  return String(value ?? "")
    .trim()
    .toLowerCase();
}

function parseKeywords(category) {
  if (Array.isArray(category.keywords)) {
    return category.keywords.map((k) => normalizeLabel(k)).filter(Boolean);
  }
  return String(category.keywords || "")
    .split(/[,;]/)
    .map((k) => normalizeLabel(k))
    .filter(Boolean);
}

function parseReferenceAliases(category) {
  const raw = category.reference_aliases;
  if (Array.isArray(raw)) {
    return raw.map((a) => normalizeLabel(a)).filter(Boolean);
  }
  return String(raw || "")
    .split(/[,;]/)
    .map((a) => normalizeLabel(a))
    .filter(Boolean);
}

/** Map free-text ground truth (e.g. CMMS part condition) to one of your defined categories. */
export function buildCategoryMatcher(categories, labelGroups = []) {
  const registered = new Set();
  const entries = (categories || [])
    .map((cat) => {
      const label = normalizeLabel(cat.label);
      const keywords = [...parseKeywords(cat), ...parseReferenceAliases(cat)];
      if (label && !keywords.includes(label)) {
        keywords.unshift(label);
      }
      if (label) registered.add(label);
      return { label, keywords };
    })
    .filter((e) => e.label);

  for (const group of normalizeLabelGroups(labelGroups)) {
    const head = group.group_label;
    if (!head || registered.has(head)) continue;
    entries.push({ label: head, keywords: [head, ...group.members] });
    registered.add(head);
  }

  return function matchCategory(raw) {
    const text = normalizeLabel(raw);
    if (!text || text === "unclassified") return null;

    for (const entry of entries) {
      if (text === entry.label) return entry.label;
    }
    for (const entry of entries) {
      for (const kw of entry.keywords) {
        if (kw && text === kw) return entry.label;
      }
    }
    for (const entry of entries) {
      for (const kw of entry.keywords) {
        if (!kw) continue;
        if (text.includes(kw) || kw.includes(text)) return entry.label;
      }
    }
    return null;
  };
}

/**
 * Macro F1 over your defined categories only (not every distinct value in the reference column).
 * Rows whose reference label cannot be mapped to a category are excluded from the score.
 */
export function macroF1FromRows(
  rows,
  { categories = [], predKey = "final_condition", actualKey = "actual_label" } = {},
) {
  if (!categories?.length) return null;

  const match = buildCategoryMatcher(categories);
  const classLabels = categories.map((c) => normalizeLabel(c.label)).filter(Boolean);
  if (!classLabels.length) return null;

  const pairs = [];
  let skipped = 0;

  rows.forEach((row) => {
    const actual = match(row[actualKey]);
    if (!actual) {
      skipped += 1;
      return;
    }
    const rawPred = row[predKey] || row.predicted_condition;
    const pred = match(rawPred) || normalizeLabel(rawPred);
    pairs.push({ actual, pred });
  });

  if (!pairs.length) return null;

  let f1Sum = 0;
  classLabels.forEach((label) => {
    let tp = 0;
    let fp = 0;
    let fn = 0;
    pairs.forEach(({ actual, pred }) => {
      const act = actual === label;
      const pr = pred === label;
      if (pr && act) tp += 1;
      else if (pr && !act) fp += 1;
      else if (!pr && act) fn += 1;
    });
    const precision = tp / (tp + fp) || 0;
    const recall = tp / (tp + fn) || 0;
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
    f1Sum += f1;
  });

  return {
    macroF1: f1Sum / classLabels.length,
    evaluatedRows: pairs.length,
    skippedRows: skipped,
    targetClasses: classLabels.length,
  };
}
