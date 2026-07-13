/** Resolve ranked label options for a prediction row (review table + XAI). */

import { buildCategoryMatcher } from "./engineerReview";
import { normalizeLabelGroups } from "./labelGroups";

function normalizeLabel(value) {
  return String(value || "").trim().toLowerCase();
}

function surfaceTermInText(text, candidates) {
  const blob = normalizeLabel(text);
  if (!blob) return false;
  return candidates.some((raw) => {
    const term = normalizeLabel(raw);
    if (!term) return false;
    const pattern = new RegExp(`\\b${term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`, "i");
    return pattern.test(blob);
  });
}

function enrichTopKWithMappedActual(choices, row, limit, categories, labelGroups) {
  const cap = Math.max(1, limit);
  const actual = row?.actual_label;
  if (!actual || !categories?.length) return choices.slice(0, cap);

  const match = buildCategoryMatcher(categories, normalizeLabelGroups(labelGroups));
  const mapped = match(actual);
  if (!mapped) return choices.slice(0, cap);

  const predicted = normalizeLabel(row?.predicted_condition || row?.final_condition);
  const inText = surfaceTermInText(row?.discrepancy, [mapped, actual]);
  const existing = new Set(choices.map((item) => normalizeLabel(item.label)));
  if (existing.has(mapped)) return choices.slice(0, cap);
  if (!inText && predicted === mapped) return choices.slice(0, cap);

  const enriched = [
    ...choices,
    {
      rank: choices.length + 1,
      label: mapped,
      confidence: 0,
      keywords: inText ? [mapped] : [],
      evidence_backed: inText,
      reference_only: !inText,
    },
  ];
  return enriched.slice(0, cap).map((item, index) => ({ ...item, rank: index + 1 }));
}

function mapRanked(items, row, keywordsByRank = {}) {
  return (items || [])
    .filter((item) => item?.label)
    .map((item, index) => {
      const rank = item.rank ?? index + 1;
      let confidence = Number(item.confidence ?? 0);
      if (confidence <= 0 && rank === 1 && row?.confidence > 0) {
        confidence = Number(row.confidence);
      }
      return {
        rank,
        label: String(item.label),
        confidence,
        keywords: item.keywords || keywordsByRank[rank] || [],
        text_spans: item.text_spans,
        evidence_backed: item.evidence_backed !== false,
      reference_only: Boolean(item.reference_only),
      };
    });
}

/**
 * @param {object} row
 * @param {number} limit
 * @returns {Array<{rank:number,label:string,confidence:number,keywords:string[],text_spans?:object[]}>}
 */
export function resolveTopKChoices(row, limit = 3, options = {}) {
  const cap = Math.max(1, limit);
  const { categories, labelGroups } = options;
  const seen = new Set();
  const details = row?.xai?.simple?.top_k_details || row?.top_k_details;
  if (Array.isArray(details) && details.length) {
    const ranked = mapRanked(details.slice(0, cap), row);
    return enrichTopKWithMappedActual(ranked, row, cap, categories, labelGroups);
  }

  const merged = [];
  const sources = [
    row?.top_predictions,
    row?.xai?.simple?.top_ranked,
    row?.xai?.simple?.top_k_details,
  ];
  for (const source of sources) {
    if (!Array.isArray(source)) continue;
    for (const item of source) {
      const label = String(item?.label || "").trim();
      const key = label.toLowerCase();
      if (!label || seen.has(key)) continue;
      seen.add(key);
      merged.push(item);
      if (merged.length >= cap) break;
    }
    if (merged.length >= cap) break;
  }

  if (merged.length) {
    const ranked = mapRanked(merged.slice(0, cap), row);
    return enrichTopKWithMappedActual(ranked, row, cap, categories, labelGroups);
  }

  const choices = [];
  seen.clear();
  const primary = row?.predicted_condition;
  if (primary) {
    choices.push({
      rank: 1,
      label: String(primary),
      confidence: Number(row?.confidence ?? 0),
      keywords: row?.xai?.simple?.keywords || [],
    });
    seen.add(normalizeLabel(primary));
  }

  const runner = row?.runner_up || row?.xai?.simple?.runner_up;
  if (runner && !seen.has(normalizeLabel(runner)) && choices.length < cap) {
    choices.push({
      rank: choices.length + 1,
      label: String(runner),
      confidence: 0,
      keywords: [],
    });
    seen.add(normalizeLabel(runner));
  }

  return enrichTopKWithMappedActual(choices.slice(0, cap), row, cap, categories, labelGroups);
}
