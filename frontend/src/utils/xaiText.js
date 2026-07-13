/** Client-side helpers for XAI display (highlights computed on server when available). */

export const HIGHLIGHT_CLASS = {
  support: "bg-emerald-500/30 text-emerald-50 rounded px-0.5",
  alternate: "bg-red-500/25 text-red-50 rounded px-0.5",
  other: "bg-amber-500/25 text-amber-50 rounded px-0.5",
};

export function pct(v) {
  return `${Math.round((v ?? 0) * 100)}%`;
}

/** Render discrepancy with optional server-provided spans. */
export function renderHighlightedText(text, spans = []) {
  if (!text) return null;
  if (!spans?.length) return text;

  const sorted = [...spans].sort((a, b) => a.start - b.start);
  const parts = [];
  let cursor = 0;

  sorted.forEach((span, idx) => {
    const start = Math.max(0, span.start ?? 0);
    const end = Math.min(text.length, span.end ?? start);
    if (start > cursor) {
      parts.push({ key: `t-${idx}-pre`, type: "text", value: text.slice(cursor, start) });
    }
    if (end > start) {
      parts.push({
        key: `t-${idx}-hl`,
        type: "mark",
        value: text.slice(start, end),
        role: span.role || "support",
        title: span.role === "support" ? "Supports prediction" : span.role === "alternate" ? "Supports alternative class" : "Other term in text",
      });
    }
    cursor = Math.max(cursor, end);
  });

  if (cursor < text.length) {
    parts.push({ key: "t-tail", type: "text", value: text.slice(cursor) });
  }
  return parts;
}

export function formatTopPredictions(topPredictions = [], limit = 3) {
  return (topPredictions || []).slice(0, limit).map((item) => ({
    label: item.label,
    confidence: item.confidence ?? 0,
  }));
}

/** Fallback when reopening runs saved before inline spans existed. */
export function fallbackSpansFromKeywords(text, keywords = []) {
  if (!text || !keywords?.length) return [];
  const spans = [];
  const lower = text.toLowerCase();
  keywords.forEach((raw) => {
    const term = String(raw).trim().toLowerCase();
    if (term.length < 3) return;
    const re = new RegExp(`\\b${term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`, "i");
    const match = re.exec(text);
    if (match) {
      spans.push({
        start: match.index,
        end: match.index + match[0].length,
        text: match[0],
        role: "support",
      });
    }
  });
  return spans.sort((a, b) => a.start - b.start);
}
