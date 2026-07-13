import { memo, useMemo } from "react";

import { HIGHLIGHT_CLASS, fallbackSpansFromKeywords, renderHighlightedText } from "../utils/xaiText";

function DiscrepancyHighlight({ text, spans, keywords }) {
  const resolvedSpans = useMemo(() => {
    if (spans?.length) return spans;
    return fallbackSpansFromKeywords(text, keywords);
  }, [text, spans, keywords]);

  const parts = useMemo(() => renderHighlightedText(text, resolvedSpans), [text, resolvedSpans]);

  if (!text) return <span className="muted">—</span>;
  if (!parts || typeof parts === "string") {
    return <span>{text}</span>;
  }

  return (
    <span className="leading-relaxed">
      {parts.map((part) =>
        part.type === "mark" ? (
          <mark
            key={part.key}
            className={HIGHLIGHT_CLASS[part.role] || HIGHLIGHT_CLASS.support}
            title={part.title}
          >
            {part.value}
          </mark>
        ) : (
          <span key={part.key}>{part.value}</span>
        ),
      )}
    </span>
  );
}

export default memo(DiscrepancyHighlight);
