import { memo, useMemo } from "react";

import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { resolveTopKChoices } from "../utils/topKChoices";

function formatShare(value) {
  const n = Number(value);
  if (!Number.isFinite(n) || n <= 0) return "—";
  if (n >= 0.01) return `${Math.round(n * 100)}%`;
  return `${(n * 100).toFixed(1)}%`;
}

function rankConfidence(item, row) {
  const stored = Number(item.confidence ?? 0);
  if (stored > 0) return stored;
  if (item.rank === 1 && row?.confidence > 0) return Number(row.confidence);
  return stored;
}

function TopKReviewCell({
  row,
  selectedRank = 1,
  activeLabel = "",
  topKLimit = 3,
  categories = [],
  labelGroups = [],
  onSelectRank,
  onPickLabel,
}) {
  const details = useMemo(
    () => resolveTopKChoices(row, topKLimit, { categories, labelGroups }),
    [row, topKLimit, categories, labelGroups],
  );

  if (!details.length) {
    return (
      <span className="text-xs text-muted-foreground" title="Re-run classification to populate top-K">
        Re-run classification for top-K
      </span>
    );
  }

  const active =
    details.find((item) => item.rank === selectedRank) || details[0];
  const keywords = active.keywords || [];
  const normalizedActive = String(activeLabel || "").trim().toLowerCase();

  function handlePick(item) {
    onSelectRank?.(item.rank);
    onPickLabel?.(item.label);
  }

  return (
    <div className="min-w-[13rem] space-y-2">
      <p className="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
        Top-{topKLimit} — click to set final
      </p>
      <div className="flex flex-col gap-1.5" role="group" aria-label="Top-K predictions">
        {details.map((item) => {
          const isSelected = item.rank === (selectedRank || active.rank);
          const isFinal =
            normalizedActive &&
            String(item.label || "").trim().toLowerCase() === normalizedActive;
          const isTop1 = item.rank === 1;
          const lowEvidence = item.evidence_backed === false && !item.reference_only;
          const isReference = Boolean(item.reference_only);
          const share = formatShare(rankConfidence(item, row));

          return (
            <button
              key={`${item.rank}-${item.label}`}
              type="button"
              className={cn(
                "w-full rounded-md border px-2.5 py-2 text-left transition-colors",
                "hover:bg-muted/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                isSelected &&
                  "border-sky-500/70 bg-sky-500/10 ring-1 ring-sky-500/25",
                !isSelected &&
                  isFinal &&
                  "border-emerald-600/50 bg-emerald-500/8",
                !isSelected &&
                  !isFinal &&
                  isTop1 &&
                  !lowEvidence &&
                  "border-border bg-card",
                !isSelected &&
                  !isFinal &&
                  lowEvidence &&
                  "border-border/80 border-dashed bg-muted/20",
                !isSelected && !isFinal && !isTop1 && !lowEvidence && "border-border bg-card",
              )}
              onClick={() => handlePick(item)}
              title={
                isReference
                  ? `${item.label} — CMMS reference label (not in your categories)`
                  : lowEvidence
                    ? `${item.label} — ensemble alternate (no keyword evidence in text)`
                    : `Set final to ${item.label} (score share ${share})`
              }
            >
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0 space-y-0.5">
                  <div className="flex flex-wrap items-center gap-1.5">
                    <span className="text-[10px] font-semibold text-muted-foreground">
                      #{item.rank}
                    </span>
                    {isTop1 && (
                      <Badge
                        variant="secondary"
                        className="h-4 px-1.5 text-[9px] font-normal uppercase"
                      >
                        predicted
                      </Badge>
                    )}
                    {isReference && (
                      <Badge
                        variant="outline"
                        className="h-4 border-amber-500/50 px-1.5 text-[9px] font-normal text-amber-700 dark:text-amber-300"
                      >
                        ref
                      </Badge>
                    )}
                    {lowEvidence && !isReference && (
                      <Badge
                        variant="outline"
                        className="h-4 border-muted-foreground/40 px-1.5 text-[9px] font-normal text-muted-foreground"
                      >
                        alt
                      </Badge>
                    )}
                    {isFinal && (
                      <Badge
                        variant="outline"
                        className="h-4 border-emerald-600/40 px-1.5 text-[9px] font-normal text-emerald-700 dark:text-emerald-400"
                      >
                        final
                      </Badge>
                    )}
                  </div>
                  <span className="block truncate text-sm font-medium capitalize text-foreground">
                    {item.label}
                  </span>
                </div>
                <span className="shrink-0 pt-0.5 text-xs tabular-nums text-muted-foreground">
                  {share}
                </span>
              </div>
            </button>
          );
        })}
      </div>
      {keywords.length > 0 ? (
        <div className="flex flex-wrap gap-1">
          {keywords.map((kw) => (
            <Badge key={kw} variant="secondary" className="text-[10px] font-normal">
              {kw}
            </Badge>
          ))}
        </div>
      ) : null}
    </div>
  );
}

export default memo(TopKReviewCell);
