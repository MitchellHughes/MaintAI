import { useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

function formatLabel(label) {
  const text = String(label || "").trim();
  if (!text) return "—";
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function truncateLabel(label, maxLen) {
  const formatted = formatLabel(label);
  if (formatted.length <= maxLen) return formatted;
  return `${formatted.slice(0, maxLen - 1)}…`;
}

function cellStyle(value, maxValue, isDiagonal) {
  if (value <= 0) {
    return {
      backgroundColor: "hsl(var(--muted) / 0.35)",
      color: "hsl(var(--muted-foreground))",
    };
  }
  const intensity = maxValue > 0 ? value / maxValue : 0;
  if (isDiagonal) {
    return {
      backgroundColor: `hsl(142 71% 35% / ${0.25 + intensity * 0.65})`,
      color: "hsl(0 0% 98%)",
    };
  }
  return {
    backgroundColor: `hsl(38 92% 45% / ${0.2 + intensity * 0.55})`,
    color: intensity > 0.45 ? "hsl(0 0% 98%)" : "hsl(var(--foreground))",
  };
}

function normalizeRows(matrix) {
  return matrix.map((row) => {
    const sum = row.reduce((a, b) => a + b, 0);
    if (!sum) return row.map(() => 0);
    return row.map((v) => v / sum);
  });
}

export default function ConfusionMatrixHeatmap({ labels = [], matrix = [] }) {
  const [mode, setMode] = useState("count");

  const n = labels.length;
  const displayMatrix = useMemo(
    () => (mode === "row_pct" ? normalizeRows(matrix) : matrix),
    [matrix, mode],
  );

  const maxValue = useMemo(() => {
    let max = 0;
    displayMatrix.forEach((row) => {
      row.forEach((v) => {
        if (v > max) max = v;
      });
    });
    return max;
  }, [displayMatrix]);

  const layout = useMemo(() => {
    if (n <= 8) {
      return { cell: 56, font: "text-sm", labelMax: 16, corner: 168 };
    }
    if (n <= 14) {
      return { cell: 46, font: "text-xs", labelMax: 12, corner: 148 };
    }
    return { cell: 40, font: "text-[11px]", labelMax: 10, corner: 132 };
  }, [n]);

  if (!n || !matrix?.length) return null;

  const verticalHeaders = n > 10;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-sm text-muted-foreground">
          Rows = <strong className="text-foreground">actual</strong> (reference) · Columns ={" "}
          <strong className="text-foreground">predicted</strong> · Brighter diagonal = correct
          matches
        </p>
        <div className="flex gap-1 rounded-lg border border-border p-0.5">
          <Button
            type="button"
            size="sm"
            variant={mode === "count" ? "default" : "ghost"}
            className="h-8"
            onClick={() => setMode("count")}
          >
            Counts
          </Button>
          <Button
            type="button"
            size="sm"
            variant={mode === "row_pct" ? "default" : "ghost"}
            className="h-8"
            onClick={() => setMode("row_pct")}
          >
            Row %
          </Button>
        </div>
      </div>

      <div className="rounded-lg border border-border bg-muted/10">
        <div className="max-h-[min(78vh,960px)] overflow-auto">
          <table
            className="border-separate border-spacing-0"
            style={{ minWidth: layout.corner + n * layout.cell }}
          >
            <thead>
              <tr>
                <th
                  className="sticky left-0 top-0 z-30 border-b border-r border-border bg-card px-2 py-2 text-left text-xs font-medium text-muted-foreground"
                  style={{ minWidth: layout.corner, width: layout.corner }}
                >
                  Actual ↓
                  <br />
                  Predicted →
                </th>
                {labels.map((label) => (
                  <th
                    key={`h-${label}`}
                    className={cn(
                      "sticky top-0 z-20 border-b border-border bg-card font-medium text-muted-foreground",
                      layout.font,
                    )}
                    style={{
                      minWidth: layout.cell,
                      width: layout.cell,
                      height: layout.corner,
                      maxHeight: layout.corner,
                    }}
                    title={formatLabel(label)}
                  >
                    <div
                      className={cn(
                        "flex h-full items-end justify-center px-0.5 pb-1",
                        verticalHeaders && "pb-2",
                      )}
                    >
                      <span
                        className={verticalHeaders ? "inline-block max-h-[120px] truncate" : ""}
                        style={
                          verticalHeaders
                            ? {
                                writingMode: "vertical-rl",
                                transform: "rotate(180deg)",
                                maxWidth: layout.cell - 4,
                              }
                            : undefined
                        }
                      >
                        {truncateLabel(label, layout.labelMax)}
                      </span>
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {labels.map((rowLabel, i) => (
                <tr key={`r-${rowLabel}`}>
                  <th
                    scope="row"
                    className={cn(
                      "sticky left-0 z-10 border-b border-r border-border bg-card px-2 text-left font-medium text-muted-foreground",
                      layout.font,
                    )}
                    style={{
                      minWidth: layout.corner,
                      width: layout.corner,
                      height: layout.cell,
                    }}
                    title={formatLabel(rowLabel)}
                  >
                    {truncateLabel(rowLabel, layout.labelMax + 6)}
                  </th>
                  {labels.map((colLabel, j) => {
                    const value = displayMatrix[i]?.[j] ?? 0;
                    const isDiagonal = i === j;
                    const display =
                      mode === "row_pct"
                        ? value > 0
                          ? `${(value * 100).toFixed(0)}%`
                          : "·"
                        : value > 0
                          ? String(value)
                          : "·";
                    return (
                      <td
                        key={`c-${rowLabel}-${colLabel}`}
                        className={cn(
                          "border-b border-r border-border/40 text-center tabular-nums font-medium",
                          layout.font,
                        )}
                        style={{
                          minWidth: layout.cell,
                          width: layout.cell,
                          height: layout.cell,
                          ...cellStyle(value, maxValue, isDiagonal),
                        }}
                        title={`Actual: ${formatLabel(rowLabel)} · Predicted: ${formatLabel(colLabel)} · ${
                          mode === "row_pct"
                            ? `${(value * 100).toFixed(1)}% of this actual class`
                            : `${value} row(s)`
                        }`}
                      >
                        {display}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <p className="text-xs text-muted-foreground">
        {n}×{n} classes · scroll to explore · hover cells for full label names
      </p>
    </div>
  );
}
