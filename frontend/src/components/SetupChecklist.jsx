import { Check, Circle, CircleDot } from "lucide-react";

import { cn } from "@/lib/utils";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

function ChecklistItem({ done, active, label, detail }) {
  const Icon = done ? Check : active ? CircleDot : Circle;

  return (
    <li
      className={cn(
        "flex gap-3 rounded-lg border px-3 py-2.5",
        done && "border-emerald-500/30 bg-emerald-500/5",
        active && !done && "border-primary/30 bg-primary/5",
        !done && !active && "border-border/60 bg-muted/10",
      )}
    >
      <Icon
        className={cn(
          "mt-0.5 size-4 shrink-0",
          done && "text-emerald-400",
          active && !done && "text-primary",
          !done && !active && "text-muted-foreground",
        )}
      />
      <div className="min-w-0">
        <p className="text-sm font-medium">{label}</p>
        {detail && <p className="text-xs text-muted-foreground">{detail}</p>}
      </div>
    </li>
  );
}

export default function SetupChecklist({
  stepUpload,
  stepCategories,
  hasTextColumn,
  hasModels,
  stepReady,
  sourceFilename,
  rowCount,
  categoriesCount,
}) {
  const items = [
    {
      done: stepUpload,
      active: !stepUpload,
      label: "Upload dataset",
      detail: stepUpload
        ? `${sourceFilename || "File loaded"} · ${rowCount?.toLocaleString() || 0} rows`
        : "CSV or Excel with discrepancy text",
    },
    {
      done: hasTextColumn,
      active: stepUpload && !hasTextColumn,
      label: "Map columns",
      detail: hasTextColumn ? "Discrepancy column selected" : "Step 2 — required before keywords",
    },
    {
      done: stepCategories,
      active: hasTextColumn && !stepCategories,
      label: "Define categories",
      detail: stepCategories
        ? `${categoriesCount} failure-mechanism label(s)`
        : "Step 3 — labels and keywords",
    },
    {
      done: hasModels,
      active: hasTextColumn && !hasModels,
      label: "Select model(s)",
      detail: hasModels ? "At least one classifier chosen" : "Ensemble recommended for best F1",
    },
    {
      done: stepReady,
      active: hasModels && !stepReady,
      label: "Run classification",
      detail: stepReady ? "Ready to start analysis" : "Complete the steps above",
    },
  ];

  const completed = items.filter((item) => item.done).length;
  const progress = Math.round((completed / items.length) * 100);

  return (
    <Card className="shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Setup progress</CardTitle>
        <CardDescription>
          {completed} of {items.length} complete · {progress}%
        </CardDescription>
        <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-muted">
          <div
            className="h-full rounded-full bg-primary transition-all duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>
      </CardHeader>
      <CardContent>
        <ul className="space-y-2">
          {items.map((item) => (
            <ChecklistItem key={item.label} {...item} />
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}
