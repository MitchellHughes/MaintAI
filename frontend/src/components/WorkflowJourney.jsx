import { useState } from "react";
import {
  ArrowRight,
  CheckCircle2,
  Download,
  FileSpreadsheet,
  Play,
  Sparkles,
  Tags,
  X,
} from "lucide-react";

import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

const STORAGE_KEY = "maintai_workflow_guide_dismissed";

const STEPS = [
  {
    id: "upload",
    icon: FileSpreadsheet,
    title: "Upload",
    short: "Upload data",
    where: "Analysis workspace · Step 1",
    detail: "Bring a CSV or Excel export with discrepancy narrative text.",
  },
  {
    id: "categories",
    icon: Tags,
    title: "Define labels",
    short: "Categories",
    where: "Analysis workspace · Step 2",
    detail: "Create ~10 failure-mechanism labels for this asset or site.",
  },
  {
    id: "classify",
    icon: Play,
    title: "Run AI",
    short: "Classify",
    where: "Analysis workspace · Step 3",
    detail: "Map columns, pick models, and classify every row automatically.",
  },
  {
    id: "review",
    icon: Sparkles,
    title: "Review",
    short: "Review",
    where: "Results & review",
    detail: "Check flagged rows, read explanations, and correct wrong labels.",
  },
  {
    id: "export",
    icon: Download,
    title: "Export",
    short: "Export",
    where: "Results & review · Finish",
    detail: "Download your labeled dataset — this is where the workflow ends.",
    isEnd: true,
  },
];

const PHASE_ORDER = STEPS.map((step) => step.id);

function phaseIndex(phase) {
  const idx = PHASE_ORDER.indexOf(phase);
  return idx >= 0 ? idx : 0;
}

export function readWorkflowGuideDismissed() {
  try {
    return localStorage.getItem(STORAGE_KEY) === "1";
  } catch {
    return false;
  }
}

export default function WorkflowJourney({
  currentPhase = "upload",
  compact = false,
  dismissible = true,
  className,
}) {
  const [dismissed, setDismissed] = useState(readWorkflowGuideDismissed);
  const activeIdx = phaseIndex(currentPhase);

  if (dismissed && dismissible) return null;

  function handleDismiss() {
    try {
      localStorage.setItem(STORAGE_KEY, "1");
    } catch {
      /* ignore */
    }
    setDismissed(true);
  }

  return (
    <Card
      className={cn(
        "overflow-hidden border-border/80 bg-gradient-to-br from-chart-1/5 via-card to-chart-2/5 shadow-sm",
        className,
      )}
    >
      <CardHeader className="space-y-3 pb-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="space-y-1">
            <div className="flex flex-wrap items-center gap-2">
              <CardTitle className={compact ? "text-base" : "text-lg"}>
                How MaintAI works
              </CardTitle>
              <Badge variant="secondary">Start → Finish</Badge>
            </div>
            <CardDescription className="max-w-3xl leading-relaxed">
              Five steps from raw discrepancy text to an exportable labeled file. You are currently
              on{" "}
              <strong className="text-foreground">
                {STEPS[activeIdx]?.title || "Upload"}
              </strong>
              . The workflow ends when you export from Results.
            </CardDescription>
          </div>
          {dismissible && (
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="shrink-0 text-muted-foreground"
              onClick={handleDismiss}
            >
              <X className="mr-1 size-4" />
              Hide guide
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <ol className="grid gap-3 md:grid-cols-5">
          {STEPS.map((step, index) => {
            const Icon = step.icon;
            const isActive = index === activeIdx;
            const isComplete = index < activeIdx;
            const isEnd = step.isEnd;

            return (
              <li
                key={step.id}
                className={cn(
                  "relative flex min-w-0 flex-col rounded-xl border p-3 transition-colors",
                  isComplete && "border-emerald-500/40 bg-emerald-500/5",
                  isActive && "border-primary/50 bg-primary/5 ring-1 ring-primary/20",
                  !isComplete && !isActive && "border-border/70 bg-background/50",
                  isEnd && isActive && "border-chart-2/50 bg-chart-2/5",
                )}
              >
                {index < STEPS.length - 1 && (
                  <ArrowRight className="absolute -right-2 top-1/2 z-10 hidden size-4 -translate-y-1/2 text-muted-foreground md:block" />
                )}
                <div className="mb-2 flex items-center gap-2">
                  <div
                    className={cn(
                      "flex size-8 shrink-0 items-center justify-center rounded-lg",
                      isComplete && "bg-emerald-500 text-white",
                      isActive && !isComplete && "bg-primary text-primary-foreground",
                      !isComplete && !isActive && "bg-muted text-muted-foreground",
                    )}
                  >
                    {isComplete ? <CheckCircle2 className="size-4" /> : <Icon className="size-4" />}
                  </div>
                  <div className="min-w-0">
                    <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                      {index + 1}. {step.short}
                    </p>
                    <p className="truncate font-semibold">{step.title}</p>
                  </div>
                </div>
                {!compact && (
                  <>
                    <p className="text-xs text-muted-foreground">{step.where}</p>
                    <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{step.detail}</p>
                  </>
                )}
                {isActive && (
                  <Badge className="mt-2 w-fit" variant={isEnd ? "success" : "default"}>
                    {isEnd ? "Finish here" : "You are here"}
                  </Badge>
                )}
              </li>
            );
          })}
        </ol>
      </CardContent>
    </Card>
  );
}
