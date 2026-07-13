import { Check } from "lucide-react";

import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

export default function StepNav({ steps, activeStep, onStepChange }) {
  return (
    <nav
      className="flex min-w-0 flex-col gap-2 sm:flex-row sm:flex-wrap"
      aria-label="Workflow steps"
    >
      {steps.map((step) => {
        const isActive = activeStep === step.id;
        const isDone = step.state === "done";

        return (
          <Button
            key={step.id}
            type="button"
            variant={isActive ? "default" : "outline"}
            className={cn(
              "h-auto min-w-0 flex-1 justify-start gap-2 px-3 py-2.5 text-left sm:flex-none sm:min-w-[140px]",
              isDone && !isActive && "border-emerald-500/40 bg-emerald-500/5 text-foreground hover:bg-emerald-500/10",
            )}
            onClick={() => onStepChange(step.id)}
          >
            <span
              className={cn(
                "flex size-6 shrink-0 items-center justify-center rounded-full text-xs font-semibold",
                isActive && "bg-primary-foreground/20 text-primary-foreground",
                isDone && !isActive && "bg-emerald-500 text-white",
                !isActive && !isDone && "bg-muted text-muted-foreground",
              )}
            >
              {isDone ? <Check className="size-3.5" /> : step.id}
            </span>
            <span className="min-w-0 truncate text-sm font-medium">{step.label}</span>
          </Button>
        );
      })}
    </nav>
  );
}
