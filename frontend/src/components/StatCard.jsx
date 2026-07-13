import { cn } from "@/lib/utils";
import { Card, CardContent } from "@/components/ui/card";

export default function StatCard({
  label,
  value,
  hint,
  icon: Icon,
  tone = "default",
  className,
}) {
  const tones = {
    default: "border-border bg-card",
    success: "border-emerald-500/30 bg-emerald-500/5",
    warning: "border-amber-500/30 bg-amber-500/5",
    info: "border-sky-500/30 bg-sky-500/5",
  };

  const valueTones = {
    default: "text-foreground",
    success: "text-emerald-400",
    warning: "text-amber-400",
    info: "text-sky-400",
  };

  return (
    <Card className={cn("shadow-sm", tones[tone], className)}>
      <CardContent className="flex items-start justify-between gap-3 p-4 md:p-5">
        <div className="min-w-0 space-y-1">
          <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
            {label}
          </p>
          <p className={cn("text-2xl font-semibold tabular-nums md:text-3xl", valueTones[tone])}>
            {value}
          </p>
          {hint && (
            <p className="truncate text-xs text-muted-foreground" title={hint}>
              {hint}
            </p>
          )}
        </div>
        {Icon && (
          <div className="rounded-lg border border-border/60 bg-muted/40 p-2 text-muted-foreground">
            <Icon className="size-4" />
          </div>
        )}
      </CardContent>
    </Card>
  );
}
