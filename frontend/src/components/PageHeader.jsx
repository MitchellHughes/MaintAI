import { Separator } from "@/components/ui/separator";
import { SidebarTrigger } from "@/components/ui/sidebar";
import { ThemeToggleButton } from "@/components/ThemeToggle";
import { cn } from "@/lib/utils";

export default function PageHeader({
  title,
  description,
  meta,
  actions,
  className,
}) {
  return (
    <div className={cn("flex min-w-0 flex-col gap-4 pb-2", className)}>
      <div className="flex min-w-0 flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
        <div className="flex min-w-0 flex-1 items-start gap-3">
          <SidebarTrigger className="-ml-1 mt-0.5 shrink-0" />
          <div className="min-w-0 flex-1 space-y-1">
            <h1 className="truncate text-2xl font-semibold tracking-tight md:text-3xl">{title}</h1>
            {description && (
              <p className="text-sm leading-relaxed text-muted-foreground md:text-base">{description}</p>
            )}
            {meta && <div className="flex min-w-0 flex-wrap gap-2 pt-1">{meta}</div>}
          </div>
        </div>
        <div className="flex w-full min-w-0 flex-wrap items-center gap-2 xl:w-auto xl:shrink-0 xl:justify-end">
          {actions}
          <ThemeToggleButton variant="ghost" size="icon" className="shrink-0" />
        </div>
      </div>
      <Separator />
    </div>
  );
}
