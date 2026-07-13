import { ArrowRight, Download, FileSpreadsheet, Sparkles, Tags } from "lucide-react";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

const tips = [
  {
    icon: FileSpreadsheet,
    title: "Upload your export",
    description:
      "Drop a CSV or Excel file from your CMMS or spreadsheet. Large files stay on the server — only a preview loads in the browser.",
    step: "Workspace · Step 1",
  },
  {
    icon: Tags,
    title: "Define failure mechanisms",
    description:
      "Create labels like leaking, corroded, or cracked. Keywords can be mined automatically from your text or typed manually.",
    step: "Workspace · Step 2",
  },
  {
    icon: Sparkles,
    title: "Run & review",
    description:
      "The ensemble classifier labels every row. You are taken to Results to review flagged rows with plain-language explanations.",
    step: "Workspace → Results",
  },
  {
    icon: Download,
    title: "Export & finish",
    description:
      "Download CSV or Excel with a new label column. That export is the deliverable — the workflow ends here.",
    step: "Results · Final step",
  },
];

export default function DashboardWelcome() {
  return (
    <Card className="overflow-hidden border-primary/20 bg-gradient-to-br from-chart-1/10 via-card to-chart-2/10 shadow-sm">
      <CardHeader className="space-y-3">
        <CardTitle className="text-xl md:text-2xl">What to expect from MaintAI</CardTitle>
        <CardDescription className="max-w-3xl text-base leading-relaxed">
          You start with raw discrepancy text and end with a labeled export. The tool does not replace
          engineer judgment — it classifies in bulk, highlights uncertain rows, and lets you correct
          labels before export.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          {tips.map((tip, index) => {
            const Icon = tip.icon;
            return (
              <div
                key={tip.title}
                className="rounded-xl border border-border/60 bg-background/80 p-4 shadow-sm backdrop-blur-sm"
              >
                <div className="mb-3 flex items-center gap-2">
                  <div className="flex size-8 items-center justify-center rounded-lg bg-primary/10 text-primary">
                    <Icon className="size-4" />
                  </div>
                  <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    {tip.step}
                  </span>
                </div>
                <h3 className="font-medium">{tip.title}</h3>
                <p className="mt-1 text-sm leading-relaxed text-muted-foreground">{tip.description}</p>
                {index < tips.length - 1 && (
                  <p className="mt-3 text-xs text-muted-foreground">Then →</p>
                )}
              </div>
            );
          })}
        </div>
        <p className="mt-4 flex items-center gap-1 text-sm text-muted-foreground">
          <ArrowRight className="size-4 shrink-0" />
          Upload a file below to begin Step 1. The journey guide above shows where you are at each stage.
        </p>
      </CardContent>
    </Card>
  );
}
