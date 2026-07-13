import { Loader2 } from "lucide-react";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";

function chunkPercent(chunkProgress) {
  if (!chunkProgress?.chunks || !chunkProgress?.chunk) return null;
  return Math.round((chunkProgress.chunk / chunkProgress.chunks) * 100);
}

export default function LoadingOverlay({ open, title, message, detail, chunkProgress }) {
  if (!open) return null;

  const pct = chunkPercent(chunkProgress);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-background/80 p-4 backdrop-blur-sm"
      role="status"
      aria-live="polite"
      aria-busy="true"
    >
      <Card className="w-full max-w-md border-border shadow-lg">
        <CardHeader className="space-y-3">
          <div className="flex items-center gap-3">
            <Loader2 className="h-5 w-5 animate-spin text-primary" />
            <CardTitle>{title || "Working…"}</CardTitle>
          </div>
          {message && <CardDescription className="text-foreground/90">{message}</CardDescription>}
        </CardHeader>
        <CardContent className="space-y-4">
          {pct != null && (
            <div className="space-y-2">
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>
                  Chunk {chunkProgress.chunk} / {chunkProgress.chunks}
                </span>
                <span>{pct}%</span>
              </div>
              <Progress value={pct} />
            </div>
          )}
          {detail && <p className="text-xs text-muted-foreground">{detail}</p>}
        </CardContent>
      </Card>
    </div>
  );
}
