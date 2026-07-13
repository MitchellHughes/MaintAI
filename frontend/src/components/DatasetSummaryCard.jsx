import { Columns3, FileSpreadsheet, Rows3 } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export default function DatasetSummaryCard({
  sourceFilename,
  rowCount,
  previewOnly,
  previewRows,
  columns = [],
  textColumn,
}) {
  return (
    <Card className="shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Current dataset</CardTitle>
        <CardDescription>Summary of the active upload session</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {sourceFilename && (
          <div className="flex items-start gap-2">
            <FileSpreadsheet className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
            <div className="min-w-0">
              <p className="truncate font-mono text-sm">{sourceFilename}</p>
              {previewOnly && (
                <p className="text-xs text-muted-foreground">
                  Showing {previewRows?.toLocaleString()} of {rowCount?.toLocaleString()} rows
                </p>
              )}
            </div>
          </div>
        )}

        <div className="grid grid-cols-2 gap-3">
          <div className="rounded-lg border border-border/60 bg-muted/20 p-3">
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <Rows3 className="size-3.5" />
              Rows
            </div>
            <p className="mt-1 text-lg font-semibold tabular-nums">{rowCount?.toLocaleString() || "—"}</p>
          </div>
          <div className="rounded-lg border border-border/60 bg-muted/20 p-3">
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <Columns3 className="size-3.5" />
              Columns
            </div>
            <p className="mt-1 text-lg font-semibold tabular-nums">{columns.length || "—"}</p>
          </div>
        </div>

        {columns.length > 0 && (
          <div className="space-y-2">
            <p className="text-xs font-medium text-muted-foreground">Detected columns</p>
            <div className="flex max-h-28 flex-wrap gap-1.5 overflow-y-auto">
              {columns.map((col) => (
                <Badge
                  key={col}
                  variant={col === textColumn ? "default" : "secondary"}
                  className="max-w-full truncate font-normal"
                  title={col}
                >
                  {col}
                </Badge>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
