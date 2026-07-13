import { useMemo, useState } from "react";
import { RotateCcw } from "lucide-react";

import { isAutoAcceptRow } from "../utils/engineerReview";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export default function ReviewBulkActions({
  predictions,
  onApply,
  onPushUndo,
  autoAcceptThreshold,
  canUndo,
  onUndo,
}) {
  const [targetLabel, setTargetLabel] = useState("__all__");

  const labelOptions = useMemo(() => {
    const labels = new Set();
    (predictions || []).forEach((row) => {
      if (row.predicted_condition) labels.add(row.predicted_condition);
    });
    return Array.from(labels).sort((a, b) => a.localeCompare(b));
  }, [predictions]);

  function runBulk(updater, description) {
    onPushUndo?.();
    onApply(updater, description);
  }

  function acceptTrustedForLabel(labelFilter) {
    runBulk(
      (rows) =>
        rows.map((row) => {
          if (!isAutoAcceptRow(row, autoAcceptThreshold)) return row;
          if (labelFilter && row.predicted_condition !== labelFilter) return row;
          return { ...row, final_condition: row.predicted_condition };
        }),
      labelFilter ? `Accepted trusted rows for ${labelFilter}` : "Accepted all trusted rows",
    );
  }

  function revertAllEdits() {
    runBulk(
      (rows) =>
        rows.map((row) => ({
          ...row,
          final_condition: row.predicted_condition,
        })),
      "Reverted all manual edits",
    );
  }

  function revertLabel(label) {
    if (!label || label === "__all__") return;
    runBulk(
      (rows) =>
        rows.map((row) =>
          row.final_condition !== row.predicted_condition &&
          (row.final_condition === label || row.predicted_condition === label)
            ? { ...row, final_condition: row.predicted_condition }
            : row,
        ),
      `Reverted edits for ${label}`,
    );
  }

  return (
    <div className="rounded-lg border border-border bg-muted/20 p-4 space-y-4">
      <div>
        <h3 className="text-sm font-semibold">Bulk review actions</h3>
        <p className="text-xs text-muted-foreground">
          Confirm trusted predictions or revert edits. Undo restores the state before your last bulk
          action.
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        <Button type="button" size="sm" variant="secondary" onClick={() => acceptTrustedForLabel()}>
          Accept all trusted rows
        </Button>
        <Button type="button" size="sm" variant="outline" onClick={revertAllEdits}>
          Revert all edits
        </Button>
        <Button type="button" size="sm" variant="outline" onClick={onUndo} disabled={!canUndo}>
          <RotateCcw className="mr-1 h-3.5 w-3.5" />
          Undo
        </Button>
      </div>

      <div className="flex flex-col gap-2 sm:flex-row sm:items-end">
        <div className="space-y-2 sm:min-w-[200px]">
          <Label className="text-xs">Trusted rows for predicted label</Label>
          <Select value={targetLabel} onValueChange={setTargetLabel}>
            <SelectTrigger>
              <SelectValue placeholder="Choose label…" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="__all__">All labels</SelectItem>
              {labelOptions.map((label) => (
                <SelectItem key={label} value={label}>
                  {label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <Button
          type="button"
          size="sm"
          variant="secondary"
          disabled={targetLabel === "__all__"}
          onClick={() => acceptTrustedForLabel(targetLabel)}
        >
          Accept trusted for label
        </Button>
        <Button
          type="button"
          size="sm"
          variant="outline"
          disabled={targetLabel === "__all__"}
          onClick={() => revertLabel(targetLabel)}
        >
          Revert edits for label
        </Button>
      </div>
    </div>
  );
}
