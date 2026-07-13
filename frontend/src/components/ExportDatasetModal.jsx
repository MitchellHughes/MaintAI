import { useState } from "react";

import { downloadLabeledDataset } from "../utils/exportLabeledDataset";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export default function ExportDatasetModal({
  open,
  onClose,
  sourceRows,
  uploadId,
  predictions,
  defaultColumnName = "FAILURE_MECHANISM",
}) {
  const [columnName, setColumnName] = useState(defaultColumnName);
  const [format, setFormat] = useState("csv");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function handleExport() {
    setError("");
    setBusy(true);
    try {
      await downloadLabeledDataset({
        sourceRows,
        uploadId,
        predictions,
        columnName,
        format,
      });
      onClose?.();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Dialog open={open} onOpenChange={(v) => !v && !busy && onClose?.()}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Export labeled dataset</DialogTitle>
          <DialogDescription>
            <strong>Dataset</strong> sheet: original columns plus one new label column in{" "}
            <strong>ALL CAPS</strong>. <strong>Findings</strong> sheet (Excel) or{" "}
            <code className="text-foreground">_findings.csv</code>: confidence, review tier,
            explanation, keywords, and model agreement.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-2">
          <div className="space-y-2">
            <Label htmlFor="export-col">Label column name (values in ALL CAPS)</Label>
            <Input
              id="export-col"
              value={columnName}
              onChange={(e) => setColumnName(e.target.value)}
              placeholder="FAILURE_MECHANISM"
            />
          </div>

          <fieldset className="space-y-2">
            <legend className="text-sm font-medium">File format</legend>
            <div className="flex gap-4 text-sm">
              <label className="flex cursor-pointer items-center gap-2">
                <input
                  type="radio"
                  name="export-format"
                  checked={format === "csv"}
                  onChange={() => setFormat("csv")}
                />
                CSV
              </label>
              <label className="flex cursor-pointer items-center gap-2">
                <input
                  type="radio"
                  name="export-format"
                  checked={format === "xlsx"}
                  onChange={() => setFormat("xlsx")}
                />
                Excel (.xlsx)
              </label>
            </div>
          </fieldset>

          {error && (
            <Alert variant="destructive">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" type="button" onClick={onClose} disabled={busy}>
            Cancel
          </Button>
          <Button type="button" onClick={handleExport} disabled={busy}>
            {busy ? "Preparing…" : "Download"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
