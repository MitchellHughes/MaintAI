import { useState } from "react";

import { generateTokens } from "../services/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

const EMPTY_ROW = { label: "", keywords: "" };

export default function TokenGenerator({
  value,
  onChange,
  corpus,
  uploadRows,
  textColumn,
  labelColumn,
  partColumn,
  embedded = false,
}) {
  const rows = value?.length ? value : [{ ...EMPTY_ROW }];
  const [busyLabel, setBusyLabel] = useState("");

  function commit(nextRows) {
    onChange(nextRows);
  }

  function updateRow(index, field, val) {
    const nextRows = rows.map((row, rowIndex) =>
      rowIndex === index ? { ...row, [field]: val } : row,
    );
    commit(nextRows);
  }

  async function autoFillKeywords(index) {
    const row = rows[index];
    const label = String(row.label || "").trim();
    if (!label) return;
    if (!textColumn) return;
    setBusyLabel(label);
    try {
      const payload = await generateTokens([label], corpus || [], {
        rows: uploadRows,
        textColumn,
        labelColumn,
        partColumn,
        customCategories: rows.map((r) => ({
          label: r.label,
          keywords: r.keywords,
        })),
      });
      const generated = payload.keywords ?? payload;
      const keywords = generated[label] || generated[label.toLowerCase()] || [];
      updateRow(index, "keywords", keywords.join(", "));
    } catch {
      /* ignore */
    } finally {
      setBusyLabel("");
    }
  }

  function addRow() {
    commit([...rows, { ...EMPTY_ROW }]);
  }

  function removeRow(index) {
    const nextRows = rows.filter((_, rowIndex) => rowIndex !== index);
    commit(nextRows.length ? nextRows : [{ ...EMPTY_ROW }]);
  }

  const inner = (
    <div className="space-y-3">
      {!embedded && (
        <div className="space-y-1">
          <h2 className="text-lg font-semibold">Failure-mechanism categories</h2>
          <p className="text-sm text-muted-foreground">
            Auto-fill uses statistical term mining on your upload only (e.g. crack, cracks for
            cracked). Requires discrepancy column mapped in step 2; reference and part columns improve
            results.
          </p>
          {!textColumn && (
            <p className="text-sm text-amber-400">Map the discrepancy text column in step 2 first.</p>
          )}
        </div>
      )}
      {rows.map((row, index) => (
        <div
          key={index}
          className="flex flex-col gap-2 rounded-lg border border-border bg-card p-3 sm:grid sm:grid-cols-[minmax(0,1fr)_minmax(0,1.5fr)_auto_auto]"
        >
          <Input
            placeholder="Label (e.g. leaking)"
            value={row.label}
            onChange={(e) => updateRow(index, "label", e.target.value)}
            onBlur={() => {
              if (!String(row.keywords || "").trim()) {
                autoFillKeywords(index);
              }
            }}
          />
          <Input
            className="sm:col-span-1"
            placeholder="Keywords (comma-separated)"
            value={row.keywords}
            onChange={(e) => updateRow(index, "keywords", e.target.value)}
          />
          <Button
            variant="outline"
            type="button"
            size="sm"
            onClick={() => autoFillKeywords(index)}
            disabled={!row.label || !textColumn || busyLabel === row.label}
          >
            {busyLabel === row.label ? "Generating…" : "Auto-fill"}
          </Button>
          <Button variant="ghost" type="button" size="sm" onClick={() => removeRow(index)}>
            Remove
          </Button>
        </div>
      ))}
      <Button variant="outline" type="button" onClick={addRow}>
        Add category
      </Button>
    </div>
  );

  if (embedded) return inner;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Categories</CardTitle>
        <CardDescription>Labels and keywords for classification.</CardDescription>
      </CardHeader>
      <CardContent>{inner}</CardContent>
    </Card>
  );
}
