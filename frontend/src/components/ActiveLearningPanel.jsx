import { useMemo, useState } from "react";
import { Sparkles } from "lucide-react";

import { fetchActiveLearning } from "../services/api";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

function formatLabel(label) {
  const text = String(label || "").trim();
  if (!text) return label;
  return text.charAt(0).toUpperCase() + text.slice(1);
}

export default function ActiveLearningPanel({
  predictions,
  customCategories,
  onApplyCategories,
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [suggestions, setSuggestions] = useState(null);
  const [editCount, setEditCount] = useState(0);
  const [applied, setApplied] = useState(false);

  const editedRows = useMemo(
    () =>
      (predictions || []).filter(
        (row) =>
          row.final_condition &&
          row.predicted_condition &&
          String(row.final_condition).trim() !== String(row.predicted_condition).trim(),
      ),
    [predictions],
  );

  async function analyze(apply = false) {
    setError("");
    setBusy(true);
    setApplied(false);
    try {
      const res = await fetchActiveLearning({
        edits: editedRows,
        custom_categories: customCategories,
        apply,
      });
      setEditCount(res.edit_count ?? editedRows.length);
      setSuggestions(res.suggestions || {});
      if (apply && res.custom_categories) {
        onApplyCategories?.(res.custom_categories);
        setApplied(true);
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  if (!editedRows.length) return null;

  const suggestionEntries = Object.entries(suggestions || {}).filter(([, tokens]) => tokens?.length);

  return (
    <Card className="border-dashed border-primary/30">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-lg">
          <Sparkles className="h-4 w-4 text-primary" />
          Active learning from your corrections
        </CardTitle>
        <CardDescription>
          {editedRows.length.toLocaleString()} row(s) edited away from the model prediction. Mine
          keywords from those narratives and merge into your categories for the next run.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {error && (
          <Alert variant="destructive">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
        {applied && (
          <Alert variant="success">
            <AlertDescription>
              Keywords merged into your categories. Re-run classification on the workspace to use
              them.
            </AlertDescription>
          </Alert>
        )}

        <div className="flex flex-wrap gap-2">
          <Button type="button" size="sm" variant="outline" disabled={busy} onClick={() => analyze(false)}>
            {busy ? "Analyzing…" : "Suggest keywords"}
          </Button>
          <Button
            type="button"
            size="sm"
            disabled={busy || !suggestionEntries.length}
            onClick={() => analyze(true)}
          >
            Apply keywords to categories
          </Button>
        </div>

        {suggestionEntries.length > 0 && (
          <ul className="space-y-2 text-sm">
            {suggestionEntries.map(([label, tokens]) => (
              <li key={label} className="rounded-md border border-border bg-muted/20 px-3 py-2">
                <span className="font-medium capitalize">{formatLabel(label)}</span>
                <span className="text-muted-foreground"> — add: </span>
                <code className="text-xs">{tokens.join(", ")}</code>
              </li>
            ))}
          </ul>
        )}

        {!suggestionEntries.length && suggestions && editCount > 0 && (
          <p className="text-sm text-muted-foreground">
            No new keyword tokens found — existing keywords may already cover those corrections.
          </p>
        )}
      </CardContent>
    </Card>
  );
}
