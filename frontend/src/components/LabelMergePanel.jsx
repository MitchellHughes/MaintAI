import { useEffect, useMemo, useState } from "react";
import { GitMerge, Plus, Sparkles, Trash2 } from "lucide-react";

import { applyLabelGroups, suggestLabelGroups } from "../services/api";
import {
  EMPTY_LABEL_GROUPS,
  normalizeLabelGroups,
  serializeLabelGroups,
} from "../utils/labelGroups";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";

function pct(value) {
  if (value == null || Number.isNaN(value)) return "—";
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function formatLabel(label) {
  const text = String(label || "").trim();
  if (!text) return "";
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function GroupNameInput({ value, onCommit }) {
  const [local, setLocal] = useState(value);

  useEffect(() => {
    setLocal(value);
  }, [value]);

  return (
    <Input
      value={local}
      onChange={(e) => setLocal(e.target.value)}
      onBlur={() => onCommit(String(local).trim().toLowerCase())}
      onKeyDown={(e) => {
        if (e.key === "Enter") e.currentTarget.blur();
      }}
      placeholder="Group name"
      className="max-w-xs"
    />
  );
}

export default function LabelMergePanel({
  predictions,
  customCategories,
  savedLabelGroups = EMPTY_LABEL_GROUPS,
  reportLabelGroups = EMPTY_LABEL_GROUPS,
  onSave,
  onApplyPreview,
}) {
  const savedKey = serializeLabelGroups(savedLabelGroups);
  const reportKey = serializeLabelGroups(reportLabelGroups);

  const [draftGroups, setDraftGroups] = useState(() => normalizeLabelGroups(savedLabelGroups));
  const [availableLabels, setAvailableLabels] = useState([]);
  const [baselineF1, setBaselineF1] = useState(null);
  const [previewF1, setPreviewF1] = useState(null);
  const [suggestedF1, setSuggestedF1] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [saved, setSaved] = useState(false);
  const [mergeCategories, setMergeCategories] = useState(false);
  const [suggestedGroups, setSuggestedGroups] = useState([]);

  const draftGroupsKey = useMemo(() => serializeLabelGroups(draftGroups), [draftGroups]);
  const hasPendingReport = draftGroups.length > 0 && draftGroupsKey !== reportKey;
  const hasUnsavedConfig = draftGroupsKey !== savedKey;

  const assignedMembers = useMemo(() => {
    const set = new Set();
    draftGroups.forEach((group) => group.members.forEach((m) => set.add(m)));
    return set;
  }, [draftGroups]);

  const ungroupedLabels = useMemo(
    () =>
      availableLabels.filter(
        (label) => !assignedMembers.has(label) && !draftGroups.some((g) => g.group_label === label),
      ),
    [availableLabels, assignedMembers, draftGroups],
  );

  useEffect(() => {
    setDraftGroups((prev) => {
      const next = normalizeLabelGroups(savedLabelGroups);
      if (serializeLabelGroups(prev) === serializeLabelGroups(next)) return prev;
      return next;
    });
  }, [savedKey]);

  useEffect(() => {
    const labels = (customCategories || [])
      .map((cat) => String(cat.label || "").trim().toLowerCase())
      .filter(Boolean);
    if (labels.length) setAvailableLabels(labels);
  }, [customCategories]);

  async function loadSuggestions() {
    setError("");
    setBusy(true);
    setSaved(false);
    try {
      const res = await suggestLabelGroups({
        predictions,
        custom_categories: customCategories,
        label_groups: draftGroups,
      });
      setBaselineF1(res.baseline_macro_f1);
      setSuggestedF1(res.suggested_macro_f1);
      setAvailableLabels(res.available_labels || []);
      setSuggestedGroups(normalizeLabelGroups(res.suggested_groups || []));
      if (res.suggested_macro_f1 != null) {
        setPreviewF1(res.suggested_macro_f1);
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  function applyPreviewToReport() {
    const normalized = normalizeLabelGroups(draftGroups);
    setError("");
    onApplyPreview?.(normalized);
    setPreviewF1(suggestedF1 ?? previewF1);
  }

  function addGroup() {
    const seed = ungroupedLabels.slice(0, 2);
    if (!seed.length) return;
    setDraftGroups((prev) => [
      ...prev,
      {
        group_label: seed.length > 1 ? `${seed[0]} / ${seed[1]}` : seed[0],
        members: seed,
        reason: "Manual group",
      },
    ]);
    setSaved(false);
  }

  function updateGroup(index, patch) {
    setDraftGroups((prev) =>
      prev.map((group, i) => (i === index ? { ...group, ...patch } : group)),
    );
    setSaved(false);
  }

  function removeGroup(index) {
    setDraftGroups((prev) => prev.filter((_, i) => i !== index));
    setSaved(false);
  }

  function toggleMember(groupIndex, label) {
    setDraftGroups((prev) =>
      prev.map((group, i) => {
        if (i === groupIndex) {
          const has = group.members.includes(label);
          return {
            ...group,
            members: has
              ? group.members.filter((m) => m !== label)
              : [...group.members, label],
          };
        }
        if (group.members.includes(label)) {
          return { ...group, members: group.members.filter((m) => m !== label) };
        }
        return group;
      }),
    );
    setSaved(false);
  }

  function applySuggestions() {
    if (!suggestedGroups.length) return;
    setDraftGroups(suggestedGroups);
    setSaved(false);
  }

  function clearAllGroups() {
    setDraftGroups([]);
    setSuggestedGroups([]);
    setSaved(false);
  }

  async function handleSave() {
    setError("");
    const normalizedDraft = normalizeLabelGroups(draftGroups);

    setBusy(true);
    setSaved(false);
    try {
      const applied = await applyLabelGroups({
        custom_categories: customCategories,
        label_groups: normalizedDraft,
        merge_categories: mergeCategories,
      });
      const savedGroups = normalizeLabelGroups(applied.label_groups || normalizedDraft);
      setDraftGroups(savedGroups);
      onSave?.({
        label_groups: savedGroups,
        custom_categories: applied.custom_categories,
        merge_categories: mergeCategories,
      });
      setSaved(true);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  const targetMet = previewF1 != null && previewF1 >= 0.8;

  return (
    <Card className="min-w-0 border-primary/20 shadow-sm">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-lg">
          <GitMerge className="size-4 text-primary" />
          Label groups for evaluation
        </CardTitle>
        <CardDescription>
          Groups change <strong>evaluation scoring only</strong> (not predictions) unless you check
          merge categories and re-run classification. Edit groups, then save or apply to the report.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {error && (
          <Alert variant="destructive">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
        {saved && !hasUnsavedConfig && !hasPendingReport && (
          <Alert variant="success">
            <AlertDescription>
              Label groups saved
              {mergeCategories ? " and categories merged for the next run" : ""}. Click{" "}
              <strong>Run evaluation report</strong> below if scores are not updated yet.
            </AlertDescription>
          </Alert>
        )}
        {hasPendingReport && (
          <Alert>
            <AlertDescription>
              Draft groups differ from the report. Click <strong>Apply to report</strong> or{" "}
              <strong>Save &amp; apply</strong> to refresh macro F1.
            </AlertDescription>
          </Alert>
        )}

        <div className="flex flex-wrap items-center gap-3">
          <Button type="button" size="sm" variant="outline" disabled={busy} onClick={loadSuggestions}>
            <Sparkles className="mr-1.5 size-4" />
            {busy ? "Analyzing…" : "Suggest merges"}
          </Button>
          <Button type="button" size="sm" variant="outline" disabled={!ungroupedLabels.length} onClick={addGroup}>
            <Plus className="mr-1.5 size-4" />
            Add group
          </Button>
          <Button
            type="button"
            size="sm"
            variant="outline"
            disabled={!suggestedGroups.length}
            onClick={applySuggestions}
          >
            Apply suggestions
          </Button>
          <Button type="button" size="sm" variant="ghost" onClick={clearAllGroups}>
            Clear all
          </Button>
          <Button
            type="button"
            size="sm"
            variant="outline"
            disabled={busy || !hasPendingReport}
            onClick={applyPreviewToReport}
          >
            Apply to report
          </Button>
          <div className="flex flex-wrap gap-2 text-sm">
            {baselineF1 != null && (
              <Badge variant="outline">Baseline macro F1 {pct(baselineF1)}</Badge>
            )}
            {previewF1 != null && draftGroups.length > 0 && (
              <Badge variant={targetMet ? "default" : "secondary"}>
                Suggested macro F1 {pct(previewF1)}
              </Badge>
            )}
          </div>
        </div>

        {!draftGroups.length && (
          <p className="text-sm text-muted-foreground">
            Click <strong>Suggest merges</strong> to cluster weak classes, then save.
          </p>
        )}

        <div className="space-y-3">
          {draftGroups.map((group, index) => (
            <div
              key={`merge-group-${index}`}
              className="rounded-lg border border-border bg-muted/15 p-3 space-y-3"
            >
              <div className="flex flex-wrap items-center gap-2">
                <GroupNameInput
                  value={group.group_label}
                  onCommit={(group_label) => updateGroup(index, { group_label })}
                />
                <Button
                  type="button"
                  size="icon"
                  variant="ghost"
                  className="size-8"
                  onClick={() => removeGroup(index)}
                  aria-label="Remove group"
                >
                  <Trash2 className="size-4" />
                </Button>
              </div>
              {group.reason && (
                <p className="text-xs text-muted-foreground">{group.reason}</p>
              )}
              <div className="flex flex-wrap gap-2">
                {availableLabels.map((label) => {
                  const inThisGroup = group.members.includes(label);
                  const inOtherGroup = !inThisGroup && assignedMembers.has(label);
                  return (
                    <label
                      key={`${index}-${label}`}
                      className={`inline-flex items-center gap-1.5 rounded-md border px-2 py-1 text-xs ${
                        inThisGroup
                          ? "border-primary bg-primary/10"
                          : inOtherGroup
                            ? "border-border opacity-40"
                            : "border-border bg-background"
                      }`}
                    >
                      <input
                        type="checkbox"
                        className="size-3.5 accent-primary"
                        checked={inThisGroup}
                        disabled={inOtherGroup}
                        onChange={() => toggleMember(index, label)}
                      />
                      <span className="capitalize">{formatLabel(label)}</span>
                    </label>
                  );
                })}
              </div>
            </div>
          ))}
        </div>

        {ungroupedLabels.length > 0 && draftGroups.length > 0 && (
          <p className="text-xs text-muted-foreground">
            Ungrouped: {ungroupedLabels.map(formatLabel).join(", ")}
          </p>
        )}

        <div className="flex flex-col gap-3 border-t border-border pt-4 sm:flex-row sm:items-center sm:justify-between">
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              className="size-4 accent-primary"
              checked={mergeCategories}
              onChange={(e) => setMergeCategories(e.target.checked)}
            />
            Also merge categories for the next classification run
          </label>
          <Button type="button" disabled={busy || !draftGroups.length} onClick={handleSave}>
            Save &amp; apply to report
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
