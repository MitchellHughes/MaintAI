import { useEffect, useMemo, useState } from "react";

import { fetchAnalysisSettings } from "../services/api";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

function IntRangeInput({ value, onChange, min, max }) {
  const pair = Array.isArray(value) ? value : [min ?? 1, max ?? 2];
  return (
    <div className="flex items-center gap-2">
      <Input
        type="number"
        min={min}
        max={max}
        className="w-20"
        value={pair[0] ?? min ?? 1}
        onChange={(e) => onChange([Number(e.target.value), pair[1]])}
      />
      <span className="text-sm text-muted-foreground">to</span>
      <Input
        type="number"
        min={min}
        max={max}
        className="w-20"
        value={pair[1] ?? max ?? 2}
        onChange={(e) => onChange([pair[0], Number(e.target.value)])}
      />
    </div>
  );
}

function isSettingVisible(entry, userSettings, catalog) {
  const rule = entry.visible_when;
  if (!rule?.setting) return true;
  const current =
    userSettings?.[rule.setting] ?? catalog?.defaults?.[rule.setting];
  if (Array.isArray(rule.equals)) {
    return rule.equals.includes(current);
  }
  return current === rule.equals;
}

function ChoiceInput({ entry, value, onChange }) {
  const selected = value ?? entry.options?.[0]?.value;
  return (
    <div className="flex flex-wrap gap-3" role="radiogroup" aria-label={entry.label}>
      {(entry.options || []).map((opt) => (
        <label key={opt.value} className="flex cursor-pointer items-center gap-2 text-sm">
          <input
            type="radio"
            name={entry.id}
            value={opt.value}
            checked={selected === opt.value}
            onChange={() => onChange(opt.value)}
            className="border-input"
          />
          <span>{opt.label}</span>
        </label>
      ))}
    </div>
  );
}

function SettingField({ entry, value, onChange }) {
  const { type, label, description, min, max, step } = entry;

  if (type === "choice") {
    return (
      <div className="space-y-2">
        <Label className="flex flex-col gap-1">
          <span className="font-medium">{label}</span>
          {description && <span className="text-xs font-normal text-muted-foreground">{description}</span>}
        </Label>
        <ChoiceInput entry={entry} value={value} onChange={onChange} />
      </div>
    );
  }

  if (type === "boolean") {
    return (
      <label className="flex cursor-pointer items-start gap-3">
        <input
          type="checkbox"
          className="mt-1 rounded border-input"
          checked={Boolean(value)}
          onChange={(e) => onChange(e.target.checked)}
        />
        <span className="space-y-1 text-sm">
          <span className="font-medium">{label}</span>
          {description && <span className="block text-xs text-muted-foreground">{description}</span>}
        </span>
      </label>
    );
  }

  if (type === "int_range") {
    return (
      <div className="space-y-2">
        <Label className="flex flex-col gap-1">
          {label}
          {description && <span className="text-xs font-normal text-muted-foreground">{description}</span>}
        </Label>
        <IntRangeInput value={value} onChange={onChange} min={min} max={max} />
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <Label className="flex flex-col gap-1">
        {label}
        {description && <span className="text-xs font-normal text-muted-foreground">{description}</span>}
      </Label>
      <Input
        type="number"
        min={min}
        max={max}
        step={step ?? (Number.isInteger(value) ? 1 : 0.01)}
        value={value ?? ""}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}

export default function AnalysisSettings({
  userSettings,
  onChange,
  selectedModels = [],
}) {
  const [catalog, setCatalog] = useState(null);
  const [loadError, setLoadError] = useState("");

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const data = await fetchAnalysisSettings();
        if (cancelled) return;
        setCatalog(data);
        if (!userSettings || Object.keys(userSettings).length === 0) {
          onChange(data.defaults || {});
        }
      } catch (e) {
        if (!cancelled) setLoadError(e.message);
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- seed defaults once on load
  }, []);

  const visible = useMemo(() => {
    if (!catalog?.settings) return [];
    const selected = new Set(selectedModels || []);
    return catalog.settings.filter((entry) => {
      if (!isSettingVisible(entry, userSettings, catalog)) return false;
      const applies = entry.applies_to;
      if (!applies?.length || !selected.size) return true;
      return applies.some((model) => selected.has(model));
    });
  }, [catalog, selectedModels, userSettings]);

  const byGroup = useMemo(() => {
    const groups = new Map();
    for (const entry of visible) {
      const name = entry.group || "General";
      if (!groups.has(name)) groups.set(name, []);
      groups.get(name).push(entry);
    }
    return [...groups.entries()];
  }, [visible]);

  function setValue(id, value) {
    onChange({ ...userSettings, [id]: value });
  }

  if (loadError) {
    return <p className="text-sm text-destructive">Could not load settings: {loadError}</p>;
  }

  if (!catalog) {
    return <p className="text-sm text-muted-foreground">Loading analysis settings…</p>;
  }

  if (!visible.length) {
    return (
      <p className="text-sm text-muted-foreground">
        Select at least one model above to configure analysis options.
      </p>
    );
  }

  return (
    <div className="space-y-6">
      {byGroup.map(([groupName, entries]) => (
        <fieldset key={groupName} className="space-y-3 rounded-lg border border-border p-4">
          <legend className="px-1 text-sm font-semibold">{groupName}</legend>
          {catalog.groups?.[groupName]?.summary && (
            <p className="text-xs text-muted-foreground">{catalog.groups[groupName].summary}</p>
          )}
          <div
            className={
              groupName === "Prediction output"
                ? "grid gap-4 md:grid-cols-2"
                : "grid gap-4 sm:grid-cols-2"
            }
          >
            {entries.map((entry) => (
              <SettingField
                key={entry.id}
                entry={entry}
                value={userSettings?.[entry.id] ?? catalog.defaults?.[entry.id]}
                onChange={(value) => setValue(entry.id, value)}
              />
            ))}
          </div>
        </fieldset>
      ))}
    </div>
  );
}
