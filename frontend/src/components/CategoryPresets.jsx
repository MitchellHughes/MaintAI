import { useEffect, useState } from "react";

import { BUILTIN_CATEGORY_TEMPLATES } from "../constants/categoryTemplates";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const STORAGE_KEY = "maintai_category_presets";

function loadPresets() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function savePresets(presets) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(presets));
}

export default function CategoryPresets({ categories, onApply }) {
  const [presets, setPresets] = useState({});
  const [name, setName] = useState("");
  const [selected, setSelected] = useState("");

  useEffect(() => {
    setPresets(loadPresets());
  }, []);

  function handleSave() {
    const key = String(name || "").trim();
    if (!key) return;
    const rows = (categories || []).filter((row) => String(row.label || "").trim());
    if (!rows.length) return;
    const next = { ...presets, [key]: rows };
    savePresets(next);
    setPresets(next);
    setName("");
    setSelected(key);
  }

  function handleLoad() {
    const rows = presets[selected];
    if (rows?.length) onApply(rows.map((row) => ({ ...row })));
  }

  function handleLoadBuiltin(templateKey) {
    const rows = BUILTIN_CATEGORY_TEMPLATES[templateKey];
    if (rows?.length) onApply(rows.map((row) => ({ ...row })));
  }

  function handleDelete() {
    if (!selected) return;
    const next = { ...presets };
    delete next[selected];
    savePresets(next);
    setPresets(next);
    setSelected("");
  }

  const names = Object.keys(presets).sort((a, b) => a.localeCompare(b));

  return (
    <div className="space-y-4 rounded-lg border border-dashed border-border bg-muted/20 p-4">
      <p className="text-sm text-muted-foreground">
        Load a built-in taxonomy or save your own per site (browser only). Keywords are always
        mined from the current upload after you load labels.
      </p>
      <div className="flex flex-wrap gap-2">
        {Object.keys(BUILTIN_CATEGORY_TEMPLATES).map((key) => (
          <Button key={key} variant="secondary" type="button" size="sm" onClick={() => handleLoadBuiltin(key)}>
            {key}
          </Button>
        ))}
      </div>
      <div className="flex flex-col gap-2 sm:flex-row">
        <Input
          placeholder="Preset name (e.g. Hydraulic — Site A)"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        <Button variant="outline" type="button" onClick={handleSave} disabled={!name.trim()}>
          Save current categories
        </Button>
      </div>
      {names.length > 0 && (
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
          <Select value={selected || "__none__"} onValueChange={(v) => setSelected(v === "__none__" ? "" : v)}>
            <SelectTrigger className="sm:flex-1" aria-label="Load saved category preset">
              <SelectValue placeholder="Load preset…" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="__none__">Load preset…</SelectItem>
              {names.map((n) => (
                <SelectItem key={n} value={n}>
                  {n} ({presets[n].length} labels)
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button variant="outline" type="button" onClick={handleLoad} disabled={!selected}>
            Load
          </Button>
          <Button variant="outline" type="button" onClick={handleDelete} disabled={!selected}>
            Delete
          </Button>
        </div>
      )}
    </div>
  );
}
