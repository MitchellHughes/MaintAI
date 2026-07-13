import { useEffect, useState } from "react";

import { Alert, AlertDescription } from "@/components/ui/alert";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";

function ColumnSelect({ id, label, hint, value, columns, onChange, required }) {
  return (
    <div className="space-y-2">
      <Label htmlFor={id}>
        {label}
        {hint && <span className="ml-1 font-normal text-muted-foreground">— {hint}</span>}
      </Label>
      <Select value={value || "__none__"} onValueChange={(v) => onChange(v === "__none__" ? "" : v)}>
        <SelectTrigger id={id} className={cn(required && !value && "border-destructive")}>
          <SelectValue placeholder="Select column…" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="__none__">{required ? "Select column…" : "None"}</SelectItem>
          {columns.map((col) => (
            <SelectItem key={col} value={col}>
              {col}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}

export default function ModelSelector({
  columns,
  onChange,
  value,
  fetchModels,
  showColumns = true,
  showModels = true,
}) {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    async function load() {
      setLoading(true);
      setError("");
      try {
        const res = await fetchModels();
        const list = res.models || [];
        setModels(list);
        if (!value.models?.length && list.some((m) => m.name === "UMECClassifier")) {
          onChange({ ...value, models: ["UMECClassifier"] });
        }
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    }
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps -- default UMEC once on mount
  }, [fetchModels]);

  function handleModelChange(e) {
    const opts = Array.from(e.target.options)
      .filter((opt) => opt.selected)
      .map((opt) => opt.value);
    onChange({ ...value, models: opts });
  }

  const missingTextColumn = !value.text_column;

  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-2">
        {showColumns && (
          <>
        <ColumnSelect
          id="text-column-select"
          label="Discrepancy text column"
          hint="Required"
          value={value.text_column}
          columns={columns}
          required
          onChange={(text_column) => onChange({ ...value, text_column })}
        />
        <ColumnSelect
          id="part-column-select"
          label="Part / asset name column"
          hint="Optional"
          value={value.part_column}
          columns={columns}
          onChange={(part_column) => onChange({ ...value, part_column })}
        />
        <ColumnSelect
          id="label-column-select"
          label="Reference label column"
          hint="Optional — macro F1 only"
          value={value.label_column}
          columns={columns}
          onChange={(label_column) => onChange({ ...value, label_column })}
        />
          </>
        )}
        {showModels && (
        <div className="space-y-2 md:col-span-2">
          <Label htmlFor="models-select">
            Classification models
            <span className="ml-1 font-normal text-muted-foreground">
              — Hold Ctrl/Cmd to select multiple
            </span>
          </Label>
          <select
            id="models-select"
            className="flex min-h-[120px] w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            multiple
            value={value.models || []}
            onChange={handleModelChange}
          >
            {models.map((m) => (
              <option key={m.name} value={m.name} title={m.description}>
                {m.display_name}
              </option>
            ))}
          </select>
        </div>
        )}
      </div>

      {showColumns && missingTextColumn && (
        <p className="text-sm text-muted-foreground">
          Choose the column that contains narrative discrepancy text.
        </p>
      )}
      {showModels && loading && (
        <p className="text-sm text-muted-foreground">Loading available models…</p>
      )}
      {showModels && error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {showModels && models.length > 0 && (
        <ul className="space-y-2 text-sm text-muted-foreground">
          {models.map((m) => (
            <li key={m.name}>
              <span className="font-medium text-foreground">{m.display_name}</span>
              {" — "}
              {m.description?.split("\n")[1] || m.description?.split("\n")[0]}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
