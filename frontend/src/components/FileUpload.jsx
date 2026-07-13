import { useCallback, useState } from "react";
import { Upload } from "lucide-react";

import { cn } from "@/lib/utils";

const ALLOWED_EXTENSIONS = [".csv", ".xlsx", ".xlsm", ".xls"];

function validateFile(file) {
  if (!file) return "No file selected.";
  const name = file.name.toLowerCase();
  if (!ALLOWED_EXTENSIONS.some((ext) => name.endsWith(ext))) {
    return "File must be a .csv, .xlsx, or .xls file.";
  }
  if (file.size === 0) return "File is empty.";
  return null;
}

export default function FileUpload({ onLoaded }) {
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");

  const handleFile = useCallback(
    (file) => {
      setError("");
      const validationError = validateFile(file);
      if (validationError) {
        setError(validationError);
        return;
      }
      onLoaded?.(file);
    },
    [onLoaded],
  );

  function onDrop(e) {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  }

  return (
    <div className="space-y-2">
      <label
        className={cn(
          "flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-border bg-muted/30 px-6 py-12 text-center transition-colors",
          dragging && "border-primary bg-accent/50",
        )}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <Upload className="mb-3 h-8 w-8 text-muted-foreground" />
        <span className="text-sm font-medium">Drag & drop CSV or Excel here</span>
        <span className="mt-1 text-xs text-muted-foreground">or click to browse</span>
        <input
          type="file"
          className="sr-only"
          accept=".csv,.xlsx,.xlsm,.xls"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) handleFile(file);
          }}
        />
      </label>
      {error && <p className="text-sm text-destructive">{error}</p>}
    </div>
  );
}
