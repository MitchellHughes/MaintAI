import { useMemo } from "react";
import { DataGrid, renderTextEditor } from "react-data-grid";
import "react-data-grid/lib/styles.css";

import { useTheme } from "../context/ThemeContext";

// Generic editable grid. `columns` is a string[] of field names; every
// column is rendered as an inline-editable text cell.
function EditableGrid({ columns, rows, onRowsChange, readOnly = false }) {
  const { isDark } = useTheme();
  const gridColumns = useMemo(
    () =>
      (columns || []).map((c) => ({
        key: c,
        name: c,
        editable: !readOnly,
        resizable: true,
        renderEditCell: readOnly ? undefined : renderTextEditor,
      })),
    [columns, readOnly],
  );

  if (!rows?.length) return null;

  return (
    <div className="min-w-0 max-w-full overflow-x-auto rounded-lg border border-border">
      <DataGrid
        columns={gridColumns}
        rows={rows}
        onRowsChange={onRowsChange}
        rowKeyGetter={(r) => r.id}
        className={isDark ? "rdg-dark" : "rdg-light"}
        style={{ blockSize: Math.min(60 + rows.length * 35, 480) }}
      />
    </div>
  );
}

export default EditableGrid;
