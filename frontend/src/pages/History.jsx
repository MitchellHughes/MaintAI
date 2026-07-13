import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

import { useApp } from "../context/AppContext";
import { getHistory, getHistoryItem } from "../services/api";
import PageHeader from "../components/PageHeader";
import PageShell from "../components/PageShell";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

function History() {
  const navigate = useNavigate();
  const { restoreRunById, setLastRunId } = useApp();
  const [items, setItems] = useState([]);
  const [selected, setSelected] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [openingId, setOpeningId] = useState("");

  useEffect(() => {
    getHistory()
      .then((res) => setItems(res.items || []))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  async function openItem(id) {
    setError("");
    try {
      setSelected(await getHistoryItem(id));
    } catch (e) {
      setError(e.message);
    }
  }

  async function openInResults(id) {
    setError("");
    setOpeningId(id);
    try {
      await restoreRunById(id);
      setLastRunId(id);
      navigate("/results");
    } catch (e) {
      setError(e.message);
    } finally {
      setOpeningId("");
    }
  }

  return (
    <PageShell>
      <PageHeader
        title="Run history"
        description="Saved classification runs. Reopen a snapshot to continue review or compare results."
      />

      <Card className="min-w-0 shadow-sm">
        <CardContent className="pt-6">
          {loading && <p className="text-sm text-muted-foreground">Loading…</p>}
          {error && (
            <Alert variant="destructive" className="mb-4">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
          {!loading && !items.length && (
            <p className="text-sm text-muted-foreground">No archived records yet.</p>
          )}
          {items.length > 0 && (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Timestamp</TableHead>
                  <TableHead>ID</TableHead>
                  <TableHead>Model</TableHead>
                  <TableHead>User</TableHead>
                  <TableHead>Rows</TableHead>
                  <TableHead>File</TableHead>
                  <TableHead />
                </TableRow>
              </TableHeader>
              <TableBody>
                {items.map((it) => (
                  <TableRow key={it.id}>
                    <TableCell>{new Date(it.timestamp).toLocaleString()}</TableCell>
                    <TableCell>
                      <code className="text-xs">{it.id.slice(0, 8)}</code>
                    </TableCell>
                    <TableCell>{it.model_version}</TableCell>
                    <TableCell>{it.user}</TableCell>
                    <TableCell>{it.num_records}</TableCell>
                    <TableCell className="max-w-[12rem] truncate font-mono text-xs">
                      {it.source_filename || "—"}
                    </TableCell>
                    <TableCell className="space-x-2 text-right">
                      <Button
                        variant="link"
                        className="h-auto p-0"
                        disabled={openingId === it.id}
                        onClick={() => openInResults(it.id)}
                      >
                        {openingId === it.id ? "Opening…" : "Open in Results"}
                      </Button>
                      <Button variant="link" className="h-auto p-0" onClick={() => openItem(it.id)}>
                        Details
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      {selected && (
        <Card className="shadow-sm">
          <CardHeader>
            <CardTitle>Record {selected.id.slice(0, 8)}</CardTitle>
            <CardDescription>
              {new Date(selected.timestamp).toLocaleString()} · {selected.model_version} ·{" "}
              {selected.num_records} rows
              {selected.type ? ` · ${selected.type}` : ""}
              {(selected.snapshot?.sourceFilename ||
                selected.snapshot?.prediction?.source_filename) && (
                <>
                  {" "}
                  ·{" "}
                  <span className="font-mono">
                    {selected.snapshot.sourceFilename ||
                      selected.snapshot.prediction?.source_filename}
                  </span>
                </>
              )}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {selected.snapshot ? (
              <>
                <p className="text-sm text-muted-foreground">
                  Full run snapshot — reopen in Results to restore predictions and workspace settings.
                </p>
                <Button
                  disabled={openingId === selected.id}
                  onClick={() => openInResults(selected.id)}
                >
                  Open in Results
                </Button>
              </>
            ) : (
              <div className="grid gap-4 md:grid-cols-2">
                <div>
                  <h3 className="mb-2 text-sm font-medium">Before (pre-edit)</h3>
                  <pre className="max-h-96 overflow-auto rounded-md border border-border bg-muted/30 p-3 text-xs">
                    {JSON.stringify(selected.before, null, 2)}
                  </pre>
                </div>
                <div>
                  <h3 className="mb-2 text-sm font-medium">After (saved)</h3>
                  <pre className="max-h-96 overflow-auto rounded-md border border-border bg-muted/30 p-3 text-xs">
                    {JSON.stringify(selected.after, null, 2)}
                  </pre>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </PageShell>
  );
}

export default History;
