import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

function formatLabel(label) {
  const text = String(label || "").trim();
  if (!text) return "—";
  return text.charAt(0).toUpperCase() + text.slice(1);
}

export default function ConfusionPairsPanel({ pairs = [] }) {
  if (!pairs.length) return null;

  return (
    <div className="space-y-3 border-t border-border pt-6">
      <div>
        <h3 className="text-base font-semibold">Label drift — top confusions</h3>
        <p className="mt-1 text-sm text-muted-foreground">
          Reference label (actual) vs what the model predicted. Use these to refine keywords for
          the actual class.
        </p>
      </div>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Actual (reference)</TableHead>
            <TableHead>Predicted instead</TableHead>
            <TableHead className="text-right">Rows</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {pairs.map((pair) => (
            <TableRow key={`${pair.actual}-${pair.predicted}`}>
              <TableCell>
                <Badge variant="outline" className="capitalize">
                  {formatLabel(pair.actual)}
                </Badge>
              </TableCell>
              <TableCell>
                <span className="text-muted-foreground">→</span>{" "}
                <Badge variant="secondary" className="capitalize">
                  {formatLabel(pair.predicted)}
                </Badge>
              </TableCell>
              <TableCell className="text-right tabular-nums">{pair.count}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
