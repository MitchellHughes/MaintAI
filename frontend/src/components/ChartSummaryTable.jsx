import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

export default function ChartSummaryTable({ summaryRows, threshold = 0.7 }) {
  if (!summaryRows?.length) {
    return <p className="text-sm text-muted-foreground">No rows to summarize — run prediction first.</p>;
  }

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Label (final)</TableHead>
          <TableHead>Auto-accept (≥ {Math.round(threshold * 100)}%)</TableHead>
          <TableHead>Needs review</TableHead>
          <TableHead>Total</TableHead>
          <TableHead>Auto-accept %</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {summaryRows.map((row) => {
          const total = row.high + row.low;
          const highPct = total ? ((row.high / total) * 100).toFixed(1) : "0.0";
          return (
            <TableRow key={row.label}>
              <TableCell>{row.label}</TableCell>
              <TableCell>{row.high}</TableCell>
              <TableCell>{row.low}</TableCell>
              <TableCell>{total}</TableCell>
              <TableCell>{highPct}%</TableCell>
            </TableRow>
          );
        })}
      </TableBody>
    </Table>
  );
}
