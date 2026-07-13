import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

const MODEL_LABELS = {
  TokenMatchingClassifier: "Token matching",
  EquipmentBasedClassifier: "Equipment based",
  SemanticSimilarityClassifier: "Semantic similarity",
  UMECClassifier: "UMEC ensemble",
};

export default function RunTimingPanel({ timing }) {
  if (!timing) return null;

  const modelEntries = Object.entries(timing.models || {});

  return (
    <Card className="min-w-0 shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="text-lg">Run timing</CardTitle>
        <CardDescription>
          {timing.row_count?.toLocaleString?.() ?? timing.row_count} rows · total{" "}
          <strong className="text-foreground">
            {timing.total_display || `${timing.total_seconds}s`}
          </strong>
          {timing.large_dataset_mode && (
            <> · large-file mode (sampled training, chunked scoring)</>
          )}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Step</TableHead>
              <TableHead className="text-right">Duration</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            <TableRow>
              <TableCell>Model training (fit on upload)</TableCell>
              <TableCell className="text-right">
                {timing.fit_display || `${timing.fit_seconds}s`}
              </TableCell>
            </TableRow>
            {modelEntries.map(([key, entry]) => (
              <TableRow key={key}>
                <TableCell>{entry.label || MODEL_LABELS[key] || key}</TableCell>
                <TableCell className="text-right">
                  {entry.display || `${entry.seconds}s`}
                </TableCell>
              </TableRow>
            ))}
            <TableRow className="font-semibold">
              <TableCell>Scoring all models</TableCell>
              <TableCell className="text-right">
                {timing.predict_display || `${timing.predict_seconds}s`}
              </TableCell>
            </TableRow>
            <TableRow className="font-semibold">
              <TableCell>Total</TableCell>
              <TableCell className="text-right">
                {timing.total_display || `${timing.total_seconds}s`}
              </TableCell>
            </TableRow>
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}
