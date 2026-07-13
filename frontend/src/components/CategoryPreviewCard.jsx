import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export default function CategoryPreviewCard({ categories = [] }) {
  const labels = categories
    .map((row) => String(row.label || "").trim())
    .filter(Boolean);

  if (!labels.length) return null;

  return (
    <Card className="shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Category preview</CardTitle>
        <CardDescription>{labels.length} label(s) for this run</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex max-h-32 flex-wrap gap-1.5 overflow-y-auto">
          {labels.map((label) => (
            <Badge key={label} variant="outline" className="capitalize">
              {label}
            </Badge>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
