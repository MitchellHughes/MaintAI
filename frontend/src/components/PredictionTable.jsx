import { memo, useMemo, useState } from "react";

import DiscrepancyHighlight from "./DiscrepancyHighlight";
import ExplainabilityCell from "./ExplainabilityCell";
import TopKReviewCell from "./TopKReviewCell";
import { resolveTopKChoices } from "../utils/topKChoices";
import { buildCategoryMatcher, isAutoAcceptRow, tierForRow } from "../utils/engineerReview";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";

const TIER_LABEL = {
  high: "Auto-accept",
  medium: "Spot-check",
  low: "Manual review",
  review: "Needs review",
};

function pct(v) {
  return `${Math.round((v ?? 0) * 100)}%`;
}

const PredictionRow = memo(function PredictionRow({
  row,
  showActual,
  showComponent,
  autoAcceptThreshold,
  topKLimit = 3,
  onEdit,
  matchCategory,
  categories = [],
  labelGroups = [],
}) {
  const [selectedRank, setSelectedRank] = useState(1);
  const tier = tierForRow(row);
  const tierLabel = isAutoAcceptRow(row, autoAcceptThreshold)
    ? "Auto-accept"
    : TIER_LABEL[tier] || tier;
  const auto = isAutoAcceptRow(row, autoAcceptThreshold);
  const topKChoices = resolveTopKChoices(row, topKLimit, { categories, labelGroups });
  const activeDetail =
    topKChoices.find((item) => item.rank === selectedRank) || topKChoices[0] || null;
  const spans = activeDetail?.text_spans || row.xai?.simple?.text_spans || [];
  const keywords = activeDetail?.keywords || row.xai?.simple?.keywords;
  const actualMapped = row.actual_label ? matchCategory(row.actual_label) : null;

  return (
    <TableRow
      className={cn(
        auto ? "bg-emerald-500/5" : tier === "medium" ? "bg-amber-500/5" : "bg-destructive/5",
      )}
    >
      <TableCell className="tabular-nums">{row.row_id}</TableCell>
      <TableCell className="max-w-md whitespace-normal text-sm">
        <DiscrepancyHighlight text={row.discrepancy} spans={spans} keywords={keywords} />
      </TableCell>
      {showComponent && (
        <TableCell className="max-w-[10rem] whitespace-normal text-sm">
          {row.component ? (
            <span className="capitalize">{row.component}</span>
          ) : (
            <span className="text-muted-foreground">—</span>
          )}
        </TableCell>
      )}
      {showActual && (
        <TableCell>
          {row.actual_label ? (
            <div className="flex flex-col gap-1">
              <Badge variant="outline">{row.actual_label}</Badge>
              {!actualMapped && (
                <span className="text-xs text-amber-400" title="Reference label is not in your category list">
                  Not in categories
                </span>
              )}
            </div>
          ) : (
            "—"
          )}
        </TableCell>
      )}
      <TableCell className="min-w-[14rem] whitespace-normal align-top text-sm">
        <TopKReviewCell
          row={row}
          selectedRank={selectedRank}
          activeLabel={row.final_condition ?? row.predicted_condition}
          topKLimit={topKLimit}
          categories={categories}
          labelGroups={labelGroups}
          onSelectRank={setSelectedRank}
          onPickLabel={(label) => onEdit?.(row.row_id, label)}
        />
      </TableCell>
      <TableCell className="align-top">
        <Input
          className="h-8 min-w-[140px]"
          type="text"
          value={row.final_condition ?? row.predicted_condition ?? ""}
          onChange={(e) => onEdit?.(row.row_id, e.target.value)}
          placeholder="Or type final label"
          list="cond-options"
        />
        {row.final_condition &&
          row.predicted_condition &&
          row.final_condition !== row.predicted_condition && (
            <span className="mt-1 block text-xs text-amber-400">edited</span>
          )}
      </TableCell>
      <TableCell>
        <div className="flex flex-col gap-1">
          <Badge variant={auto ? "success" : tier === "medium" ? "warning" : "destructive"}>
            {tierLabel}
          </Badge>
          <span
            className={cn("text-xs tabular-nums", auto ? "text-emerald-400" : "text-amber-400")}
          >
            {pct(row.confidence)}
          </span>
        </div>
      </TableCell>
      <TableCell className="max-w-sm whitespace-normal text-sm">
        <ExplainabilityCell xai={row.xai} tier={tier} tierLabel={tierLabel} />
      </TableCell>
    </TableRow>
  );
});

function PredictionTable({
  predictions,
  onEdit,
  showActualColumn = false,
  autoAcceptThreshold = 0.85,
  defaultTierFilter = "all",
  categories = [],
  labelGroups = [],
  showComponentColumn = false,
  topKLimit = 3,
}) {
  const [query, setQuery] = useState("");
  const [sortKey, setSortKey] = useState("confidence");
  const [sortDir, setSortDir] = useState("desc");
  const [showChangedOnly, setShowChangedOnly] = useState(false);
  const [tierFilter, setTierFilter] = useState(defaultTierFilter);
  const [pageSize, setPageSize] = useState(50);
  const [page, setPage] = useState(1);
  const safeRows = predictions ?? [];
  const showActual = Boolean(showActualColumn);
  const showComponent =
    Boolean(showComponentColumn) || safeRows.some((row) => Boolean(row.component));

  const matchCategory = useMemo(
    () => buildCategoryMatcher(categories, labelGroups),
    [categories, labelGroups],
  );

  const acceptCounts = useMemo(() => {
    let auto = 0;
    safeRows.forEach((row) => {
      if (isAutoAcceptRow(row, autoAcceptThreshold)) auto += 1;
    });
    return { auto, review: Math.max(0, safeRows.length - auto) };
  }, [safeRows, autoAcceptThreshold]);

  const labelOptions = useMemo(() => {
    const labels = new Set();
    safeRows.forEach((p) => {
      if (p.predicted_condition) labels.add(p.predicted_condition);
      if (p.final_condition) labels.add(p.final_condition);
    });
    return Array.from(labels).sort((a, b) => a.localeCompare(b));
  }, [safeRows]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    const rows = safeRows.filter((p) => {
      if (showChangedOnly && p.final_condition === p.predicted_condition) {
        return false;
      }
      const tier = tierForRow(p);
      if (tierFilter === "auto" && !isAutoAcceptRow(p, autoAcceptThreshold)) return false;
      if (tierFilter === "review" && isAutoAcceptRow(p, autoAcceptThreshold)) return false;
      if (!q) return true;
      const hay = [
        String(p.row_id ?? ""),
        p.discrepancy,
        p.predicted_condition,
        p.final_condition,
        p.xai?.simple?.one_liner,
        ...(p.xai?.simple?.keywords || []),
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      return hay.includes(q);
    });

    const dir = sortDir === "asc" ? 1 : -1;
    return [...rows].sort((a, b) => {
      const getVal = (row) => {
        if (sortKey === "confidence") return row.confidence ?? 0;
        if (sortKey === "row_id") return Number(row.row_id ?? 0);
        return String(row[sortKey] ?? "").toLowerCase();
      };
      const av = getVal(a);
      const bv = getVal(b);
      if (typeof av === "number" && typeof bv === "number") {
        return (av - bv) * dir;
      }
      return String(av).localeCompare(String(bv)) * dir;
    });
  }, [safeRows, query, showChangedOnly, sortKey, sortDir, tierFilter, autoAcceptThreshold]);

  const total = filtered.length;
  const maxPage = Math.max(1, Math.ceil(total / pageSize));
  const safePage = Math.min(page, maxPage);
  const start = (safePage - 1) * pageSize;
  const paged = filtered.slice(start, start + pageSize);

  if (!safeRows.length) return null;

  function toggleSort(nextKey) {
    if (sortKey === nextKey) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(nextKey);
      setSortDir(nextKey === "confidence" ? "desc" : "asc");
    }
  }

  const SortBtn = ({ col, children }) => (
    <Button variant="ghost" size="sm" className="h-7 px-2" type="button" onClick={() => toggleSort(col)}>
      {children} {sortKey === col ? (sortDir === "asc" ? "▲" : "▼") : ""}
    </Button>
  );

  return (
    <div className="min-w-0 space-y-4">
      <div className="flex flex-wrap gap-2">
        <Badge variant="success">Auto-accept: {acceptCounts.auto}</Badge>
        <Badge variant="warning">Needs review: {acceptCounts.review}</Badge>
      </div>

      <p className="text-xs text-muted-foreground">
        Use <strong>Top-K</strong> to set the final label (or type in Final). Percentages are model
        score share; <span className="text-amber-600 dark:text-amber-400">alt</span> ranks lack
        keyword evidence. Highlights follow the selected rank:{" "}
        <span className="rounded bg-emerald-500/30 px-1">green</span> supports that class,{" "}
        <span className="rounded bg-red-500/25 px-1">red</span> supports an alternative,{" "}
        <span className="rounded bg-amber-500/25 px-1">amber</span> other salient term (e.g. reference
        wording not in your categories).
      </p>

      <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div className="flex flex-wrap items-center gap-2">
          <Input
            className="max-w-xs"
            type="search"
            placeholder="Filter rows…"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setPage(1);
            }}
          />
          <label className="flex items-center gap-2 text-sm text-muted-foreground">
            <input
              type="checkbox"
              className="rounded border-input"
              checked={showChangedOnly}
              onChange={(e) => {
                setShowChangedOnly(e.target.checked);
                setPage(1);
              }}
            />
            Edited only
          </label>
          <Select
            value={tierFilter}
            onValueChange={(v) => {
              setTierFilter(v);
              setPage(1);
            }}
          >
            <SelectTrigger className="w-[200px]" aria-label="Filter by review tier">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All rows</SelectItem>
              <SelectItem value="auto">Auto-accept only</SelectItem>
              <SelectItem value="review">Needs review</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <SortBtn col="row_id">ID</SortBtn>
          <SortBtn col="final_condition">Final</SortBtn>
          <SortBtn col="confidence">Confidence</SortBtn>
          <Select
            value={String(pageSize)}
            onValueChange={(v) => {
              setPageSize(Number(v));
              setPage(1);
            }}
          >
            <SelectTrigger className="w-[120px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="25">25 rows</SelectItem>
              <SelectItem value="50">50 rows</SelectItem>
              <SelectItem value="100">100 rows</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <p className="text-xs text-muted-foreground">
        Showing {paged.length} of {total} rows (paginated for performance)
      </p>

      <datalist id="cond-options">
        {labelOptions.map((label) => (
          <option key={label} value={label} />
        ))}
      </datalist>

      <div className="overflow-x-auto rounded-md border border-border">
        <Table className="min-w-[1000px]">
          <TableHeader>
            <TableRow>
              <TableHead className="w-12">#</TableHead>
              <TableHead className="min-w-[16rem]">Discrepancy</TableHead>
              {showComponent && <TableHead>Component</TableHead>}
              {showActual && <TableHead>Actual</TableHead>}
              <TableHead className="min-w-[14rem]">Top-K predictions</TableHead>
              <TableHead className="min-w-[9rem]">Final label</TableHead>
              <TableHead>Confidence</TableHead>
              <TableHead className="min-w-[10rem]">Why this label?</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {paged.map((p) => (
              <PredictionRow
                key={p.row_id}
                row={p}
                showActual={showActual}
                showComponent={showComponent}
                autoAcceptThreshold={autoAcceptThreshold}
                topKLimit={topKLimit}
                onEdit={onEdit}
                matchCategory={matchCategory}
                categories={categories}
                labelGroups={labelGroups}
              />
            ))}
          </TableBody>
        </Table>
      </div>

      {maxPage > 1 && (
        <div className="flex items-center justify-center gap-4">
          <Button
            variant="outline"
            type="button"
            onClick={() => setPage((v) => Math.max(1, v - 1))}
            disabled={safePage === 1}
          >
            Previous
          </Button>
          <span className="text-sm text-muted-foreground">
            Page {safePage} of {maxPage}
          </span>
          <Button
            variant="outline"
            type="button"
            onClick={() => setPage((p) => Math.min(maxPage, p + 1))}
            disabled={safePage === maxPage}
          >
            Next
          </Button>
        </div>
      )}
    </div>
  );
}

export default memo(PredictionTable);
