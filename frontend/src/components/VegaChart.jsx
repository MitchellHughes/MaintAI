import { useEffect, useMemo, useRef, useState } from "react";
import { VegaEmbed } from "react-vega";

import { cn } from "@/lib/utils";

const EMBED_OPTIONS = { actions: false, renderer: "svg" };

/** Replace `{ name: "foo" }` data refs with inline `{ values: [...] }`. */
function inlineSpecData(spec, data) {
  if (!spec) return null;
  const next = { ...spec };

  if (next.data?.name && data?.[next.data.name]) {
    next.data = { values: data[next.data.name] };
  }

  if (Array.isArray(next.layer)) {
    next.layer = next.layer.map((layer) => {
      if (layer.data?.name && data?.[layer.data.name]) {
        return { ...layer, data: { values: data[layer.data.name] } };
      }
      return layer;
    });
  }

  delete next.datasets;
  return next;
}

export default function VegaChart({ spec, data, className, minHeight = 220 }) {
  const containerRef = useRef(null);
  const [width, setWidth] = useState(0);
  const [error, setError] = useState("");

  useEffect(() => {
    const node = containerRef.current;
    if (!node) return undefined;

    const measure = () => {
      const w = node.getBoundingClientRect().width;
      setWidth(Math.max(0, Math.floor(w)));
    };

    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  const fullSpec = useMemo(() => {
    const base = inlineSpecData(spec, data);
    if (!base || width < 120) return null;

    return {
      ...base,
      width: Math.max(120, width - 8),
      height: base.height ?? minHeight,
      autosize: { type: "fit", contains: "padding", resize: true },
    };
  }, [spec, data, width, minHeight]);

  return (
    <div
      ref={containerRef}
      className={cn("relative min-h-[220px] w-full min-w-0", className)}
      style={{ minHeight }}
    >
      {error && (
        <p className="mb-2 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
          Chart error: {error}
        </p>
      )}
      {fullSpec ? (
        <VegaEmbed
          spec={fullSpec}
          options={EMBED_OPTIONS}
          onError={(err) => setError(err instanceof Error ? err.message : String(err))}
        />
      ) : (
        <div className="flex h-full min-h-[inherit] items-center justify-center text-xs text-muted-foreground">
          Loading chart…
        </div>
      )}
    </div>
  );
}
