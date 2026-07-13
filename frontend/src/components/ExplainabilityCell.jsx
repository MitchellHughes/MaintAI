import { memo, useState } from "react";

const TIER_CLASS = {
  high: "tier-high",
  medium: "tier-medium",
  low: "tier-low",
  review: "tier-review",
};

const MODEL_SHORT = {
  token: "Token",
  equipment: "Equipment",
  semantic: "Semantic",
  TokenMatchingClassifier: "Token",
  EquipmentBasedClassifier: "Equipment",
  SemanticSimilarityClassifier: "Semantic",
};

function BaseModelLine({ base }) {
  const short = MODEL_SHORT[base.model] || base.model_label || base.model;
  const vote = base.base_vote ?? "—";
  const agrees = base.agrees_with_ensemble;
  const terms = (base.contributions || [])
    .slice(0, 3)
    .map((c) => c.term)
    .filter(Boolean)
    .join(", ");

  return (
    <li className={agrees === false ? "xai-detail-mismatch" : ""}>
      <span className="xai-detail-name">{short}</span>
      <span className="xai-detail-vote">
        {agrees ? "✓ agrees" : "✗ differs"} — <strong>{vote}</strong>
      </span>
      {terms ? <span className="xai-detail-terms"> · {terms}</span> : null}
    </li>
  );
}

function ExplainabilityCell({ xai, tier, tierLabel }) {
  const [open, setOpen] = useState(false);

  if (!xai) {
    return <span className="muted">—</span>;
  }

  const simple = xai.simple || {};
  const resolvedTier = tier || simple.tier || "review";
  const resolvedLabel = tierLabel || simple.tier_label || "Review";
  const oneLiner = simple.one_liner || xai.explanation || "";
  const bases = xai.bases || [];

  return (
    <div className="xai-simple">
      <div className="xai-simple-head">
        <span className={`tier-badge ${TIER_CLASS[resolvedTier] || "tier-review"}`}>
          {resolvedLabel}
        </span>
      </div>
      {oneLiner ? <p className="xai-one-liner text-muted-foreground">{oneLiner}</p> : null}
      {bases.length > 0 && (
        <button
          type="button"
          className="xai-toggle"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
        >
          {open ? "Hide model breakdown" : "Show model breakdown"}
        </button>
      )}
      {open && bases.length > 0 && (
        <div className="xai-details">
          <ul className="xai-detail-list">
            {bases.map((base) => (
              <BaseModelLine key={base.model} base={base} />
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default memo(ExplainabilityCell);
