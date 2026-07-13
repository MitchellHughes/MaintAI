/** Stable helpers for label-group state (avoid re-render loops from new [] references). */

export const EMPTY_LABEL_GROUPS = Object.freeze([]);

export function normalizeLabelGroups(groups) {
  return (groups || [])
    .map((group) => ({
      group_label: String(group.group_label || "").trim().toLowerCase(),
      members: [
        ...new Set(
          (group.members || []).map((m) => String(m).trim().toLowerCase()).filter(Boolean),
        ),
      ].sort(),
      reason: group.reason || "",
    }))
    .filter((group) => group.group_label && group.members.length)
    .sort((a, b) => a.group_label.localeCompare(b.group_label));
}

export function serializeLabelGroups(groups) {
  return JSON.stringify(normalizeLabelGroups(groups));
}
