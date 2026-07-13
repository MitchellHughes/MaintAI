/**
 * Built-in category sets derived from SDR maintenance_records.csv (~170k rows).
 * Keywords are seeded from a stratified research sample; re-run auto-fill per upload
 * to mine site-specific phrases.
 */
export const BUILTIN_CATEGORY_TEMPLATES = {
  "SDR 170k (recommended)": [
    {
      label: "corroded",
      keywords:
        "corrosion, corroded, corrosion was, corrosion around, corrosion level, corrosion floor",
      reference_aliases: ["corroded"],
    },
    {
      label: "cracked",
      keywords: "cracked, crack, cracks, cracked frame, crack fuselage, cracked shear plate",
      reference_aliases: ["cracked"],
    },
    {
      label: "leaking",
      keywords: "leaking, leak, leaks noted, leak check, saturated, hydraulic leak",
      reference_aliases: ["leaking", "saturated", "clogged"],
    },
    {
      label: "worn",
      keywords: "worn, wear, worn out, worn seals, chafed, vibration, lack of lube",
      reference_aliases: ["worn", "vibration", "lack of lube", "chafed"],
    },
    {
      label: "damaged",
      keywords:
        "damaged, damage, broken, bent, gouged, punctured, sheared, torn, deformed",
      reference_aliases: [
        "damaged",
        "broken",
        "dented",
        "punctured",
        "gouged",
        "sheared",
        "separated",
        "bent",
        "torn",
        "deformed",
        "burned",
        "chafed",
      ],
    },
    {
      label: "faulty",
      keywords:
        "faulty, failed, fault, fault message, inoperative, malfunctioned, defective, intermittent",
      reference_aliases: [
        "failed",
        "faulty",
        "inoperative",
        "malfunctioned",
        "faulted",
        "defective",
        "unserviceable",
        "intermittent",
        "suspect faulty",
        "failed test",
        "fault message",
        "false indication",
        "tripped",
        "stuck",
      ],
    },
    {
      label: "loose",
      keywords:
        "loose, out of adjust, excess play, unsecure, detached, disconnected, seized, binding",
      reference_aliases: [
        "loose",
        "out of adjust",
        "excess play",
        "unsecure",
        "detached",
        "disconnected",
        "out of position",
        "misinstalled",
        "seized",
        "locked",
        "sticking",
        "binding",
      ],
    },
    {
      label: "missing",
      keywords: "missing, missing from, foreign object, fod, dislodged",
      reference_aliases: ["missing", "dislodged", "fod"],
    },
    {
      label: "delaminated",
      keywords: "delaminated, delamination, delamination floor panel",
      reference_aliases: ["delaminated", "debonded"],
    },
    {
      label: "low pressure",
      keywords: "low pressure, low bottle, bottle pressure, pressure low, low psi",
      reference_aliases: ["low pressure", "low"],
    },
    {
      label: "odor",
      keywords: "odor, smell, fumes, odor cabin, odor detected, dirty sock",
      reference_aliases: ["odor", "odour"],
    },
    {
      label: "contaminated",
      keywords:
        "contaminated, contamination, decontamination, debris, blocked, metal contamination, grease contamination",
      reference_aliases: ["contaminated"],
    },
    {
      label: "smoke",
      keywords: "smoke, galley, oven, burnt, electrical smoke, cockpit smoke",
      reference_aliases: ["smoke"],
    },
    {
      label: "deteriorated",
      keywords: "deteriorated, deterioration, degraded",
      reference_aliases: ["deteriorated"],
    },
    {
      label: "dirty",
      keywords: "dirty, obstructed, clogged",
      reference_aliases: ["dirty", "obstructed"],
    },
    {
      label: "discharged",
      keywords: "discharged, discharge",
      reference_aliases: ["discharged"],
    },
    {
      label: "false activation",
      keywords: "false activation, unwanted deploy, partial deploy",
      reference_aliases: ["false activation", "unwanted deploy", "partial deploy"],
    },
    {
      label: "no test",
      keywords: "no test, not tested",
      reference_aliases: ["no test"],
    },
  ],
  "Aircraft maintenance (merged loose)": [
    { label: "corroded", keywords: "" },
    { label: "cracked", keywords: "" },
    { label: "leaking", keywords: "" },
    { label: "worn", keywords: "" },
    { label: "damaged", keywords: "" },
    { label: "dirty", keywords: "" },
    { label: "loose", keywords: "" },
    { label: "faulty", keywords: "" },
    { label: "missing", keywords: "" },
    { label: "delaminated", keywords: "" },
    { label: "low pressure", keywords: "" },
    { label: "odor", keywords: "" },
    { label: "discharged", keywords: "" },
    { label: "false activation", keywords: "" },
    { label: "no test", keywords: "" },
  ],
  "Aircraft maintenance (full)": [
    { label: "corroded", keywords: "" },
    { label: "cracked", keywords: "" },
    { label: "leaking", keywords: "" },
    { label: "worn", keywords: "" },
    { label: "damaged", keywords: "" },
    { label: "dirty", keywords: "" },
    { label: "loose", keywords: "" },
    { label: "unsecure", keywords: "" },
    { label: "out of adjust", keywords: "" },
    { label: "excess play", keywords: "" },
    { label: "detached", keywords: "" },
    { label: "faulty", keywords: "" },
    { label: "faulted", keywords: "" },
    { label: "missing", keywords: "" },
    { label: "delaminated", keywords: "" },
    { label: "low pressure", keywords: "" },
    { label: "odor", keywords: "" },
    { label: "discharged", keywords: "" },
    { label: "false activation", keywords: "" },
    { label: "no test", keywords: "" },
  ],
};
