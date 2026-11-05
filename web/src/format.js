// One number format for the whole page. A share is a percentage, whole above 10% and to one decimal
// below it; a count is a whole number with thousands separators; anything else has two decimals.

export function share(v) {
  if (typeof v !== "number") return String(v);
  const pct = v * 100;
  if (pct === 0) return "0%";
  // One decimal would print a real share this small as 0.0%, which reads as none.
  if (Math.abs(pct) < 0.1) return `under 0.1%`;
  return `${Math.abs(pct) >= 10 ? pct.toFixed(0) : pct.toFixed(1)}%`;
}

export function count(v) {
  if (typeof v !== "number") return String(v);
  return Math.round(v).toLocaleString("en-US");
}

export function num(v) {
  if (typeof v !== "number") return String(v);
  return Number.isInteger(v) ? count(v) : v.toFixed(2);
}

export const FORMATS = { share, count, num };

// A placeholder names its format; one that names none is shown as a count or a two-decimal number.
export function show(v, kind) {
  return (FORMATS[kind] || num)(v);
}
