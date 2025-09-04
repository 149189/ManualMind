export function prettyBytes(bytes) {
  if (!bytes && bytes !== 0) return "—";
  const units = ["B", "KB", "MB", "GB"];
  let i = 0;
  let n = Number(bytes);
  while (n >= 1024 && i < units.length - 1) {
    n /= 1024;
    i++;
  }
  return `${n.toFixed(1)} ${units[i]}`;
}

export function normalizeConfidence(raw) {
  if (raw == null) return 0;
  let c = Number(raw);
  if (Number.isNaN(c)) return 0;
  if (c >= 0 && c <= 1) return c * 100;
  return Math.max(0, Math.min(100, c));
}

export function escapeHtml(unsafe) {
  return (unsafe || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}
