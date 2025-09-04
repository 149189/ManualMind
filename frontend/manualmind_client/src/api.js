const API_BASE = process.env.REACT_APP_API_BASE_URL || "http://localhost:8000";

async function jsonFetch(path, opts = {}) {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, opts);
  const text = await res.text();
  try {
    const data = JSON.parse(text);
    if (!res.ok) {
      const err = new Error(data.detail || res.statusText || "API error");
      err.payload = data;
      throw err;
    }
    return data;
  } catch (e) {
    // non-JSON or error JSON
    if (!res.ok) {
      const err = new Error(text || res.statusText || "API error");
      err.payload = text;
      throw err;
    }
    // parse fallback: return raw text
    return { text };
  }
}

export async function getHealth() {
  return jsonFetch("/health");
}
export async function getStats() {
  return jsonFetch("/stats");
}
export async function postQuery(payload) {
  return jsonFetch("/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}
export async function postIngest(formData) {
  return jsonFetch("/ingest", {
    method: "POST",
    body: formData,
  });
}
