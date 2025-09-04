import React from "react";
import { prettyBytes } from "../utils";

export default function Sidebar({ health, stats, onRefresh }) {
  const services = health?.services || {};
  const overall = health?.status || "unknown";

  return (
    <aside className="sidebar">
      <div className="card small">
        <div className="card-title">System Status</div>
        <div className="status-row">
          <div
            className={`pill ${
              services.api === "healthy" ? "pill-ok" : "pill-bad"
            }`}
          >
            API
          </div>
          <div
            className={`pill ${
              services.embedder === "healthy" ? "pill-ok" : "pill-bad"
            }`}
          >
            Embedder
          </div>
          <div
            className={`pill ${
              services.store === "healthy" ? "pill-ok" : "pill-bad"
            }`}
          >
            Store
          </div>
          <div
            className={`pill ${
              services.llm === "healthy" ? "pill-ok" : "pill-bad"
            }`}
          >
            LLM
          </div>
        </div>
        <div className="muted">
          Overall: <strong className="nowrap">{overall}</strong>
        </div>
        <div style={{ marginTop: 10 }}>
          <button className="btn" onClick={onRefresh}>
            Refresh status
          </button>
        </div>
      </div>

      <div style={{ height: 12 }} />

      <div className="card small">
        <div className="card-title">Index</div>
        {stats ? (
          <div className="stat-grid">
            <div>
              <div className="stat-value">
                {stats.total_chunks ?? stats.total_vectors ?? 0}
              </div>
              <div className="muted">Chunks</div>
            </div>
            <div>
              <div className="stat-value">
                {stats.total_documents ?? stats.unique_sources ?? 0}
              </div>
              <div className="muted">Documents</div>
            </div>
            <div>
              <div className="stat-value">
                {stats.index_size ??
                  prettyBytes(stats.index_file_size_bytes ?? 0)}
              </div>
              <div className="muted">Size</div>
            </div>
          </div>
        ) : (
          <div className="muted">No stats available</div>
        )}
      </div>

      <div className="card">
        <div className="card-title">Quick Actions</div>
        <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
          <button
            className="btn"
            onClick={() =>
              window.dispatchEvent(
                new CustomEvent("exampleQuery", {
                  detail: "How do I reset the device?",
                })
              )
            }
          >
            Example Query
          </button>
          <button
            className="btn ghost"
            onClick={() => window.dispatchEvent(new Event("clearUI"))}
          >
            Clear
          </button>
        </div>
      </div>
    </aside>
  );
}
