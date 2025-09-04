import React from "react";
import { prettyBytes } from "../utils";

export default function Rightbar({ lastQuery, indexStats, onRefreshStats }) {
  return (
    <aside className="rightbar">
      <div className="card">
        <div className="card-title">Last Query</div>
        <div className="muted">{lastQuery || "—"}</div>
      </div>

      <div className="card">
        <div className="card-title">Quick Tips</div>
        <ul>
          <li>Use product names or error codes for best retrieval.</li>
          <li>When confidence is low, check cited snippets.</li>
          <li>Ingest updated manuals after firmware changes.</li>
        </ul>
      </div>

      <div className="card">
        <div className="card-title">Index Size</div>
        <div className="muted">
          {indexStats
            ? prettyBytes(indexStats.index_file_size_bytes ?? 0)
            : "n/a"}
        </div>
        <div style={{ marginTop: 8 }}>
          <button className="btn ghost" onClick={onRefreshStats}>
            Refresh Stats
          </button>
        </div>
      </div>
    </aside>
  );
}
