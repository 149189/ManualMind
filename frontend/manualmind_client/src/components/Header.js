import React from "react";

export default function Header({ onRefresh, connected }) {
  return (
    <header className="topbar">
      <div className="brand">
        <div className="logo" aria-hidden>
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
            <rect x="2" y="3" width="20" height="18" rx="2" fill="#0b5cff" />
            <path
              d="M6 7h12M6 11h12M6 15h8"
              stroke="#fff"
              strokeWidth="1.2"
              strokeLinecap="round"
            />
          </svg>
        </div>
        <div>
          <div className="title">ManualMind</div>
          <div className="subtitle">RAG assistant — product manuals</div>
        </div>
      </div>

      <div className="right-controls">
        <div
          className={`health-dot ${connected ? "ok" : "bad"}`}
          title={connected ? "Backend connected" : "Backend unavailable"}
        />
        <button className="btn ghost small" onClick={onRefresh}>
          Refresh
        </button>
      </div>
    </header>
  );
}
