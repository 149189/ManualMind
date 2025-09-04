import React from "react";

export default function QueryForm({
  query,
  setQuery,
  topK,
  setTopK,
  minScore,
  setMinScore,
  useLLM,
  setUseLLM,
  onSearch,
  searching,
}) {
  return (
    <form
      className="query-panel"
      onSubmit={(e) => {
        e.preventDefault();
        onSearch();
      }}
    >
      <div className="query-controls">
        <textarea
          placeholder="Ask a question about the ingested manuals (e.g., 'How to update firmware?')"
          className="query-input"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          rows={4}
        />
        <div className="controls-row">
          <div className="control-group">
            <label>Top K</label>
            <input
              type="range"
              min="1"
              max="20"
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
            />
            <div className="control-value">{topK}</div>
          </div>

          <div className="control-group">
            <label>Min similarity</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={minScore}
              onChange={(e) => setMinScore(Number(e.target.value))}
            />
            <div className="control-value">{minScore.toFixed(2)}</div>
          </div>

          <div className="control-group tighten">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={useLLM}
                onChange={(e) => setUseLLM(e.target.checked)}
              />
              Use LLM refine
            </label>
          </div>

          <div className="action-col">
            <button
              type="submit"
              className={`btn primary ${searching ? "loading" : ""}`}
              disabled={searching}
            >
              {searching ? "Searching…" : "Search"}
            </button>
          </div>
        </div>
      </div>
    </form>
  );
}
