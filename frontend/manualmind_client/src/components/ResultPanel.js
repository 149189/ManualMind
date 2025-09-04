import React, { useState } from "react";
import { normalizeConfidence } from "../utils";

export default function ResultPanel({ result, error }) {
  const [showAll, setShowAll] = useState(false);
  if (!result && !error) return null;

  if (error) {
    return <div className="alert danger">{String(error)}</div>;
  }

  // parse returned result
  let summary = result.answer || "";
  let citations = result.citations || result.citation || [];
  let llm_conf = result.confidence ?? result.llm_confidence ?? null;
  const retrieval = result.retrieval_stats || {};
  const snippets = result.snippets || [];

  // if answer is JSON string
  try {
    if (typeof summary === "string" && summary.trim().startsWith("{")) {
      const p = JSON.parse(summary);
      if (p?.answer) {
        summary = p.answer;
        citations = p.citations || citations;
        llm_conf = p.llm_confidence ?? llm_conf;
      }
    }
  } catch (e) {
    /* ignore */
  }

  const confPct = (() => {
    if (llm_conf == null) {
      const avg = retrieval.avg_score ?? 0;
      return Math.round((avg || 0) * 100);
    }
    let num = Number(llm_conf);
    if (num >= 0 && num <= 1) num = num * 100;
    return Math.round(Math.max(0, Math.min(100, num)));
  })();

  const findSnippetById = (cid) =>
    snippets.find(
      (s) =>
        s.id === cid ||
        s.chunk_id === cid ||
        (s.id && s.id.toUpperCase() === cid)
    );

  return (
    <div className="panel">
      <div className="panel-header">
        <h3>Answer</h3>
        <div className="confidence">
          <div className="conf-label">Confidence</div>
          <div className="conf-bar">
            <div className="conf-fill" style={{ width: `${confPct}%` }} />
          </div>
          <div className="conf-num">{confPct}%</div>
        </div>
      </div>

      <div className="answer-card">
        {summary ? (
          <div
            className="answer-text"
            dangerouslySetInnerHTML={{
              __html: summary.replace(/\n/g, "<br/>"),
            }}
          />
        ) : (
          <div className="muted">No summary generated.</div>
        )}
        {citations && citations.length > 0 && (
          <div className="citations">
            <strong>Citations:</strong> {citations.join(", ")}
          </div>
        )}
      </div>

      <div className="snippet-list">
        <h4>Used snippets</h4>
        {citations && citations.length > 0 ? (
          citations.map((cid) => {
            const s = findSnippetById(cid);
            if (!s)
              return (
                <div key={cid} className="snippet missing">
                  <div className="snippet-title">{cid} — (not returned)</div>
                </div>
              );
            return (
              <details key={cid} className="snippet" open>
                <summary className="snippet-title">
                  {cid} — {s.source || "unknown"} page {s.page ?? "?"}{" "}
                  <span className="score">({(s.score || 0).toFixed(3)})</span>
                </summary>
                <div className="snippet-body">{s.text}</div>
              </details>
            );
          })
        ) : (
          <div className="muted">No snippets cited by the model.</div>
        )}

        <details className="snippet-all">
          <summary>Show all retrieved snippets ({snippets.length})</summary>
          {snippets.map((s, i) => (
            <details key={i} className="snippet small">
              <summary className="snippet-title">
                {s.id ?? `S${i + 1}`} — {s.source ?? "unknown"} page{" "}
                {s.page ?? "?"}{" "}
                <span className="score">({(s.score || 0).toFixed(3)})</span>
              </summary>
              <div className="snippet-body">{s.text}</div>
            </details>
          ))}
        </details>
      </div>
    </div>
  );
}
