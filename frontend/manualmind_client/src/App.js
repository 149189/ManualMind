import React, { useEffect, useState } from "react";
import Header from "./components/Header";
import Sidebar from "./components/Sidebar";
import QueryForm from "./components/Queryform";
import ResultPanel from "./components/ResultPanel";
import IngestPanel from "./components/IngestPanel";
import Rightbar from "./components/Rightbar";
import Footer from "./components/Footer";
import { getHealth, getStats, postQuery, postIngest } from "./api";
import { normalizeConfidence } from "./utils";

export default function App() {
  const [health, setHealth] = useState(null);
  const [stats, setStats] = useState(null);
  const [connected, setConnected] = useState(false);

  // Query state
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5);
  const [minScore, setMinScore] = useState(0.0);
  const [useLLM, setUseLLM] = useState(true);
  const [searching, setSearching] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    refreshAll();
    // global quick actions events
    const ex = (e) => setQuery(e.detail || "");
    const clear = () => {
      setQuery("");
      setResult(null);
      setError(null);
    };
    window.addEventListener("exampleQuery", ex);
    window.addEventListener("clearUI", clear);
    return () => {
      window.removeEventListener("exampleQuery", ex);
      window.removeEventListener("clearUI", clear);
    };
  }, []);

  async function refreshAll() {
    try {
      const h = await getHealth();
      setHealth(h);
      setConnected(true);
    } catch (e) {
      setHealth(null);
      setConnected(false);
    }
    try {
      const s = await getStats();
      setStats(s);
    } catch (e) {
      setStats(null);
    }
  }

  async function handleSearch() {
    setError(null);
    setResult(null);
    if (!query || !query.trim()) {
      setError("Please enter a question.");
      return;
    }
    setSearching(true);
    try {
      const payload = { q: query, top_k: topK, min_score: minScore };
      const res = await postQuery(payload);
      // Normalize confidence if needed
      if (res?.confidence == null && res?.llm_confidence != null) {
        res.confidence = normalizeConfidence(res.llm_confidence);
      }
      setResult(res);
    } catch (e) {
      console.error(e);
      setError(e.message || "Query failed");
    } finally {
      setSearching(false);
    }
  }

  async function handleIngest(formData) {
    const res = await postIngest(formData);
    // refresh stats after successful ingest
    await refreshAll();
    return res;
  }

  return (
    <div className="app-root">
      <Header onRefresh={refreshAll} connected={connected} />

      <main className="main-grid">
        <Sidebar health={health} stats={stats} onRefresh={refreshAll} />

        <section className="content">
          <QueryForm
            query={query}
            setQuery={setQuery}
            topK={topK}
            setTopK={setTopK}
            minScore={minScore}
            setMinScore={setMinScore}
            useLLM={useLLM}
            setUseLLM={setUseLLM}
            onSearch={handleSearch}
            searching={searching}
          />

          <div className="result-area">
            {error && <div className="alert danger">{error}</div>}
            <ResultPanel result={result} error={null} />
          </div>

          <IngestPanel onIngest={handleIngest} />
        </section>

        <Rightbar
          lastQuery={query}
          indexStats={stats}
          onRefreshStats={() =>
            getStats()
              .then(setStats)
              .catch(() => {})
          }
        />
      </main>

      <Footer />
    </div>
  );
}
