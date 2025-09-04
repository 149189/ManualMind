import React, { useRef, useState } from "react";

export default function IngestPanel({ onIngest }) {
  const fileRef = useRef();
  const [uploading, setUploading] = useState(false);

  const handleUpload = async () => {
    const file = fileRef.current?.files?.[0];
    if (!file) return alert("Choose a PDF file.");
    if (!file.name.toLowerCase().endsWith(".pdf"))
      return alert("Only PDFs supported.");

    setUploading(true);
    const fd = new FormData();
    fd.append("file", file, file.name);

    try {
      const res = await onIngest(fd);
      // onIngest should return parsed result or throw
      console.log("Ingest result", res);
    } catch (e) {
      console.error(e);
      alert("Ingest failed: " + (e.message || e));
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="ingest-panel card">
      <div className="card-title">Ingest a PDF</div>
      <div className="muted">
        Upload a manual PDF. System will extract text and index it.
      </div>
      <div style={{ marginTop: 12 }}>
        <input ref={fileRef} type="file" accept="application/pdf" />
      </div>
      <div style={{ marginTop: 10, display: "flex", gap: 8 }}>
        <button
          className={`btn ${uploading ? "loading" : ""}`}
          onClick={handleUpload}
          disabled={uploading}
        >
          {uploading ? "Uploading…" : "Upload & Ingest"}
        </button>
        <button
          className="btn ghost"
          onClick={() => {
            fileRef.current.value = null;
          }}
        >
          Reset
        </button>
      </div>
      <div style={{ marginTop: 12 }}>
        <small className="muted">Max file size enforced by server.</small>
      </div>
    </div>
  );
}
