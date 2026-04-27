import { useEffect, useRef, useState } from "react";
import type { SampleImage } from "../types";

interface Props {
  busy: boolean;
  samples: SampleImage[];
  onAnalyzeFile: (file: File) => void;
  onAnalyzeSample: (name: string) => void;
}

export default function UploadCard({ busy, samples, onAnalyzeFile, onAnalyzeSample }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);
  const [selected, setSelected] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!selected) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(selected);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [selected]);

  function handleFiles(files: FileList | null) {
    if (!files || !files.length) return;
    const file = files[0];
    if (!file.type.startsWith("image/")) {
      alert("Please choose an image file (jpg/png/bmp/tiff).");
      return;
    }
    setSelected(file);
  }

  return (
    <div className="card">
      <h2>Input</h2>
      <label
        className={`dropzone ${dragActive ? "active" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragActive(false);
          handleFiles(e.dataTransfer.files);
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          onChange={(e) => handleFiles(e.target.files)}
        />
        <strong>Drop a blood smear image</strong>
        <p>or click to browse (jpg, png, bmp, tiff — max 15 MB)</p>
      </label>

      {previewUrl && (
        <div className="preview">
          <img src={previewUrl} alt="upload preview" />
          <span className="meta">
            {selected?.name} • {(selected?.size ?? 0 / 1024).toLocaleString()} bytes
          </span>
        </div>
      )}

      <div className="action-row">
        <button
          className="btn"
          disabled={!selected || busy}
          onClick={() => selected && onAnalyzeFile(selected)}
        >
          {busy ? <span className="spinner" /> : null}
          {busy ? "Analyzing…" : "Analyze image"}
        </button>
        <button
          className="btn secondary"
          disabled={busy}
          onClick={() => {
            setSelected(null);
            if (inputRef.current) inputRef.current.value = "";
          }}
        >
          Clear
        </button>
      </div>

      {samples.length > 0 && (
        <div className="samples-row">
          <h3>Bundled samples</h3>
          <div className="chips">
            {samples.map((s) => (
              <button
                key={s.name}
                className="chip"
                disabled={busy}
                onClick={() => onAnalyzeSample(s.name)}
                title={`${(s.size_bytes / 1024).toFixed(0)} KB`}
              >
                {s.name}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
