# Blood Smear Domain Expert

A safety-aware blood smear analysis pipeline that combines computer vision and retrieval-augmented reasoning.

This repository implements a three-stage workflow:

1. Stage 1: YOLOv8 cell detection (WBC, RBC, Platelet)
2. Stage 2: EfficientNet WBC subtype classification with Monte Carlo Dropout uncertainty
3. Stage 3: RAG plus GPT-4o clinical reasoning with safety flags and expert-review escalation

## Overview

The project is designed as a configuration-driven CLI pipeline for reproducible inference, not notebook-only execution.

Key characteristics:
- Modular Python code under src/
- Config-first behavior via config.yaml
- Safety-focused uncertainty propagation
- Structured JSON outputs for each stage
- Smoke-test command for first-run validation

## Requirements

- Python 3.8+ (3.10 recommended)
- Windows PowerShell (commands below are Windows-first)
- OpenAI API key for Stage 3
- Model checkpoints in models/
- PDF knowledge base files for Stage 3

## Quick Start (Windows PowerShell)

```powershell
# 1) Open repo root
cd C:\Users\qkafr\Desktop\BSc_Thesis

# 2) Create and activate virtual environment
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3) Install dependencies
pip install -r requirements.txt

# 4) Configure environment
Copy-Item .env.example .env
# Edit .env and set OPENAI_API_KEY=your-key

# 5) Verify required model files exist
# models/yolov8s_blood.pt
# models/efficientnet_wbc_finetuned.pt

# 6) Verify required Stage 3 PDFs exist
# data/pdfs/essentials_haematology.pdf
# data/pdfs/consie_haematology.pdf

# 7) Validate configuration and setup
python main.py test-config
python main.py smoke-test
```

## CLI Usage

### 1) Analyze a single image

```powershell
python main.py analyze .\path\to\blood_smear.jpg
```

### 2) Analyze multiple images as batch

```powershell
python main.py analyze .\img1.jpg .\img2.jpg .\img3.jpg --batch
```

### 3) Verbose run

```powershell
python main.py analyze .\path\to\blood_smear.jpg --verbose
```

### 4) Run without saving output files

```powershell
python main.py analyze .\path\to\blood_smear.jpg --no-save
```

### 5) Use a custom config file

```powershell
python main.py analyze .\path\to\blood_smear.jpg --config .\config.yaml
```

### 6) Setup checks

```powershell
python main.py test-config
python main.py smoke-test
```

### 7) Version

```powershell
python main.py version
```

## Current Asset and Path Expectations

The active config expects:

- Detection model: models/yolov8s_blood.pt
- Classification model: models/efficientnet_wbc_finetuned.pt
- Stage 3 PDF directory: data/pdfs
- Stage 3 PDF sources:
  - essentials_haematology.pdf
  - consie_haematology.pdf
- Vector DB directory (generated at runtime): data/chroma_db
- Internet augmentation: enabled by default to supplement local PDFs with best-effort web evidence

## Repository Structure

```text
BSc_Thesis/
├── main.py                       # CLI entrypoint (analyze / test-config / smoke-test)
├── config.yaml                   # Single source of truth for runtime parameters
├── requirements.txt
├── README.md
│
├── src/                          # Core inference library (importable as `src.*`)
│   ├── pipeline.py               # End-to-end orchestrator (Stage 1 → 2 → 3)
│   ├── config/                   # YAML/env loader
│   ├── detection/                # YOLOv8 cell detector
│   ├── classification/           # EfficientNet WBC classifier + MC-Dropout
│   ├── rag/                      # PDF ingest, retriever, GPT-4o reasoner
│   └── utils/                    # Logging, validators, metrics, reproducibility
│
├── backend/                      # FastAPI HTTP layer over `src.pipeline`
│   ├── main.py                   # `app` factory + uvicorn entry
│   ├── config.py                 # CORS, upload limits
│   ├── dependencies.py           # Pipeline singleton (lazy init)
│   ├── schemas.py                # Pydantic response models
│   ├── routes/                   # /api/health, /api/samples, /api/analyze
│   └── services/                 # Bounding-box overlay renderer
│
├── frontend/                     # Vite + React + TypeScript UI
│   ├── package.json
│   ├── vite.config.ts            # Proxies /api → http://localhost:8000
│   └── src/
│       ├── api.ts                # Typed client
│       ├── types.ts              # Mirrors backend schemas
│       ├── components/           # Header, UploadCard, DetectionPanel, ...
│       └── App.tsx
│
├── models/                       # Trained checkpoints (.pt) — gitignored
├── data/pdfs/                     # Stage-3 hematology knowledge base
├── examples/                     # Scripted demos + sample_images/
├── scripts/                      # Dev utilities (diag_yolo)
├── tests/                        # pytest suite
├── Notebooks/                    # Training + evaluation notebooks
├── results/                      # Per-stage JSON artefacts (gitignored)
├── logs/                         # Pipeline log files (gitignored)
└── data/                          # Stage-3 vector store + source PDFs
```

## Web App (FastAPI + React)

The repo ships a thin web layer for interactive demos. It re-uses the same
`BloodSmearPipeline` instance the CLI uses (no duplicate model loads), and the
React frontend talks to it via a typed `/api` client.

### Run the backend (FastAPI)

```powershell
# Activate the venv first, then:
cd backend
uvicorn main:app --reload --port 8000   # http://127.0.0.1:8000
```

Key endpoints:
- `GET  /api/health`            — liveness + pipeline-warmth probe
- `GET  /api/samples`           — list bundled demo images
- `GET  /api/samples/{name}`    — preview a sample
- `POST /api/analyze`           — multipart upload (`image`), returns full pipeline output + base64 overlay
- `GET  /docs`                  — interactive OpenAPI

### Run the frontend (Vite + React + TS)

```powershell
cd frontend
npm install                            # first time only
npm run dev                            # http://localhost:5173
```

Vite proxies `/api/*` to the FastAPI server, so both run side-by-side in dev.
For production, run `npm run build` and serve `frontend/dist/` from any static
host (or behind the same origin as the API to skip CORS).

## Agentic Mode (LangChain ReAct + Grad-CAM)

The pipeline ships with two upgrades on top of the linear baseline:

1. **Grad-CAM saliency** — per-WBC class-activation overlays produced by
   [src/classification/gradcam.py](src/classification/gradcam.py). Toggle via
   `classification.gradcam.enabled: true` in `config.yaml`. The frontend
   renders them in the *Grad-CAM Saliency* card.
2. **ReAct reasoning agent** — a LangChain agent
   ([src/rag/agent.py](src/rag/agent.py)) that replaces the single-shot LLM
   call with iterative tool use. Five tools are available:
   `query_knowledge_base`, `lookup_lab_reference_ranges`,
   `interpret_differential`, `get_uncertainty_summary`, and
   `get_detection_counts`. Switch on with `reasoning.mode: agent` in
   `config.yaml`; the trace is shown in the *Agent Trace* card.

Both flags are independent — Grad-CAM works in linear mode too.

## Configuration Notes

All runtime parameters are centralized in config.yaml.

Important sections:
- detection.confidence_threshold
- classification.mc_dropout_passes
- rag.pdf_directory and rag.pdf_sources
- rag.vector_store.chromadb_path
- llm.model_name and llm.api_key_env_var
- pipeline.enable_stage1, enable_stage2, enable_stage3

To disable Stage 3 while keeping Stage 1 and Stage 2:

```yaml
pipeline:
  enable_stage3: false
```

## Outputs

When saving is enabled, the pipeline writes JSON artifacts in results/:

- stage1_detection.json
- stage2_classification.json
- stage3_reasoning.json (only if Stage 3 enabled)

Logs are written to logs/ with timestamped filenames.

## Troubleshooting

### OPENAI_API_KEY not found

- Ensure .env exists and contains OPENAI_API_KEY=...
- Or set it directly in your shell before running

### Model file not found

- Place checkpoints in models/
- Confirm filenames match config.yaml exactly
- Optional override: set THESIS_MODELS_DIR

### Stage 3 PDF error

- Ensure data/pdfs exists
- Ensure all files listed in rag.pdf_sources are present
- If you want to run without Stage 3, set pipeline.enable_stage3: false

### Web retrieval warning

- The retriever now supplements PDFs with best-effort web evidence when internet augmentation is enabled.
- If your network is unavailable, the pipeline continues with local evidence and logs a warning.

### Batch wildcard issues in PowerShell

PowerShell wildcard expansion can vary by context. If needed, pass explicit file paths:

```powershell
python main.py analyze .\images\a.jpg .\images\b.jpg --batch
```

## Development and Testing

Run test suite:

```powershell
python -m pytest -q
```

Use smoke-test after config or asset changes:

```powershell
python main.py smoke-test
```

## Safety Notice

This is research software for decision support and experimentation.

- Not a certified medical device
- Not for standalone diagnosis
- Human expert review is required, especially when uncertainty flags are present

## License

MIT License.
