# Blood Smear Domain Expert

A safety-aware, three-stage pipeline that combines computer vision, retrieval-augmented generation, and an agentic LLM reasoner to produce clinician-style interpretations of peripheral blood smear images. Built for a BSc thesis on **trustworthy multimodal AI in hematology**.

> **Not a medical device.** Outputs are advisory and intended for research, education, and decision support. Every result includes uncertainty signals and a `requires_expert_review` flag.

---

## Table of Contents

1. [What it does](#what-it-does)
2. [Architecture at a glance](#architecture-at-a-glance)
3. [Models and data](#models-and-data)
4. [Repository layout](#repository-layout)
5. [Prerequisites](#prerequisites)
6. [First-time setup](#first-time-setup)
7. [Daily workflow](#daily-workflow)
8. [CLI reference](#cli-reference)
9. [Web app (FastAPI + React)](#web-app-fastapi--react)
10. [Configuration guide](#configuration-guide)
11. [Output schema](#output-schema)
12. [Testing and validation](#testing-and-validation)
13. [Troubleshooting](#troubleshooting)
14. [Design decisions and trade-offs](#design-decisions-and-trade-offs)
15. [Project conventions](#project-conventions)
16. [Safety, ethics, and limitations](#safety-ethics-and-limitations)
17. [License](#license)

---

## What it does

Given a peripheral blood smear photomicrograph, the system answers three questions:

| Stage | Question | Technology | Output |
|-------|----------|------------|--------|
| **1. Detection** | *Where are the cells?* | YOLOv8s, fine-tuned on BCCD | Bounding boxes for WBC, RBC, Platelet |
| **2. Classification** | *What WBC subtype is each one, and how confident are we?* | EfficientNet-B0 + Monte Carlo Dropout + Grad-CAM | Per-cell class, entropy, margin, saliency heatmap |
| **3. Reasoning** | *What does this mean clinically, and what should be reviewed?* | RAG over hematology textbooks + GPT-4o agent (LangChain ReAct) | Grounded interpretation, differential diagnosis, safety flags |

The whole chain is **configuration-driven** ([config.yaml](config.yaml)) and **reproducible** (every run emits structured JSON with metadata).

---

## Architecture at a glance

```
                    ┌─────────────────────────────────────────────┐
                    │                  config.yaml                │
                    │       (single source of truth, YAML)        │
                    └──────────────────────┬──────────────────────┘
                                           │
                                           ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌────────────────────┐
│   Image      │──▶│  Stage 1     │──▶│  Stage 2     │──▶│  Stage 3           │
│  (.jpg/.png) │   │  YOLOv8      │   │ EfficientNet │   │  RAG + LLM Agent   │
└──────────────┘   │  detector    │   │  + MC-Dropout│   │  (LangChain ReAct) │
                   └──────────────┘   │  + Grad-CAM  │   └────────┬───────────┘
                                      └──────────────┘            │
                                                                  ▼
                                                       ┌──────────────────────┐
                                                       │ ChromaDB vector store│
                                                       │ (1,049 PDF chunks,   │
                                                       │  MiniLM embeddings)  │
                                                       └──────────────────────┘

                                  ┌────────────────────────────────────┐
                                  │  Two consumers of src.pipeline:    │
                                  │  • main.py             (CLI)       │
                                  │  • backend/main.py     (FastAPI)   │
                                  └────────────────────────────────────┘
                                                  │
                                                  ▼
                                       ┌──────────────────┐
                                       │  React frontend  │
                                       │ (Vite + TS, /api)│
                                       └──────────────────┘
```

**Key property:** [src/pipeline.py](src/pipeline.py) is the single orchestrator. The CLI and the HTTP server both import the same `BloodSmearPipeline` class — there is no duplicated inference code, no double model load.

---

## Models and data

### Stage 1 — Detection (YOLOv8s)

- **Checkpoint**: [models/yolov8s_blood.pt](models/yolov8s_blood.pt) (~22 MB)
- **Source**: Ultralytics YOLOv8s, fine-tuned on the BCCD dataset (Blood Cell Count and Detection).
- **Classes**: `WBC`, `RBC`, `Platelet`
- **Default confidence threshold**: `0.50` ([config.yaml](config.yaml) → `detection.confidence_threshold`)
- **Hardware**: auto-selects CUDA if available, falls back to CPU.

### Stage 2 — Classification (EfficientNet-B0)

- **Checkpoint**: [models/efficientnet_wbc_finetuned.pt](models/efficientnet_wbc_finetuned.pt) (~16 MB)
- **Architecture**: `timm/efficientnet_b0` with a custom 8-class head, fine-tuned on the **PBC dataset** (Acevedo *et al.*, Mendeley Data — peripheral blood cells).
- **Classes (8)**: `basophil`, `eosinophil`, `erythroblast`, `ig` (immature granulocyte), `lymphocyte`, `monocyte`, `neutrophil`, `platelet`
- **Uncertainty quantification**: **Monte Carlo Dropout** with 20 stochastic forward passes per cell. Produces:
  - `entropy` — predictive uncertainty across passes
  - `margin` — gap between top-1 and top-2 class probability
  - `bucket` ∈ {`low`, `medium`, `high`} per the thresholds in `config.yaml` → `classification.uncertainty`
- **Interpretability**: **Grad-CAM** on the last MBConv block — see [src/classification/](src/classification/). Each WBC gets a heatmap overlay so reviewers can verify the model attended to the cell, not artefacts.

### Stage 3 — Reasoning (RAG + GPT-4o)

- **Knowledge base**: two open hematology textbooks under [data/pdfs/](data/pdfs/):
  - `essentials_haematology.pdf` (505 pages → 996 chunks)
  - `consie_haematology.pdf` (27 pages → 53 chunks)
  - Total: **1,049 chunks** at 200 words / 40-word overlap.
- **Embedder**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, 256-token window).
- **Vector store**: ChromaDB (cosine distance), persisted to `data/chroma_db/` (~29 MB after first build).
- **LLM**: OpenAI **GPT-4o** at `temperature=0.1` with structured JSON output ([src/rag/llm_reasoner.py](src/rag/llm_reasoner.py)).
- **Optional internet augmentation**: domain-allowlisted web retrieval (NCBI, PubMed, MedlinePlus, WHO, NIH, hematology.org). Disabled by default; enable via `config.yaml` → `rag.internet.enabled`.
- **Agentic mode** (`reasoning.mode: agent`): a **LangChain ReAct agent** (`langgraph.prebuilt.create_react_agent`) with five tools:
  1. `query_knowledge_base` — semantic search over the textbook corpus
  2. `lookup_lab_reference_ranges` — normal value tables for CBC analytes
  3. `interpret_differential` — heuristics over WBC class proportions
  4. `get_uncertainty_summary` — surfaces high-uncertainty cells
  5. `get_detection_counts` — exposes Stage 1 totals to the agent

  The agent loops Thought → Action → Observation up to `agent.max_iterations: 6` before producing the final JSON. The full trace is returned and rendered in the frontend's *Agent Trace* card.

### Datasets used during training (notebooks)

- **TBL-PBC** combined dataset for detection ([Notebooks/YOLOv8_detection/](Notebooks/YOLOv8_detection/)).
- **PBC** (Acevedo et al.) for classification ([Notebooks/Efficientnet_classification/](Notebooks/Efficientnet_classification/)).
- **RAG / LLM ablations** ([Notebooks/LLM_RAG_Pipline/](Notebooks/LLM_RAG_Pipline/)).

Inference does **not** require the training datasets — the checkpoints under `models/` are self-contained.

---

## Repository layout

```
BSc_Thesis/
├── main.py                 # CLI entrypoint:   analyze, smoke-test, test-config
├── config.yaml             # Single source of truth for all runtime parameters
├── requirements.txt        # Python deps (one venv for the whole project)
├── .env / .env.example     # OPENAI_API_KEY (gitignored)
├── README.md               # ← you are here
│
├── src/                    # Core inference library — importable as `src.*`
│   ├── pipeline.py         # End-to-end orchestrator (Stage 1 → 2 → 3)
│   ├── config/             # YAML + .env loader, validation
│   ├── detection/          # YOLOv8 wrapper
│   ├── classification/     # EfficientNet, MC-Dropout, Grad-CAM
│   ├── rag/                # PDF processor, retriever, reasoner, ReAct agent
│   └── utils/              # Logging, validators, metrics, helpers
│
├── backend/                # FastAPI HTTP layer over src.pipeline
│   ├── main.py             # `app` factory + uvicorn entry
│   ├── version.py          # __version__ string
│   ├── config.py           # Web-only settings (CORS, upload limits)
│   ├── dependencies.py     # Lazy pipeline singleton
│   ├── schemas.py          # Pydantic request/response DTOs
│   ├── routes/             # /api/health, /api/samples, /api/analyze
│   └── services/           # Bounding-box overlay renderer
│
├── frontend/               # Vite + React + TypeScript SPA
│   ├── package.json
│   ├── vite.config.ts      # Proxies /api → http://localhost:8000
│   └── src/
│       ├── api.ts          # Typed fetch client
│       ├── types.ts        # Mirrors backend Pydantic schemas
│       ├── components/     # Header, UploadCard, DetectionPanel,
│       │                   # ClassificationPanel, GradCamPanel,
│       │                   # ReasoningPanel, AgentTracePanel
│       └── App.tsx
│
├── models/                 # Trained .pt checkpoints (gitignored — too large)
├── data/
│   ├── pdfs/               # Stage-3 hematology textbooks
│   └── chroma_db/          # Persisted vector store (built on first run)
├── examples/               # Scripted demos + sample_images/
├── scripts/                # Dev utilities (currently: diag_yolo.py)
├── tests/                  # pytest suite (13 tests)
├── Notebooks/              # Training + evaluation notebooks (not used at inference)
├── results/                # Per-run JSON artefacts (gitignored)
├── logs/                   # Timestamped log files (gitignored)
└── figures/                # Optional matplotlib outputs
```

### Why `.env`, `requirements.txt`, and `src/rag/` live at the repo root (not under `backend/`)

Because the **backend is one of three consumers** of the same library:

| Consumer | Entry point | Imports |
|----------|-------------|---------|
| CLI | `python main.py analyze ...` | `src.pipeline` |
| HTTP | `uvicorn main:app` (in `backend/`) | `src.pipeline` |
| Tests / notebooks | `pytest`, Jupyter | `src.pipeline`, `src.rag.*` |

A separate `backend/.env` or `backend/requirements.txt` would mean the CLI and tests can't find the API key or the dependencies. **One project, one venv, one `.env`, one `requirements.txt`** — and `backend/` contains *only* HTTP-layer code.

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | **3.10** (3.8+ minimum) | Pipeline + backend |
| Node.js | 18+ (LTS) | Frontend dev server |
| npm | 9+ | Frontend package manager |
| Git | any recent | Version control |
| OpenAI API key | — | Stage 3 reasoning (GPT-4o) |
| Internet | — | First-time `pip install`, OpenAI API calls; optional Stage-3 web augmentation |
| Disk | ~3 GB free | venv, models, vector store |
| RAM | 8 GB | CPU inference is comfortable; 16 GB recommended for parallel batch runs |
| GPU | Optional | CUDA accelerates Stages 1 + 2 by ~10× |

> Windows PowerShell is the primary supported shell. The instructions also work in `bash` on macOS / Linux with the obvious path-separator changes.

---

## First-time setup

```powershell
# 1) Open the repo
cd C:\Users\qkafr\Desktop\BSc_Thesis

# 2) Create the Python virtual environment (one-time, repo-root only)
py -3.10 -m venv .venv

# 3) Activate it (do this in EVERY new Python terminal you open)
.\.venv\Scripts\Activate.ps1
# Your prompt should now start with (.venv)

# 4) Install all Python dependencies
pip install -r requirements.txt

# 5) Configure secrets
Copy-Item .env.example .env
notepad .env
#   → set OPENAI_API_KEY=sk-...

# 6) Verify model checkpoints exist (already in repo for the thesis defence)
#    models/yolov8s_blood.pt
#    models/efficientnet_wbc_finetuned.pt

# 7) Verify Stage-3 PDFs exist
#    data/pdfs/essentials_haematology.pdf
#    data/pdfs/consie_haematology.pdf

# 8) Smoke-test the whole chain (also builds the ChromaDB index on first run)
python main.py test-config
python main.py smoke-test

# 9) Install frontend deps (Node, separate from the Python venv)
cd frontend
npm install
cd ..
```

After step 8 you should see:

```
✓ Configuration loaded successfully
✓ Stage 1 (Detection) ready
✓ Stage 2 (Classification) ready
✓ Stage 3 (RAG + Reasoning) ready
```

The first Stage 3 invocation builds the ChromaDB index (~60 s on CPU). Subsequent runs reuse the persisted store — typical end-to-end latency on a sample image is **8–12 s**.

---

## Daily workflow

You'll typically need **two terminals**: one for the backend, one for the frontend.

### Terminal 1 — backend (FastAPI on :8000)

```powershell
cd C:\Users\qkafr\Desktop\BSc_Thesis
.\.venv\Scripts\Activate.ps1
cd backend
uvicorn main:app --reload --port 8000
```

Wait for `Uvicorn running on http://127.0.0.1:8000`. The first request lazily warms the pipeline (one-time ~10 s); subsequent requests are fast.

### Terminal 2 — frontend (Vite on :5173)

```powershell
cd C:\Users\qkafr\Desktop\BSc_Thesis\frontend
npm run dev
```

Open **http://localhost:5173**. The header health pill should turn green (`pipeline_ready: true`) within ~10 s.

### Mental model: when to activate the venv

- **Activate at the repo root** (`.\.venv\Scripts\Activate.ps1`).
- **Once activated, every `cd` in that terminal still uses the venv.** You don't re-activate when you `cd backend`.
- **Frontend terminals don't need the venv** — `npm` is Node, not Python.

---

## CLI reference

All CLI commands live in [main.py](main.py).

```powershell
# Setup checks
python main.py test-config              # Validate config.yaml + asset paths
python main.py smoke-test               # End-to-end check on a bundled image
python main.py version                  # Print pipeline version

# Single image
python main.py analyze .\examples\sample_images\BCCD_00100.jpg

# Multiple images as a batch
python main.py analyze .\img1.jpg .\img2.jpg .\img3.jpg --batch

# Verbose logging
python main.py analyze .\img.jpg --verbose

# Don't write results/*.json (useful for ad-hoc tests)
python main.py analyze .\img.jpg --no-save

# Custom config
python main.py analyze .\img.jpg --config .\config.alt.yaml
```

> PowerShell wildcards behave inconsistently — pass explicit paths if a glob like `images\*.jpg` doesn't expand.

---

## Web app (FastAPI + React)

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/health` | Liveness + pipeline-warmth probe |
| `GET` | `/api/samples` | List bundled demo images |
| `GET` | `/api/samples/{name}` | Stream a sample image |
| `POST` | `/api/analyze` | Multipart upload (`image`), returns full pipeline output + base64 overlay |
| `GET` | `/docs` | Swagger UI (auto-generated) |
| `GET` | `/openapi.json` | OpenAPI 3 spec |

### Frontend cards

The SPA renders five panels stacked top-to-bottom:

1. **Detection** — annotated overlay (boxes color-coded per class) + per-class count chips.
2. **Classification** — bar chart of WBC class proportions, per-cell rows with confidence and uncertainty bucket.
3. **Grad-CAM Saliency** — thumbnail grid of heatmap-overlaid WBC crops (interpretability check).
4. **Reasoning** — final markdown answer + grounded citations + safety flags + `requires_expert_review` badge.
5. **Agent Trace** — collapsible list of ReAct steps (Thought / Tool call / Observation / Final). Only populated in `reasoning.mode: agent`.

A *Raw response (debug)* `<details>` block at the bottom shows the JSON contract with base64 image fields elided so the browser doesn't choke on multi-megabyte strings.

### Production build

```powershell
cd frontend
npm run build      # outputs frontend/dist/
```

Serve `frontend/dist/` from any static host. To skip CORS, host it behind the same origin as the API (e.g. behind nginx, or have FastAPI mount it as a static directory).

---

## Configuration guide

[config.yaml](config.yaml) is the single source of truth. The most-used knobs:

| Section | Key | Default | What it controls |
|---------|-----|---------|------------------|
| `models` | `yolo_detection` | `models/yolov8s_blood.pt` | Stage-1 weights |
| `models` | `efficientnet_classification` | `models/efficientnet_wbc_finetuned.pt` | Stage-2 weights |
| `detection` | `confidence_threshold` | `0.50` | Min YOLO confidence |
| `detection` | `device` | `auto` | `auto`, `cpu`, `cuda`, or device id |
| `classification` | `mc_dropout_passes` | `20` | MC-Dropout stochastic passes |
| `classification` | `uncertainty.low.min_confidence` | `0.85` | Threshold for `low` bucket |
| `classification` | `gradcam.enabled` | `true` | Compute Grad-CAM heatmaps |
| `classification` | `gradcam.alpha` | `0.45` | Heatmap blend strength |
| `reasoning` | `mode` | `agent` | `linear` (single-shot) or `agent` (ReAct) |
| `reasoning` | `agent.max_iterations` | `6` | Cap on ReAct loop |
| `rag` | `pdf_directory` | `data/pdfs` | Source folder for textbooks |
| `rag` | `chunking.chunk_size` | `200` | Words per chunk |
| `rag` | `embedding.model_name` | `sentence-transformers/all-MiniLM-L6-v2` | Embedder |
| `rag` | `vector_store.chromadb_path` | `data/chroma_db` | On-disk vector store |
| `rag` | `retrieval.top_k` | `5` | Chunks per query |
| `rag` | `internet.enabled` | `false` | Web augmentation (allowlisted) |
| `llm` | `model_name` | `gpt-4o` | OpenAI model |
| `llm` | `temperature` | `0.1` | Low for clinical consistency |
| `llm` | `safety.require_citations` | `true` | Enforce grounded claims |
| `pipeline` | `enable_stage1/2/3` | `true` | Toggle entire stages |
| `pipeline` | `continue_on_error` | `true` | Don't abort on single-stage failure |

### Common configuration recipes

**Run only Stage 1 + 2 (no LLM cost):**
```yaml
pipeline:
  enable_stage3: false
```

**Switch from agentic back to linear reasoning (faster, cheaper, less explainable):**
```yaml
reasoning:
  mode: linear
```

**Use a different OpenAI-compatible endpoint** — set `OPENAI_BASE_URL` in `.env` and change `llm.model_name` (works with Azure OpenAI, vLLM, Groq's OpenAI shim, etc.).

---

## Output schema

A successful `/api/analyze` (or `python main.py analyze ...`) returns:

```jsonc
{
  "metadata": {
    "version": "1.0.0",
    "timestamp": "2026-04-27T23:33:08Z",
    "duration_seconds": 8.60,
    "config_hash": "..."
  },
  "stage1_detection": {
    "per_image": [{
      "image": "upload.jpg",
      "boxes": [
        {"class": "WBC", "confidence": 0.93, "xyxy": [120, 88, 240, 210]}
      ]
    }],
    "totals": {"WBC": 2, "RBC": 17, "Platelet": 0}
  },
  "stage2_classification": {
    "predictions": [
      {
        "class": "neutrophil",
        "confidence": 0.91,
        "entropy": 0.18,
        "margin": 0.62,
        "uncertainty_bucket": "low",
        "gradcam_base64": "iVBORw0K..."
      }
    ],
    "distribution": {"neutrophil": 0.5, "monocyte": 0.5}
  },
  "stage3_reasoning": {
    "reasoning_mode": "agent",
    "interpretation": "Markdown text...",
    "differential_diagnosis": ["...", "..."],
    "safety_flags": ["High uncertainty in AI analysis"],
    "references": [
      {"source": "essentials_haematology.pdf", "page": 142, "snippet": "..."}
    ],
    "requires_expert_review": true,
    "agent_trace": [
      {"type": "thought",     "content": "I need to check..."},
      {"type": "tool_call",   "tool": "query_knowledge_base", "input": "..."},
      {"type": "tool_result", "content": "..."}
    ]
  },
  "annotated_image_base64": "iVBORw0K..."
}
```

---

## Testing and validation

```powershell
# Full pytest suite (13 tests, ~20 s)
pytest -q

# Specific test file
pytest tests/test_pipeline_batch_logic.py -v

# Smoke-test the live pipeline (uses examples/sample_images/)
python main.py smoke-test

# Frontend type-check + production build
cd frontend
npm run build
```

The test suite covers:

- Config + model path validation
- `.env` loading
- Pipeline batch logic (multi-image runs)
- LLM reasoner JSON parsing (with malformed-input regression cases)
- Retriever hybrid mode (PDF-only / PDF + ChromaDB / fallback)
- Uncertainty schema correctness

---

## Troubleshooting

### `OPENAI_API_KEY not found` on Stage 3

- Ensure `.env` exists at the **repo root** (not under `backend/`) and contains `OPENAI_API_KEY=sk-...`.
- Verify it loads:
  ```powershell
  python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(bool(os.getenv('OPENAI_API_KEY')))"
  ```
  Should print `True`.

### `Address already in use` on port 8000

Some other process is holding the socket (sometimes a ghost PID after a crash).
- Check: `Get-NetTCPConnection -LocalPort 8000 -State Listen`.
- Kill it: `Stop-Process -Id <PID> -Force`.
- If the kernel won't release it (rare), reboot — or just use another port: `uvicorn main:app --reload --port 8002` and set `$env:VITE_BACKEND_URL = "http://localhost:8002"` before `npm run dev`.

### `ModuleNotFoundError: No module named 'src'` from inside `backend/`

`backend/main.py` injects the repo root into `sys.path` automatically. If you see this, you're probably running uvicorn with the **old** `backend.main:app` form. Use one of:

```powershell
# Preferred
cd backend; uvicorn main:app --reload --port 8000

# Or from repo root
uvicorn --app-dir backend main:app --reload --port 8000
```

### Frontend says "Backend unreachable"

- Backend isn't running, *or* it's on a non-default port. Verify with `Invoke-WebRequest http://127.0.0.1:8000/api/health`.
- Vite dev server caches its proxy config — restart `npm run dev` after editing [frontend/vite.config.ts](frontend/vite.config.ts).

### Stage 3 takes ~60 s on first run

That's the one-time ChromaDB index build over the two PDFs (1,049 chunks × MiniLM encoding). Subsequent runs reuse `data/chroma_db/` and reasoning is ~2–8 s.

### Grad-CAM panel is empty

- `classification.gradcam.enabled: false` in `config.yaml` — flip it to `true`.
- Stage 1 detected zero WBCs — check the *Detection* card.

### Agent trace is empty

- `reasoning.mode: linear` in `config.yaml` — set to `agent`.
- Agent ran but produced no intermediate tool calls (rare; the model went straight to the final answer). The empty-state message says exactly this.

### Web retrieval warning in logs

`rag.internet.enabled: true` and the network is down or a domain isn't on the allowlist. The pipeline degrades gracefully to PDF-only retrieval.

### PowerShell blocks `Activate.ps1`

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

---

## Design decisions and trade-offs

These are the choices a reviewer is most likely to ask about.

**1. Why three stages instead of an end-to-end model?**
Each stage uses the right tool for its sub-problem (object detection ≠ fine-grained classification ≠ language reasoning) and produces a verifiable artefact. The system stays **inspectable**: a reviewer can see exactly which cell the LLM was reasoning about, with what confidence, and which textbook page grounded each claim.

**2. Why MC-Dropout for uncertainty?**
It's a Bayesian approximation that requires **no architectural change** — just keeping dropout active at inference and averaging over N stochastic forward passes. It captures epistemic uncertainty (model "doesn't know") which is the kind we care about for safety flagging.

**3. Why an agent instead of a single LLM call?**
A single-shot RAG call answers *what* but rarely *why*, and never asks for clarifying evidence. The ReAct agent decomposes the problem: it can call `lookup_lab_reference_ranges` for a numeric anchor, *then* `query_knowledge_base` for context, *then* `interpret_differential` to sanity-check class proportions. The trace is rendered to the user — **explainability comes for free**.

**4. Why ChromaDB and not FAISS?**
Persistence and metadata filtering. ChromaDB's on-disk store survives process restarts, supports per-chunk source/page metadata for grounded citations, and offers cosine distance out of the box. FAISS would be faster at >1 M chunks; we have ~1 K.

**5. Why a separate `backend/` module instead of mounting FastAPI inside `src/`?**
Separation of concerns. `src/` contains domain logic and is testable / Jupyter-importable without spinning up a web server. `backend/` is a thin HTTP adapter — schemas, routes, DI. Either layer can be replaced (e.g. swap FastAPI for gRPC) without touching the other.

**6. Why one venv at the repo root, not per-component?**
The CLI, the API, the tests, and the notebooks all import the same `src.*` library. Multiple venvs would force the requirements to be kept in sync manually, slow down installs, and make `pytest` fragile. The convention "one Python project, one venv at the repo root" is the boring, correct default.

**7. Why GPT-4o specifically?**
Higher quality than open-weight alternatives on clinical reasoning benchmarks at the time of writing, and the JSON-mode + function-calling support is rock-solid for the agent's tool-use loop. The system is **provider-agnostic** at the config level — switch `llm.model_name` and set `OPENAI_BASE_URL` to use any OpenAI-compatible endpoint.

---

## Project conventions

- **Code style**: `black` (88 cols) + `flake8`.
- **Type hints**: pervasive in `src/` and `backend/`; less strict in notebooks.
- **Imports**: backend uses **flat imports** (`from config import ...`) so `cd backend; uvicorn main:app` works without packaging gymnastics. The repo root is added to `sys.path` by [backend/main.py](backend/main.py) so `from src.pipeline import ...` still resolves.
- **Logging**: stdlib `logging`, configured by [src/utils/logging_config.py](src/utils/logging_config.py). Logs go to both console and `logs/pipeline_<timestamp>.log`.
- **Errors**: typed exceptions in `src/`; HTTP errors translated to FastAPI `HTTPException` at the route boundary.
- **Reproducibility**: `metadata.config_hash` and `metadata.version` are emitted on every run.
- **Secrets**: `.env` only, never committed. `python-dotenv` loads it from the repo root.
- **Gitignore**: `models/*.pt`, `data/chroma_db/`, `results/`, `logs/`, `__pycache__/`, `node_modules/`, `frontend/dist/`, `.venv/`.

---

## Safety, ethics, and limitations

- **Not a medical device.** No regulatory clearance, not validated for diagnostic use.
- **Distribution shift.** Stage 1 was trained on BCCD; Stage 2 on PBC. Performance on smears from other staining protocols, microscopes, or magnifications is unverified.
- **Class imbalance.** The `erythroblast` and `basophil` classes are minority in the training data — predictions on these are correspondingly less reliable, which is reflected in the uncertainty scores.
- **LLM hallucinations.** Mitigated by RAG grounding, citation enforcement (`llm.safety.require_citations: true`), and the `requires_expert_review` flag, but not eliminated. Always verify against the cited sources.
- **PHI.** The system does not store uploaded images beyond the temp directory of a single request. Do not upload images containing patient identifiers.
- **OpenAI data policy.** Image content used in Stage 3 prompts (textual descriptors only — actual pixels never leave the local machine) is subject to OpenAI's API data policy. Disable Stage 3 for sensitive workloads.

---

## License

MIT. See `LICENSE` if present, otherwise the MIT terms apply by default for thesis-defence purposes.

---

## Citation

If this code informs your work, please cite the BSc thesis:

```bibtex
@misc{bloodsmear_domain_expert_2026,
  title  = {Blood Smear Domain Expert: A Safety-Aware Multimodal Pipeline
            for Peripheral Blood Smear Analysis},
  author = {<author>},
  year   = {2026},
  note   = {BSc Thesis}
}
```
