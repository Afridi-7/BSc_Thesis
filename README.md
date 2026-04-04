# Blood Smear Domain Expert 🔬

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A clinically safer AI assistant for peripheral blood smear interpretation** combining computer vision and retrieval-augmented clinical reasoning.

End-to-end BSc thesis repository implementing a hematology decision-support pipeline with three integrated stages:

1. **Stage 1**: YOLOv8 blood cell detection (WBC, RBC, Platelet)
2. **Stage 2**: EfficientNet WBC subtype classification with Monte Carlo Dropout uncertainty quantification
3. **Stage 3**: RAG + LLM (GPT-4o) clinical reasoning with safety constraints and evidence-grounded abstention

## 🎯 Key Features

- ✅ **Production-Ready Code**: Clean, modular Python architecture (no notebooks required)
- ✅ **Safety-First Design**: 3-tier uncertainty propagation → explicit abstention when evidence is weak
- ✅ **Configuration-Driven**: All parameters externalized in `config.yaml`
- ✅ **Comprehensive Logging**: Structured logging with file and console output
- ✅ **Easy CLI Interface**: Simple command-line usage with rich output formatting
- ✅ **Full Documentation**: Every function documented with examples
- ✅ **Reproducible**: Complete dependency management and metadata tracking

## 📋 Research Objective

Build a clinically safer AI assistant for peripheral blood smear interpretation by:

1. **Detecting** cell categories (WBC, RBC, Platelet) with YOLOv8
2. **Classifying** WBC subtypes with calibrated uncertainty signals (Monte Carlo Dropout)
3. **Reasoning** with evidence-grounded clinical interpretations, citations, and abstention when evidence is weak

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, but recommended)
- OpenAI API key for GPT-4o

### Installation

```bash
# Clone repository
cd BSc_Thesis

# Install dependencies
pip install -r requirements.txt

# Create local environment file (recommended)
copy .env.example .env  # Windows
# cp .env.example .env  # Linux/Mac

# Edit .env and set:
# OPENAI_API_KEY=your-api-key-here

# Set OpenAI API key
$env:OPENAI_API_KEY='your-api-key-here'    # Windows PowerShell
set OPENAI_API_KEY=your-api-key-here       # Windows CMD
export OPENAI_API_KEY='your-api-key-here'  # Linux/Mac

# Verify installation
python main.py version
python main.py smoke-test
```

### Quick Test

```bash
# Analyze a single blood smear image
python main.py analyze path/to/blood_smear.jpg

# Analyze batch of images
python main.py analyze images/*.jpg --batch

# Verbose logging
python main.py analyze smear.jpg --verbose

# Quick smoke checks
python main.py smoke-test

# Unit tests
python -m pytest -q
```

---

## 📁 Repository Structure

```
BSc_Thesis/
├── src/                          # Production Python modules
│   ├── config/                   # Configuration management
│   │   ├── __init__.py
│   │   └── config_loader.py      # YAML config loading & validation
│   ├── detection/                # YOLOv8 detection module
│   │   ├── __init__.py
│   │   └── detector.py           # Cell detection implementation
│   ├── classification/           # EfficientNet classification
│   │   ├── __init__.py
│   │   └── classifier.py         # WBC classification + MC Dropout
│   ├── rag/                      # RAG components
│   │   ├── __init__.py
│   │   ├── pdf_processor.py      # PDF extraction & chunking
│   │   ├── retriever.py          # ChromaDB semantic retrieval
│   │   └── llm_reasoner.py       # GPT-4o clinical reasoning
│   ├── utils/                    # Utilities
│   │   ├── __init__.py
│   │   ├── logging_config.py     # Logging setup
│   │   ├── validators.py         # Input validation
│   │   └── metrics.py            # Uncertainty & statistics
│   └── pipeline.py               # Main orchestrator
├── config/
│   └── config.yaml               # Configuration file (ALL parameters)
├── examples/                     # Example usage scripts
│   ├── single_image_inference.py
│   ├── batch_inference.py
│   └── custom_configuration.py
├── Notebooks/                    # Original Jupyter notebooks (archived)
│   ├── YOLOv8_on_TBL_PBC_dataset.ipynb
│   ├── Efficientnet_classification.ipynb
│   └── LLM_Rag_pipline.ipynb
├── models/                       # Trained model checkpoints
│   ├── yolov8s_blood.pt
│   └── efficientnet_wbc_finetuned.pt
├── LLM_RAG_Pipline/
│   └── pdfs/                     # Hematology knowledge base PDFs
│       ├── essentials_haematology.pdf
│       ├── concise_haematology.pdf
│       └── lab_guide_hematology.pdf
├── results/                      # Analysis outputs (auto-generated)
├── figures/                      # Generated visualizations
├── logs/                         # Log files
├── main.py                       # CLI entry point
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🔧 Configuration

All parameters are externalized in `config.yaml`. Key sections:

### Detection (YOLOv8)
```yaml
detection:
  confidence_threshold: 0.25  # Detection confidence (0.0-1.0)
  device: "auto"              # 'auto', 'cpu', 'cuda', or device ID
  canonical_classes:
    - "WBC"
    - "RBC"
    - "Platelet"
```

### Classification (EfficientNet + MC Dropout)
```yaml
classification:
  mc_dropout_passes: 20  # Number of stochastic forward passes
  uncertainty:
    low:
      min_confidence: 0.85
      max_entropy: 0.3
    medium:
      min_confidence: 0.65
      max_entropy: 0.6
```

### RAG (Retrieval-Augmented Generation)
```yaml
rag:
  chunking:
    chunk_size: 500  # Words per chunk
    overlap: 50      # Word overlap
  retrieval:
    top_k: 5  # Number of chunks to retrieve
  embedding:
    model_name: "sentence-transformers/all-MiniLM-L6-v2"
```

### LLM (OpenAI GPT-4o)
```yaml
llm:
  model_name: "gpt-4o"
  temperature: 0.1  # Low for consistent medical reasoning
  safety:
    enable_abstention: true
    propagate_vision_uncertainty: true
    require_citations: true
```

---

## 📖 Usage Examples

### Python API

```python
from src.pipeline import BloodSmearPipeline

# Initialize pipeline
pipeline = BloodSmearPipeline()

# Analyze single image
results = pipeline.analyze('blood_smear.jpg')

# Access results
print(results['stage1_detection']['total_counts'])
print(results['stage2_classification']['summary'])
print(results['stage3_reasoning']['clinical_interpretation'])
```

### Command Line

```bash
# Single image analysis
python main.py analyze smear.jpg

# Batch processing
python main.py analyze images/*.jpg --batch

# Custom config
python main.py analyze smear.jpg --config custom_config.yaml

# Test configuration
python main.py test-config

# Smoke test setup and asset paths
python main.py smoke-test

# Verbose output
python main.py analyze smear.jpg --verbose
```

### Example Scripts

See `examples/` directory:
- `single_image_inference.py` - Basic single-image analysis
- `batch_inference.py` - Batch processing with statistics
- `custom_configuration.py` - Using custom parameters

---

## 🏗️ System Architecture

### Three-Stage Pipeline

```
Input Image(s)
      ↓
┌─────────────────────────────────────┐
│ Stage 1: YOLOv8 Detection           │
│  - Detect WBC, RBC, Platelet        │
│  - Extract bounding boxes           │
│  - Count cells by type              │
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│ Stage 2: EfficientNet + MC Dropout  │
│  - Classify WBC subtypes            │
│  - 20 stochastic forward passes     │
│  - Calculate uncertainty metrics    │
│  - Flag high-uncertainty cells      │
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│ Stage 3: RAG + LLM Reasoning        │
│  - Build query from vision results  │
│  - Retrieve evidence from PDFs      │
│  - Generate clinical reasoning      │
│  - Apply safety constraints         │
│  - Abstain if evidence insufficient │
└─────────────────────────────────────┘
      ↓
JSON Output + Clinical Report
```

### Safety Mechanisms

**3-Tier Uncertainty Propagation:**

1. **Vision Layer** (Stage 2)
   - Monte Carlo Dropout uncertainty quantification
   - Flags cells with high entropy/low confidence

2. **Retrieval Layer** (Stage 3)
   - Checks retrieval quality
   - Sets `INSUFFICIENT_EVIDENCE` flag if < 80 chars retrieved

3. **Reasoning Layer** (Stage 3)
   - Propagates uncertainty flags to LLM prompt
   - LLM trained to abstain when evidence is weak
   - Explicit safety flags in output

**Example Safety Flow:**
```
High entropy cell (0.8) detected
  → Flag set in uncertainty_summary
  → RAG query includes "uncertain morphology"
  → Retrieved chunks on differential diagnoses
  → OpenAI prompted with "HIGH_UNCERTAINTY flag"
  → Output includes "requires_expert_review": true
```

---

## 🔬 Technical Details

### Stage 1: Cell Detection
- **Model**: YOLOv8 small (`yolov8s_blood.pt`)
- **Classes**: WBC, RBC, Platelet
- **Dataset**: TXL-PBC public dataset
- **Confidence Threshold**: 0.25 (configurable)

### Stage 2: WBC Classification
- **Model**: EfficientNet-B0 with custom classifier head
- **Classes**: 8 WBC subtypes (basophil, eosinophil, erythroblast, ig, lymphocyte, monocyte, neutrophil, platelet)
- **Uncertainty**: Monte Carlo Dropout (20 passes)
- **Metrics**: Confidence, entropy, variance
- **Thresholds**: LOW (conf≥0.85, H<0.3), MEDIUM (conf≥0.65, H<0.6), HIGH (flagged)

### Stage 3: RAG + LLM
- **Embedding**: sentence-transformers/all-MiniLM-L6-v2 (384-dim)
- **Vector Store**: ChromaDB (with numpy fallback)
- **Chunking**: 500 words, 50-word overlap
- **Retrieval**: Top-5 cosine similarity
- **LLM**: OpenAI GPT-4o (temperature=0.1)
- **Knowledge Base**: 3 hematology PDFs (~200+ chunks)

### Dependencies
- **PyTorch**: 2.0+ (CUDA support recommended)
- **Ultralytics**: YOLOv8 implementation
- **timm**: EfficientNet implementation
- **ChromaDB**: Vector database
- **OpenAI**: GPT-4o API
- **sentence-transformers**: Embedding models

Full list: see `requirements.txt`

---

## 📊 Output Format

### JSON Structure
```json
{
  "stage1_detection": {
    "total_counts": {"WBC": 15, "RBC": 120, "Platelet": 40},
    "cell_count_stats": {...},
    "per_image": [...]
  },
  "stage2_classification": {
    "predictions": [...],
    "summary": {
      "sample_count": 15,
      "class_distribution": {"neutrophil": 8, "lymphocyte": 5, ...},
      "flagged_count": 2,
      "requires_expert_review": true
    },
    "uncertainty_summary": {...}
  },
  "stage3_reasoning": {
    "clinical_interpretation": "...",
    "key_findings": ["...", "..."],
    "differential_diagnoses": ["... [Reference 1]", "..."],
    "recommendations": ["...", "..."],
    "safety_flags": ["HIGH_UNCERTAINTY"],
    "requires_expert_review": true,
    "citations_used": [1, 2, 3]
  },
  "metadata": {
    "timestamp": "2024-01-01T12:00:00",
    "execution_time_seconds": 45.2
  }
}
```

---

## ⚙️ Environment Setup

### Required Environment Variables
```bash
# OpenAI API key (REQUIRED)
$env:OPENAI_API_KEY='sk-...'           # Windows PowerShell
set OPENAI_API_KEY=sk-...              # Windows CMD
export OPENAI_API_KEY='sk-...'         # Linux/Mac

# Optional: Custom model directory
export THESIS_MODELS_DIR='/path/to/models'

# Optional: Custom config directory
export THESIS_CONFIG_DIR='/path/to/config'
```

### Model Files
Place trained models in `models/` directory:
- `yolov8s_blood.pt` - YOLOv8 detection weights
- `efficientnet_wbc_finetuned.pt` - EfficientNet classification weights

Model resolution fallback order:
1. Config path (`models.yolo_detection`, `models.efficientnet_classification`)
2. `THESIS_MODELS_DIR` + model filename
3. `models/` + model filename

### PDF Knowledge Base
Place hematology PDFs in `LLM_RAG_Pipline/pdfs/`:
- `essentials_haematology.pdf`
- `concise_haematology.pdf`
- `lab_guide_hematology.pdf`

Missing PDF strategy is configurable via `rag.pdf_missing_strategy`:
- `fail` (default): stop with actionable error
- `warn`: continue with available PDFs
- `download`: fetch missing PDFs from `rag.pdf_download_base_url`

---

## 🧪 Testing & Validation

### Test Configuration
```bash
python main.py test-config
```

### Smoke Test
```bash
python main.py smoke-test
```

### Unit Tests
```bash
python -m pytest -q
```

### Run Examples
```bash
# From examples directory
cd examples
python single_image_inference.py
python batch_inference.py
python custom_configuration.py
```

---

## ⚠️ Troubleshooting

- `OpenAI API key not found`:
  - Add `OPENAI_API_KEY=...` to `.env` or set the environment variable in your shell.
- `Model file not found`:
  - Ensure checkpoints exist in `models/` or set `THESIS_MODELS_DIR`.
- `Missing required PDF files`:
  - Add PDFs under `LLM_RAG_Pipline/pdfs/` or use `rag.pdf_missing_strategy`.
- PowerShell wildcard paths:
  - Pass explicit paths if glob expansion fails, e.g. `python main.py analyze .\images\a.jpg .\images\b.jpg --batch`.

---

## 🔄 Migration Note

Notebook-to-Python parity status:
- Stage 2 now classifies WBC crops across all input images (not first image only).
- Uncertainty summary now uses canonical `flagged_count` with backward-compatible `flagged_samples`.
- Stage 3 keeps safety-aware abstention and citation grounding.
- Notebook visualization extras remain outside CLI runtime by design.

---

## 📝 Thesis Writing Guide

### Methods Chapter Should Include:
1. Three-stage architecture diagram (Detection → Classification+Uncertainty → RAG Reasoning)
2. Monte Carlo Dropout uncertainty equations
3. Safety protocol and abstention policy
4. Data contract and interoperability schema

### Results Chapter Should Include:
1. Detection performance (mAP, precision, recall)
2. Classification accuracy + uncertainty distributions
3. Qualitative reasoning examples with citations
4. Comparative analysis: baseline vs RAG vs RAG+uncertainty
5. Safety effectiveness: cases triggering `INSUFFICIENT_EVIDENCE`

### Appendix Should Include:
1. Run metadata JSON files
2. Example output schemas
3. Complete prompt templates
4. Configuration reference

---

## ⚠️ Known Limitations

1. **Performance**: Depends on available GPU/CPU resources
2. **PDF Quality**: Retrieval quality depends on PDF text extraction success (no OCR fallback)
3. **LLM Output**: Structure is validated but remains probabilistic
4. **Clinical Validation**: Research software only - **NOT a certified medical device**
5. **Scalability**: Single-node inference (no distributed computing)

---

## 🔐 Safety & Ethics Notice

This repository contains an **academic prototype for decision support research**.

**CRITICAL DISCLAIMERS:**
- ❌ NOT a certified medical device
- ❌ NOT for standalone diagnostic use
- ❌ NOT clinically validated for deployment
- ✅ Requires qualified human oversight
- ✅ For research and educational purposes only

**Clinical Use Warning:**
This system is designed to **assist**, not replace, qualified medical professionals. All automated interpretations must be reviewed and validated by licensed practitioners before any clinical decision-making.

---

## 📚 References & Datasets

### Datasets Used:
- **TXL-PBC**: Public blood cell dataset for detection training
  - Repository: `lugan113/TXL-PBC_Dataset`

### Knowledge Base:
- Essentials of Haematology (public domain)
- Concise Haematology textbook (public domain)
- Laboratory Guide to Hematology (public domain)

### Models:
- **YOLOv8**: Ultralytics implementation
- **EfficientNet-B0**: timm implementation
- **Sentence Transformers**: all-MiniLM-L6-v2
- **GPT-4o**: OpenAI API

---

## 🤝 Contributing

This is a thesis project repository. If you find issues or have suggestions:

1. Check existing issues
2. Open a new issue with detailed description
3. For code contributions, open a pull request

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@misc{bloodsmear2024,
  title={Blood Smear Domain Expert: A Safety-Aware Clinical AI Assistant},
  author={[Your Name]},
  year={2024},
  school={[Your University]},
  type={Bachelor's Thesis}
}
```

---

## 📧 Contact

For questions about this thesis project:
- **Author**: [Your Name]
- **Institution**: [Your University]
- **Email**: [Your Email]

---

## 🙏 Acknowledgments

- Dataset providers (TXL-PBC)
- Open-source community (PyTorch, Ultralytics, OpenAI)
- Thesis supervisor(s)
- [Other acknowledgments]

---

**Built with ❤️ for safer clinical AI**
