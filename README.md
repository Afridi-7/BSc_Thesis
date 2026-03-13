# 🩸 Blood Cell Domain Expert — LLM-Powered Medical AI

> **Thesis:** Development of a Domain Expert using a Large Language Model  
> A hybrid medical AI system combining computer vision with LLM reasoning
> for automated blood smear analysis and clinical interpretation.

---

## 🏗️ System Architecture
```
Blood Smear Image
       ↓
┌─────────────────────────┐
│  Stage 1 — YOLOv8s      │  ✅ Complete
│  Detects WBC·RBC·Plt    │
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│  Stage 2 — EfficientNet │  🔄 In Progress
│  Classifies WBC subtypes│
└────────────┬────────────┘
             ↓
      Structured JSON
             ↓
┌─────────────────────────┐
│  Stage 3 — LLM + RAG    │  ⏳ Upcoming
│  Clinical Interpretation│
└────────────┬────────────┘
             ↓
      Clinical Report
```

---

## 📁 Project Stages

### ✅ Stage 1 — Blood Cell Detection (YOLOv8s)
**Goal:** Detect and count WBC, RBC, and Platelets in blood smear images

| Metric | Score |
|--------|-------|
| mAP@0.50 | 0.9849 |
| mAP@0.50:0.95 | 0.8762 |
| Precision | 0.9759 |
| Recall | 0.9606 |

👉 [View Stage 1 Details](stage1_detection/README.md)

---

### 🔄 Stage 2 — WBC Classification (EfficientNet)
**Goal:** Sub-classify white blood cells into 5 types
(Neutrophil · Lymphocyte · Monocyte · Eosinophil · Basophil)

> Results will be added after training

👉 [View Stage 2 Details](stage2_classification/README.md)

---

### ⏳ Stage 3 — Clinical LLM + RAG
**Goal:** Generate clinical interpretation from vision model outputs

> Results will be added after implementation

👉 [View Stage 3 Details](stage3_llm_rag/README.md)

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Cell Detection | YOLOv8s (Ultralytics) |
| WBC Classification | EfficientNet-B0 (PyTorch) |
| LLM | GPT-4o / Mistral-7B |
| RAG | LangChain + ChromaDB |
| Embeddings | sentence-transformers |
| Knowledge Base | PubMed Open Access |
| Training | Google Colab (T4 GPU) |

---

## 📦 Datasets

| Stage | Dataset | Images | Classes |
|-------|---------|--------|---------|
| Detection | TXL-PBC | 1,260 | 3 |
| Classification | PBC Mendeley | 17,092 | 8 |
| RAG Knowledge | PubMed Hematology | ~100 papers | - |

---

## ⚠️ Disclaimer
Research prototype only. Not a certified medical device.
Always consult a qualified hematologist.
