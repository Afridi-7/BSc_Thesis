# 🩸 Blood Smear Domain Expert
### Development of a Domain Expert using a Large Language Model

> A complete end-to-end AI pipeline that analyses peripheral blood smear images and generates clinical reports — combining object detection, image classification, and LLM reasoning into a single medical AI system.

<p align="center">
  <img src="stage1_detection/results/predictions.png" width="800"/>
</p>

---

## 🎯 What This System Does

A doctor uploads a blood smear image. In seconds the system returns:

- ✅ **Cell counts** — how many WBC, RBC, and Platelets are present
- ✅ **WBC subtype** — neutrophil, lymphocyte, monocyte, eosinophil, basophil, etc.
- ✅ **Clinical interpretation** — what the findings mean medically
- ✅ **Differential diagnosis** — 2-3 conditions consistent with the findings
- ✅ **Recommended tests** — what the clinician should order next

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    BLOOD SMEAR IMAGE INPUT                       │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                  STAGE 1 — YOLOv8s Detection                    │
│                                                                  │
│   Scans entire image → draws bounding boxes around every cell   │
│   Detects: WBC ● RBC ● Platelet                                 │
│   Crops each WBC for Stage 2                                     │
│                                                                  │
│   mAP@0.50 = 0.9849  │  Precision = 0.9759  │  Recall = 0.9606 │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│               STAGE 2 — EfficientNet-B0 Classification          │
│                                                                  │
│   Examines each WBC crop → identifies subtype                   │
│   Classes: Neutrophil ● Lymphocyte ● Monocyte ● Eosinophil     │
│            Basophil ● Erythroblast ● Immature Granulocyte       │
│                                                                  │
│   Accuracy = 98.48%  (fine-tuned from 87.22% baseline)         │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                      JSON BRIDGE                                 │
│                                                                  │
│   Converts raw model outputs into structured clinical summary   │
│   { cell_counts, wbc_differential, total_cells }               │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 3 — RAG + Gemini 2.5 Flash                   │
│                                                                  │
│   ChromaDB retrieves relevant haematology knowledge             │
│   Gemini reasons step by step like a senior haematologist       │
│   Produces: interpretation ● diagnosis ● recommendations        │
└─────────────────────────────────────────────────────────────────┘
                             ↓
                    CLINICAL REPORT OUTPUT
```

---

## 📊 Results at a Glance

### Stage 1 — YOLOv8s Blood Cell Detection

| Metric | Score |
|--------|-------|
| **mAP @ 0.50** | **0.9849** |
| mAP @ 0.50:0.95 | 0.8762 |
| Precision | 0.9759 |
| Recall | 0.9606 |

| Class | mAP@0.50 | Notes |
|-------|----------|-------|
| WBC | 0.9950 | Perfect recall — 1.000 |
| RBC | 0.9900 | Most common cell |
| Platelet | 0.9700 | Smallest cell type |

### Stage 2 — EfficientNet-B0 WBC Classification

| Training Phase | Accuracy | Strategy |
|----------------|----------|----------|
| Phase 1 — Frozen backbone | 87.22% | Feature extraction only |
| **Phase 2 — Fine-tuned** | **98.48%** | Full network update |
| **Improvement** | **+11.26%** | |

> **98.48% accuracy outperforms published benchmarks on the PBC dataset and exceeds typical expert haematologist inter-rater agreement (~96-97%)**

### Stage 3 — RAG + Gemini Clinical Reasoning

Sample findings from a real test image:

```json
{
  "cell_counts": { "WBC": 1, "RBC": 13, "Platelet": 0 },
  "wbc_differential": {
    "erythroblast": { "count": 1, "percent": "100.0%" }
  },
  "total_cells": 14
}
```

**Gemini's conclusion:**
> *"The combined findings present severe pancytopenia with a circulating nucleated red blood cell — highly suggestive of bone marrow dysfunction. Differential diagnosis: Aplastic Anaemia, Acute Myeloid Leukaemia, Myelodysplastic Syndrome. Urgent bone marrow biopsy required."*

---

## 📁 Repository Structure

```
blood-smear-domain-expert/
│
├── README.md
│
├── stage1_detection/
│   ├── README.md
│   ├── YOLOv8_BloodCell_Detection.ipynb
│   └── results/
│       ├── predictions.png
│       ├── training_curves.png
│       ├── confusion_matrix.png
│       ├── PR_curve.png
│       ├── F1_curve.png
│       └── report.pdf
│
├── stage2_classification/
│   ├── README.md
│   ├── EfficientNet_WBC_Classification.ipynb
│   └── results/
│       ├── predictions.png
│       ├── training_curves.png
│       ├── confusion_matrix.png
│       └── overfitting_check.png
│
└── stage3_llm_rag/
    ├── README.md
    ├── Stage3_RAG_Gemini.ipynb
    └── results/
        ├── pipeline_output.png
        └── clinical_report.txt
```

---

## 🔬 Technical Deep Dive

### Stage 1 — Why YOLOv8?

YOLO (You Only Look Once) detects all objects in a single forward pass through the network — making it extremely fast. YOLOv8s was chosen for its balance of speed and accuracy (11.2M parameters).

**Transfer Learning:** We started with COCO pretrained weights (118,000 images, 80 classes). The backbone already knew how to detect shapes, edges and textures. We replaced the detection head for our 3 classes and fine-tuned on TXL-PBC — training took only **0.301 hours** on a T4 GPU.

**Why mAP@0.50 matters:** A detection is correct only if the predicted bounding box overlaps ground truth by at least 50% (IoU ≥ 0.5). Our score of 0.9849 means 98.49% of detections were correct at this threshold.

---

### Stage 2 — Why EfficientNet?

EfficientNet uses **compound scaling** — simultaneously scaling network depth, width, and resolution in a principled ratio. This achieves better accuracy per parameter than ResNet or VGG.

**Two-phase training strategy:**

```
Phase 1 — Frozen backbone
  ImageNet weights preserved
  Only classifier head trains
  Fast convergence → 87.22%
         ↓
Phase 2 — Fine-tuning
  Entire network unfrozen
  Very small lr (0.0001)
  Backbone adapts to microscopy
  → 98.48% accuracy
```

The second phase improved accuracy by +11.26% because unfreezing allowed the CNN to adapt its low-level feature detectors specifically to blood cell morphology — nuclear shape, cytoplasm granularity, cell size.

**Overfitting check:** Training vs validation accuracy gap = 0.54% — confirming the model generalises well to unseen images.

---

### Stage 3 — Why RAG instead of plain LLM?

| Plain LLM | RAG + LLM |
|-----------|-----------|
| May hallucinate reference ranges | Grounded in verified medical text |
| Generic answers | Specific to the findings |
| No evidence trail | Retrieved passages cited |
| Outdated knowledge possible | Knowledge base controlled by us |

**How RAG works in our system:**

```
1. Medical textbooks chunked into 500-word passages
2. Each passage embedded as a semantic vector
   (sentence-transformers/all-MiniLM-L6-v2)
3. Vectors stored in ChromaDB
4. At inference: query built from JSON findings
5. ChromaDB retrieves top-5 most relevant passages
   (cosine similarity search)
6. Gemini receives JSON + retrieved knowledge + reasoning prompt
7. Chain-of-thought response generated
```

**Chain-of-thought prompting** forces Gemini to reason step by step:

```
STEP 1 → Assess each finding vs normal reference ranges
STEP 2 → Identify the overall pattern across all findings
STEP 3 → Differential diagnosis with justification for each
STEP 4 → Specific recommended investigations
STEP 5 → Final clinical interpretation summary
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Cell Detection | YOLOv8s (Ultralytics) | Locate all cells in smear |
| WBC Classification | EfficientNet-B0 (timm) | Identify WBC subtype |
| Vector Database | ChromaDB | Store medical knowledge |
| Embeddings | sentence-transformers | Text → semantic vectors |
| LLM | Gemini 2.5 Flash | Clinical reasoning |
| Framework | PyTorch 2.10 | Deep learning |
| Training | Google Colab T4 GPU | Free cloud GPU |

---

## 📦 Datasets

| Stage | Dataset | Images | Classes | Source |
|-------|---------|--------|---------|--------|
| Detection | TXL-PBC | 1,260 (18,143 boxes) | 3 | [GitHub](https://github.com/lugan113/TXL-PBC_Dataset) |
| Classification | PBC Mendeley | 17,092 | 8 | [Kaggle](https://www.kaggle.com/datasets/differentiatedthyroidcancer/peripheral-blood-cell-images) |
| RAG Knowledge | Open Haematology Textbooks | — | — | Open Textbook Library (CC-BY) |

---

## ⚙️ Training Configuration

### Stage 1 — YOLOv8s

| Parameter | Value |
|-----------|-------|
| Base model | YOLOv8s (pretrained COCO) |
| Epochs | 50 (early stopping) |
| Image size | 640 × 640 |
| Batch size | 16 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Augmentation | HSV, flip, rotation, mosaic |
| Training time | **0.301 hours** |

### Stage 2 — EfficientNet-B0

| Parameter | Phase 1 (Frozen) | Phase 2 (Fine-tuned) |
|-----------|-----------------|---------------------|
| Backbone | Frozen | Unfrozen |
| Learning rate | 0.001 | 0.0001 |
| Batch size | 32 | 32 |
| Max epochs | 30 | 10 |
| Early stopping | patience=5 | patience=5 |
| Scheduler | ReduceLROnPlateau | ReduceLROnPlateau |
| Best accuracy | 87.22% | **98.48%** |

---

## 🚀 Quick Start

### 1. Clone and install

```bash
git clone https://github.com/yourusername/blood-smear-domain-expert.git
cd blood-smear-domain-expert
pip install ultralytics timm chromadb sentence-transformers \
            google-genai torch torchvision opencv-python
```

### 2. Run inference with trained models

```python
from ultralytics import YOLO
import torch, timm, torch.nn as nn

# Load Stage 1
yolo_model = YOLO('stage1_detection/results/best.pt')

# Load Stage 2
checkpoint    = torch.load('stage2_classification/results/best.pt')
CATEGORIES    = checkpoint['categories']
effnet        = timm.create_model('efficientnet_b0', pretrained=False)
in_feat       = effnet.classifier.in_features
effnet.classifier = nn.Sequential(
    nn.Linear(in_feat, 1024), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(1024, 512),     nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(512, len(CATEGORIES))
)
effnet.load_state_dict(checkpoint['model_state'])
effnet.eval()

# Run full pipeline
summary = build_json_summary('blood_smear.jpg', yolo_model, effnet)
report  = ask_gemini_with_rag(summary)
print(report)
```

### 3. Gemini API Key (free)

1. Go to **aistudio.google.com**
2. Click **Get API key → Create API key**
3. In Colab: add as secret named `GEMINI_API_KEY`

```python
from google.colab import userdata
GEMINI_API_KEY = userdata.get('GEMINI_API_KEY')
```

---

## 🧠 Key Contributions

**1. Multimodal Pipeline Design**
Combining object detection, image classification, and language generation into a coherent medical reasoning system — each stage contributing capabilities the others cannot provide alone.

**2. The JSON Bridge**
A structured architectural connector that translates quantitative vision model outputs into qualitative language model inputs — enabling two fundamentally different AI paradigms to collaborate.

**3. RAG-Grounded Medical Reasoning**
Using retrieval augmented generation ensures the LLM's clinical reasoning is anchored to verified haematology literature — not hallucinated training data.

**4. Chain-of-Thought Clinical Prompting**
Prompt design that forces transparent, step-by-step reasoning mirroring actual clinical practice — making outputs interpretable and auditable by clinicians.

**5. Two-Phase Training Strategy**
Demonstrating that frozen-backbone feature extraction followed by full fine-tuning can improve classification accuracy by +11.26% — a reproducible training recipe for medical image classification.

---

## 📋 WBC Types and Clinical Meaning

| WBC Type | Normal % | High Suggests | Low Suggests |
|----------|----------|--------------|--------------|
| Neutrophil | 50-70% | Bacterial infection | Viral infection, drug effect |
| Lymphocyte | 20-40% | Viral infection, CLL | HIV, immunosuppression |
| Monocyte | 2-8% | TB, chronic infection | — |
| Eosinophil | 1-4% | Allergy, parasites | — |
| Basophil | 0-1% | CML, allergy | — |
| Erythroblast | 0% | Bone marrow stress | — |
| Immature Granulocyte | 0-5% | Severe infection, leukaemia | — |

---

## ⚠️ Limitations

- Cell counts are from a **single microscopy field** — not a full quantitative slide analysis
- System has **not been clinically validated** on real patient data
- RAG knowledge quality depends on the embedded medical sources
- Erythroblast appearing under WBC class follows TXL-PBC dataset convention

---

## 🔭 Future Work

- [ ] Whole slide image (WSI) analysis across multiple fields
- [ ] Blast cell detection for acute leukaemia screening
- [ ] RAG knowledge base expansion with PubMed open access papers
- [ ] Clinical validation study with expert haematologist ground truth
- [ ] Gradio web interface for non-technical clinical users
- [ ] ONNX export for CPU deployment without GPU dependency

---

## 📚 References

- Jocher, G. et al. (2023). *Ultralytics YOLOv8*. https://github.com/ultralytics/ultralytics
- Tan, M. & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. ICML 2019.
- Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020.
- Acevedo, A. et al. (2020). *A dataset of microscopic peripheral blood cell images for development of automatic recognition systems*. Mendeley Data.
- TXL-PBC Dataset. https://github.com/lugan113/TXL-PBC_Dataset
- Hendry, B.M. et al. *A Laboratory Guide to Clinical Hematology*. Open Textbook Library. (CC-BY)
- WHO Reference Ranges for Blood Cell Counts. (Public Domain)

---

## ⚠️ Disclaimer

This system is a **research prototype** developed for academic purposes only. It is **not** a certified medical device and must **not** be used for clinical decision-making without review and confirmation by a qualified haematologist. All outputs are AI-generated and carry inherent uncertainty.

---

<p align="center">
  <b>Computer Science Thesis Project</b><br>
  Development of a Domain Expert using a Large Language Model<br><br>
  Built with PyTorch · Ultralytics · Gemini 2.5 Flash · ChromaDB · Google Colab
</p>
