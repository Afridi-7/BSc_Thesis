readme = f"""# 🔬 Stage 2 — WBC Subtype Classification (EfficientNet-B0)
> **Thesis Project** — Development of a Domain Expert using a Large Language Model
> **Stage 2** — White Blood Cell subtype classification

---

## Results

| Training Phase | Accuracy |
|----------------|----------|
| Frozen backbone (baseline) | 87.22% |
| **Fine-tuned (final)** | **98.48%** |
| Improvement | +11.26% |

### Per-Class Performance

| Class | Clinical Meaning |
|-------|-----------------|
| Neutrophil | Bacterial infection |
| Lymphocyte | Viral infection |
| Monocyte | Chronic infection |
| Eosinophil | Allergy / parasites |
| Basophil | Inflammatory response |
| Erythroblast | Immature RBC |
| Immature Granulocyte | Early WBC forms |
| Platelet | Thrombocyte |

---

## Sample Predictions

![Predictions](results/predictions.png)

---

## Training Curves

![Training Curves](results/training_curves.png)

---

## Confusion Matrix

![Confusion Matrix](results/confusion_matrix.png)

---

## DataSet

| | |
|---|---|
| **Name** | PBC — Peripheral Blood Cell Images |
| **Source** | Kaggle / Mendeley Data |
| **Images** | 17,092 |
| **Classes** | 8 WBC subtypes |
| **Split** | 80% train / 20% t