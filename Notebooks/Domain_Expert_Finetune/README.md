# Domain-Expert Fine-Tuning

End-to-end recipe for replacing GPT-4o in Stage 3 with a hematology-specialised
**Llama-3.1-8B + QLoRA** model. Plugs back into the existing pipeline through
the OpenAI-compatible interface — no Python code changes required.

## Files

| File | Role |
|---|---|
| [scripts/build_finetune_dataset.py](../scripts/build_finetune_dataset.py) | Builds `train.jsonl` from your ChromaDB textbook chunks + MedQA + hand-curated gold. |
| [Notebooks/Domain_Expert_Finetune/domain_expert_finetune.ipynb](domain_expert_finetune.ipynb) | Colab QLoRA training notebook (free T4 16 GB). |
| [scripts/evaluate_finetuned_model.py](../scripts/evaluate_finetuned_model.py) | Hold-out evaluation: MCQ exact-match + GPT-4-as-judge rubric. |

## Step 1 — Build the dataset (local)

```powershell
# install minimal extras (already in your .venv if you use the project)
.\.venv\Scripts\python.exe -m pip install openai datasets

# Set your key for synthetic Q&A generation
$env:OPENAI_API_KEY = "sk-…"

# Quick dry-run on 20 chunks to validate plumbing
.\.venv\Scripts\python.exe scripts/build_finetune_dataset.py `
    --max-chunks 20 --pairs-per-chunk 1 --out data/finetune/sample.jsonl

# Full run (~$2–10 depending on --model)
.\.venv\Scripts\python.exe scripts/build_finetune_dataset.py `
    --pairs-per-chunk 2 --model gpt-4o-mini --out data/finetune/train.jsonl
```

**Hand-curated gold:** drop ~30–50 examples in `data/finetune/gold.jsonl`, one
JSON per line. Either format is accepted:
```json
{"question": "...", "answer": "..."}
{"messages": [{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```
Hand-write cases that test: refusal/insufficient evidence, citation format,
uncertainty calibration, cross-modal CBC + image reasoning.

## Step 2 — Train (Colab)

Open `Notebooks/Domain_Expert_Finetune/domain_expert_finetune.ipynb` in Colab
(GPU runtime → T4). Upload `train.jsonl` when prompted. Run all cells.

| Setup | Time for 3 epochs × 2 500 samples |
|---|---|
| Free Colab T4 | 2.5–3 h |
| Colab Pro A100 40 GB | 30–45 min |
| Colab Pro+ L4 24 GB | ~1.5 h |

The notebook saves and (optionally) pushes the LoRA adapter to your HF Hub
account (`YOU/hematology-llama-3.1-8b-lora`, ~50 MB).

## Step 3 — Serve & plug in

On any machine with a GPU (≥10 GB for 8B-Q4):
```bash
pip install vllm
vllm serve YOU/hematology-llama-3.1-8b-lora --port 8001 --enable-lora
```

Then in `config.yaml`:
```yaml
llm:
  model_name: YOU/hematology-llama-3.1-8b-lora
  temperature: 0.1
```
And `.env`:
```
OPENAI_API_KEY=dummy
OPENAI_BASE_URL=http://localhost:8001/v1
```

Stage 3 of the existing pipeline now talks to your domain-expert model —
no code changes.

## Step 4 — Evaluate

Hold out ~200 records when you build the dataset (`tail -n 200 train.jsonl >
eval.jsonl && head -n -200 …`), then:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_finetuned_model.py `
    --eval-file data/finetune/eval.jsonl `
    --model YOU/hematology-llama-3.1-8b-lora `
    --base-url http://localhost:8001/v1 `
    --judge-model gpt-4o `
    --out data/finetune/eval_report.json
```

Reports MCQ accuracy and average faithfulness / clinical correctness /
calibration scores. Compare to the same eval run against vanilla GPT-4o
(`--model gpt-4o --base-url https://api.openai.com/v1`) for the ablation.
