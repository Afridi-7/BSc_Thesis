"""Evaluate a fine-tuned hematology LLM against a held-out set.

Run after fine-tuning. Targets the standard OpenAI-compatible endpoint that
your repo already uses (vLLM, Ollama, Azure OpenAI, raw OpenAI).

Two metrics:
  * MCQ exact-match for any record whose user message ends in "Options:" block.
  * GPT-4-as-judge (1-5 rubric) for free-text rationales.

Usage:
    python scripts/evaluate_finetuned_model.py \
        --eval-file data/finetune/eval.jsonl \
        --model YOU/hematology-llama-3.1-8b-lora \
        --base-url http://localhost:8001/v1 \
        --judge-model gpt-4o \
        --out data/finetune/eval_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


JUDGE_PROMPT = """You are grading a hematology assistant's answer.
Score 1–5 on each axis (1=worst, 5=best). Return STRICT JSON.
{{"faithfulness": int, "clinical_correctness": int, "calibration": int, "rationale": str}}

QUESTION:
{question}

GOLD ANSWER:
{gold}

MODEL ANSWER:
{prediction}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def split_messages(record: Dict[str, Any]) -> Optional[Dict[str, str]]:
    msgs = record.get("messages") or []
    if len(msgs) < 2:
        return None
    user = next((m["content"] for m in msgs if m["role"] == "user"), None)
    assistant = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
    system = next((m["content"] for m in msgs if m["role"] == "system"), None)
    if not user or not assistant:
        return None
    return {"system": system or "", "user": user, "gold": assistant}


def is_mcq(user_msg: str) -> bool:
    return bool(re.search(r"\boptions:\s*\n", user_msg, flags=re.IGNORECASE))


_MCQ_LETTER_RE = re.compile(r"\b([A-E])\b")


def extract_letter(text: str) -> Optional[str]:
    """Pull the first standalone capital letter A–E from a free-text answer."""
    # Prefer 'answer is X' / 'option X' / '**X.' patterns.
    for pat in (r"answer is\s*\**\s*([A-E])\b",
                r"option\s*\**\s*([A-E])\b",
                r"\*\*([A-E])\."):
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
    m = _MCQ_LETTER_RE.search(text)
    return m.group(1).upper() if m else None


# ---------------------------------------------------------------------------
# Inference + judging
# ---------------------------------------------------------------------------

def call_model(client: Any, model: str, system: str, user: str, temperature: float = 0.2) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system or "You are a clinical hematology expert."},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=600,
    )
    return resp.choices[0].message.content or ""


def judge_freetext(judge: Any, judge_model: str, q: str, gold: str, pred: str) -> Dict[str, Any]:
    prompt = JUDGE_PROMPT.format(question=q, gold=gold, prediction=pred)
    resp = judge.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(resp.choices[0].message.content or "{}")
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Model name on the target endpoint.")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"),
                        help="OpenAI-compatible base URL (e.g. vLLM at http://localhost:8001/v1).")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "dummy"))
    parser.add_argument("--judge-model", default="gpt-4o")
    parser.add_argument("--judge-base-url", default=None,
                        help="Defaults to OpenAI cloud (uses OPENAI_API_KEY).")
    parser.add_argument("--judge-api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("data/finetune/eval_report.json"))
    args = parser.parse_args()

    if not args.judge_api_key:
        print("[warn] OPENAI_API_KEY not set; free-text judging will be skipped.",
              file=sys.stderr)

    from openai import OpenAI  # type: ignore

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    judge = OpenAI(base_url=args.judge_base_url, api_key=args.judge_api_key) \
        if args.judge_api_key else None

    records = load_jsonl(args.eval_file)
    if args.limit:
        records = records[: args.limit]
    print(f"Evaluating {len(records)} records against {args.model}")

    mcq_total = mcq_correct = 0
    free_scores: List[Dict[str, int]] = []
    detail: List[Dict[str, Any]] = []

    for i, rec in enumerate(records, 1):
        parts = split_messages(rec)
        if not parts:
            continue
        try:
            pred = call_model(client, args.model, parts["system"], parts["user"])
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}] inference failed: {exc}", file=sys.stderr)
            continue

        item: Dict[str, Any] = {
            "question": parts["user"][:300],
            "gold": parts["gold"][:300],
            "prediction": pred[:500],
        }

        if is_mcq(parts["user"]):
            mcq_total += 1
            gold_letter = extract_letter(parts["gold"])
            pred_letter = extract_letter(pred)
            ok = bool(gold_letter and gold_letter == pred_letter)
            mcq_correct += int(ok)
            item.update({"type": "mcq", "gold_letter": gold_letter,
                         "pred_letter": pred_letter, "correct": ok})
        elif judge is not None:
            score = judge_freetext(judge, args.judge_model,
                                   parts["user"], parts["gold"], pred)
            if all(k in score for k in ("faithfulness", "clinical_correctness", "calibration")):
                free_scores.append(score)
            item.update({"type": "free", "judge": score})
        else:
            item["type"] = "free_skipped"

        detail.append(item)
        if i % 10 == 0:
            print(f"  {i}/{len(records)}")

    summary: Dict[str, Any] = {
        "n_records": len(detail),
        "mcq": {
            "n": mcq_total,
            "correct": mcq_correct,
            "accuracy": mcq_correct / mcq_total if mcq_total else None,
        },
    }
    if free_scores:
        summary["freetext_avg"] = {
            k: round(statistics.mean(s[k] for s in free_scores), 2)
            for k in ("faithfulness", "clinical_correctness", "calibration")
        }
        summary["freetext_n"] = len(free_scores)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "items": detail}, indent=2),
                        encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Full report: {args.out}")


if __name__ == "__main__":
    main()
