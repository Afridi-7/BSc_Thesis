"""Extract embedded image outputs from the three stage notebooks and copy
standalone result files into figures/stageN_<name>/ folders with descriptive
names suitable for inclusion in the thesis.

Run from repo root:
    python scripts/extract_notebook_figures.py
"""
from __future__ import annotations

import base64
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parent.parent
FIGS = REPO / "figures"

EXT_FOR_MIME = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/svg+xml": ".svg",
}


@dataclass
class NotebookSpec:
    nb_path: Path
    out_dir: Path
    prefix: str
    # (absolute_cell_index, image_index_within_cell) -> descriptive slug.
    labels: Dict[Tuple[int, int], str] = field(default_factory=dict)
    # Cells whose embedded outputs duplicate files we copy from results/.
    skip_cells: List[int] = field(default_factory=list)


SPECS = [
    NotebookSpec(
        nb_path=REPO
        / "Notebooks"
        / "YOLOv8_detection"
        / "YOLOv8_on_TBL_PBC_dataset.ipynb",
        out_dir=FIGS / "stage1_detection",
        prefix="stage1",
        labels={
            (6, 0): "annotated_training_samples",
            (6, 1): "class_distribution_per_split",
            (15, 0): "test_predictions_grid",
        },
        skip_cells=[12],  # cell 12 just re-displays files from results/
    ),
    NotebookSpec(
        nb_path=REPO
        / "Notebooks"
        / "Efficientnet_classification"
        / "Efficientnet_classification.ipynb",
        out_dir=FIGS / "stage2_classification",
        prefix="stage2",
        labels={
            (18, 0): "confusion_matrix_test",
            (23, 0): "predictions_grid_display",
            (25, 0): "mc_dropout_uncertainty_demo",
        },
    ),
    NotebookSpec(
        nb_path=REPO / "Notebooks" / "LLM_RAG_Pipline" / "LLM_Rag_pipline.ipynb",
        out_dir=FIGS / "stage3_reasoning",
        prefix="stage3",
    ),
]


def extract_embedded(spec: NotebookSpec) -> int:
    if not spec.nb_path.exists():
        print(f"  SKIP (missing notebook): {spec.nb_path}")
        return 0
    spec.out_dir.mkdir(parents=True, exist_ok=True)
    nb = json.loads(spec.nb_path.read_text(encoding="utf-8"))
    saved = 0
    for cell_idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        if cell_idx in spec.skip_cells:
            continue
        img_idx = 0
        for output in cell.get("outputs", []):
            data = output.get("data") or {}
            for mime, payload in data.items():
                if not mime.startswith("image/"):
                    continue
                ext = EXT_FOR_MIME.get(mime, ".png")
                label = spec.labels.get(
                    (cell_idx, img_idx),
                    f"cell{cell_idx:02d}_img{img_idx:02d}",
                )
                fname = f"{spec.prefix}_embedded_{label}{ext}"
                target = spec.out_dir / fname
                body = payload if isinstance(payload, str) else "".join(payload)
                if mime == "image/svg+xml":
                    target.write_text(body, encoding="utf-8")
                else:
                    target.write_bytes(base64.b64decode(body))
                print(f"  + {fname}  ({target.stat().st_size / 1024:.0f} KB)")
                saved += 1
                img_idx += 1
    return saved


def copy_standalone(spec: NotebookSpec) -> int:
    res_dir = spec.nb_path.parent / "results"
    if not res_dir.exists():
        return 0
    spec.out_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for f in sorted(res_dir.iterdir()):
        if f.suffix.lower() not in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}:
            continue
        target = spec.out_dir / f"{spec.prefix}_results_{f.name}"
        shutil.copy2(f, target)
        print(
            f"  + {target.name}  ({target.stat().st_size / 1024:.0f} KB)  (from results/)"
        )
        copied += 1
    return copied


def clean_dir(path: Path) -> None:
    if not path.exists():
        return
    for f in path.iterdir():
        if f.is_file():
            f.unlink()


def main() -> None:
    total = 0
    for spec in SPECS:
        print(f"\n=> {spec.nb_path.relative_to(REPO)}")
        clean_dir(spec.out_dir)
        n_emb = extract_embedded(spec)
        n_res = copy_standalone(spec)
        print(f"   embedded: {n_emb}, standalone copied: {n_res}")
        total += n_emb + n_res
    print(f"\nDone. Total figures written: {total}")
    print(f"Output root: {FIGS}")


if __name__ == "__main__":
    main()
