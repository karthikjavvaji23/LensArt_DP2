"""CLIP-embedding visual similarity.

Encodes every reference image (from sample_images/ AND optionally from
data/images/train/<style>/) once and persists the resulting matrix at
index_cache/embeddings.npy. Subsequent queries are an O(N) cosine product.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from . import clip_model, config

_INDEX_FILE = config.INDEX_DIR / "embeddings.npy"
_META_FILE = config.INDEX_DIR / "metadata.json"

_INDEX_MATRIX: np.ndarray | None = None
_INDEX_META: list[dict] | None = None


def _gather_reference_paths(*, max_per_style: int = 25) -> list[dict]:
    """Pick reference images for the index.

    Priority: sample_images/ (always) + a sampled subset of
    data/images/train/<style>/ (if present, capped to keep boot fast).
    """
    out: list[dict] = []
    exts = {".jpg", ".jpeg", ".png", ".webp"}

    if config.SAMPLE_DIR.exists():
        for p in sorted(config.SAMPLE_DIR.iterdir()):
            if p.is_file() and p.suffix.lower() in exts:
                out.append({"path": str(p), "rel": p.name,
                            "title": p.stem.replace("_", " ").title(),
                            "style": "sample", "is_sample": True})

    if config.TRAIN_DIR.exists():
        for style_dir in sorted(config.TRAIN_DIR.iterdir()):
            if not style_dir.is_dir():
                continue
            files = [p for p in style_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in exts]
            random.Random(42).shuffle(files)
            for p in files[:max_per_style]:
                out.append({
                    "path": str(p),
                    "rel": f"train/{style_dir.name}/{p.name}",
                    "title": p.stem.replace("_", " ").title(),
                    "style": style_dir.name,
                    "is_sample": False,
                })
    return out


def build_index(force: bool = False, *, verbose: bool = True) -> None:
    """Encode every reference image and persist the embedding matrix."""
    global _INDEX_MATRIX, _INDEX_META

    if not force and _INDEX_FILE.exists() and _META_FILE.exists():
        _INDEX_MATRIX = np.load(_INDEX_FILE)
        _INDEX_META = json.loads(_META_FILE.read_text(encoding="utf-8"))
        return

    refs = _gather_reference_paths()
    if not refs:
        _INDEX_MATRIX = np.empty((0, 512), dtype="float32")
        _INDEX_META = []
        return

    if verbose:
        print(f"[LensArt] Encoding {len(refs)} reference images for similarity index…",
              flush=True)
    feats: list[np.ndarray] = []
    meta: list[dict] = []
    for r in refs:
        try:
            with Image.open(r["path"]) as im:
                feats.append(clip_model.encode_image(im))
            meta.append({
                "title": r["title"],
                "rel":   r["rel"],
                "style": r["style"],
                "is_sample": r["is_sample"],
            })
        except Exception:
            continue

    _INDEX_MATRIX = np.stack(feats).astype("float32")
    _INDEX_META = meta
    np.save(_INDEX_FILE, _INDEX_MATRIX)
    _META_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if verbose:
        print(f"[LensArt] Index built: {_INDEX_MATRIX.shape}", flush=True)


def topk_similar(query: np.ndarray, k: int = config.TOP_K_SIMILAR) -> list[dict]:
    global _INDEX_MATRIX, _INDEX_META
    if _INDEX_MATRIX is None or _INDEX_META is None:
        build_index()
    if _INDEX_MATRIX is None or _INDEX_MATRIX.shape[0] == 0:
        return []
    sims = _INDEX_MATRIX @ query
    order = np.argsort(-sims)
    out = []
    seen = set()
    for i in order:
        if sims[i] >= 0.999:  # drop trivial self-match
            continue
        m = _INDEX_META[int(i)]
        # Avoid showing two near-identical hits with the same title
        if m["title"] in seen:
            continue
        seen.add(m["title"])
        out.append({
            "title":      m["title"],
            "rel":        m["rel"],
            "style":      m["style"],
            "is_sample":  m["is_sample"],
            "similarity": float(sims[i]),
        })
        if len(out) >= k:
            break
    return out
