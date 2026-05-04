"""ResNet50 art-style classifier with auto-locate of the trained checkpoint.

Searches the obvious places where Karthik's `resnet_model.pth` might live —
the project itself, the previous LensArt_ArtBench_Final folder next door,
common Downloads locations — and auto-copies into models/ on first launch.
"""
from __future__ import annotations

import shutil
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from . import config

_LOCK = threading.Lock()
_MODEL = None
_PREPROCESS = None
_LOADED_FROM: Optional[Path] = None


def _candidate_paths() -> list[Path]:
    """Probable locations where the trained ResNet checkpoint may live."""
    here = Path(__file__).resolve().parents[1]
    parents = [here, here.parent, here.parent.parent]  # up to ~3 levels
    candidates: list[Path] = []
    # 1. Inside this project
    candidates.append(config.RESNET_WEIGHTS)
    # 2. Common sibling-folder names from the user's previous attempts
    for parent in parents:
        for sibling in (
            "LensArt_ArtBench_Final",
            "LensArt_v1",
            "LensArt_MVP",
            "LensArt",
            "lensart_project",
            "ArtBench",
        ):
            for fname in (
                "resnet_model.pth",
                "models/resnet_model.pth",
                "model.pth",
                "models/model.pth",
                "resnet50.pth",
                "models/resnet50.pth",
                "best_model.pth",
                "models/best_model.pth",
            ):
                candidates.append(parent / sibling / fname)
    # 3. Bare files in any nearby folder
    for parent in parents:
        for fname in ("resnet_model.pth", "model.pth", "resnet50.pth", "best_model.pth"):
            candidates.append(parent / fname)
    # Dedupe while preserving order
    seen = set()
    uniq = []
    for p in candidates:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(p)
    return uniq


def find_weights() -> Optional[Path]:
    """Return the first existing checkpoint path, or None."""
    for p in _candidate_paths():
        if p.exists() and p.is_file() and p.stat().st_size > 1_000_000:
            return p
    return None


def _ensure_model():
    """Load ResNet50 from disk on first call. Cached thereafter."""
    global _MODEL, _PREPROCESS, _LOADED_FROM
    if _MODEL is not None:
        return
    with _LOCK:
        if _MODEL is not None:
            return
        import torch
        import torch.nn as nn
        from torchvision import models, transforms

        weights_path = config.RESNET_WEIGHTS
        if not weights_path.exists():
            found = find_weights()
            if found is not None:
                # Copy into the project's models/ so subsequent runs are fast.
                weights_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(found, weights_path)
                    print(f"[LensArt] Auto-located ResNet weights at {found}")
                    print(f"[LensArt]   → copied to {weights_path}")
                except Exception as e:
                    print(f"[LensArt] Found weights at {found} but couldn't copy: {e}")
                    weights_path = found  # use them in-place
            else:
                raise FileNotFoundError(
                    f"ResNet weights not found at {weights_path} and could not "
                    f"be auto-located in the usual sibling folders. Place "
                    f"resnet_model.pth into models/ and restart."
                )

        # Build a vanilla ResNet50 with a fresh head.
        net = models.resnet50(weights=None)
        net.fc = nn.Linear(net.fc.in_features, config.RESNET_NUM_CLASSES)

        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        else:
            state = ckpt
        if isinstance(state, dict):
            state = {k.replace("module.", ""): v for k, v in state.items()}
        head_keys = ("fc.weight", "fc.bias")
        if all(k in state for k in head_keys):
            head_shape = state["fc.weight"].shape
            if head_shape[0] != config.RESNET_NUM_CLASSES:
                print(f"[LensArt] Checkpoint has {head_shape[0]} classes; "
                      f"adapting head from {config.RESNET_NUM_CLASSES}.")
                net.fc = nn.Linear(net.fc.in_features, head_shape[0])
        try:
            net.load_state_dict(state, strict=False)
        except Exception as e:
            print(f"[LensArt] Non-strict ResNet load: {e}")
        net.eval()

        preprocess = transforms.Compose([
            transforms.Resize(int(config.RESNET_INPUT_SIZE * 1.15)),
            transforms.CenterCrop(config.RESNET_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        _MODEL = net
        _PREPROCESS = preprocess
        _LOADED_FROM = weights_path
        print(f"[LensArt] ResNet50 loaded from {weights_path}")


def is_loaded() -> bool:
    return _MODEL is not None


def loaded_from() -> Optional[str]:
    return str(_LOADED_FROM) if _LOADED_FROM is not None else None


def predict_style(img: Image.Image, top_k: int = 3) -> list[dict]:
    """Top-K predicted style classes with confidence."""
    import torch

    try:
        _ensure_model()
    except FileNotFoundError as e:
        return [{"label": "(ResNet weights missing)",
                 "confidence": 0.0, "error": str(e)}]

    with torch.no_grad():
        x = _PREPROCESS(img.convert("RGB")).unsqueeze(0)
        logits = _MODEL(x)[0]
        probs = torch.softmax(logits, dim=-1)
    topp, topi = probs.topk(min(top_k, probs.shape[0]))
    out = []
    for p, i in zip(topp.cpu().numpy(), topi.cpu().numpy()):
        raw = (config.ARTBENCH_STYLES[int(i)]
               if int(i) < len(config.ARTBENCH_STYLES) else f"class_{i}")
        out.append({
            "label": raw.replace("_", " ").title(),
            "raw_label": raw,
            "confidence": float(p),
        })
    return out
