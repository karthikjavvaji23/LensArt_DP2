"""CLIP wrapper. Hybrid system uses CLIP only for artist + similarity;
style classification is delegated to the trained ResNet50."""
from __future__ import annotations

import threading
from typing import Iterable

import numpy as np
from PIL import Image

from . import config

_LOCK = threading.Lock()
_MODEL = None
_PREPROC = None
_TOKENIZER = None
_ARTIST_TEXT_FEATS = None


def _ensure_model():
    global _MODEL, _PREPROC, _TOKENIZER
    if _MODEL is not None:
        return
    with _LOCK:
        if _MODEL is not None:
            return
        import torch  # noqa: F401
        import open_clip
        model, _, preproc = open_clip.create_model_and_transforms(
            config.CLIP_MODEL, pretrained=config.CLIP_PRETRAINED
        )
        model.eval()
        _MODEL = model
        _PREPROC = preproc
        _TOKENIZER = open_clip.get_tokenizer(config.CLIP_MODEL)


def _encode_text(prompts: Iterable[str]) -> np.ndarray:
    import torch
    _ensure_model()
    with torch.no_grad():
        toks = _TOKENIZER(list(prompts))
        feats = _MODEL.encode_text(toks).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype("float32")


def _ensure_artist_cache():
    global _ARTIST_TEXT_FEATS
    if _ARTIST_TEXT_FEATS is None:
        _ARTIST_TEXT_FEATS = _encode_text(
            [f"a painting by {a}" for a in config.ARTISTS]
        )


def encode_image(img: Image.Image) -> np.ndarray:
    """Return a (D,) L2-normalised image embedding."""
    import torch
    _ensure_model()
    with torch.no_grad():
        x = _PREPROC(img.convert("RGB")).unsqueeze(0)
        feat = _MODEL.encode_image(x).float()
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat[0].cpu().numpy().astype("float32")


def predict_artist_topk(img_feat: np.ndarray, k: int = config.TOP_K_ARTIST) -> list[dict]:
    """Cosine similarity → softmax → top-K artists."""
    _ensure_artist_cache()
    sims = _ARTIST_TEXT_FEATS @ img_feat
    logits = sims * 30.0
    z = logits - logits.max()
    p = np.exp(z); p /= p.sum()
    idx = np.argsort(-p)[:k]
    return [{"label": config.ARTISTS[i], "confidence": float(p[i])} for i in idx]


def classify_artist(img: Image.Image) -> dict:
    """End-to-end zero-shot artist prediction."""
    feat = encode_image(img)
    return {"artist_top": predict_artist_topk(feat),
            "image_feature": feat}
