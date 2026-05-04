"""Flask web UI for LensArt — hybrid backend with model attribution."""
from __future__ import annotations

import io
import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image

from lensart import clip_model, config, enrichment, resnet_classifier, similarity

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024


@app.route("/")
def index():
    samples = []
    if config.SAMPLE_DIR.exists():
        for p in sorted(config.SAMPLE_DIR.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                samples.append({"name": p.name,
                                "title": p.stem.replace("_", " ").title()})
    return render_template("index.html",
                           title=config.APP_TITLE, samples=samples)


@app.route("/sample/<path:filename>")
def serve_sample(filename: str):
    return send_from_directory(str(config.SAMPLE_DIR), filename)


@app.route("/dataset/<path:relpath>")
def serve_dataset(relpath: str):
    base = config.DATA_DIR / "images"
    return send_from_directory(str(base), relpath)


def _model_status() -> dict:
    """Snapshot of which models are currently active."""
    resnet_path = resnet_classifier.find_weights()
    return {
        "style": {
            "model": "ResNet50",
            "active": resnet_classifier.is_loaded() or resnet_path is not None,
            "weights": resnet_classifier.loaded_from() or (str(resnet_path) if resnet_path else None),
            "label":  "Trained ResNet50 (ArtBench-10)",
        },
        "artist": {
            "model": "CLIP ViT-B/32",
            "active": True,
            "weights": "OpenAI pretrained",
            "label":  "CLIP ViT-B/32 zero-shot",
        },
        "similarity": {
            "model": "CLIP ViT-B/32",
            "active": True,
            "weights": "OpenAI pretrained",
            "label":  "CLIP image embedding + cosine",
        },
        "biography": {
            "model": "Wikipedia REST",
            "active": True,
            "weights": None,
            "label":  "Wikipedia → Met Museum fallback",
        },
    }


@app.route("/api/model-status")
def model_status():
    return jsonify(_model_status())


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """ResNet50 (style) → CLIP (artist + similarity) → Wikipedia (biography)."""
    t_total = time.perf_counter()

    img: Image.Image | None = None
    if "image" in request.files and request.files["image"].filename:
        try:
            img = Image.open(io.BytesIO(request.files["image"].read())).convert("RGB")
        except Exception as e:
            return jsonify({"error": f"Could not read image: {e}"}), 400
    elif request.form.get("sample"):
        sample_path = config.SAMPLE_DIR / request.form["sample"]
        if not sample_path.exists():
            return jsonify({"error": "Unknown sample image."}), 404
        img = Image.open(sample_path).convert("RGB")
    else:
        return jsonify({"error": "Upload a file or pick a sample."}), 400

    timings = {}

    t0 = time.perf_counter()
    style_top = resnet_classifier.predict_style(img, top_k=3)
    timings["style_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    cls = clip_model.classify_artist(img)
    artist_top = cls["artist_top"]
    timings["artist_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    sim = similarity.topk_similar(cls["image_feature"], k=config.TOP_K_SIMILAR)
    timings["similarity_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    bio = enrichment.enrich(artist_top[0]["label"]) if artist_top else None
    timings["biography_ms"] = (time.perf_counter() - t0) * 1000

    timings["total_ms"] = (time.perf_counter() - t_total) * 1000

    return jsonify({
        "style":      style_top[0]["label"]      if style_top  else "",
        "confidence": style_top[0]["confidence"] if style_top  else 0.0,
        "artist":     artist_top[0]["label"]     if artist_top else "",
        "biography":  (bio or {}).get("summary", ""),
        "similar_images": [
            {"title": s["title"],
             "url": (f"/sample/{s['rel']}" if s["is_sample"]
                     else f"/dataset/{s['rel']}"),
             "similarity": s["similarity"],
             "style": s["style"]}
            for s in sim
        ],
        "time": timings["total_ms"] / 1000.0,

        "style_top":   style_top,
        "artist_top":  artist_top,
        "biography_full": bio,
        "timings":     timings,
        "models":      _model_status(),
        "backend": (
            f"ResNet50 (style) + CLIP {config.CLIP_MODEL} "
            f"({config.CLIP_PRETRAINED}) (artist + similarity)"
        ),
    })


@app.route("/api/health")
def health():
    return jsonify({"ok": True})


def _warmup():
    print("[LensArt] Warming up models and similarity index…", flush=True)
    img = Image.new("RGB", (224, 224), (200, 180, 150))
    try:
        resnet_classifier.predict_style(img, top_k=1)
    except Exception as e:
        print(f"[LensArt] (info) ResNet warmup skipped: {e}", flush=True)
    clip_model.classify_artist(img)
    similarity.build_index()
    status = _model_status()
    print("[LensArt] Active models:", flush=True)
    for k, v in status.items():
        mark = "OK " if v["active"] else "—  "
        print(f"  {mark} {k:11s}  {v['label']}", flush=True)
        if v.get("weights"):
            print(f"            weights: {v['weights']}", flush=True)
    print("[LensArt] Warmup complete. Open http://localhost:5000", flush=True)


if __name__ == "__main__":
    _warmup()
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
