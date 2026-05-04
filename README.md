# LensArt — Hybrid (ResNet50 + CLIP)

**Final hybrid pipeline for the Data Science Practicum 2 submission.**

| Component | Model | What it does |
|-----------|-------|--------------|
| Style classification | **ResNet50** (your trained weights) | Predicts one of the 10 ArtBench-10 styles |
| Artist prediction | **CLIP ViT-B/32** (zero-shot) | Ranks 41 well-known painters via text-prompt similarity |
| Visual similarity  | **CLIP ViT-B/32** (image embeddings) | Top-5 cosine-similar paintings from the indexed corpus |
| Biography          | Wikipedia REST → Met fallback | Short summary + thumbnail |
| UI                 | Flask + HTML / CSS / vanilla JS | Single-page gallery aesthetic |

## How to run

1. Place your trained ResNet50 checkpoint at `models/resnet_model.pth`.
2. (Optional) Drop ArtBench dataset images at `data/images/train/<style>/`
   so the similarity index can index real paintings.
3. **First time only:** double-click `setup_venv.bat` (creates `.venv\` and
   installs CPU-only PyTorch + CLIP + Flask, ~5 minutes).
4. **Every time:** double-click `run.bat`. Browser opens at
   `http://localhost:5000`.

## Folder layout

```
LensArt_Hybrid/
├── app.py                    Flask routes + JSON API
├── requirements.txt
├── setup_venv.bat            one-time installer
├── run.bat                   launcher
├── README.md
├── lensart/
│   ├── config.py             ArtBench styles + artist universe
│   ├── resnet_classifier.py  loads models/resnet_model.pth
│   ├── clip_model.py         CLIP zero-shot artist
│   ├── similarity.py         CLIP-embedding nearest neighbours
│   └── enrichment.py         Wikipedia + Met clients
├── templates/index.html      single-page UI
├── static/
│   ├── style.css             gallery aesthetic
│   └── app.js                renders the hybrid output
├── sample_images/            5 demo paintings shipped with the app
├── models/                   put resnet_model.pth here
├── data/                     ArtBench-10 dataset (images + metadata.csv)
└── index_cache/              auto-built CLIP embedding matrix
```

## API contract

`POST /api/predict` returns:

```json
{
  "style": "Post Impressionism",
  "confidence": 0.81,
  "artist": "Vincent van Gogh",
  "biography": "Vincent van Gogh was a Dutch Post-Impressionist painter ...",
  "similar_images": [
    {"title": "Starry Night", "url": "/sample/...", "similarity": 0.89, "style": "post_impressionism"}
  ],
  "time": 0.82,

  "style_top":  [...],    /* top-3 with confidence — for the UI bars */
  "artist_top": [...],    /* top-5 with confidence */
  "biography_full": {...},
  "timings": {...},
  "backend": "ResNet50 (style) + CLIP ViT-B-32 (openai) (artist + similarity)"
}
```

## What changed vs the previous Streamlit version

* **Streamlit replaced by Flask** — a single-page web app served on
  `localhost:5000`, matching the existing LensArt MVP front-end.
* **Style now uses your trained ResNet50** instead of CLIP zero-shot.
* **Artist classification still uses CLIP zero-shot** — no labelled-artist
  corpus required, generalises beyond the 10 ArtBench styles.
* **Similarity reuses the CLIP image embedding** computed during artist
  prediction, so we get three tasks out of two model passes.

## Why the hybrid design

CLIP is excellent at fine-grained zero-shot recognition (artist names, style
keywords) because its 400M (image, caption) training set names exactly those
concepts. ResNet50 fine-tuned on ArtBench is excellent at the well-defined
10-class style taxonomy because the labels are deterministic and the dataset
is balanced. Combining them lets each model do what it does best.
