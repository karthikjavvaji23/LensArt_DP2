# LensArt —  (ResNet50 + CLIP)
LensArt is a Art Recognition and Discovery System which works on a deep learning system that takes the
uploaded image of any painting and provides more context to it like the artist name, a simple amount of information
regarding the artist and the historical context behind the painting giving depth to the work 
<p align="center">
  <img src="media/Recording%202026-04-30%20150925.gif" width="50%"/>
</p>
## 🚀 Features

- 🖼️ Upload any artwork image (JPEG/PNG)
- 🎭 Art style classification using ResNet50
- 🧠 Artist identification using CLIP (zero-shot)
- 🔍 Visual similarity search using embeddings
- 📚 Artist biography via Wikipedia & Met Museum APIs
- ⚡ Fast inference (1–2 seconds on CPU)
- 🌐 Flask-based interactive web interface
  
## How to run

1. Place your trained ResNet50 checkpoint at `models/resnet_model.pth`.
2. Drop ArtBench dataset images at `data/images/train/<style>/`
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


