"""Project-wide constants for the hybrid LensArt system."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "sample_images"
INDEX_DIR = ROOT / "index_cache"
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
TRAIN_DIR = DATA_DIR / "images" / "train"
METADATA_CSV = DATA_DIR / "processed" / "metadata.csv"

INDEX_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ArtBench-10 — the 10 style classes the ResNet50 was trained on.
ARTBENCH_STYLES = [
    "art_nouveau",
    "baroque",
    "expressionism",
    "impressionism",
    "post_impressionism",
    "realism",
    "renaissance",
    "romanticism",
    "surrealism",
    "ukiyo_e",
]

# Artist universe used by the CLIP zero-shot prompt bank.
# Loaded dynamically from data/processed/metadata.csv if available; otherwise
# we fall back to a curated list of 41 well-known painters.
_DEFAULT_ARTISTS = [
    "Vincent van Gogh", "Claude Monet", "Pablo Picasso", "Rembrandt",
    "Leonardo da Vinci", "Salvador Dali", "Henri Matisse", "Edvard Munch",
    "Gustav Klimt", "Wassily Kandinsky", "Paul Cezanne", "Edgar Degas",
    "Pierre-Auguste Renoir", "Edouard Manet", "Albrecht Durer",
    "Michelangelo", "Raphael", "Caravaggio", "Johannes Vermeer",
    "Diego Velazquez", "Francisco Goya", "J.M.W. Turner",
    "Andy Warhol", "Jackson Pollock", "Mark Rothko", "Georgia O'Keeffe",
    "Frida Kahlo", "Marc Chagall", "Joan Miro", "Rene Magritte",
    "Paul Klee", "Piet Mondrian", "Egon Schiele", "Amedeo Modigliani",
    "Hieronymus Bosch", "El Greco", "Titian", "Sandro Botticelli",
    "Jan van Eyck", "Katsushika Hokusai", "Caspar David Friedrich",
]


def _artists_from_csv() -> list[str]:
    if not METADATA_CSV.exists():
        return []
    out: set[str] = set()
    try:
        with METADATA_CSV.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # ArtBench typically has "artist" column; fall back to common variants.
                for key in ("artist", "Artist", "author", "painter"):
                    if key in row and row[key]:
                        a = row[key].strip().replace("_", " ").title()
                        if a:
                            out.add(a)
                        break
    except Exception:
        return []
    return sorted(out)


def get_artists() -> list[str]:
    csv_list = _artists_from_csv()
    return csv_list if csv_list else _DEFAULT_ARTISTS


ARTISTS = get_artists()

# CLIP
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "openai"

# ResNet50
RESNET_WEIGHTS = MODELS_DIR / "resnet_model.pth"
RESNET_NUM_CLASSES = len(ARTBENCH_STYLES)
RESNET_INPUT_SIZE = 224

# UI defaults
APP_TITLE = "LensArt — Art Recognition and Discovery"
TOP_K_ARTIST = 5
TOP_K_SIMILAR = 5

# Enrichment
USER_AGENT = "LensArt/1.0 (academic; karthikjavvaji97@gmail.com)"
WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/summary"
MET_API = "https://collectionapi.metmuseum.org/public/collection/v1"
REQUEST_TIMEOUT = 10
