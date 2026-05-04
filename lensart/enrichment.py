"""Wikipedia (primary) + Met Museum (fallback) artist biography enrichment."""
from __future__ import annotations

import urllib.parse
from typing import Optional

import requests

from . import config


def _wiki_lookup(name: str) -> Optional[dict]:
    title = urllib.parse.quote(name.replace(" ", "_"))
    url = f"{config.WIKI_API}/{title}"
    headers = {"User-Agent": config.USER_AGENT, "Accept": "application/json"}
    try:
        r = requests.get(url, headers=headers, timeout=config.REQUEST_TIMEOUT)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    try:
        data = r.json()
    except ValueError:
        return None
    extract = (data.get("extract") or "").strip()
    if len(extract) < 60:
        return None
    return {
        "title":         data.get("title", name),
        "summary":       extract,
        "url":           data.get("content_urls", {}).get("desktop", {}).get("page", ""),
        "thumbnail_url": (data.get("thumbnail") or {}).get("source"),
        "source":        "Wikipedia",
    }


def _met_lookup(name: str) -> Optional[dict]:
    headers = {"User-Agent": config.USER_AGENT, "Accept": "application/json"}
    try:
        r = requests.get(
            f"{config.MET_API}/search",
            params={"q": name, "hasImages": "true"},
            headers=headers, timeout=config.REQUEST_TIMEOUT,
        )
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    ids = (r.json() or {}).get("objectIDs") or []
    for oid in ids[:8]:
        try:
            o = requests.get(f"{config.MET_API}/objects/{oid}",
                             headers=headers,
                             timeout=config.REQUEST_TIMEOUT).json()
        except (requests.RequestException, ValueError):
            continue
        if name.lower() not in (o.get("artistDisplayName") or "").lower():
            continue
        title = o.get("title", "Untitled")
        year = o.get("objectDate", "")
        bio = o.get("artistDisplayBio") or ""
        return {
            "title":         o.get("artistDisplayName") or name,
            "summary":       (
                f"{name} is represented in the Metropolitan Museum of Art collection. "
                f"{bio} A representative work in the collection is "
                f"\"{title}\" ({year})."
            ).strip(),
            "url":           o.get("objectURL", ""),
            "thumbnail_url": o.get("primaryImageSmall") or o.get("primaryImage"),
            "source":        "Met Museum",
        }
    return None


def enrich(name: str) -> dict:
    """Always returns a non-None biography dict, even if APIs fail."""
    out = _wiki_lookup(name) or _met_lookup(name)
    if out is None:
        out = {
            "title":         name,
            "summary":       "No biographical context available right now. "
                              "Wikipedia and the Met Museum did not return a result.",
            "url":           "",
            "thumbnail_url": None,
            "source":        "—",
        }
    return out
