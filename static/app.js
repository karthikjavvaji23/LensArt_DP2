// LensArt hybrid front-end controller — with model attribution
const $ = (id) => document.getElementById(id);
const empty   = $("empty");
const loading = $("loading");
const output  = $("output");
const errorBox = $("error");

const show = el => el.classList.remove("hidden");
const hide = el => el.classList.add("hidden");
function showOnly(el) {
  for (const x of [empty, loading, output, errorBox]) hide(x);
  show(el);
}
const fmtPct = p => (100 * p).toFixed(1) + "%";

function bar(item) {
  const label = (item.label || "").replace(/_/g, " ");
  const pct = Math.max(0, Math.min(item.confidence || 0, 1)) * 100;
  return `
    <div class="bar">
      <div class="bar-label">
        <span class="label">${label}</span>
        <span class="pct">${fmtPct(item.confidence || 0)}</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill" style="width:${pct.toFixed(1)}%"></div>
      </div>
    </div>`;
}

function renderTimings(t) {
  if (!t) return "";
  return `
    <span class="stage"><strong>Total ${t.total_ms.toFixed(0)} ms</strong></span>
    <span class="stage">Style ${t.style_ms.toFixed(0)} ms (ResNet)</span>
    <span class="stage">Artist ${t.artist_ms.toFixed(0)} ms (CLIP)</span>
    <span class="stage">Similarity ${t.similarity_ms.toFixed(0)} ms (CLIP)</span>
    <span class="stage">Biography ${t.biography_ms.toFixed(0)} ms (Wiki)</span>`;
}

function renderBio(bio) {
  if (!bio || !bio.summary) {
    return `<i style="color:#5a4a3a;">No biographical context available right now.</i>`;
  }
  const link = bio.url
    ? `<a href="${bio.url}" target="_blank" rel="noopener">Read more on ${bio.source} →</a>`
    : "";
  const thumb = bio.thumbnail_url
    ? `<div class="bio-thumb"><img src="${bio.thumbnail_url}" alt=""></div>`
    : "";
  return `
    <div class="bio-row">
      ${thumb}
      <div>
        <h4>${bio.title || ""}</h4>
        <p>${bio.summary}</p>
        ${link}
      </div>
    </div>`;
}

function renderSimilar(items) {
  if (!items || items.length === 0) {
    return `<div class="card"><i style="color:#5a4a3a;">
            Index is empty — drop reference images into sample_images/ or
            data/images/train/&lt;style&gt;/ and restart.</i></div>`;
  }
  return items.map(it => `
    <div class="tile">
      <img src="${it.url}" alt="">
      <div class="tile-title">${it.title}</div>
      <div class="tile-meta">${(it.style || "").replace(/_/g, " ")} · sim ${it.similarity.toFixed(2)}</div>
    </div>
  `).join("");
}

// Update the top model-status badges based on what's actually loaded
function updateModelStatus(models) {
  if (!models) return;
  const bar = $("model-status");
  if (!bar) return;
  bar.innerHTML = `
    <span class="ms-badge ms-resnet ${models.style.active ? '' : 'ms-inactive'}">
      ResNet50 — style ${models.style.active ? '✓' : '(weights missing)'}
    </span>
    <span class="ms-badge ms-clip">CLIP — artist ✓</span>
    <span class="ms-badge ms-clip">CLIP — similarity ✓</span>
    <span class="ms-badge ms-wiki">Wikipedia — biography ✓</span>
  `;
}

// On page load, fetch /api/model-status to show current state before any prediction
fetch("/api/model-status").then(r => r.ok ? r.json() : null).then(s => {
  if (s) updateModelStatus(s);
}).catch(() => {});

async function predict(formData, previewSrc) {
  showOnly(loading);
  try {
    const r = await fetch("/api/predict", { method: "POST", body: formData });
    if (!r.ok) {
      const text = await r.text().catch(() => "");
      throw new Error(`Server returned ${r.status}: ${text}`);
    }
    const data = await r.json();

    $("preview").src = previewSrc;
    $("backend").textContent = "Inference backend: " + data.backend;
    $("timing").innerHTML  = renderTimings(data.timings);
    $("style-bars").innerHTML  = (data.style_top  || []).map(bar).join("");
    $("artist-bars").innerHTML = (data.artist_top || []).map(bar).join("");
    $("bio").innerHTML         = renderBio(data.biography_full);
    $("similar").innerHTML     = renderSimilar(data.similar_images);
    updateModelStatus(data.models);
    showOnly(output);
  } catch (err) {
    errorBox.textContent = "Error: " + err.message;
    showOnly(errorBox);
  }
}

$("upload-form").addEventListener("submit", e => {
  e.preventDefault();
  const file = $("file-input").files[0];
  if (!file) {
    errorBox.textContent = "Please choose a file first, or pick a sample.";
    showOnly(errorBox);
    return;
  }
  const fd = new FormData();
  fd.append("image", file);
  predict(fd, URL.createObjectURL(file));
});

document.querySelectorAll(".sample-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    const name = btn.dataset.sample;
    const fd = new FormData();
    fd.append("sample", name);
    predict(fd, "/sample/" + encodeURIComponent(name));
  });
});

$("file-input").addEventListener("change", e => {
  const f = e.target.files[0];
  document.querySelector(".file-drop span").textContent =
    f ? f.name : "Drop a JPEG or PNG, or click to browse";
});
