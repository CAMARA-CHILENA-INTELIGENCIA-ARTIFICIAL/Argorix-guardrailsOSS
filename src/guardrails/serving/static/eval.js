const $ = (id) => document.getElementById(id);

let selectedPair = null;

const labels = [
  "SAFE",
  "PROMPT_INJECTION",
  "JAILBREAK",
  "HARMFUL",
  "HATE",
  "VIOLENCE",
  "SEXUAL",
  "POLITICS",
  "INVALID",
];

function fmt(value, digits = 1) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) return "--";
  return Number(value)
    .toFixed(digits)
    .replace(/(\.\d*?)0+$/, "$1")
    .replace(/\.$/, "");
}

function pct(value) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) return "--";
  return `${fmt(Number(value) * 100, 1)}%`;
}

function escapeHtml(value) {
  return String(value || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function cleanLog(line) {
  return String(line || "")
    .replace(/[^\x09\x0a\x0d\x20-\x7eáéíóúñÁÉÍÓÚÑüÜ¿¡]/g, "")
    .trim();
}

function renderFlow(data) {
  const metrics = data.metrics || {};
  const done = data.done || 0;
  const total = data.total || 1000;
  const fp = metrics.false_positives || 0;
  const fn = metrics.false_negatives || 0;
  const ok = Math.max(0, (metrics.decision_correct || 0) - fp - fn);
  const blockSeen = data.block_seen || 0;
  const safeSeen = data.safe_seen || 0;
  const svg = $("flowDiagram");
  const goodWidth = Math.max(8, Math.min(240, ok * 1.2));
  const fpWidth = Math.max(8, Math.min(220, fp * 1.2));
  const fnWidth = Math.max(8, Math.min(220, fn * 2));

  svg.innerHTML = `
    <defs>
      <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="#b7ff39"></path>
      </marker>
      <filter id="glow"><feGaussianBlur stdDeviation="4" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
    </defs>
    <g font-family="Bahnschrift, Segoe UI, sans-serif">
      <rect x="40" y="92" width="210" height="126" rx="8" fill="#11170f" stroke="#33402f"/>
      <text x="64" y="130" fill="#a1aa9a" font-size="13" font-weight="900">GOLDEN SET</text>
      <text x="64" y="170" fill="#f8f3e8" font-size="34" font-weight="900">${done}/${total}</text>
      <text x="64" y="198" fill="#73e0c6" font-size="15">SAFE ${safeSeen} · BLOCK ${blockSeen}</text>

      <path d="M260 155 C330 155 330 155 402 155" stroke="#b7ff39" stroke-width="4" fill="none" marker-end="url(#arrow)" filter="url(#glow)"/>

      <rect x="410" y="72" width="260" height="166" rx="8" fill="#11170f" stroke="#33402f"/>
      <text x="438" y="112" fill="#a1aa9a" font-size="13" font-weight="900">ADAPTER V3 CORRECTIVE</text>
      <text x="438" y="154" fill="#f8f3e8" font-size="28" font-weight="900">Qwen 2.5 QLoRA</text>
      <text x="438" y="188" fill="#73e0c6" font-size="16">${data.status || "--"} · GPU ${fmt(data.gpu?.utilization, 0)}%</text>

      <path d="M680 155 C744 155 744 155 810 155" stroke="#73e0c6" stroke-width="4" fill="none" marker-end="url(#arrow)" filter="url(#glow)"/>

      <rect x="820" y="36" width="248" height="92" rx="8" fill="#10170e" stroke="#73e0c6"/>
      <text x="846" y="72" fill="#a1aa9a" font-size="13" font-weight="900">CORRECTAS</text>
      <text x="846" y="106" fill="#73e0c6" font-size="34" font-weight="900">${ok}</text>
      <rect x="846" y="116" width="${goodWidth}" height="5" fill="#73e0c6"/>

      <rect x="820" y="150" width="248" height="78" rx="8" fill="#17150d" stroke="#ffd35c"/>
      <text x="846" y="184" fill="#a1aa9a" font-size="13" font-weight="900">FALSE POSITIVES</text>
      <text x="1010" y="184" fill="#ffd35c" font-size="26" font-weight="900">${fp}</text>
      <rect x="846" y="202" width="${fpWidth}" height="5" fill="#ffd35c"/>

      <rect x="820" y="250" width="248" height="78" rx="8" fill="#1a0f0f" stroke="#ff6666"/>
      <text x="846" y="284" fill="#a1aa9a" font-size="13" font-weight="900">FALSE NEGATIVES</text>
      <text x="1010" y="284" fill="#ff6666" font-size="26" font-weight="900">${fn}</text>
      <rect x="846" y="302" width="${fnWidth}" height="5" fill="#ff6666"/>
    </g>
  `;
}

function renderMatrix(data) {
  const confusion = data.metrics?.label_confusion || {};
  const activeLabels = labels.filter((label) => {
    if (label === "INVALID") return Object.keys(confusion).some((key) => key.endsWith("->INVALID"));
    return (
      Object.keys(confusion).some((key) => key.startsWith(`${label}->`)) ||
      Object.keys(confusion).some((key) => key.endsWith(`->${label}`))
    );
  });
  const matrix = $("confusionMatrix");
  matrix.style.setProperty("--cols", activeLabels.length || 1);
  const max = Math.max(1, ...Object.values(confusion));
  let html = `<div class="cell head">Esperado \\ Pred</div>`;
  html += activeLabels.map((label) => `<div class="cell head">${label}</div>`).join("");
  for (const expected of activeLabels) {
    html += `<div class="cell head">${expected}</div>`;
    for (const predicted of activeLabels) {
      const key = `${expected}->${predicted}`;
      const count = confusion[key] || 0;
      const alpha = count ? Math.max(0.14, count / max) : 0;
      const color = expected === predicted ? "115, 224, 198" : "255, 102, 102";
      const selected = selectedPair === key ? " selected" : "";
      html += `<button class="cell value${selected}" data-pair="${key}" data-count="${count}" style="background: rgba(${color}, ${alpha})">${count}</button>`;
    }
  }
  matrix.innerHTML = html;
  matrix.querySelectorAll(".value").forEach((cell) => {
    cell.addEventListener("click", () => {
      selectedPair = cell.dataset.pair;
      $("matrixTitle").textContent = selectedPair;
      renderRecent(window.__lastEvalData);
      renderMatrix(window.__lastEvalData);
    });
  });
}

function renderRecent(data) {
  const rows = data.recent || [];
  const filtered = selectedPair
    ? rows.filter((row) => `${row.expected_label}->${row.predicted_label}` === selectedPair)
    : rows;
  $("recentList").innerHTML =
    filtered
      .slice()
      .reverse()
      .map((row) => {
        const cls = row.ok ? "ok" : "bad";
        return `
          <div class="recent-item ${cls}">
            <div class="tags">
              <span class="tag ${cls}">${row.ok ? "OK" : "ERROR"}</span>
              <span class="tag">Esperado ${escapeHtml(row.expected_decision)} / ${escapeHtml(row.expected_label)}</span>
              <span class="tag">Predicho ${escapeHtml(row.predicted_decision)} / ${escapeHtml(row.predicted_label)}</span>
            </div>
            <p>${escapeHtml(row.text)}</p>
          </div>
        `;
      })
      .join("") || `<div class="recent-item"><p>No hay ejemplos recientes para este filtro.</p></div>`;
}

function renderBias(data) {
  const metrics = data.metrics || {};
  const fpr = metrics.false_positive_rate;
  const fnr = metrics.false_negative_rate;
  $("safeSeen").textContent = data.safe_seen || 0;
  $("blockSeen").textContent = data.block_seen || 0;
  $("allowCount").textContent = data.decision_counts?.ALLOW || 0;
  $("blockCount").textContent = data.decision_counts?.BLOCK || 0;

  if ((data.safe_seen || 0) > 0 && (fpr || 0) === 0) {
    $("biasTitle").textContent = "SAFE estable";
    $("biasCopy").textContent = "Hasta ahora no está castigando ejemplos benignos. Esa era la falla principal de v2.";
  } else if ((fpr || 0) > 0.2) {
    $("biasTitle").textContent = "Sobrebloqueo detectado";
    $("biasCopy").textContent = "El modelo todavía está enviando demasiados ejemplos SAFE a bloqueo. Conviene extraer esos casos para otra ronda correctiva.";
  } else if ((data.block_seen || 0) === 0) {
    $("biasTitle").textContent = "Solo SAFE procesado";
    $("biasCopy").textContent = "Aún falta pasar por ataques reales. La lectura de falsos negativos todavía no existe.";
  } else if ((fnr || 0) > 0.08) {
    $("biasTitle").textContent = "Riesgo permisivo";
    $("biasCopy").textContent = "Bajaron falsos positivos, pero están apareciendo falsos negativos. Hay que reforzar ataques.";
  } else {
    $("biasTitle").textContent = "Balance saludable";
    $("biasCopy").textContent = "La corrección mantiene bajo el sobrebloqueo sin perder demasiada detección de ataques.";
  }
}

function render(data) {
  window.__lastEvalData = data;
  const metrics = data.metrics || {};
  const percent = Math.max(0, Math.min(100, data.percent || 0));
  $("evalLabel").textContent = data.label || "Golden eval";
  $("progress").textContent = `${data.done || 0} / ${data.total || 1000}`;
  $("progressBar").style.width = `${percent}%`;
  $("decisionAccuracy").textContent = pct(metrics.decision_accuracy);
  $("falsePositiveRate").textContent = pct(metrics.false_positive_rate);
  $("falseNegativeRate").textContent = pct(metrics.false_negative_rate);
  $("statusPill").textContent = data.status || "--";
  $("gpu").textContent = data.gpu?.utilization !== undefined ? `${fmt(data.gpu.utilization, 0)}%` : "--";
  $("vram").textContent =
    data.gpu?.memory_used_mb !== undefined
      ? `${fmt(data.gpu.memory_used_mb / 1024, 1)} / ${fmt(data.gpu.memory_total_mb / 1024, 0)} GB`
      : "--";
  $("temp").textContent = data.gpu?.temperature_c !== undefined ? `${fmt(data.gpu.temperature_c, 0)} C` : "--";
  $("pid").textContent = data.process?.pid || "--";
  $("updated").textContent = new Date().toLocaleTimeString();
  $("logs").textContent = [...(data.stdout || []), ...(data.stderr || [])]
    .slice(-16)
    .map(cleanLog)
    .filter(Boolean)
    .join("\n") || "Esperando logs...";
  renderFlow(data);
  renderBias(data);
  renderMatrix(data);
  renderRecent(data);
}

async function refresh() {
  const response = await fetch("/api/eval/golden-status", { cache: "no-store" });
  render(await response.json());
}

$("clearFilter").addEventListener("click", () => {
  selectedPair = null;
  $("matrixTitle").textContent = "Todas las clases";
  if (window.__lastEvalData) {
    renderMatrix(window.__lastEvalData);
    renderRecent(window.__lastEvalData);
  }
});

refresh().catch((error) => {
  $("logs").textContent = error.message;
});
setInterval(() => refresh().catch(() => {}), 4000);
