const $ = (id) => document.getElementById(id);

function fmt(value, digits = 2) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) return "--";
  return Number(value)
    .toFixed(digits)
    .replace(/(\.\d*?)0+$/, "$1")
    .replace(/\.$/, "");
}

function explain(status) {
  if (status === "tokenizing") return "Tokenizando el dataset antes de usar fuerte la GPU.";
  if (status === "preparing") return "Preparando ejemplos y tensores para el trainer.";
  if (status === "running") return "Entrenando adaptadores QLoRA sobre Qwen 2.5.";
  if (status === "finished") return "Entrenamiento terminado. Sigue evaluación.";
  return "Esperando estado del proceso.";
}

function cleanLog(line) {
  return line
    .replace(/[^\x09\x0a\x0d\x20-\x7eáéíóúñÁÉÍÓÚÑüÜ¿¡]/g, "")
    .replace(/\|+/g, "|")
    .trim();
}

async function refresh() {
  const query = new URLSearchParams(window.location.search);
  const run = query.get("run");
  const endpoint = run ? `/api/train/status?run=${encodeURIComponent(run)}` : "/api/train/status";
  const res = await fetch(endpoint);
  const data = await res.json();
  const logs = data.logs || {};
  const gpu = data.gpu || {};
  const pct = Math.max(0, Math.min(100, Number(data.percent || 0)));

  $("runLabel").textContent = data.label || data.run || "QLoRA";
  $("stage").textContent = data.stage || "--";
  $("status").textContent = data.status || "--";
  $("eta").textContent = data.eta || (data.status === "running" ? "calculando" : "--");
  $("percent").textContent = `${fmt(pct, pct < 10 ? 1 : 0)}%`;
  $("detail").textContent = explain(data.status);
  $("step").textContent = `${data.step || 0} / ${data.max_steps || 4000}`;
  $("bar").style.width = `${pct}%`;
  $("loss").textContent = fmt(logs.loss ?? logs.eval_loss, 4);
  $("lr").textContent = logs.learning_rate ? Number(logs.learning_rate).toExponential(2) : "--";
  $("gpu").textContent = gpu.utilization !== undefined ? `${fmt(gpu.utilization, 0)}%` : "--";
  $("vram").textContent =
    gpu.memory_used_mb !== undefined
      ? `${fmt(gpu.memory_used_mb / 1024, 1)} / ${fmt(gpu.memory_total_mb / 1024, 0)} GB`
      : "--";
  $("temp").textContent = gpu.temperature_c !== undefined ? `${fmt(gpu.temperature_c, 0)} C` : "--";
  $("output").textContent = data.output_dir || "--";
  $("updated").textContent = new Date().toLocaleTimeString();

  const lines = (data.stderr || []).slice(-10).map(cleanLog).filter(Boolean);
  $("logText").textContent = lines.length ? lines.join("\n") : "Esperando logs...";
}

refresh().catch((error) => {
  $("stage").textContent = "Monitor offline";
  $("detail").textContent = error.message;
});
setInterval(() => refresh().catch(() => {}), 5000);
