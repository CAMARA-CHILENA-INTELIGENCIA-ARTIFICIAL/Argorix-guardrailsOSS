from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from guardrails import paths, taxonomy
from guardrails.evaluation import metrics
from guardrails.serving.normalization import (
    merge_results,
    normalize_for_guardrail,
)

ROOT = paths.PROJECT_ROOT
APP_DIR = paths.STATIC_DIR.parent
WORKER_PATH = paths.WORKER_PATH
DEFAULT_WSL_DISTRO = os.environ.get("GUARDRAIL_WSL_DISTRO", "kali-linux")
DEFAULT_PROJECT = paths.wsl_project_root()
DEFAULT_VENV = os.environ.get("GUARDRAIL_WSL_VENV", "/home/root/.venvs/guardrails-governance")
DEFAULT_BASE_MODEL = os.environ.get("GUARDRAIL_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
TRAIN_RUNS = {
    "v2": {
        "label": "QLoRA v2",
        "dir": ROOT / "models" / "guardrail-qwen25-1_5b-qlora-v2",
        "stdout": ROOT / "logs" / "train_v2_stdout.log",
        "stderr": ROOT / "logs" / "train_v2_stderr.log",
        "max_steps": int(os.environ.get("GUARDRAIL_TRAIN_V2_MAX_STEPS", "4000")),
    },
    "v3": {
        "label": "QLoRA v3 corrective",
        "dir": ROOT / "models" / "guardrail-qwen25-1_5b-qlora-v3-corrective",
        "stdout": ROOT / "logs" / "train_v3_stdout.log",
        "stderr": ROOT / "logs" / "train_v3_stderr.log",
        "max_steps": int(os.environ.get("GUARDRAIL_TRAIN_V3_MAX_STEPS", "1000")),
    },
}
GOLDEN_EVAL = {
    "label": "Golden eval v3 corrective",
    "adapter": "models/guardrail-qwen25-1_5b-qlora-v3-corrective",
    "input_csv": ROOT / "eval_golden_es.csv",
    "predictions": ROOT / "reports" / "v3_golden_predictions.jsonl",
    "metrics": ROOT / "reports" / "v3_golden_metrics.json",
    "stdout": ROOT / "logs" / "eval_v3_golden_stdout.log",
    "stderr": ROOT / "logs" / "eval_v3_golden_stderr.log",
}


PROGRESS_RE = re.compile(
    r"(?P<stage>[^:\r\n]+):\s+"
    r"(?P<pct>\d+)%\|.*?\|\s+"
    r"(?P<done>\d+)/(?P<total>\d+)\s+"
    r"\[(?P<elapsed>[^<\]]+)(?:<(?P<eta>[^,\]]+))?"
)
TRAIN_PROGRESS_RE = re.compile(
    r"^\s*(?P<pct>\d+)%\|.*?\|\s+"
    r"(?P<done>\d+)/(?P<total>\d+)\s+"
    r"\[(?P<elapsed>[^<\]]+)(?:<(?P<eta>[^,\]]+))?"
)
EVAL_PROGRESS_RE = re.compile(r"eval_progress\s+(?P<done>\d+)/(?P<total>\d+)")
def select_default_adapter() -> str:
    configured = os.environ.get("GUARDRAIL_ADAPTER")
    if configured:
        return configured

    preferred = [
        ROOT / "models" / "guardrail-qwen25-1_5b-qlora-v3-corrective",
        ROOT / "models" / "guardrail-qwen25-1_5b-qlora-v2",
        ROOT / "models" / "guardrail-qwen25-1_5b-qlora-v1",
    ]
    for path in preferred:
        if (path / "adapter_config.json").exists() or (path / "adapter_model.safetensors").exists():
            return str(path.relative_to(ROOT)).replace("\\", "/")
    return "models/guardrail-qwen25-1_5b-qlora-v1"


DEFAULT_ADAPTER = select_default_adapter()


def wsl_project_path(path: Path) -> str:
    relative = path.resolve().relative_to(ROOT.resolve())
    return f"{DEFAULT_PROJECT}/{relative.as_posix()}"


def tail_lines(path: Path, limit: int = 80) -> list[str]:
    if not path.exists():
        return []
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]
    except OSError:
        return []


def latest_progress(lines: list[str]) -> dict[str, Any] | None:
    for line in reversed(lines):
        clean_line = line.replace("\r", "")
        match = PROGRESS_RE.search(clean_line)
        if not match:
            train_match = TRAIN_PROGRESS_RE.search(clean_line)
            if not train_match:
                continue
            done = int(train_match.group("done"))
            total = int(train_match.group("total"))
            return {
                "stage": "Entrenando QLoRA",
                "percent": round((done / total) * 100, 2) if total else int(train_match.group("pct")),
                "done": done,
                "total": total,
                "elapsed": train_match.group("elapsed"),
                "eta": train_match.group("eta") or "",
            }
        done = int(match.group("done"))
        total = int(match.group("total"))
        pct = int(match.group("pct"))
        return {
            "stage": match.group("stage").strip(),
            "percent": pct,
            "done": done,
            "total": total,
            "elapsed": match.group("elapsed"),
            "eta": match.group("eta") or "",
        }
    return None


def gpu_status() -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except Exception:
        return {}
    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 5:
        return {}
    return {
        "utilization": float(parts[0]),
        "memory_used_mb": float(parts[1]),
        "memory_total_mb": float(parts[2]),
        "temperature_c": float(parts[3]),
        "power_w": float(parts[4]),
    }


def running_command_fragment(fragment: str) -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "wsl",
                "-d",
                DEFAULT_WSL_DISTRO,
                "--",
                "bash",
                "-lc",
                f"ps -eo pid,etime,cmd | grep {json.dumps(fragment)} | grep -v grep || true",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
    except Exception:
        return {"running": False}

    lines = [line.strip() for line in result.stdout.splitlines() if fragment in line]
    line = lines[0] if lines else ""
    if not line:
        return {"running": False}
    parts = line.split(None, 2)
    return {
        "running": True,
        "pid": int(parts[0]) if parts and parts[0].isdigit() else None,
        "elapsed": parts[1] if len(parts) > 1 else "",
        "command": parts[2] if len(parts) > 2 else line,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        return rows
    return rows


def golden_total(path: Path) -> int:
    if not path.exists():
        return 1000
    try:
        return max(0, len(path.read_text(encoding="utf-8", errors="replace").splitlines()) - 1)
    except OSError:
        return 1000


def golden_eval_snapshot() -> dict[str, Any]:
    predictions = read_jsonl(GOLDEN_EVAL["predictions"])
    total = golden_total(GOLDEN_EVAL["input_csv"])
    done = len(predictions)
    stdout = tail_lines(GOLDEN_EVAL["stdout"], 40)
    stderr = tail_lines(GOLDEN_EVAL["stderr"], 30)
    process = running_command_fragment("eval_guardrail.py")
    final_metrics = None
    if GOLDEN_EVAL["metrics"].exists():
        try:
            final_metrics = json.loads(GOLDEN_EVAL["metrics"].read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            final_metrics = None

    last_progress = None
    for line in reversed(stdout):
        match = EVAL_PROGRESS_RE.search(line)
        if match:
            last_progress = {"done": int(match.group("done")), "total": int(match.group("total"))}
            break

    reporte = metrics.Report()
    expected_counts: dict[str, int] = {}
    predicted_counts: dict[str, int] = {}
    decision_counts: dict[str, int] = {}

    for row in predictions:
        expected = row.get("expected") or {}
        prediction = row.get("prediction") or {}
        if not expected.get("expected_decision"):
            continue
        observacion = metrics.observation_from(
            expected, prediction, group_id=str(row.get("group_id") or "")
        )
        reporte.add(observacion)
        expected_counts[observacion.expected_label] = (
            expected_counts.get(observacion.expected_label, 0) + 1
        )
        predicted_counts[observacion.predicted_label] = (
            predicted_counts.get(observacion.predicted_label, 0) + 1
        )
        decision_counts[observacion.predicted_decision] = (
            decision_counts.get(observacion.predicted_decision, 0) + 1
        )

    safe_seen = sum(1 for o in reporte.observations if o.expected_decision == taxonomy.ALLOW)
    block_seen = sum(1 for o in reporte.observations if o.expected_decision == taxonomy.BLOCK)
    confusion = (
        reporte.as_dict().get("primary_label", {}).get("confusion", {})
        if reporte.observations
        else {}
    )

    percent = round((done / total) * 100, 2) if total else 0
    status = "finished" if final_metrics else ("running" if process.get("running") else "idle")
    parciales = reporte.as_dict() if reporte.observations else {"total": 0}
    metricas = final_metrics or parciales

    top_confusions = sorted(
        [{"pair": key, "count": value} for key, value in confusion.items()],
        key=lambda item: item["count"],
        reverse=True,
    )[:10]
    recent = []
    for row in predictions[-12:]:
        expected = row.get("expected") or {}
        prediction = row.get("prediction") or {}
        recent.append(
            {
                "text": row.get("text", ""),
                "expected_decision": expected.get("expected_decision"),
                "expected_label": expected.get("expected_primary_label"),
                "predicted_decision": prediction.get("decision"),
                "predicted_label": prediction.get("primary_label"),
                "ok": expected.get("expected_decision") == prediction.get("decision")
                and expected.get("expected_primary_label") == prediction.get("primary_label"),
            }
        )

    return {
        "label": GOLDEN_EVAL["label"],
        "adapter": GOLDEN_EVAL["adapter"],
        "status": status,
        "done": done,
        "total": total,
        "percent": percent,
        "progress": last_progress,
        "process": process,
        "metrics": metricas,
        "safe_seen": safe_seen,
        "block_seen": block_seen,
        "expected_counts": expected_counts,
        "predicted_counts": predicted_counts,
        "decision_counts": decision_counts,
        "top_confusions": top_confusions,
        "recent": recent,
        "stdout": stdout[-20:],
        "stderr": stderr[-20:],
        "gpu": gpu_status(),
        "paths": {
            "predictions": str(GOLDEN_EVAL["predictions"].relative_to(ROOT)),
            "metrics": str(GOLDEN_EVAL["metrics"].relative_to(ROOT)),
        },
    }


def active_train_run() -> str:
    for name in ("v3", "v2"):
        run = TRAIN_RUNS[name]
        state_path = run["dir"] / "train_state.json"
        if not state_path.exists():
            continue
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if state.get("status") != "finished":
            return name
    return "v3" if (TRAIN_RUNS["v3"]["dir"] / "train_state.json").exists() else "v2"


def train_snapshot(run_name: str | None = None) -> dict[str, Any]:
    run_name = run_name or active_train_run()
    run = TRAIN_RUNS.get(run_name, TRAIN_RUNS["v2"])
    train_dir = run["dir"]
    state_path = train_dir / "train_state.json"
    stderr = tail_lines(run["stderr"], 120)
    stdout = tail_lines(run["stdout"], 60)
    state = None
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = None

    progress = latest_progress(stderr)
    status = "preparing"
    percent = progress["percent"] if progress else 0
    eta = progress["eta"] if progress else ""
    stage = progress["stage"] if progress else "Esperando logs"
    step = 0
    max_steps = int(run["max_steps"])
    logs = {}

    if state:
        status = state.get("status", "running")
        step = int(state.get("global_step") or 0)
        logs = state.get("logs") or {}
        if status == "finished":
            percent = 100
            eta = "00:00"
            stage = "Entrenamiento finalizado"
        else:
            if step > 0:
                percent = round((step / max_steps) * 100, 2) if max_steps else 0
            elif progress and progress["stage"] == "Entrenando QLoRA":
                step = int(progress["done"])
                max_steps = int(progress["total"])
                percent = progress["percent"]
            stage = "Entrenando QLoRA"
            eta = progress["eta"] if progress and progress["stage"] == "Entrenando QLoRA" else ""

    if not state and progress and progress["stage"].lower().startswith("tokenizing"):
        status = "tokenizing"
    elif not state and progress and "adding eos" in progress["stage"].lower():
        status = "preparing"

    return {
        "status": status,
        "stage": stage,
        "percent": percent,
        "eta": eta,
        "progress": progress,
        "step": step,
        "max_steps": max_steps,
        "logs": logs,
        "state": state,
        "stdout": stdout[-20:],
        "stderr": stderr[-30:],
        "run": run_name,
        "label": run["label"],
        "output_dir": str(train_dir.relative_to(ROOT)),
        "has_output": train_dir.exists(),
        "gpu": gpu_status(),
    }


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1)


class AnalyzeBatchRequest(BaseModel):
    items: list[str | dict[str, Any]]


class WorkerClient:
    def __init__(self) -> None:
        self.process: subprocess.Popen[str] | None = None
        self.lock = threading.Lock()
        self.stderr_lines: queue.Queue[str] = queue.Queue(maxsize=200)
        self.started_at: float | None = None

    def start(self) -> None:
        if self.process and self.process.poll() is None:
            return

        command = [
            "wsl",
            "-d",
            DEFAULT_WSL_DISTRO,
            "--",
            "bash",
            "-lc",
            (
                f"cd {json.dumps(DEFAULT_PROJECT)} && "
                f"source {json.dumps(DEFAULT_VENV + '/bin/activate')} && "
                f"python {json.dumps(wsl_project_path(WORKER_PATH))} "
                f"--base-model {json.dumps(DEFAULT_BASE_MODEL)} "
                f"--adapter {json.dumps(DEFAULT_ADAPTER)} "
                "--load-4bit"
            ),
        ]
        self.process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self.started_at = time.time()
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def _drain_stderr(self) -> None:
        assert self.process and self.process.stderr
        for line in self.process.stderr:
            try:
                self.stderr_lines.put_nowait(line.rstrip())
            except queue.Full:
                try:
                    self.stderr_lines.get_nowait()
                except queue.Empty:
                    pass
                self.stderr_lines.put_nowait(line.rstrip())

    def analyze(self, text: str) -> dict[str, Any]:
        with self.lock:
            self.start()
            assert self.process and self.process.stdin and self.process.stdout
            if self.process.poll() is not None:
                raise HTTPException(status_code=503, detail="Model worker stopped unexpectedly.")

            request_id = uuid.uuid4().hex
            payload = {"id": request_id, "text": text}
            self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.process.stdin.flush()

            line = self.process.stdout.readline()
            if not line:
                raise HTTPException(status_code=503, detail="Model worker returned no output.")

            response = json.loads(line)
            if response.get("id") != request_id:
                raise HTTPException(status_code=500, detail="Model worker response id mismatch.")
            if "error" in response:
                raise HTTPException(status_code=500, detail=response["error"])
            return response

    def status(self) -> dict[str, Any]:
        running = self.process is not None and self.process.poll() is None
        return {
            "running": running,
            "pid": self.process.pid if self.process else None,
            "started_at": self.started_at,
            "adapter": DEFAULT_ADAPTER,
            "base_model": DEFAULT_BASE_MODEL,
            "wsl_distro": DEFAULT_WSL_DISTRO,
            "logs": list(self.stderr_lines.queue)[-20:],
        }


worker = WorkerClient()
app = FastAPI(title="Guardrail Governance", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    # Lazy-starting still happens on first request; this warms the process in the background.
    threading.Thread(target=worker.start, daemon=True).start()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(APP_DIR / "static" / "index.html")


@app.get("/train")
def train_dashboard() -> FileResponse:
    return FileResponse(APP_DIR / "static" / "train.html")


@app.get("/eval")
def eval_dashboard() -> FileResponse:
    return FileResponse(APP_DIR / "static" / "eval.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "worker": worker.status()}


@app.get("/api/train-v2/status")
def train_v2_status() -> dict[str, Any]:
    return train_snapshot("v2")


@app.get("/api/train/status")
def train_status(run: str | None = None) -> dict[str, Any]:
    return train_snapshot(run)


@app.get("/api/eval/golden-status")
def eval_golden_status() -> dict[str, Any]:
    return golden_eval_snapshot()


def analyze_text(text: str) -> dict[str, Any]:
    started = time.perf_counter()
    original = worker.analyze(text)
    normalization = normalize_for_guardrail(text)
    normalized_result = None
    if normalization["changed"] and normalization["score"] > 0:
        normalized_result = worker.analyze(normalization["text"])
    result = merge_results(original, normalized_result, normalization)
    result["text"] = text
    result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return result


@app.post("/api/analyze")
def analyze(request: AnalyzeRequest) -> dict[str, Any]:
    return analyze_text(request.text)


def item_to_text(item: str | dict[str, Any]) -> str:
    if isinstance(item, str):
        return item
    for key in ("text", "prompt", "input", "content", "attack"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return json.dumps(item, ensure_ascii=False)


@app.post("/api/analyze-batch")
def analyze_batch(request: AnalyzeBatchRequest) -> dict[str, Any]:
    if not request.items:
        raise HTTPException(status_code=400, detail="items cannot be empty")
    if len(request.items) > 200:
        raise HTTPException(status_code=400, detail="maximum batch size is 200")

    started = time.perf_counter()
    results = []
    for index, item in enumerate(request.items):
        text = item_to_text(item)
        result = analyze_text(text)
        result["index"] = index
        results.append(result)

    return {
        "count": len(results),
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "results": results,
    }


@app.post("/api/upload-json")
async def upload_json(file: UploadFile = File(...)) -> dict[str, Any]:
    raw = await file.read()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid JSON: {exc}") from exc

    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict) and isinstance(payload.get("items"), list):
        items = payload["items"]
    else:
        raise HTTPException(status_code=400, detail="JSON must be a list or an object with items[]")

    return analyze_batch(AnalyzeBatchRequest(items=items))


app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
