"""Configuración de la batería de tests.

Varios módulos del paquete importan dependencias pesadas en el nivel superior
(``torch``, ``transformers``, ``peft``, ``fastapi``, ``pydantic``). Los tests de
caracterización sólo ejercitan funciones puras, así que aquí se instalan
sustitutos mínimos cuando la dependencia real no está disponible.

Esto permite ejecutar ``uv run pytest`` con solo ``uv sync --group dev``, sin
descargar la pila de entrenamiento ni requerir GPU. Si las dependencias reales
sí están instaladas, no se sustituye nada.
"""

from __future__ import annotations

import sys
import types


def _ensure(name: str, build: object) -> None:
    """Registra un módulo sustituto sólo si el real no puede importarse."""
    try:
        __import__(name)
    except ImportError:
        sys.modules[name] = build()  # type: ignore[operator]


def _stub_torch() -> types.ModuleType:
    module = types.ModuleType("torch")
    module.float16 = "float16"  # type: ignore[attr-defined]
    module.bfloat16 = "bfloat16"  # type: ignore[attr-defined]
    module.inference_mode = lambda *a, **k: None  # type: ignore[attr-defined]
    return module


def _stub_transformers() -> types.ModuleType:
    module = types.ModuleType("transformers")
    for attr in (
        "AutoModelForCausalLM",
        "AutoModelForSeq2SeqLM",
        "AutoTokenizer",
        "BitsAndBytesConfig",
        "TrainerCallback",
    ):
        setattr(module, attr, object)
    utils = types.ModuleType("transformers.utils")
    logging_mod = types.ModuleType("transformers.utils.logging")
    logging_mod.set_verbosity_error = lambda *a, **k: None  # type: ignore[attr-defined]
    utils.logging = logging_mod  # type: ignore[attr-defined]
    sys.modules["transformers.utils"] = utils
    sys.modules["transformers.utils.logging"] = logging_mod
    return module


def _stub_peft_full() -> types.ModuleType:
    module = types.ModuleType("peft")
    module.PeftModel = object  # type: ignore[attr-defined]
    module.LoraConfig = object  # type: ignore[attr-defined]
    module.prepare_model_for_kbit_training = lambda *a, **k: None  # type: ignore[attr-defined]
    return module


def _stub_datasets() -> types.ModuleType:
    module = types.ModuleType("datasets")
    module.Dataset = object  # type: ignore[attr-defined]
    module.load_dataset = lambda *a, **k: None  # type: ignore[attr-defined]
    return module


def _stub_tqdm() -> types.ModuleType:
    module = types.ModuleType("tqdm")
    module.tqdm = lambda iterable=None, *a, **k: iterable  # type: ignore[attr-defined]
    return module


def _stub_pyarrow_parquet() -> types.ModuleType:
    module = types.ModuleType("pyarrow.parquet")
    module.ParquetFile = object  # type: ignore[attr-defined]
    return module


def _stub_trl() -> types.ModuleType:
    module = types.ModuleType("trl")
    module.SFTConfig = object  # type: ignore[attr-defined]
    module.SFTTrainer = object  # type: ignore[attr-defined]
    return module


def _stub_fastapi() -> types.ModuleType:
    module = types.ModuleType("fastapi")

    class _App:
        def __init__(self, *a: object, **k: object) -> None: ...
        def add_middleware(self, *a: object, **k: object) -> None: ...
        def mount(self, *a: object, **k: object) -> None: ...
        def on_event(self, *a: object, **k: object):
            return lambda fn: fn

        def get(self, *a: object, **k: object):
            return lambda fn: fn

        def post(self, *a: object, **k: object):
            return lambda fn: fn

    module.FastAPI = _App  # type: ignore[attr-defined]
    module.HTTPException = type("HTTPException", (Exception,), {})  # type: ignore[attr-defined]
    module.UploadFile = object  # type: ignore[attr-defined]
    module.File = lambda *a, **k: None  # type: ignore[attr-defined]

    cors = types.ModuleType("fastapi.middleware.cors")
    cors.CORSMiddleware = object  # type: ignore[attr-defined]
    responses = types.ModuleType("fastapi.responses")

    class _FileResponse:
        def __init__(self, *a: object, **k: object) -> None: ...

    responses.FileResponse = _FileResponse  # type: ignore[attr-defined]
    staticfiles = types.ModuleType("fastapi.staticfiles")

    class _StaticFiles:
        def __init__(self, *a: object, **k: object) -> None: ...

    staticfiles.StaticFiles = _StaticFiles  # type: ignore[attr-defined]
    middleware = types.ModuleType("fastapi.middleware")

    sys.modules["fastapi.middleware"] = middleware
    sys.modules["fastapi.middleware.cors"] = cors
    sys.modules["fastapi.responses"] = responses
    sys.modules["fastapi.staticfiles"] = staticfiles
    return module


def _stub_pydantic() -> types.ModuleType:
    module = types.ModuleType("pydantic")

    class BaseModel:
        def __init__(self, **kwargs: object) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    module.BaseModel = BaseModel  # type: ignore[attr-defined]
    module.Field = lambda *a, **k: None  # type: ignore[attr-defined]
    return module


_ensure("torch", _stub_torch)
_ensure("peft", _stub_peft_full)
_ensure("transformers", _stub_transformers)
_ensure("datasets", _stub_datasets)
_ensure("trl", _stub_trl)
_ensure("tqdm", _stub_tqdm)
_ensure("fastapi", _stub_fastapi)
_ensure("pydantic", _stub_pydantic)
