"""Resolución portable de rutas del proyecto.

Todas las rutas del proyecto se derivan de aquí, de modo que el código no
depende de dónde esté instalado ni de qué sistema operativo lo ejecute.

La raíz se determina, en este orden:

1. la variable de entorno ``GUARDRAIL_PROJECT_ROOT``, si está definida;
2. el primer directorio ascendente que contenga ``pyproject.toml``;
3. como último recurso, tres niveles por encima de este archivo.

Funciona igual en Windows y en Linux porque todo pasa por :mod:`pathlib`.
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV_VAR = "GUARDRAIL_PROJECT_ROOT"


def _discover_root() -> Path:
    override = os.environ.get(_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()

    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return here.parents[2]


PROJECT_ROOT: Path = _discover_root()

PACKAGE_DIR: Path = Path(__file__).resolve().parent
STATIC_DIR: Path = PACKAGE_DIR / "serving" / "static"
WORKER_PATH: Path = PACKAGE_DIR / "serving" / "worker.py"

CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_ES_DIR: Path = PROJECT_ROOT / "data_es"
DATA_FINETUNE_DIR: Path = PROJECT_ROOT / "data_finetune"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
EXAMPLES_DIR: Path = PROJECT_ROOT / "examples"


def relative_to_root(path: Path) -> str:
    """Devuelve ``path`` relativa a la raíz, con separadores POSIX.

    Se usa para construir rutas que se envían a un intérprete Linux (WSL o
    contenedor) desde un proceso que puede estar corriendo en Windows.
    """
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def to_wsl_path(path: Path | str) -> str:
    """Traduce una ruta de Windows a su equivalente montada en WSL.

    ``D:\\Proyectos\\X`` -> ``/mnt/<letra>/Proyectos/X``. Una ruta POSIX se
    devuelve sin cambios, de modo que la función es segura de llamar en Linux.
    """
    raw = str(path)
    drive, sep, rest = raw.partition(":")
    if sep and len(drive) == 1 and drive.isalpha():
        tail = rest.replace("\\", "/").lstrip("/")
        return f"/mnt/{drive.lower()}/{tail}"
    return raw.replace("\\", "/")


def wsl_project_root() -> str:
    """Raíz del proyecto tal como la ve un intérprete dentro de WSL.

    Puede forzarse con ``GUARDRAIL_WSL_PROJECT`` cuando el montaje no siga la
    convención ``/mnt/<letra>``.
    """
    override = os.environ.get("GUARDRAIL_WSL_PROJECT")
    if override:
        return override
    return to_wsl_path(PROJECT_ROOT)


__all__ = [
    "CONFIGS_DIR",
    "DATA_DIR",
    "DATA_ES_DIR",
    "DATA_FINETUNE_DIR",
    "EXAMPLES_DIR",
    "LOGS_DIR",
    "MODELS_DIR",
    "PACKAGE_DIR",
    "PROJECT_ROOT",
    "REPORTS_DIR",
    "STATIC_DIR",
    "WORKER_PATH",
    "relative_to_root",
    "to_wsl_path",
    "wsl_project_root",
]
