"""Estructura del paquete: importabilidad, portabilidad de rutas y CLI."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from guardrails import cli, paths

MODULOS = [
    "guardrails.data.build_dataset",
    "guardrails.data.corrective",
    "guardrails.data.ingest",
    "guardrails.evaluation.golden_set",
    "guardrails.evaluation.run_eval",
    "guardrails.serving.backend",
    "guardrails.training.qlora",
]


@pytest.mark.parametrize("nombre", MODULOS)
def test_los_modulos_migrados_se_importan(nombre: str) -> None:
    assert importlib.import_module(nombre) is not None


def test_la_raiz_del_proyecto_se_descubre_sola() -> None:
    """La raíz se descubre sola, sin rutas codificadas."""
    assert (paths.PROJECT_ROOT / "pyproject.toml").is_file()
    assert paths.STATIC_DIR.is_dir()
    assert paths.WORKER_PATH.is_file()


def test_no_quedan_rutas_absolutas_en_el_codigo_fuente() -> None:
    """Ninguna ruta absoluta de una máquina concreta en el código fuente."""
    ofensores = [
        ruta
        for ruta in (paths.PROJECT_ROOT / "src").rglob("*.py")
        if "/mnt/d/Proyectos" in ruta.read_text(encoding="utf-8")
    ]
    assert ofensores == []


@pytest.mark.parametrize(
    ("entrada", "esperado"),
    [
        (r"D:\Proyectos\GuardrailsGovernance", "/mnt/d/Proyectos/GuardrailsGovernance"),
        (r"C:\Users\ana\proj", "/mnt/c/Users/ana/proj"),
        ("/home/ana/proj", "/home/ana/proj"),
    ],
)
def test_traduccion_de_rutas_windows_a_wsl(entrada: str, esperado: str) -> None:
    """Windows y Linux comparten el mismo código de serving."""
    assert paths.to_wsl_path(entrada) == esperado


def test_todos_los_subcomandos_del_cli_apuntan_a_modulos_reales() -> None:
    """Cada subcomando apunta a un módulo existente que expone ``main()``.

    Se comprueba sin importar: algunos módulos requieren dependencias
    opcionales (``huggingface_hub``, ``torch``) que no están en el grupo dev.
    """
    for clave, (modulo, _) in cli._COMMANDS.items():
        spec = importlib.util.find_spec(modulo)
        assert spec is not None and spec.origin, f"{clave}: módulo inexistente"

        arbol = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
        funciones = {n.name for n in arbol.body if isinstance(n, ast.FunctionDef)}
        assert "main" in funciones, f"{clave}: {modulo} no define main()"


def test_el_ejecutable_guardrails_responde() -> None:
    resultado = subprocess.run(
        [sys.executable, "-m", "guardrails.cli", "doctor"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        cwd=paths.PROJECT_ROOT,
    )
    assert resultado.returncode == 0, resultado.stderr
    assert "raíz del proyecto" in resultado.stdout
