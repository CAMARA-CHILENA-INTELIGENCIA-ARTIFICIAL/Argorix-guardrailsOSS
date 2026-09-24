"""Los registros de ``configs/`` deben coincidir con los lanzadores.

El código todavía no lee ``configs/`` (ver configs/*/README.md). Para que esos
archivos no se conviertan en documentación muerta, estos tests comprueban que
cada valor registrado aparece realmente en el script que lo aplica.

Cuando se cableen los configs como entrada real (tanda 1), estos tests se
sustituyen por la lectura directa.
"""

from __future__ import annotations

import pytest
import yaml

from guardrails import paths

CONFIGS = paths.CONFIGS_DIR
SCRIPTS = paths.PROJECT_ROOT / "scripts"

# (config, script linux, script windows, claves con bandera CLI equivalente)
CASOS = [
    ("train/v1.yaml", "linux/full_train.sh", "windows/full_train.ps1"),
    ("train/v2.yaml", "linux/v2_train.sh", "windows/v2_train.ps1"),
    ("train/v3_corrective.yaml", "linux/v3_corrective_train.sh", "windows/v3_corrective_train.ps1"),
]

BANDERAS = {
    "dataset": "--dataset",
    "model": "--model",
    "output_dir": "--output-dir",
    "adapter_init": "--adapter-init",
    "max_steps": "--max-steps",
    "max_eval_samples": "--max-eval-samples",
    "eval_steps": "--eval-steps",
    "save_steps": "--save-steps",
    "logging_steps": "--logging-steps",
    "train_batch_size": "--train-batch-size",
    "eval_batch_size": "--eval-batch-size",
    "gradient_accumulation_steps": "--gradient-accumulation-steps",
    "max_length": "--max-length",
}


def _cargar(nombre: str) -> dict:
    return yaml.safe_load((CONFIGS / nombre).read_text(encoding="utf-8"))


@pytest.mark.parametrize(("config", "linux", "windows"), CASOS)
def test_el_config_coincide_con_ambos_lanzadores(config: str, linux: str, windows: str) -> None:
    datos = _cargar(config)
    for script in (linux, windows):
        texto = (SCRIPTS / script).read_text(encoding="utf-8")
        for clave, bandera in BANDERAS.items():
            valor = datos.get(clave)
            if valor is None:
                assert bandera not in texto, f"{script}: {bandera} sobra respecto a {config}"
                continue
            assert f"{bandera} {valor}" in texto, f"{script}: falta `{bandera} {valor}` ({config})"


@pytest.mark.parametrize(("config", "linux", "windows"), CASOS)
def test_la_precision_declarada_coincide(config: str, linux: str, windows: str) -> None:
    datos = _cargar(config)
    for script in (linux, windows):
        texto = (SCRIPTS / script).read_text(encoding="utf-8")
        assert f"--{datos['precision']}" in texto


def test_linux_y_windows_aplican_los_mismos_hiperparametros() -> None:
    """La paridad entre plataformas es una propiedad, no una coincidencia."""
    for _, linux, windows in CASOS:

        def banderas(ruta: str) -> set[str]:
            texto = (SCRIPTS / ruta).read_text(encoding="utf-8")
            return {t for t in texto.split() if t.startswith("--")}

        assert banderas(linux) == banderas(windows), f"{linux} vs {windows}"
