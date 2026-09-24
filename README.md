# Argorix — Guardrails OSS

Guardrail generativo en español sobre `Qwen/Qwen2.5-1.5B-Instruct`, ajustado con **QLoRA**
para clasificar texto de usuario y devolver un JSON compacto con decisión y clase primaria.

```json
{"decision":"BLOCK","primary_label":"PROMPT_INJECTION","labels":["PROMPT_INJECTION"]}
```

| Decisiones | Clases primarias |
|---|---|
| `ALLOW`, `BLOCK` | `SAFE`, `PROMPT_INJECTION`, `JAILBREAK`, `HARMFUL`, `VIOLENCE`, `HATE`, `SEXUAL`, `POLITICS` |

El esquema, el orden de severidad y la política de decisión viven en un único módulo,
[`guardrails.taxonomy`](src/guardrails/taxonomy.py), del que dependen la generación de datos, el
runtime y el generador del conjunto de medición:

```
VIOLENCE > HARMFUL > PROMPT_INJECTION > JAILBREAK > HATE > SEXUAL > POLITICS > SAFE
```

---

## Requisitos

- **Python 3.11 o 3.12**
- [**uv**](https://docs.astral.sh/uv/) para gestionar entorno y dependencias
- Para entrenar o inferir: **GPU NVIDIA con CUDA**, en Linux o Windows

El proyecto está pensado para funcionar igual en **Windows y Linux**. La resolución de
dependencias es específica por plataforma:

| Plataforma | Origen de `torch` | `bitsandbytes` |
|---|---|---|
| Linux, Windows | índice CUDA 12.8 de PyTorch | sí |
| macOS | PyPI (CPU) | excluido |

macOS se soporta solo para desarrollo, lint, tests y análisis: el entrenamiento QLoRA
aborta explícitamente sin CUDA.

## Puesta en marcha

```bash
uv sync --group dev
```

Eso instala el paquete en modo editable más `ruff` y `pytest`, sin descargar la pila de
entrenamiento. Para añadir lo demás:

```bash
uv sync --extra train
uv sync --extra serve
uv sync --all-extras
```

Comprueba el entorno y la GPU:

```bash
uv run guardrails doctor
```

## Uso

Un único punto de entrada, idéntico en Windows y en Linux:

```bash
uv run guardrails data ingest --dataset-id J1N2/mix-prompt-injection-dataset
uv run guardrails data translate --profile finetune
uv run guardrails data build --output data_finetune/guardrail_es.parquet
uv run guardrails train qlora --bf16 --max-steps 3000
uv run guardrails eval golden-set --output holdout.csv
uv run guardrails eval run --adapter <ruta> --input-csv holdout.csv --load-4bit
uv run guardrails serve
```

O la evaluación completa de una vez, que genera el conjunto de medición y reporta tanto el modelo
aislado como la ruta de producción:

```bash
./scripts/linux/evaluate.sh <ruta-al-adaptador>     # Linux
.\scripts\windows\evaluate.ps1 <ruta-al-adaptador>  # Windows
```

Cada subcomando reenvía sus argumentos al módulo correspondiente, de modo que las banderas
del pipeline original siguen siendo válidas. `--help` funciona en cualquier nivel.

Los lanzadores por plataforma están en [`scripts/linux/`](scripts/linux/) y
[`scripts/windows/`](scripts/windows/); son envoltorios finos sobre estos comandos.

## Estructura

```
configs/                 parámetros del pipeline por etapa
  data/  train/  eval/
docs/
  analisis-base/         auditoría técnica (PDF, 73 págs.)
  partes-clave/          partes del código con problemas
examples/                entradas de ejemplo para pruebas manuales
hf_publish/              fichas y artefactos publicados en Hugging Face
reports/                 métricas y salidas de evaluación
scripts/
  linux/  windows/       lanzadores por plataforma
src/guardrails/
  data/                  ingesta, traducción, construcción y partición de datasets
  training/              fine-tuning QLoRA
  evaluation/            conjunto de medición, inferencia y métricas
  serving/               consola de gobernanza (FastAPI), worker y normalización
  publishing/            publicación en Hugging Face
  taxonomy.py            esquema de etiquetas, severidad y política de decisión
  paths.py               resolución portable de rutas
  cli.py                 punto de entrada
tests/                   tests de caracterización
```

**Cada carpeta hoja tiene un `README.md`** con lo que hace ese nivel, cómo se usa y las notas
relevantes para trabajar en él.

## Calidad

```bash
uv run ruff check .          # limpio
uv run ruff format --check . # limpio
uv run pytest                # sin GPU ni red
```

La integración continua ejecuta lo mismo en **Ubuntu y Windows**, en Python 3.11 y 3.12, y
comprueba que `uv.lock` corresponde a `pyproject.toml`.

Ambas comprobaciones pasan. Las supresiones vigentes están declaradas archivo por archivo y regla
por regla en `pyproject.toml`, bajo `[tool.ruff.lint.per-file-ignores]` y `[tool.ruff.format]`,
cada una con su motivo. Se retiran a medida que se reescriben los módulos correspondientes.

La batería comprueba, entre otras cosas, que una variante léxica nunca cae en una partición
distinta de su origen, que un texto con etiquetas contradictorias no llega al dataset, que la
construcción del dataset correctivo aborta si el conjunto de medición aparece en él, y que las
tasas de error están acotadas sea cual sea la entrada. Ver [`tests/`](tests/).

## Datos y modelos

No se versionan (ver `.gitignore`): `data/`, `data_es/`, `data_finetune/`, `models/`, `logs/`
y todo `*.parquet`, `*.csv`, `*.jsonl`. Los datasets y adaptadores publicados están en
Hugging Face; las fichas correspondientes, bajo `hf_publish/`.

## Licencia y procedencia

Este repositorio consolida y transforma datasets de terceros. Antes de reutilizar o
redistribuir, revisa la licencia de cada fuente aguas arriba: ver
[`hf_publish/dataset-governance-ai-guardrail-es/SOURCES.md`](hf_publish/dataset-governance-ai-guardrail-es/SOURCES.md).
