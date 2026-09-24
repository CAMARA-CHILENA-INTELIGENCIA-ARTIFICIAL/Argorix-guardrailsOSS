"""Guardrail generativo en español sobre Qwen2.5-1.5B (QLoRA).

Subpaquetes:
    guardrails.data        ingesta, traducción, construcción y partición de datasets
    guardrails.training    fine-tuning QLoRA
    guardrails.evaluation  golden set, inferencia de evaluación y métricas
    guardrails.serving     consola de gobernanza (FastAPI) y worker de inferencia
    guardrails.publishing  publicación de artefactos en Hugging Face

Cada subpaquete tiene un README.md en su carpeta con lo que hace, cómo se usa y
las notas relevantes para trabajar en él.
"""

__version__ = "0.2.0"
