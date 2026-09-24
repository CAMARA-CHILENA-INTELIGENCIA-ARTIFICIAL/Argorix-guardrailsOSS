---
pretty_name: Governance AI Guardrail ES Dataset
license: other
language:
- es
task_categories:
- text-classification
task_ids:
- multi-class-classification
- multi-label-classification
tags:
- guardrails
- prompt-injection
- jailbreak
- spanish
- cybersecurity
- safety
- governance-ai
- ai-security
- runtime-guardrails
size_categories:
- 1M<n<10M
---

<p align="center">
  <img src="https://www.gobiernatuia.cl/logo/gobiernatuialogo.png" alt="Gobierna Tu IA" height="56">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://www.gobiernatuia.cl/logo/governancetransparency.png" alt="Governance AI" height="56">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://www.gobiernatuia.cl/logo/subproductos/guardrails.png" alt="Runtime Guardrails &amp; Agents" height="56">
</p>

<p align="center">
  <sub><b>Gobierna Tu IA</b> &nbsp;·&nbsp; <b>Governance AI</b> &nbsp;·&nbsp; <b>Runtime Guardrails &amp; Agents</b></sub>
</p>

# Governance AI Guardrail ES Dataset

Dataset principal en español para entrenar y evaluar guardrails orientados a:

- `PROMPT_INJECTION`
- `JAILBREAK`
- `HARMFUL`
- `VIOLENCE`
- `HATE`
- `SEXUAL`
- `POLITICS`
- `SAFE`

Este repositorio quedó estructurado en formato estándar de Hugging Face para que el Dataset Viewer detecte correctamente los splits:

- `train.parquet`
- `validation.parquet`
- `test.parquet`

## Creador

**Gustavo Venegas**  
AI Security Research | AI Security SpA Chile Co Funder  
[LinkedIn](https://www.linkedin.com/in/gvenegascc/)  
[Papers en Zenodo](https://zenodo.org/records/18918455)

## Qué es este dataset

Este dataset forma parte del trabajo de **Governance AI** en seguridad y gobernanza operacional de IA.  
Su objetivo no es solo moderación genérica, sino la construcción de una capa de control para asistentes y agentes que permita:

1. detectar ataques de `prompt injection`
2. detectar `jailbreaks`
3. distinguir instrucciones benignas de solicitudes peligrosas
4. reducir falsos positivos en español real y coloquial
5. servir como base para fine-tuning con QLoRA / LoRA

## Estructura

Los archivos publicados en este repositorio son:

- `train.parquet`
- `validation.parquet`
- `test.parquet`
- `README.md`
- `SOURCES.md`

## Estadísticas del dataset

Filas totales: **918,219**

| Split | Filas |
|---|---:|
| train | 772,713 |
| validation | 72,746 |
| test | 72,760 |

Distribución global por clase:

| Clase | Filas |
|---|---:|
| SAFE | 184,000 |
| PROMPT_INJECTION | 184,000 |
| JAILBREAK | 181,819 |
| HARMFUL | 104,118 |
| VIOLENCE | 102,141 |
| HATE | 94,485 |
| SEXUAL | 40,798 |
| POLITICS | 26,858 |

## Fuentes integradas

Este dataset consolida y transforma ejemplos provenientes de:

- `Necent/llm-jailbreak-prompt-injection-dataset`
- `bogdanminko/Catch_the_prompt_injection_or_jailbreak_or_benign`
- `J1N2/mix-prompt-injection-dataset`
- `huyhoangdinhcong/guardrails-dataset-full`

Las transformaciones incluyen:

- traducción a español
- mutaciones lingüísticas
- balanceo por clase
- normalización de etiquetas
- consolidación en parquet

## Columnas principales

- `id`
- `parent_id`
- `source_dataset`
- `source_split`
- `source_row`
- `text_role`
- `original_language`
- `original_text`
- `text_es`
- `mutation_type`
- `split`
- `primary_label`
- `decision`
- `target_json`
- `sft_text`
- `label_safe`
- `label_harmful`
- `label_sexual`
- `label_violence`
- `label_hate`
- `label_politics`
- `label_prompt_injection`
- `label_jailbreak`

## Esquema de decisión

### Decisiones

- `ALLOW`
- `BLOCK`

### Etiquetas primarias

- `SAFE`
- `PROMPT_INJECTION`
- `JAILBREAK`
- `HARMFUL`
- `VIOLENCE`
- `HATE`
- `SEXUAL`
- `POLITICS`

## Uso recomendado

Este dataset está orientado a:

- fine-tuning guardrail
- clasificación de seguridad en español
- evaluación adversarial
- entrenamiento de capas runtime para agentes
- investigación aplicada en AI security

## Ejemplo rápido

```python
import pandas as pd

train_df = pd.read_parquet("train.parquet")
print(train_df[["text_es", "primary_label", "decision"]].head())
```

## Modelo relacionado

Modelo publicado con este dataset:

- [governanceai/governance-ai-guardrail-qwen25-1_5b-v3-corrective](https://huggingface.co/governanceai/governance-ai-guardrail-qwen25-1_5b-v3-corrective)

## Nota de procedencia

Este repositorio contiene transformaciones, traducciones y balanceo de fuentes múltiples.  
Revisar licencias aguas arriba antes de redistribución o uso comercial extendido.

Ver `SOURCES.md`.
