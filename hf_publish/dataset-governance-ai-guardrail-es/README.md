---
pretty_name: Governance AI Guardrail ES Dataset
license: other
language:
- es
task_categories:
- text-classification
task_ids:
- text-classification
tags:
- guardrails
- prompt-injection
- jailbreak
- spanish
- cybersecurity
- safety
- governance-ai
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

Dataset de entrenamiento y evaluación para construir guardrails en español orientados a:

- `PROMPT_INJECTION`
- `JAILBREAK`
- `HARMFUL`
- `VIOLENCE`
- `HATE`
- `SEXUAL`
- `POLITICS`
- `SAFE`

Este repositorio concentra el trabajo de preparación de datos de **Governance AI** y **Runtime Guardrails & Agents**, con foco en tráfico realista en español, ataques adversariales, mutaciones lingüísticas y reducción de falsos positivos.

## Creador

**Gustavo Venegas**  
AI Security Research | AI Security SpA Chile Co Funder  
[LinkedIn](https://www.linkedin.com/in/gvenegascc/)  
[Papers en Zenodo](https://zenodo.org/records/18918455)

## Qué contiene este repositorio

### Parquets principales

- `guardrail_es.parquet`
  - primera versión consolidada del dataset en español
- `guardrail_es_v2.parquet`
  - versión balanceada para fine-tuning principal
- `guardrail_es_v3_corrective.parquet`
  - dataset correctivo construido para reducir falsos positivos
- `guardrail_es_sample.parquet`
  - muestra pequeña para inspección rápida

### Archivos auxiliares

- `*.summary.json`
  - estadísticas de balanceo, fuentes y distribución por clase
- `v2_golden_metrics.json`
  - evaluación golden del modelo v2
- `v3_golden_metrics.json`
  - evaluación golden del modelo v3 corrective
- `test_prompt_injections_20_curated.json`
  - prompts reales curados para smoke testing manual
- `SOURCES.md`
  - detalle de procedencia y notas de licencias aguas arriba

## Objetivo del dataset

Este dataset no intenta resolver solo “toxicidad”. Su objetivo real es entrenar un guardrail operativo para LLMs en español que sea capaz de:

1. detectar ataques de `prompt injection` y `jailbreak`
2. distinguir instrucciones benignas de solicitudes peligrosas
3. mantener bajo el falso positivo en español coloquial
4. funcionar como capa de decisión previa a agentes, copilotos o asistentes
5. soportar entrenamiento incremental y rondas correctivas

## Diseño del pipeline

La construcción siguió una estrategia práctica:

1. recolección de datasets heterogéneos
2. traducción masiva al español
3. mutaciones y variaciones lingüísticas
4. limpieza y balanceo
5. normalización de etiquetas
6. consolidación en parquet
7. creación de dataset correctivo a partir de errores observados

## Fuentes principales

Este trabajo integra y transforma ejemplos provenientes de:

- `Necent/llm-jailbreak-prompt-injection-dataset`
- `bogdanminko/Catch_the_prompt_injection_or_jailbreak_or_benign`
- `J1N2/mix-prompt-injection-dataset`
- `huyhoangdinhcong/guardrails-dataset-full`

Además, la versión correctiva agrega:

- hard negatives derivados del golden set v2
- contrastive attacks creados para corrección de sesgo

## Esquema de columnas

Los parquets principales usan una estructura orientada a entrenamiento SFT y análisis de procedencia:

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
- columnas binarias por etiqueta, por ejemplo:
  - `label_safe`
  - `label_harmful`
  - `label_prompt_injection`
  - `label_jailbreak`

## Etiquetas y política de decisión

### Decisiones

- `ALLOW`
- `BLOCK`

### Clases primarias

- `SAFE`
- `PROMPT_INJECTION`
- `JAILBREAK`
- `HARMFUL`
- `VIOLENCE`
- `HATE`
- `SEXUAL`
- `POLITICS`

## Estadísticas clave

### `guardrail_es_v2.parquet`

- filas totales: **918,219**
- train: **772,713**
- validation: **72,746**
- test: **72,760**

Distribución por clase:

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

### `guardrail_es_v3_corrective.parquet`

- filas totales: **34,766**
- soporte base: **16,662**
- filas correctivas: **18,104**
- hard negatives SAFE tomados del golden v2: **600**
- falsos positivos usados para corrección: **539**

Distribución por clase:

| Clase | Filas |
|---|---:|
| SAFE | 22,700 |
| PROMPT_INJECTION | 1,788 |
| JAILBREAK | 1,768 |
| HARMFUL | 1,726 |
| HATE | 1,716 |
| VIOLENCE | 1,716 |
| SEXUAL | 1,686 |
| POLITICS | 1,666 |

## Evolución de evaluación

### Golden evaluation v2

- `decision_accuracy`: **0.458**
- `primary_label_accuracy`: **0.246**
- `false_positive_rate`: **0.8983**
- `false_negative_rate`: **0.0075**

### Golden evaluation v3 corrective

- `decision_accuracy`: **0.999**
- `primary_label_accuracy`: **0.887**
- `false_positive_rate`: **0.0**
- `false_negative_rate`: **0.0025**

Interpretación práctica:

- v2 detectaba ataques, pero sobrebloqueaba texto benigno
- v3 corrective corrige el sobrebloqueo sin deteriorar de forma material la sensibilidad

## Uso recomendado

Este dataset está pensado para:

- fine-tuning con QLoRA / LoRA
- clasificación guardrail generativa con salida JSON
- entrenamiento de modelos de moderación en español
- benchmarking de `prompt injection` y `jailbreak`
- construcción de conjuntos correctivos

No está pensado como dataset universal de moderación generalista. Su diseño es claramente operativo y enfocado en seguridad aplicada a asistentes y agentes.

## Ejemplo de uso en Python

```python
import pandas as pd

df = pd.read_parquet("guardrail_es_v2.parquet")
print(df[["text_es", "primary_label", "decision"]].head())
```

## Consideraciones de licencia y procedencia

Este repositorio contiene transformaciones, traducciones, balanceo y consolidación de datasets provenientes de múltiples fuentes.  
Se debe revisar la licencia y restricciones de cada fuente aguas arriba antes de reutilizar el contenido en producción o redistribución comercial.

Ver `SOURCES.md`.

## Gobernanza del proyecto

Este dataset forma parte de la línea de trabajo de **Governance AI** en torno a:

- seguridad de modelos
- guardrails en runtime
- defensas para agentes
- reducción de riesgo operacional en IA generativa

## Contacto

Para colaboración, investigación aplicada o despliegue empresarial:

- **Gustavo Venegas**
- AI Security Research
- AI Security SpA Chile Co Funder
- [LinkedIn](https://www.linkedin.com/in/gvenegascc/)
- [Zenodo](https://zenodo.org/records/18918455)
