# `configs/data/` — parámetros de construcción de datasets

| Archivo | Dataset |
|---|---|
| `pipeline_v1.yaml` | `data_finetune/guardrail_es.parquet` |
| `pipeline_v2.yaml` | `data_finetune/guardrail_es_v2.parquet` |

## Estado: registro declarativo

Mismo criterio que `configs/train/`: registran los parámetros de las ejecuciones históricas y no
son leídos por el pipeline.

El bloque `splits:` ya es operativo: las proporciones y la sal se pasan por línea de comandos
(`--split-train`, `--split-validation`, `--split-test`, `--split-salt`) y las aplica
`guardrails.data.splits`, que es la única función de partición del proyecto.
