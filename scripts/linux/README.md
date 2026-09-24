# `scripts/linux/` — lanzadores para Linux (con CUDA)

Envoltorios finos sobre `uv run guardrails …`. Cada archivo tiene un **gemelo exacto** en
`scripts/windows/` con los mismos hiperparámetros; `tests/test_configs.py` verifica esa paridad.

| Script | Equivale a |
|---|---|
| `setup.sh` | `uv sync --extra train --group dev` + `guardrails doctor` |
| `pipeline.sh` | traducción + construcción del dataset v1 |
| `smoke_train.sh` | entrenamiento de humo (20 pasos) |
| `full_train.sh` | entrenamiento v1 (3.000 pasos) |
| `v2_train.sh` | entrenamiento v2 (4.000 pasos) |
| `v3_corrective_train.sh` | entrenamiento v3 correctivo (1.000 pasos, desde v2) |
| `serve.sh` | consola de gobernanza |
| `evaluate.sh` | evaluación de extremo a extremo: conjunto de medición, modelo aislado y ruta completa |

```bash
./scripts/linux/setup.sh
./scripts/linux/full_train.sh
```

Todos derivan la raíz del proyecto de su propia ubicación, así que funcionan desde cualquier
directorio de trabajo. Los parámetros que aplican están registrados en `configs/`.

`evaluate.sh` deja los resultados en `reports/<marca-de-tiempo>/`, con las métricas del modelo
aislado y las de la ruta completa por separado. La segunda nunca bloquea menos que la primera.
