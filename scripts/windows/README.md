# `scripts/windows/` — lanzadores para Windows (con CUDA)

Envoltorios finos sobre `uv run guardrails …`. Cada archivo tiene un **gemelo exacto** en
`scripts/linux/` con los mismos hiperparámetros; `tests/test_configs.py` verifica esa paridad.

| Script | Equivale a |
|---|---|
| `setup.ps1` | `uv sync --extra train --group dev` + `guardrails doctor` |
| `pipeline.ps1` | traducción + construcción del dataset v1 |
| `smoke_train.ps1` | entrenamiento de humo (20 pasos) |
| `full_train.ps1` | entrenamiento v1 (3.000 pasos) |
| `v2_train.ps1` | entrenamiento v2 (4.000 pasos) |
| `v3_corrective_train.ps1` | entrenamiento v3 correctivo (1.000 pasos, desde v2) |
| `v2_prepare_after_translate.ps1` | espera a la traducción y construye el dataset v2 |
| `serve.ps1` | consola de gobernanza |
| `evaluate.ps1` | evaluación de extremo a extremo: conjunto de medición, modelo aislado y ruta completa |

Monitores de sólo lectura: `pipeline_status.ps1`, `train_status.ps1`, `train_v2_status.ps1`,
`train_v3_status.ps1`, `v2_status.ps1`.

```powershell
.\scripts\windows\setup.ps1
.\scripts\windows\full_train.ps1
```

`evaluate.ps1` deja los resultados en `reports/<marca-de-tiempo>/`, con las métricas del modelo
aislado y las de la ruta completa por separado. La segunda nunca bloquea menos que la primera.

Con `torch+cu128` el entrenamiento corre **nativo en Windows**: no hace falta WSL. Si aun así se
prefiere ejecutar dentro de WSL, `GUARDRAIL_WSL_PROJECT` fuerza el punto de montaje.

`.gitattributes` fuerza `*.ps1 text eol=crlf`.
