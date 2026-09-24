# `configs/eval/` — parámetros de evaluación

| Archivo | Evaluación |
|---|---|
| `golden_v3.yaml` | La que produjo `reports/v3_golden_metrics.json` |

## Estado: reconstruido a partir de los artefactos

A diferencia de `configs/train/` y `configs/data/`, este archivo no se pudo reconstruir desde un
lanzador porque no existía ninguno: la evaluación se ejecutaba a mano. Se reconstruyó a partir de
los artefactos de `reports/` y del código.

Un campo queda sin determinar:

```yaml
load_4bit: unknown
```

Esa bandera decide si el modelo base se carga cuantizado en 4 bits, como en entrenamiento y
serving, o sin cuantizar. No hay registro de cuál se usó.

Conviene que `run_eval.py` emita un bloque `run_metadata` junto a las métricas: commit, comando,
adaptador y versiones de biblioteca.
