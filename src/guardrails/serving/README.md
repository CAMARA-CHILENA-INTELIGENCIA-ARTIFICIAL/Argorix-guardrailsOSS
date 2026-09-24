# `guardrails.serving` — consola de gobernanza y worker de inferencia

| Módulo | Qué hace |
|---|---|
| `backend.py` | API FastAPI y paneles de entrenamiento y evaluación |
| `normalization.py` | Normalización de entrada y arbitraje de resultados |
| `worker.py` | Proceso persistente que carga el adaptador y responde por stdin/stdout |
| `static/` | Frontend estático |

```bash
uv sync --extra serve
uv run guardrails serve            # http://127.0.0.1:8000
```

## Endpoints

| Método | Ruta | Qué hace |
|---|---|---|
| `POST` | `/api/analyze` | Clasifica un texto |
| `POST` | `/api/analyze-batch` | Clasifica hasta 200 textos |
| `POST` | `/api/upload-json` | Clasifica los elementos de un JSON subido |
| `GET` | `/api/train/status` | Progreso del entrenamiento |
| `GET` | `/api/eval/golden-status` | Progreso y resultados de la evaluación |
| `GET` | `/api/health` | Estado del worker |

## Cómo procesa una petición

`analyze_text()` normaliza el texto con `normalize_for_guardrail()`, que revierte ofuscaciones
comunes: sustitución *leet* (`0`→`o`, `3`→`e`…), separadores insertados entre letras y letras
espaciadas. Si la normalización cambia el texto, se invoca al modelo una segunda vez con la
versión normalizada y `merge_results()` se queda con el resultado de mayor severidad.

Ese arbitraje es **monótono hacia el bloqueo**: `BLOCK` siempre supera a `ALLOW`. La ruta de
producción, por tanto, nunca bloquea menos que el modelo aislado.

`normalization.py` vive fuera de `backend.py` y no depende de FastAPI, de modo que la evaluación
puede medir exactamente la misma función que se sirve:

```bash
uv run guardrails eval run --production-path
```

El orden de severidad lo importa de `guardrails.taxonomy`, el mismo que usan la generación de
datos y el generador del conjunto de medición.

El worker corre en un proceso aparte, con el modelo cargado en 4 bits, y se comunica por
stdin/stdout con JSON por línea. En Windows puede ejecutarse dentro de WSL:
`GUARDRAIL_WSL_PROJECT` fuerza el punto de montaje y `guardrails.paths.to_wsl_path()` traduce
`D:\…` a `/mnt/d/…`.

## Rutas

Este módulo deriva la raíz del proyecto de `guardrails.paths`, de modo que funciona igual desde
cualquier ubicación y en ambos sistemas operativos. `GUARDRAIL_PROJECT_ROOT` permite forzarla.

## Deuda de lint declarada

En `pyproject.toml`, `backend.py` tiene suprimida `SIM105` (un `try/except/pass` en el drenaje de
stderr del worker) y está excluido del formateador, para mantener revisable el diff de la
migración. `normalization.py`, al ser código nuevo, cumple el conjunto completo de reglas.
