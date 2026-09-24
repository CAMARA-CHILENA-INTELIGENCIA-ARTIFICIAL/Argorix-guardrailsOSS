# `guardrails.publishing` — publicación en Hugging Face

| Módulo | Qué hace |
|---|---|
| `publish_hf.py` | Sube las carpetas de `hf_publish/` a la organización configurada |

```bash
uv sync --extra publish
uv run guardrails publish hf --help
```

Crea o reutiliza los repositorios de dataset y modelo y sube el contenido con
`upload_large_folder`. Acepta `--org`, `--dataset-repo`, `--model-repo`, las rutas de origen y
`--private`.

Requiere estar autenticado en Hugging Face (`huggingface-cli login`) con permisos de escritura
sobre la organización de destino.

## Antes de publicar

La subida no registra qué commit ni qué artefactos se enviaron. Conviene anotarlo manualmente y
revisar que las fichas de `hf_publish/` describan el estado real de los artefactos que acompañan.
