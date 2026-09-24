# `examples/` — entradas de ejemplo para pruebas manuales

| Archivo | Contenido |
|---|---|
| `eval_cases_es.txt` | 10 prompts en español, uno por línea |

```bash
uv run guardrails eval run --input-file examples/eval_cases_es.txt
```

`--input-file` imprime las predicciones pero no calcula métricas: para eso hace falta un CSV con
las columnas `text`, `expected_decision` y `expected_primary_label`, que es lo que consume
`--input-csv`.
