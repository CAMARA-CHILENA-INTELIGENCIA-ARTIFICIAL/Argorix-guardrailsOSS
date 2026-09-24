# `reports/` — artefactos de evaluación

| Archivo | Qué es |
|---|---|
| `v2_golden_metrics.json` | Métricas del adaptador v2 sobre el golden set |
| `v3_golden_metrics.json` | Métricas del adaptador v3 correctivo sobre el golden set |
| `test_prompt_injections_20.json` | Prompts reales traducidos, sin etiquetar |
| `test_prompt_injections_20_curated.json` | Prompts curados con `source`, `label` y `text` |

Los archivos `*_predictions.jsonl` que produce `guardrails eval run` no se versionan
(ver `.gitignore`). Ten en cuenta que `guardrails data corrective` los necesita como entrada.

## Formato de las métricas

```json
{
  "total": 1000,
  "decision_correct": 999,
  "primary_label_correct": 887,
  "false_positives": 0,
  "false_negatives": 1,
  "decision_accuracy": 0.999,
  "primary_label_accuracy": 0.887,
  "false_positive_rate": 0.0,
  "false_negative_rate": 0.0025,
  "label_confusion": { "SAFE->SAFE": 600, "...": 0 }
}
```

Los archivos no registran el commit, el comando ni las versiones de biblioteca con que se
generaron. `tests/test_cost_metrics.py` comprueba que los conteos publicados son aritméticamente
coherentes entre sí.
