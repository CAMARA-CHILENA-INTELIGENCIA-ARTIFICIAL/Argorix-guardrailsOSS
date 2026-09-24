# `guardrails.data` — ingesta, traducción y construcción de datasets

Produce el parquet de fine-tuning a partir de cuatro datasets de Hugging Face.

| Módulo | Qué hace |
|---|---|
| `ingest.py` | Descarga un dataset de Hugging Face a `data/<id>/<split>.parquet` |
| `translate.py` | Traduce al español con `facebook/nllb-200-distilled-600M` |
| `build_dataset.py` | Une los orígenes, etiqueta, muta, deduplica, particiona y balancea |
| `splits.py` | Partición en train / validation / test, por grupo |
| `corrective.py` | Construye el dataset correctivo a partir de predicciones previas |

## Orígenes

| Dataset | Idioma | Columnas de texto |
|---|---|---|
| `huyhoangdinhcong/guardrails-dataset-full` | vi | `text` |
| `Necent/llm-jailbreak-prompt-injection-dataset` | mixto | `prompt`, `response` |
| `J1N2/mix-prompt-injection-dataset` | en | `prompt`, `base_prompt` |
| `bogdanminko/Catch_the_prompt_injection_or_jailbreak_or_benign` | en | `prompt` |

Cada origen tiene su propia función de etiquetado (`labels_*`), que traduce sus columnas al
esquema común de ocho etiquetas booleanas.

## Partición

`splits.py` reparte en train / validation / test con proporciones y sal configurables. La
partición se calcula sobre la **clave de grupo**, no sobre la fila: las mutaciones heredan el
`group_id` de su origen, de modo que todas las derivaciones de un mismo texto caen juntas. Sin
eso, una paráfrasis puede acabar en `test` mientras su original está en `train`.

```bash
uv run guardrails data build --split-train 80 --split-validation 10 --split-test 10 --split-salt ronda2
```

Cambiar la sal rota el reparto sin tocar los datos, que es lo que hace falta para estimar cuánta
varianza aporta la partición.

## Deduplicación

La clave es el texto normalizado. Si dos orígenes etiquetan el mismo texto igual, se conserva una
fila; si lo etiquetan distinto, **se descartan todas**: resolver el empate en favor de una fuente
arbitraria mete ruido silencioso. El número de conflictos queda en el `summary.json`.

## Dataset correctivo

`corrective.py` exige dos conjuntos distintos y **aborta** si se solapan:

| Argumento | Qué es |
|---|---|
| `--corrections` | Predicciones sobre el conjunto de corrección; de aquí salen los ejemplos de refuerzo |
| `--holdout` | Conjunto de medición reservado; no aporta filas, sólo se usa para verificar |

La verificación detecta el solapamiento literal y también el de paráfrasis, midiendo qué fracción
de los n-gramas del texto reservado aparece contenida en el generado. Quitar sólo los literales no
basta: envolver un texto reservado en un prefijo lo filtra igual.

## Esquema de salida

23 columnas, definidas en `base_record()`. Las principales:

| Columna | Contenido |
|---|---|
| `id`, `parent_id`, `group_id` | Identificador, origen de la mutación y clave que agrupa a todas las derivaciones |
| `source_dataset`, `source_split`, `source_row`, `text_role` | Procedencia |
| `original_text`, `text_es` | Texto de origen y su traducción |
| `mutation_type` | `translation` o `mutation_es_N` |
| `split` | `train` / `validation` / `test` |
| `primary_label`, `decision`, `target_json` | Etiqueta resuelta y objetivo de entrenamiento |
| `sft_text` | Secuencia completa que consume el entrenador |
| `label_*` (8) | Etiquetas booleanas |

## Uso

```bash
uv run guardrails data ingest --dataset-id J1N2/mix-prompt-injection-dataset
uv run guardrails data translate --profile finetune
uv run guardrails data build --output data_finetune/guardrail_es.parquet
```

Cada comando acepta `--help`. Los parámetros de las ejecuciones históricas están en
`configs/data/`.

## Tests

```bash
uv run pytest tests/test_splits.py tests/test_deduplication.py \
              tests/test_corrective_dataset.py tests/test_taxonomy.py
```
