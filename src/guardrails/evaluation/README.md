# `guardrails.evaluation` — golden set, inferencia y métricas

| Módulo | Qué hace |
|---|---|
| `golden_set.py` | Genera el conjunto de medición de forma determinista |
| `metrics.py` | Agregación de resultados: única implementación del proyecto |
| `scoring.py` | Probabilidad continua de la decisión |
| `run_eval.py` | Infiere sobre cada caso y agrega con `metrics` |

## Conjunto de evaluación

`golden_set.py` produce 1.418 filas sin aleatoriedad, reproducibles bit a bit: las seis categorías
benignas declaradas, los diez contextos, las ocho clases del esquema, variantes ofuscadas y prompts
largos.

Columnas: `id`, `group_id`, `text`, `expected_decision`, `expected_primary_label`,
`expected_labels`, `category`, `variant`, `length_bucket`.

`group_id` identifica la unidad semántica: varias filas derivadas de un mismo tópico o de una misma
frase semilla no son observaciones independientes, y las métricas lo tienen en cuenta.

Sobre `POLITICS`: la etiqueta designa **manipulación** política —propaganda engañosa, suplantación,
desinformación electoral—, no información cívica. Explicar cómo funciona una elección es `SAFE`.

## Métricas

`metrics.py` es la única implementación; la consola de gobernanza consume el mismo módulo. Emite:

| Bloque | Contenido |
|---|---|
| `confusion_matrix` | TP, TN, FP, FN, más las salidas inválidas por lado |
| `decision` | exactitud, exactitud balanceada, precisión, recall, F1, MCC, FPR, FNR |
| `primary_label` | exactitud micro y macro, por clase, y matriz de confusión |
| `parse_errors` | cuántas salidas no se pudieron parsear |
| `cost` | costo total, por observación y esperado por petición |
| `latency_ms` | p50, p95, p99 y máximo |

Cada proporción lleva su intervalo de confianza exacto de Clopper–Pearson, calculado sin
dependencias externas. `effective_n` informa cuántas unidades semánticas independientes hay
detrás de las filas.

Una salida que no se puede parsear **cuenta como error**: en un guardrail es un fallo operativo,
no una fila a descartar.

### Costo

Los pesos de cada tipo de error son un parámetro, no un supuesto implícito:

```bash
uv run guardrails eval run --cost-false-positive 5 --cost-false-negative 1 --arrival-rate-block 0.02
```

Los valores por defecto son unitarios y se declaran como marcador de posición. Mientras no se fije
la razón real entre el costo de un falso positivo y el de un falso negativo, el costo total sirve
para comparar modelos bajo el mismo supuesto, no como magnitud absoluta.

Con `--arrival-rate-block` se calcula además el costo esperado por petición, que es lo que cambia
con el volumen operacional: con pocos ataques en el tráfico, el falso positivo domina.

### Puntuación continua

La salida del modelo es discreta, así que por sí sola no permite trazar curvas ni mover el umbral.
`--with-scores` añade una pasada hacia adelante por caso y recupera la probabilidad de `BLOCK`:

```bash
uv run guardrails eval run --with-scores --max-false-positive-rate 0.02
```

La probabilidad no sale del texto generado. Se construye el prompt seguido del prefijo del JSON
hasta la posición en que el siguiente token fija la clase —`{"decision":"`— y se comparan los
logits de los primeros tokens de `ALLOW` y `BLOCK`. Es determinista y no depende de que la salida
se pueda parsear.

Con puntuación disponible, el informe añade un bloque `score`:

| Campo | Contenido |
|---|---|
| `roc_auc` | Área bajo la curva ROC |
| `average_precision` | Área bajo precisión-recall, más informativa con clase minoritaria |
| `threshold_sweep` | FPR y FNR a lo largo del rango de umbrales |
| `operating_points` | Umbral más permisivo que respeta techos de 1 %, 2 %, 5 % y 10 % de falsos positivos |

`--max-false-positive-rate` añade el punto correspondiente al techo que se le indique. Es la
forma habitual de fijar el punto de operación de un guardrail: se declara cuánta fricción se
tolera sobre tráfico benigno y se mide qué sensibilidad se obtiene a cambio.

### Dos cifras, no una

```bash
uv run guardrails eval run --production-path
```

Sin esa bandera se mide el **modelo aislado**. Con ella se mide la **ruta completa de producción**:
normaliza la entrada, consulta al modelo una segunda vez si el texto cambió y arbitra por
severidad. Ese arbitraje es monótono hacia el bloqueo, así que la ruta completa nunca bloquea
menos que el modelo aislado. Conviene reportar ambas.

## Uso

```bash
# de extremo a extremo: conjunto, modelo aislado y ruta completa
./scripts/linux/evaluate.sh models/guardrail-qwen25-1_5b-qlora-v3-corrective

# o paso a paso
uv run guardrails eval golden-set --output holdout.csv
uv run guardrails eval run --adapter <ruta> --input-csv holdout.csv --load-4bit \
  --metrics-output reports/metrics.json

# sobre la partición test del propio parquet
uv run guardrails eval run --adapter <ruta> --dataset data_finetune/guardrail_es.parquet --split test
```

`--load-4bit` carga el modelo base cuantizado, igual que en entrenamiento y serving. Sin esa
bandera se carga sin cuantizar, y el resultado puede diferir.

Las métricas se escriben junto a un bloque `run_metadata` con commit, comando, adaptador,
plataforma y versiones de biblioteca.

## Tests

```bash
uv run pytest tests/test_golden_set.py tests/test_cost_metrics.py tests/test_production_path.py
```
