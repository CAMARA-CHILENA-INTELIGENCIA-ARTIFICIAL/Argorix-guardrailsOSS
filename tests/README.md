# `tests/` — batería de pruebas

```bash
uv run pytest                       # 131 tests, < 5 s, sin GPU ni red
uv run pytest -m characterization   # sólo los de caracterización
```

`conftest.py` instala sustitutos mínimos de las dependencias pesadas (`torch`, `transformers`,
`peft`, `trl`, `datasets`, `tqdm`, `fastapi`, `pydantic`) **sólo si no están instaladas**, de modo
que la batería corre con `uv sync --group dev` y también en una máquina con la pila completa.

## Contenido

| Archivo | Qué cubre |
|---|---|
| `test_taxonomy.py` | Orden de severidad, resolución multi-clase y política de decisión |
| `test_prompting.py` | Plantilla de prompt y formato prompt-completion |
| `test_scoring.py` | Probabilidad de la decisión, curvas y punto de operación |
| `test_labeling.py` | Qué texto se clasifica y con qué etiqueta según el origen |
| `test_splits.py` | Determinismo, proporciones configurables y agrupación de derivaciones |
| `test_deduplication.py` | Textos repetidos y conflictos de etiqueta entre orígenes |
| `test_corrective_dataset.py` | Separación entre el conjunto de corrección y el de medición |
| `test_golden_set.py` | Cobertura y reproducibilidad del conjunto de evaluación |
| `test_cost_metrics.py` | Matriz de confusión, tasas, intervalos y función de costo |
| `test_production_path.py` | Normalización, arbitraje y equivalencia con lo que se sirve |
| `test_configs.py` | Que `configs/` coincida con los lanzadores y que Windows y Linux estén a la par |
| `test_packaging.py` | Importabilidad, rutas portables entre sistemas y CLI |

## Invariantes que se comprueban

Estos son los que conviene no romper:

- Una variante léxica **nunca** cae en una partición distinta de su origen.
- Un texto con etiquetas contradictorias entre orígenes **no** llega al dataset.
- La construcción del dataset correctivo **aborta** si el conjunto de medición aparece en él,
  sea literalmente o como paráfrasis.
- Las tasas de error están acotadas en `[0, 1]` sea cual sea la entrada.
- Una salida que no se puede parsear cuenta como error.
- Datos, runtime y generador del conjunto importan la severidad del mismo módulo.
- Sólo se clasifica texto de entrada de usuario; la respuesta del modelo no hereda la etiqueta
  del prompt salvo que se pida explícitamente.

## Tests de integración

`test_training_integration.py` ejecuta el entrenamiento completo sobre un modelo de pruebas de
unos pocos MB, en CPU y en segundos. Comprueba que la configuración del entrenador es aceptable
para la versión de TRL instalada, que es lo que no puede verificarse por inspección estática.

Requiere el extra `train` y va marcado con `integration`, de modo que la batería por defecto no lo
ejecuta:

```bash
uv sync --extra train
uv run pytest -m integration
```

## Integración continua

`.github/workflows/ci.yml` ejecuta:

| Trabajo | Dónde | Qué |
|---|---|---|
| `check` | Ubuntu y Windows × Python 3.11 y 3.12 | lint, formato, batería sin integración, CLI |
| `integration` | macOS | entrenamiento de extremo a extremo en CPU |
| `lock` | Ubuntu | que `uv.lock` corresponda a `pyproject.toml` |

## Tests de caracterización

Los marcados con `@pytest.mark.characterization` fijan el comportamiento actual de funciones
puras del pipeline, para que cualquier cambio en ellas aparezca como un fallo visible en el diff
en lugar de pasar inadvertido.

No se cubren `translate.py` (requiere el modelo NLLB), `qlora.py` (requiere GPU) ni el frontend.
