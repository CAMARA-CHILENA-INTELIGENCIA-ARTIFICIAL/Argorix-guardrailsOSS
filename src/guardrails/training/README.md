# `guardrails.training` — fine-tuning QLoRA

| Módulo | Qué hace |
|---|---|
| `qlora.py` | Ajusta un adaptador LoRA sobre el campo `sft_text` del parquet |

## Configuración

| Aspecto | Valor |
|---|---|
| Modelo base | `Qwen/Qwen2.5-1.5B-Instruct` |
| Cuantización | 4-bit NF4 con doble cuantización |
| LoRA | `r=16`, `alpha=32`, `dropout=0.05`, sobre las siete proyecciones |
| Optimizador | `paged_adamw_8bit`, planificador coseno, `warmup_ratio=0.03` |
| Longitud | 512 tokens |
| Semilla | 42 |
| Pérdida | sólo sobre el completado (`completion_only_loss`) |
| Punto de control | el de menor `eval_loss`, no el último paso |

`sft_text` contiene instrucción, texto de usuario y JSON objetivo. La pérdida se calcula **sólo
sobre el JSON**: entrenar sobre el prompt gasta la mayor parte de la señal en enseñar al modelo a
reproducir la entrada, que no es lo que se le pide en inferencia. Con `--no-best-model` se
conserva el último paso en lugar del mejor.

`SaveStateCallback` escribe `train_state.json` en el directorio de salida en cada log, para que la
consola de gobernanza pueda mostrar el progreso.

## Qué vio el modelo

Con `--max-steps` el entrenamiento corta antes de completar una época, de modo que el dataset no
basta para saber qué filas consumió. Junto al adaptador se escribe `consumed_ids.json` con los
identificadores de train y validation, y `guardrail_training.json` registra semilla, lote efectivo
y número de pasos.

Es lo que permite comprobar después que el conjunto de medición no estaba entre ellas.

## Requisitos

Para entrenar de verdad, GPU NVIDIA con CUDA, en **Linux o Windows**.

```bash
uv sync --extra train
uv run guardrails doctor          # confirma que torch ve la GPU
uv run guardrails train qlora --help
```

### Sin GPU

`--device cpu --no-quantization` ejecuta la cadena completa sobre CPU. La cuantización de 4 bits
depende de bitsandbytes, que requiere CUDA, y el optimizador cambia a `adamw_torch`.

No sirve para producir un modelo —es órdenes de magnitud más lento— pero sí para validar que la
configuración es correcta antes de reservar una GPU. `tests/test_training_integration.py` hace
exactamente eso con un modelo de pruebas de unos pocos MB, en segundos:

```bash
uv sync --extra train
uv run pytest -m integration
```

### Sesiones que se cortan

En cualquier GPU de nivel gratuito la sesión termina antes que el entrenamiento. El estado se
guarda cada `--save-steps`, y `--resume-from-checkpoint auto` continúa desde el punto de control
más reciente de `--output-dir`:

```bash
uv run guardrails train qlora --dataset ... --output-dir models/v4 --resume-from-checkpoint auto
```

## Uso

```bash
uv run guardrails train qlora \
  --dataset data_finetune/guardrail_es.parquet \
  --output-dir models/guardrail-qwen25-1_5b-qlora-v1 \
  --max-steps 3000 --bf16
```

`--adapter-init` continúa el entrenamiento desde un adaptador existente en lugar de partir de
cero. Los parámetros de las ejecuciones históricas están en `configs/train/`, y los lanzadores
equivalentes en `scripts/linux/` y `scripts/windows/`.

## Nota sobre la plantilla de prompt

`sft_text` usa delimitadores `<start_of_turn>` / `<end_of_turn>`. El tokenizador de Qwen no los
trata como tokens especiales, así que se tokenizan como texto ordinario. La plantilla está
duplicada en cuatro módulos (`data/build_dataset.py`, `data/corrective.py`,
`evaluation/run_eval.py`, `serving/worker.py`) y debe mantenerse sincronizada a mano.
