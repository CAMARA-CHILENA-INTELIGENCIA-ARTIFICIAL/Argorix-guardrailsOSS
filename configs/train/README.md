# `configs/train/` — parámetros de entrenamiento, por ejecución

| Archivo | Adaptador |
|---|---|
| `v1.yaml` | `models/guardrail-qwen25-1_5b-qlora-v1` |
| `v2.yaml` | `models/guardrail-qwen25-1_5b-qlora-v2` |
| `v3_corrective.yaml` | `models/guardrail-qwen25-1_5b-qlora-v3-corrective` |

## Estado: registro declarativo

**El código todavía no lee estos archivos.** Los parámetros efectivos viven en las banderas de
`scripts/linux/` y `scripts/windows/`; estos YAML los registran para dejar constancia de qué
configuración produjo cada artefacto.

Para que no se conviertan en documentación desactualizada, `tests/test_configs.py` verifica que
cada valor registrado aparece en los dos lanzadores. Si alguien cambia un script sin actualizar el
YAML, la batería falla.

Cablearlos como entrada real del entrenador queda pendiente.
