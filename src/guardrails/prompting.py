"""Plantilla de prompt del guardrail.

Definición única, compartida por la construcción del dataset, el dataset
correctivo, la evaluación y el worker de inferencia. El prompt debe ser
idéntico en entrenamiento y en inferencia.

El par ``(prompt, completion)`` es además el formato que espera TRL para
calcular la pérdida **sólo sobre el completado**. Con un único campo de texto,
TRL trata el ejemplo como *language modeling* y rechaza esa opción.
"""

from __future__ import annotations

INSTRUCTION = (
    "Clasifica el texto del usuario para un sistema guardrail en español. "
    "Responde solo JSON compacto con decision, primary_label y labels."
)

#: Marca donde termina el prompt y empieza lo que el modelo debe generar.
COMPLETION_MARKER = "<start_of_turn>model\n"


def build_prompt(text: str) -> str:
    """Parte de la secuencia sobre la que NO se calcula la pérdida."""
    return f"<start_of_turn>user\n{INSTRUCTION}\n\nTexto:\n{text}<end_of_turn>\n{COMPLETION_MARKER}"


def build_completion(target_json: str) -> str:
    """Parte de la secuencia sobre la que SÍ se calcula la pérdida."""
    return f"{target_json}<end_of_turn>"


def build_sft_text(text: str, target_json: str) -> str:
    """Secuencia completa. Se conserva para inspección y compatibilidad."""
    return build_prompt(text) + build_completion(target_json)


def split_sft_text(sft_text: str) -> tuple[str, str]:
    """Separa una secuencia completa en ``(prompt, completion)``.

    Permite consumir parquets que sólo traen la secuencia completa.
    """
    indice = sft_text.find(COMPLETION_MARKER)
    if indice < 0:
        raise ValueError("la secuencia no contiene la marca de completado")
    corte = indice + len(COMPLETION_MARKER)
    return sft_text[:corte], sft_text[corte:]


__all__ = [
    "COMPLETION_MARKER",
    "INSTRUCTION",
    "build_completion",
    "build_prompt",
    "build_sft_text",
    "split_sft_text",
]
