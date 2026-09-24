"""Puntuación continua de la decisión.

El guardrail emite JSON, de modo que su salida es discreta: ``ALLOW`` o
``BLOCK``. Sin una puntuación continua no hay curva ROC ni PR, no hay
calibración y el único modo de mover el equilibrio entre falsos positivos y
falsos negativos es reentrenar.

La probabilidad se obtiene de una sola pasada hacia adelante, no del texto
generado. Se construye el prompt seguido del prefijo del JSON hasta el punto en
que el siguiente token decide la clase::

    <prompt>{"decision":"

y se comparan los logits de los primeros tokens de ``ALLOW`` y ``BLOCK`` en esa
posición. Es determinista y no depende de que la salida se pueda parsear.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from guardrails import prompting, taxonomy

#: Prefijo del JSON hasta la posición en que el siguiente token fija la decisión.
DECISION_PREFIX = '{"decision":"'


def first_token_id(tokenizer: Any, texto: str) -> int:
    """Identificador del primer token de ``texto``, sin prefijo de espacio."""
    ids = tokenizer.encode(texto, add_special_tokens=False)
    if not ids:
        raise ValueError(f"el tokenizador no produce tokens para {texto!r}")
    return int(ids[0])


@dataclass(frozen=True)
class DecisionHead:
    """Identificadores de los tokens que distinguen una decisión de la otra.

    Se resuelven una vez por tokenizador. Si ``ALLOW`` y ``BLOCK`` empezaran por
    el mismo token, el primer token no bastaría para separarlos y la puntuación
    no sería válida.
    """

    allow_id: int
    block_id: int

    @classmethod
    def from_tokenizer(cls, tokenizer: Any) -> DecisionHead:
        allow_id = first_token_id(tokenizer, taxonomy.ALLOW)
        block_id = first_token_id(tokenizer, taxonomy.BLOCK)
        if allow_id == block_id:
            raise ValueError(
                "ALLOW y BLOCK comparten primer token con este tokenizador: "
                "la puntuación por token único no los distingue"
            )
        return cls(allow_id=allow_id, block_id=block_id)


def block_probability(
    model: Any,
    tokenizer: Any,
    text: str,
    head: DecisionHead,
    *,
    max_length: int | None = None,
) -> float:
    """Probabilidad que el modelo asigna a ``BLOCK`` para un texto.

    Normalizada sobre las dos alternativas, de modo que el resultado está en
    ``[0, 1]`` y ``1 - p`` es la probabilidad de ``ALLOW``.
    """
    import torch

    prefijo = prompting.build_prompt(text) + DECISION_PREFIX
    entradas = tokenizer(
        prefijo,
        return_tensors="pt",
        truncation=max_length is not None,
        max_length=max_length,
    ).to(model.device)

    with torch.inference_mode():
        salida = model(**entradas)

    logits = salida.logits[0, -1, :]
    par = torch.stack([logits[head.allow_id], logits[head.block_id]]).float()
    return float(torch.softmax(par, dim=-1)[1])


def decision_from_probability(probabilidad: float, threshold: float = 0.5) -> str:
    """Decisión que corresponde a una probabilidad y un umbral."""
    return taxonomy.BLOCK if probabilidad >= threshold else taxonomy.ALLOW


__all__ = [
    "DECISION_PREFIX",
    "DecisionHead",
    "block_probability",
    "decision_from_probability",
    "first_token_id",
]
