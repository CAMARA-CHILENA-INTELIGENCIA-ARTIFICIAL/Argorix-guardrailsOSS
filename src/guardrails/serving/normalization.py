"""Normalización de entrada y arbitraje de resultados de la ruta de producción.

Vive fuera de :mod:`guardrails.serving.backend` para que la evaluación pueda
medir exactamente la misma función que se sirve, sin arrastrar FastAPI.

La ruta de producción no clasifica el texto tal como llega: primero revierte
ofuscaciones habituales —sustitución *leet*, separadores insertados entre
letras, letras espaciadas— y, si la normalización cambió algo, consulta al
modelo una segunda vez con la versión normalizada. Entre los dos resultados se
queda con el de mayor rango.

Ese arbitraje es monótono hacia el bloqueo: ``BLOCK`` siempre supera a
``ALLOW``. La ruta de producción, por tanto, nunca bloquea menos que el modelo
aislado. Evaluar sólo el modelo subestima el sobrebloqueo real.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from guardrails import taxonomy

LEET_TRANSLATION = str.maketrans(
    {
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "6": "g",
        "7": "t",
        "8": "b",
        "@": "a",
        "$": "s",
        "!": "i",
        "€": "e",
        "|": "i",
    }
)
SPLIT_JOIN_RE = re.compile(r"(?<=\w)[\.\-_~`'\"\\\/]+(?=\w)")
LETTER_SPACED_RE = re.compile(r"\b(?:[a-zA-ZáéíóúñÁÉÍÓÚÑ]\s+){2,}[a-zA-ZáéíóúñÁÉÍÓÚÑ]\b")
# El orden de severidad vive en guardrails.taxonomy, que es también el que usa
# la generación de datos y el generador del conjunto de evaluación.
LABEL_PRIORITY = taxonomy.SEVERITY


def join_spaced_letters(match: re.Match[str]) -> str:
    return match.group(0).replace(" ", "")


def normalize_for_guardrail(text: str) -> dict[str, Any]:
    ascii_text = unicodedata.normalize("NFKC", text)
    # LEET_TRANSLATION mapea carácter a carácter, así que la longitud se
    # conserva; strict=True hace explícita esa garantía.
    traducido = ascii_text.translate(LEET_TRANSLATION)
    replaced = sum(1 for a, b in zip(ascii_text, traducido, strict=True) if a != b)
    normalized = traducido
    separator_hits = len(SPLIT_JOIN_RE.findall(normalized))
    normalized = SPLIT_JOIN_RE.sub("", normalized)
    spaced_hits = len(LETTER_SPACED_RE.findall(normalized))
    normalized = LETTER_SPACED_RE.sub(join_spaced_letters, normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return {
        "text": normalized,
        "changed": normalized != text,
        "replacements": replaced,
        "separator_hits": separator_hits,
        "spaced_hits": spaced_hits,
        "score": replaced + separator_hits + spaced_hits,
    }


def result_rank(result: dict[str, Any]) -> tuple[int, int]:
    return taxonomy.rank(
        result.get("decision") or taxonomy.ALLOW,
        result.get("primary_label") or "SAFE",
    )


def merge_results(
    original: dict[str, Any], normalized: dict[str, Any] | None, normalization: dict[str, Any]
) -> dict[str, Any]:
    final = dict(original)
    final["analysis_mode"] = "original"
    final["normalization"] = normalization
    final["variants"] = {"original": original}
    if not normalized:
        return final

    final["variants"]["normalized"] = normalized
    if not normalization.get("changed") or normalization.get("score", 0) <= 0:
        return final

    original_rank = result_rank(original)
    normalized_rank = result_rank(normalized)
    if normalized_rank > original_rank:
        final = dict(normalized)
        final["analysis_mode"] = "normalized_override"
    final["normalization"] = normalization
    final["variants"] = {"original": original, "normalized": normalized}
    return final
