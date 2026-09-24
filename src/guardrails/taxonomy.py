"""Taxonomía canónica de etiquetas.

Definición única del esquema de etiquetas, del orden de severidad y de la
política de decisión. La importan la generación de datos, el runtime y el
generador del conjunto de evaluación, de modo que un mismo texto se resuelve
igual en los tres.

El orden es el de severidad operacional: el daño físico pesa más que el tipo de
ataque.

    VIOLENCE > HARMFUL > PROMPT_INJECTION > JAILBREAK > HATE > SEXUAL > POLITICS > SAFE

``HARMFUL`` es una etiqueta hoja, no un agregado: significa «dañino sin
categoría más específica». No se deriva de las demás. Derivarla haría el orden
degenerado, porque cualquier texto de odio, sexual o político quedaría
resuelto como ``HARMFUL`` al tener mayor severidad que su categoría propia.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

# Orden de severidad. El valor numérico sólo define el orden relativo.
SEVERITY: dict[str, int] = {
    "VIOLENCE": 80,
    "HARMFUL": 70,
    "PROMPT_INJECTION": 65,
    "JAILBREAK": 64,
    "HATE": 60,
    "SEXUAL": 55,
    "POLITICS": 45,
    "SAFE": 0,
}

#: Etiquetas ordenadas de mayor a menor severidad.
LABELS: tuple[str, ...] = tuple(sorted(SEVERITY, key=lambda name: -SEVERITY[name]))

#: Clases que no son SAFE, en orden de severidad.
BLOCKING_LABELS: tuple[str, ...] = tuple(name for name in LABELS if name != "SAFE")

#: Nombre de la columna booleana de cada etiqueta.
COLUMNS: tuple[str, ...] = ("label_safe", *(f"label_{name.lower()}" for name in BLOCKING_LABELS))

_COLUMN_TO_LABEL: dict[str, str] = {f"label_{name.lower()}": name for name in BLOCKING_LABELS}

ALLOW = "ALLOW"
BLOCK = "BLOCK"


def empty_labels() -> dict[str, bool]:
    """Diccionario de etiquetas con todas las columnas en ``False``."""
    return dict.fromkeys(COLUMNS, False)


def active_labels(labels: Mapping[str, bool]) -> list[str]:
    """Etiquetas activas, ordenadas de mayor a menor severidad."""
    return [name for name in BLOCKING_LABELS if labels.get(f"label_{name.lower()}", False)]


def resolve(labels: Mapping[str, bool]) -> str:
    """Etiqueta primaria: la activa de mayor severidad, o ``SAFE``."""
    activas = active_labels(labels)
    return activas[0] if activas else "SAFE"


def most_severe(names: Iterable[str]) -> str:
    """La más severa de un conjunto de etiquetas ya resueltas."""
    candidatas = [name for name in names if name in SEVERITY]
    if not candidatas:
        return "SAFE"
    return max(candidatas, key=lambda name: SEVERITY[name])


def decision_for(labels: Mapping[str, bool]) -> str:
    """``ALLOW`` si ninguna etiqueta de bloqueo está activa, ``BLOCK`` si no."""
    return ALLOW if resolve(labels) == "SAFE" else BLOCK


def decision_for_label(label: str) -> str:
    """Decisión que corresponde a una etiqueta primaria ya resuelta."""
    return ALLOW if label == "SAFE" else BLOCK


def normalize(labels: Mapping[str, bool]) -> dict[str, bool]:
    """Copia con ``label_safe`` recalculado como «ninguna otra activa»."""
    resultado = {column: bool(labels.get(column, False)) for column in COLUMNS}
    resultado["label_safe"] = not any(resultado[column] for column in COLUMNS if column != "label_safe")
    return resultado


def labels_from_primary(label: str) -> dict[str, bool]:
    """Vector de etiquetas correspondiente a una única etiqueta primaria."""
    resultado = empty_labels()
    if label == "SAFE":
        resultado["label_safe"] = True
    else:
        columna = f"label_{label.lower()}"
        if columna not in _COLUMN_TO_LABEL:
            raise ValueError(f"etiqueta desconocida: {label!r}")
        resultado[columna] = True
    return resultado


def rank(decision: str, label: str) -> tuple[int, int]:
    """Rango para comparar dos resultados: primero la decisión, luego la severidad.

    Lo usa el runtime para arbitrar entre la variante original de un texto y su
    versión normalizada.
    """
    severidad = SEVERITY.get(str(label).upper(), 10)
    bloqueo = 100 if str(decision).upper() == BLOCK else 0
    return bloqueo + severidad, severidad


__all__ = [
    "ALLOW",
    "BLOCK",
    "BLOCKING_LABELS",
    "COLUMNS",
    "LABELS",
    "SEVERITY",
    "active_labels",
    "decision_for",
    "decision_for_label",
    "empty_labels",
    "labels_from_primary",
    "most_severe",
    "normalize",
    "rank",
    "resolve",
]
