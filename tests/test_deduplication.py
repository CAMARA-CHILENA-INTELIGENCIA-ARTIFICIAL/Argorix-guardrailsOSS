"""Deduplicación consciente de conflictos de etiqueta."""

from __future__ import annotations

import pandas as pd

from guardrails import taxonomy
from guardrails.data import build_dataset as bd

TEXTO = "Como funciona una inyeccion SQL"


def _fila(fuente: str, columna: str | None, texto: str = TEXTO) -> dict:
    etiquetas = taxonomy.empty_labels()
    if columna:
        etiquetas[columna] = True
    return bd.base_record(
        source_dataset=fuente,
        source_split="train",
        source_row=1,
        text_role="prompt",
        original_language="es",
        original_text=texto,
        text_es=texto,
        labels=etiquetas,
        mutation_type="translation",
    )


def test_un_texto_repetido_y_coherente_se_conserva_una_vez() -> None:
    df = pd.DataFrame.from_records([_fila("A", None), _fila("B", None)])
    limpio, stats = bd.deduplicate(df)
    assert len(limpio) == 1
    assert stats["duplicates_removed"] == 1
    assert stats["conflicts"] == 0


def test_un_texto_con_etiquetas_contradictorias_se_descarta() -> None:
    """Resolver el empate en favor de una fuente arbitraria mete ruido silencioso."""
    df = pd.DataFrame.from_records([_fila("A", None), _fila("B", "label_harmful")])
    limpio, stats = bd.deduplicate(df)

    assert len(limpio) == 0
    assert stats["conflicts"] == 1
    assert stats["rows_dropped_by_conflict"] == 2


def test_el_conflicto_no_puede_repartirse_entre_particiones() -> None:
    """El caso que se buscaba evitar: el mismo texto en train y en test con verdades opuestas."""
    df = pd.DataFrame.from_records([_fila("A", None), _fila("B", "label_harmful"), _fila("C", None)])
    limpio, _ = bd.deduplicate(df)
    assert limpio.empty


def test_la_deduplicacion_ignora_mayusculas_y_espacios() -> None:
    df = pd.DataFrame.from_records([_fila("A", None, TEXTO), _fila("B", None, f"  {TEXTO.upper()}  ")])
    limpio, stats = bd.deduplicate(df)
    assert len(limpio) == 1
    assert stats["duplicates_removed"] == 1


def test_textos_distintos_no_se_tocan() -> None:
    df = pd.DataFrame.from_records([_fila("A", None, "uno"), _fila("B", "label_hate", "dos")])
    limpio, stats = bd.deduplicate(df)
    assert len(limpio) == 2
    assert stats == {"duplicates_removed": 0, "conflicts": 0, "rows_dropped_by_conflict": 0}
