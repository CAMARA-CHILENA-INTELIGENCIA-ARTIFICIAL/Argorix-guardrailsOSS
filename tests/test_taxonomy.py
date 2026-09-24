"""Taxonomía canónica: severidad, resolución y política de decisión."""

from __future__ import annotations

import pandas as pd
import pytest

from guardrails import taxonomy
from guardrails.data import build_dataset as bd

ORDEN_ESPERADO = [
    "VIOLENCE",
    "HARMFUL",
    "PROMPT_INJECTION",
    "JAILBREAK",
    "HATE",
    "SEXUAL",
    "POLITICS",
    "SAFE",
]


def _etiquetas(*activas: str) -> dict[str, bool]:
    labels = taxonomy.empty_labels()
    for nombre in activas:
        labels[f"label_{nombre.lower()}"] = True
    return labels


def test_el_orden_de_severidad_es_el_documentado() -> None:
    assert list(taxonomy.LABELS) == ORDEN_ESPERADO


def test_resuelve_siempre_la_etiqueta_mas_severa() -> None:
    for i, mayor in enumerate(ORDEN_ESPERADO[:-1]):
        for menor in ORDEN_ESPERADO[i + 1 : -1]:
            assert taxonomy.resolve(_etiquetas(mayor, menor)) == mayor


def test_sin_etiquetas_activas_es_safe() -> None:
    assert taxonomy.resolve(taxonomy.empty_labels()) == "SAFE"
    assert taxonomy.decision_for(taxonomy.empty_labels()) == taxonomy.ALLOW


@pytest.mark.parametrize("etiqueta", [e for e in ORDEN_ESPERADO if e != "SAFE"])
def test_toda_etiqueta_distinta_de_safe_bloquea(etiqueta: str) -> None:
    assert taxonomy.decision_for(_etiquetas(etiqueta)) == taxonomy.BLOCK


def test_harmful_es_una_hoja_no_un_agregado() -> None:
    """Si HARMFUL se derivara de las demás, absorbería a HATE, SEXUAL y POLITICS.

    Con el orden de severidad adoptado, HARMFUL pesa más que esas tres, así que
    derivarla haría la taxonomía degenerada: cualquier texto de odio se
    resolvería como HARMFUL y las categorías específicas desaparecerían.
    """
    for especifica in ("HATE", "SEXUAL", "POLITICS"):
        assert taxonomy.resolve(_etiquetas(especifica)) == especifica

    fila = pd.Series({"POLITICS": 1, "HARMFULNESS": 0, "SEXUAL": 0, "VIOLENCE": 0, "HATE SPEECH": 0})
    etiquetas = bd.labels_guardrails(fila)
    assert etiquetas["label_politics"] is True
    assert etiquetas["label_harmful"] is False, "HARMFUL no debe derivarse"
    assert bd.primary_label(etiquetas) == "POLITICS"


def test_el_dano_fisico_pesa_mas_que_el_tipo_de_ataque() -> None:
    """Criterio del orden adoptado."""
    assert taxonomy.resolve(_etiquetas("VIOLENCE", "JAILBREAK")) == "VIOLENCE"
    assert taxonomy.resolve(_etiquetas("VIOLENCE", "PROMPT_INJECTION")) == "VIOLENCE"
    assert taxonomy.resolve(_etiquetas("HARMFUL", "JAILBREAK")) == "HARMFUL"


def test_hay_una_sola_definicion_del_orden() -> None:
    """Datos, runtime y conjunto de evaluación importan la severidad del mismo módulo."""
    from guardrails.evaluation import golden_set
    from guardrails.serving import normalization

    assert normalization.LABEL_PRIORITY is taxonomy.SEVERITY
    assert normalization.result_rank({"decision": "BLOCK", "primary_label": "VIOLENCE"}) == (
        taxonomy.rank("BLOCK", "VIOLENCE")
    )
    assert bd.LABEL_COLUMNS is taxonomy.COLUMNS
    # El generador del conjunto deriva la decisión de la taxonomía, no de listas fijas.
    assert golden_set.taxonomy is taxonomy


def test_normalize_recalcula_label_safe() -> None:
    activas = taxonomy.normalize(_etiquetas("HATE"))
    assert activas["label_safe"] is False

    vacias = taxonomy.normalize(taxonomy.empty_labels())
    assert vacias["label_safe"] is True


def test_labels_from_primary_es_inverso_de_resolve() -> None:
    for etiqueta in taxonomy.LABELS:
        assert taxonomy.resolve(taxonomy.labels_from_primary(etiqueta)) == etiqueta


def test_rank_ordena_primero_por_decision() -> None:
    """El runtime arbitra entre la variante original y la normalizada."""
    bloqueo_leve = taxonomy.rank(taxonomy.BLOCK, "POLITICS")
    permiso_alto = taxonomy.rank(taxonomy.ALLOW, "SAFE")
    assert bloqueo_leve > permiso_alto
    assert taxonomy.rank(taxonomy.BLOCK, "VIOLENCE") > taxonomy.rank(taxonomy.BLOCK, "SEXUAL")
