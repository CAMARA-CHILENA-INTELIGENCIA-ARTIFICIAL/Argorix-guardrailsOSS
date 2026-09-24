"""Puntuación continua de la decisión y métricas que habilita."""

from __future__ import annotations

import pytest

from guardrails import taxonomy
from guardrails.evaluation import metrics, scoring


class _Tokenizer:
    """Tokenizador mínimo: una palabra, un identificador."""

    def __init__(self, mapa: dict[str, int]) -> None:
        self.mapa = mapa

    def encode(self, texto: str, add_special_tokens: bool = False) -> list[int]:
        return [self.mapa[texto[0]]] if texto and texto[0] in self.mapa else []


def test_la_cabeza_resuelve_tokens_distintos() -> None:
    cabeza = scoring.DecisionHead.from_tokenizer(_Tokenizer({"A": 11, "B": 22}))
    assert (cabeza.allow_id, cabeza.block_id) == (11, 22)


def test_falla_si_las_decisiones_comparten_primer_token() -> None:
    """Con un primer token común la puntuación no distingue las clases."""
    with pytest.raises(ValueError, match="comparten primer token"):
        scoring.DecisionHead.from_tokenizer(_Tokenizer({"A": 7, "B": 7}))


def test_falla_si_el_tokenizador_no_produce_tokens() -> None:
    with pytest.raises(ValueError, match="no produce tokens"):
        scoring.first_token_id(_Tokenizer({}), "ALLOW")


def test_el_prefijo_deja_la_decision_en_la_siguiente_posicion() -> None:
    assert scoring.DECISION_PREFIX.endswith('"')
    assert '"decision"' in scoring.DECISION_PREFIX


@pytest.mark.parametrize(
    ("probabilidad", "umbral", "esperado"),
    [
        (0.90, 0.5, taxonomy.BLOCK),
        (0.10, 0.5, taxonomy.ALLOW),
        (0.50, 0.5, taxonomy.BLOCK),
        (0.30, 0.2, taxonomy.BLOCK),
        (0.30, 0.8, taxonomy.ALLOW),
    ],
)
def test_el_umbral_desplaza_la_decision(probabilidad, umbral, esperado) -> None:
    assert scoring.decision_from_probability(probabilidad, umbral) == esperado


# ---------------------------------------------------------------------------
# Curvas
# ---------------------------------------------------------------------------


def test_roc_auc_de_un_separador_perfecto_es_uno() -> None:
    scores = [0.9, 0.8, 0.2, 0.1]
    positivos = [True, True, False, False]
    assert metrics.roc_auc(scores, positivos) == 1.0


def test_roc_auc_de_una_puntuacion_invertida_es_cero() -> None:
    assert metrics.roc_auc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 0.0


def test_roc_auc_con_empates_totales_es_media() -> None:
    assert metrics.roc_auc([0.5] * 6, [True, True, True, False, False, False]) == 0.5


def test_roc_auc_sin_alguna_clase_no_esta_definida() -> None:
    assert metrics.roc_auc([0.5, 0.6], [True, True]) is None
    assert metrics.roc_auc([0.5, 0.6], [False, False]) is None


def test_average_precision_de_un_separador_perfecto_es_uno() -> None:
    assert metrics.average_precision([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0


def test_average_precision_con_empates_totales_es_la_prevalencia() -> None:
    positivos = [True] * 3 + [False] * 7
    assert metrics.average_precision([0.5] * 10, positivos) == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Punto de operación
# ---------------------------------------------------------------------------


def test_el_barrido_recorre_el_rango_de_umbrales() -> None:
    puntos = metrics.threshold_sweep([0.2, 0.8], [False, True], pasos=11)
    assert len(puntos) == 11
    assert puntos[0]["threshold"] == 0.0
    assert puntos[-1]["threshold"] == 1.0
    # Con umbral 0 se bloquea todo: ningún falso negativo, todos falsos positivos.
    assert puntos[0]["false_negative_rate"] == 0.0
    assert puntos[0]["false_positive_rate"] == 1.0


def test_el_punto_de_operacion_respeta_el_techo_de_falsos_positivos() -> None:
    """Forma habitual de fijar el umbral: declarar la fricción que se tolera."""
    scores = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
    positivos = [False] * 4 + [True] * 4

    punto = metrics.operating_point(scores, positivos, max_false_positive_rate=0.0)
    assert punto is not None
    assert punto["false_positive_rate"] <= 0.0
    assert punto["false_negative_rate"] == 0.0


def test_un_techo_inalcanzable_no_devuelve_punto() -> None:
    # Todo empatado: cualquier umbral que bloquee algo bloquea todo.
    assert metrics.operating_point([0.5] * 4, [True, True, False, False], -1.0) is None


def test_el_informe_incluye_el_bloque_de_puntuacion() -> None:
    obs = [
        metrics.Observation("BLOCK", "HATE", "BLOCK", "HATE", score=0.9),
        metrics.Observation("BLOCK", "HATE", "BLOCK", "HATE", score=0.8),
        metrics.Observation("ALLOW", "SAFE", "ALLOW", "SAFE", score=0.2),
        metrics.Observation("ALLOW", "SAFE", "ALLOW", "SAFE", score=0.1),
    ]
    bloque = metrics.summarize(obs)["score"]

    assert bloque["scored"] == 4
    assert bloque["roc_auc"] == 1.0
    assert bloque["average_precision"] == 1.0
    assert bloque["threshold_sweep"]
    assert "fpr<=0.01" in bloque["operating_points"]


def test_sin_puntuaciones_el_bloque_queda_vacio() -> None:
    obs = [metrics.Observation("ALLOW", "SAFE", "ALLOW", "SAFE")]
    assert metrics.summarize(obs)["score"] is None
