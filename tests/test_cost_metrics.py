"""Métricas de evaluación y función de costo."""

from __future__ import annotations

import json

import pytest

from guardrails import paths
from guardrails.evaluation import metrics


def _obs(esperado_dec, esperado_lab, predicho=None, **kw):
    return metrics.observation_from(
        {"expected_decision": esperado_dec, "expected_primary_label": esperado_lab},
        predicho,
        **kw,
    )


# ---------------------------------------------------------------------------
# Tasas acotadas
# ---------------------------------------------------------------------------


def test_las_tasas_estan_acotadas_con_entradas_repetidas() -> None:
    """Numerador y denominador se cuentan sobre el mismo conjunto de filas."""
    obs = [_obs("ALLOW", "SAFE", {"decision": "BLOCK", "primary_label": "HATE"}) for _ in range(3)]
    r = metrics.summarize(obs)
    fpr = r["decision"]["false_positive_rate"]

    assert fpr["numerator"] == 3
    assert fpr["denominator"] == 3
    assert 0.0 <= fpr["value"] <= 1.0


@pytest.mark.parametrize("n_allow,n_fp", [(1, 1), (10, 3), (600, 0), (600, 600)])
def test_fpr_siempre_en_el_intervalo_unitario(n_allow: int, n_fp: int) -> None:
    obs = [
        _obs("ALLOW", "SAFE", {"decision": "BLOCK" if i < n_fp else "ALLOW", "primary_label": "SAFE"})
        for i in range(n_allow)
    ]
    valor = metrics.summarize(obs)["decision"]["false_positive_rate"]["value"]
    assert 0.0 <= valor <= 1.0


# ---------------------------------------------------------------------------
# Salidas inválidas
# ---------------------------------------------------------------------------


def test_una_salida_no_parseable_cuenta_como_error() -> None:
    """Un JSON inválido es un fallo operativo del guardrail, no una fila a descartar."""
    obs = [
        _obs("ALLOW", "SAFE", None),
        _obs("ALLOW", "SAFE", {"decision": "ALLOW", "primary_label": "SAFE"}),
        _obs("BLOCK", "HATE", None),
        _obs("BLOCK", "HATE", {"decision": "BLOCK", "primary_label": "HATE"}),
    ]
    r = metrics.summarize(obs)

    assert r["decision"]["accuracy"]["value"] == 0.5
    assert r["decision"]["false_positive_rate"]["value"] == 0.5
    assert r["decision"]["false_negative_rate"]["value"] == 0.5
    assert r["parse_errors"]["count"] == 2
    assert r["parse_errors"]["parse_error_rate"]["value"] == 0.5
    assert r["confusion_matrix"]["invalid_on_allow"] == 1
    assert r["confusion_matrix"]["invalid_on_block"] == 1


def test_la_tasa_de_error_nunca_es_menor_que_la_de_parseo() -> None:
    """Si todo falla al parsear, ambas tasas de error son 1."""
    obs = [_obs("ALLOW", "SAFE", None), _obs("BLOCK", "HATE", None)]
    r = metrics.summarize(obs)
    assert r["decision"]["false_positive_rate"]["value"] == 1.0
    assert r["decision"]["false_negative_rate"]["value"] == 1.0
    assert r["decision"]["accuracy"]["value"] == 0.0


# ---------------------------------------------------------------------------
# Matriz de confusión y métricas derivadas
# ---------------------------------------------------------------------------


def test_se_registran_los_cuatro_cuadrantes() -> None:
    obs = [
        _obs("BLOCK", "HATE", {"decision": "BLOCK", "primary_label": "HATE"}),
        _obs("ALLOW", "SAFE", {"decision": "ALLOW", "primary_label": "SAFE"}),
        _obs("ALLOW", "SAFE", {"decision": "BLOCK", "primary_label": "HATE"}),
        _obs("BLOCK", "HATE", {"decision": "ALLOW", "primary_label": "SAFE"}),
    ]
    m = metrics.summarize(obs)["confusion_matrix"]
    assert (m["true_positive"], m["true_negative"], m["false_positive"], m["false_negative"]) == (
        1,
        1,
        1,
        1,
    )


def test_precision_recall_f1_y_mcc_son_calculables() -> None:
    obs = [_obs("BLOCK", "HATE", {"decision": "BLOCK", "primary_label": "HATE"}) for _ in range(8)]
    obs += [_obs("ALLOW", "SAFE", {"decision": "BLOCK", "primary_label": "HATE"}) for _ in range(2)]
    obs += [_obs("ALLOW", "SAFE", {"decision": "ALLOW", "primary_label": "SAFE"}) for _ in range(10)]
    d = metrics.summarize(obs)["decision"]

    assert d["precision"]["value"] == pytest.approx(8 / 10)
    assert d["recall"]["value"] == pytest.approx(1.0)
    assert d["f1"] == pytest.approx(2 * 0.8 * 1.0 / 1.8)
    assert d["mcc"] is not None
    assert d["balanced_accuracy"] is not None


def test_exactitud_de_etiqueta_macro_y_micro() -> None:
    """El macro-promedio no deja que la clase mayoritaria domine."""
    obs = [_obs("ALLOW", "SAFE", {"decision": "ALLOW", "primary_label": "SAFE"}) for _ in range(90)]
    obs += [_obs("BLOCK", "HATE", {"decision": "BLOCK", "primary_label": "SEXUAL"}) for _ in range(10)]
    r = metrics.summarize(obs)["primary_label"]

    assert r["accuracy_micro"]["value"] == pytest.approx(0.9)
    assert r["accuracy_macro"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Incertidumbre
# ---------------------------------------------------------------------------


def test_toda_proporcion_lleva_intervalo_de_confianza() -> None:
    obs = [_obs("ALLOW", "SAFE", {"decision": "ALLOW", "primary_label": "SAFE"}) for _ in range(600)]
    d = metrics.summarize(obs)["decision"]
    for clave in ("accuracy", "false_positive_rate", "precision", "recall"):
        ic = d[clave]["ci95"]
        assert ic is None or (len(ic) == 2 and ic[0] <= ic[1])


def test_el_intervalo_contiene_la_estimacion_puntual() -> None:
    for exitos, total in [(0, 600), (1, 400), (999, 1000), (27, 50)]:
        bajo, alto = metrics.clopper_pearson(exitos, total)
        assert bajo <= exitos / total <= alto


def test_n_efectivo_agrupa_por_unidad_semantica() -> None:
    """Filas derivadas de una misma unidad no son observaciones independientes."""
    obs = [
        _obs("ALLOW", "SAFE", {"decision": "ALLOW", "primary_label": "SAFE"}, group_id=f"g{i // 10}")
        for i in range(600)
    ]
    r = metrics.summarize(obs)
    assert r["total"] == 600
    assert r["effective_n"] == 60


def test_sin_group_id_cada_fila_es_su_propia_unidad() -> None:
    obs = [_obs("ALLOW", "SAFE", {"decision": "ALLOW", "primary_label": "SAFE"}) for _ in range(20)]
    assert metrics.summarize(obs)["effective_n"] == 20


# ---------------------------------------------------------------------------
# Costo
# ---------------------------------------------------------------------------


def test_el_costo_pondera_cada_tipo_de_error() -> None:
    obs = [_obs("ALLOW", "SAFE", {"decision": "BLOCK", "primary_label": "HATE"}) for _ in range(10)]
    obs += [_obs("BLOCK", "HATE", {"decision": "ALLOW", "primary_label": "SAFE"}) for _ in range(4)]

    barato = metrics.summarize(obs, metrics.CostModel(false_positive=1.0, false_negative=1.0))
    caro = metrics.summarize(obs, metrics.CostModel(false_positive=5.0, false_negative=1.0))

    assert barato["cost"]["total"] == 14.0
    assert caro["cost"]["total"] == 54.0


def test_el_costo_esperado_depende_del_volumen_operacional() -> None:
    """Con tasas asimétricas, la proporción de ataques cambia el costo esperado.

    FPR = 0,5 sobre 20 filas ALLOW; FNR = 1,0 sobre 10 filas BLOCK.
    """
    obs = [_obs("ALLOW", "SAFE", {"decision": "BLOCK", "primary_label": "HATE"}) for _ in range(10)]
    obs += [_obs("ALLOW", "SAFE", {"decision": "ALLOW", "primary_label": "SAFE"}) for _ in range(10)]
    obs += [_obs("BLOCK", "HATE", {"decision": "ALLOW", "primary_label": "SAFE"}) for _ in range(10)]

    r = metrics.summarize(obs, metrics.CostModel(arrival_rate_block=0.01))
    assert r["decision"]["false_positive_rate"]["value"] == pytest.approx(0.5)
    assert r["decision"]["false_negative_rate"]["value"] == pytest.approx(1.0)

    raro = r["cost"]["expected_per_request"]
    frecuente = metrics.summarize(obs, metrics.CostModel(arrival_rate_block=0.50))["cost"][
        "expected_per_request"
    ]

    assert raro == pytest.approx(0.99 * 0.5 + 0.01 * 1.0)
    assert frecuente == pytest.approx(0.50 * 0.5 + 0.50 * 1.0)
    assert frecuente > raro


def test_los_pesos_por_defecto_se_declaran_como_marcador() -> None:
    """Los pesos unitarios son un supuesto explícito, no una estimación."""
    modelo = metrics.CostModel().as_dict()
    assert modelo["weights"]["false_positive"] == 1.0
    assert modelo["weights"]["false_negative"] == 1.0
    assert "marcador de posición" in modelo["note"]


# ---------------------------------------------------------------------------
# Una sola implementación
# ---------------------------------------------------------------------------


def test_backend_y_evaluador_usan_el_mismo_modulo() -> None:
    """Una sola implementación de métricas para el panel y para el evaluador."""
    from guardrails.evaluation import run_eval
    from guardrails.serving import backend

    assert backend.metrics is metrics
    assert run_eval.metrics is metrics


def test_metricas_publicadas_son_coherentes_con_los_conteos() -> None:
    """Integridad aritmética del artefacto histórico, en su formato antiguo."""
    ruta = paths.REPORTS_DIR / "v3_golden_metrics.json"
    if not ruta.is_file():
        pytest.skip("reports/v3_golden_metrics.json no está disponible")

    m = json.loads(ruta.read_text(encoding="utf-8"))
    if "decision" in m:
        pytest.skip("el artefacto ya usa el formato nuevo")
    assert m["decision_correct"] / m["total"] == pytest.approx(m["decision_accuracy"])
