"""Cálculo de métricas de evaluación.

Implementación única, compartida por el evaluador y por la consola de
gobernanza, de modo que ambos reportan las mismas cifras para una misma
ejecución.

Propiedades que garantiza:

* **Tasas acotadas.** Numerador y denominador se cuentan sobre el mismo
  conjunto de observaciones, en una sola pasada.
* **Las salidas inválidas cuentan como error.** Una predicción que no se pudo
  parsear es un fallo operativo del guardrail: se contabiliza en las tasas de
  error y además se reporta por separado en ``parse_error_rate``.
* **Matriz de confusión completa.** Se registran TP, TN, FP y FN, de modo que
  precisión, recall, F1 y MCC son derivables.
* **Incertidumbre explícita.** Cada proporción se acompaña de su intervalo de
  confianza exacto de Clopper-Pearson.
* **Agrupación.** Cuando las observaciones no son independientes —varias filas
  derivadas de una misma unidad semántica— se agregan por ``group_id`` antes de
  calcular la incertidumbre.
* **Costo explícito.** Los pesos de cada tipo de error son un parámetro, no una
  suposición implícita.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from guardrails import taxonomy

#: Valor que toma la decisión cuando la salida del modelo no se pudo parsear.
INVALID = "INVALID"


# ---------------------------------------------------------------------------
# Intervalos de confianza
# ---------------------------------------------------------------------------


def _betainv(a: float, b: float, p: float) -> float:
    """Inversa de la beta regularizada por bisección.

    Evita depender de scipy en tiempo de evaluación; la precisión es suficiente
    para reportar tres decimales.
    """
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _betainc(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _betacf(a: float, b: float, x: float) -> float:
    """Fracción continua de Lentz para la beta incompleta."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = tiny if abs(d) < tiny else d
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Beta incompleta regularizada I_x(a, b).

    Usa la transformación de simetría I_x(a,b) = 1 - I_{1-x}(b,a) cuando x cae
    del lado en que la fracción continua no converge.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(ln_beta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return min(max(front * _betacf(a, b, x) / a, 0.0), 1.0)
    return min(max(1.0 - front * _betacf(b, a, 1.0 - x) / b, 0.0), 1.0)


def clopper_pearson(exitos: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    """Intervalo de confianza exacto para una proporción binomial."""
    if total <= 0:
        return (0.0, 1.0)
    bajo = 0.0 if exitos == 0 else _betainv(exitos, total - exitos + 1, alpha / 2)
    alto = 1.0 if exitos == total else _betainv(exitos + 1, total - exitos, 1 - alpha / 2)
    return (round(bajo, 6), round(alto, 6))


def proportion(exitos: int, total: int, alpha: float = 0.05) -> dict[str, Any]:
    """Proporción con su intervalo de confianza y sus conteos."""
    if total <= 0:
        return {"value": None, "ci95": None, "numerator": exitos, "denominator": 0}
    bajo, alto = clopper_pearson(exitos, total, alpha)
    return {
        "value": round(exitos / total, 6),
        "ci95": [bajo, alto],
        "numerator": exitos,
        "denominator": total,
    }


# ---------------------------------------------------------------------------
# Observaciones
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Observation:
    """Una fila evaluada.

    ``group_id`` identifica la unidad semántica de la que deriva la fila. Varias
    filas que comparten grupo no son observaciones independientes.
    """

    expected_decision: str
    expected_label: str
    predicted_decision: str
    predicted_label: str
    group_id: str = ""
    latency_ms: float | None = None
    #: Probabilidad que el modelo asigna a BLOCK, si se solicitó.
    score: float | None = None

    @property
    def parse_failed(self) -> bool:
        return self.predicted_decision == INVALID

    @property
    def decision_ok(self) -> bool:
        return self.predicted_decision == self.expected_decision

    @property
    def label_ok(self) -> bool:
        return self.predicted_label == self.expected_label


def observation_from(
    expected: dict[str, str],
    predicted: dict[str, Any] | None,
    *,
    group_id: str = "",
    latency_ms: float | None = None,
    score: float | None = None,
) -> Observation:
    """Construye una observación a partir de lo esperado y lo predicho.

    ``predicted`` en ``None`` o sin ``decision`` significa que la salida del
    modelo no se pudo parsear: se registra como ``INVALID``, no se descarta.
    """
    payload = predicted or {}
    decision = str(payload.get("decision") or "").upper() or INVALID
    label = str(payload.get("primary_label") or "").upper() or INVALID
    return Observation(
        expected_decision=str(expected["expected_decision"]).upper(),
        expected_label=str(expected["expected_primary_label"]).upper(),
        predicted_decision=decision,
        predicted_label=label,
        group_id=group_id,
        latency_ms=latency_ms,
        score=score,
    )


# ---------------------------------------------------------------------------
# Costo
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostModel:
    """Pesos del costo de cada tipo de error.

    Los valores por defecto son unitarios: equivalen a contar errores sin
    ponderar. **Son un marcador de posición**, no una estimación. Mientras no se
    fije la razón real entre el costo de un falso positivo y el de un falso
    negativo, el costo total sólo sirve para comparar modelos entre sí bajo el
    mismo supuesto, no como magnitud absoluta.

    ``arrival_rate_block`` es la proporción esperada de tráfico que debería
    bloquearse en producción. Si se declara, se calcula además el costo esperado
    por petición, que es lo que cambia con el volumen operacional.
    """

    false_positive: float = 1.0
    false_negative: float = 1.0
    true_positive: float = 0.0
    true_negative: float = 0.0
    arrival_rate_block: float | None = None
    currency: str = "unidades"

    def total(self, tp: int, tn: int, fp: int, fn: int) -> float:
        return (
            fp * self.false_positive
            + fn * self.false_negative
            + tp * self.true_positive
            + tn * self.true_negative
        )

    def expected_per_request(self, fpr: float | None, fnr: float | None) -> float | None:
        """Costo esperado por petición bajo la tasa de llegada declarada."""
        if self.arrival_rate_block is None or fpr is None or fnr is None:
            return None
        lambda_block = self.arrival_rate_block
        lambda_allow = 1.0 - lambda_block
        return round(lambda_allow * fpr * self.false_positive + lambda_block * fnr * self.false_negative, 8)

    def as_dict(self) -> dict[str, Any]:
        return {
            "weights": {
                "false_positive": self.false_positive,
                "false_negative": self.false_negative,
                "true_positive": self.true_positive,
                "true_negative": self.true_negative,
            },
            "arrival_rate_block": self.arrival_rate_block,
            "currency": self.currency,
            "note": (
                "Pesos por defecto unitarios: marcador de posición, no una estimación. "
                "Fijar la razón real entre falso positivo y falso negativo antes de "
                "interpretar el costo como magnitud absoluta."
            ),
        }


# ---------------------------------------------------------------------------
# Agregación
# ---------------------------------------------------------------------------


@dataclass
class ConfusionMatrix:
    """Matriz de confusión binaria de la decisión.

    Se toma ``BLOCK`` como clase positiva: bloquear correctamente es un TP.
    Una salida inválida nunca es un acierto.
    """

    true_positive: int = 0
    true_negative: int = 0
    false_positive: int = 0
    false_negative: int = 0
    invalid_on_allow: int = 0
    invalid_on_block: int = 0

    @property
    def total(self) -> int:
        return self.true_positive + self.true_negative + self.false_positive + self.false_negative

    def as_dict(self) -> dict[str, int]:
        return {
            "true_positive": self.true_positive,
            "true_negative": self.true_negative,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "invalid_on_allow": self.invalid_on_allow,
            "invalid_on_block": self.invalid_on_block,
        }


def _confusion(observaciones: Sequence[Observation]) -> ConfusionMatrix:
    m = ConfusionMatrix()
    for o in observaciones:
        espera_bloqueo = o.expected_decision == taxonomy.BLOCK
        if o.parse_failed:
            # Una salida no parseable es un fallo: cuenta como el error que
            # corresponde al lado esperado, no se descarta.
            if espera_bloqueo:
                m.false_negative += 1
                m.invalid_on_block += 1
            else:
                m.false_positive += 1
                m.invalid_on_allow += 1
            continue
        predice_bloqueo = o.predicted_decision == taxonomy.BLOCK
        if espera_bloqueo and predice_bloqueo:
            m.true_positive += 1
        elif espera_bloqueo:
            m.false_negative += 1
        elif predice_bloqueo:
            m.false_positive += 1
        else:
            m.true_negative += 1
    return m


def _mcc(m: ConfusionMatrix) -> float | None:
    tp, tn, fp, fn = m.true_positive, m.true_negative, m.false_positive, m.false_negative
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return None
    return round((tp * tn - fp * fn) / denom, 6)


def _percentiles(valores: list[float]) -> dict[str, float] | None:
    if not valores:
        return None
    ordenados = sorted(valores)

    def p(q: float) -> float:
        idx = min(len(ordenados) - 1, round(q * (len(ordenados) - 1)))
        return round(ordenados[idx], 3)

    return {"p50": p(0.50), "p95": p(0.95), "p99": p(0.99), "max": round(ordenados[-1], 3)}


def roc_auc(scores: Sequence[float], positives: Sequence[bool]) -> float | None:
    """Área bajo la curva ROC, por el estadístico de Mann-Whitney.

    Equivale a la probabilidad de que un caso positivo tomado al azar reciba
    mayor puntuación que uno negativo. Los empates cuentan como medio acierto.
    """
    pos = [s for s, p in zip(scores, positives, strict=True) if p]
    neg = [s for s, p in zip(scores, positives, strict=True) if not p]
    if not pos or not neg:
        return None

    ordenados = sorted(zip(scores, positives, strict=True), key=lambda par: par[0])
    rangos: dict[int, float] = {}
    i = 0
    while i < len(ordenados):
        j = i
        while j + 1 < len(ordenados) and ordenados[j + 1][0] == ordenados[i][0]:
            j += 1
        rango_medio = (i + j) / 2 + 1
        for k in range(i, j + 1):
            rangos[k] = rango_medio
        i = j + 1

    suma_pos = sum(rangos[k] for k, (_, es_pos) in enumerate(ordenados) if es_pos)
    n_pos, n_neg = len(pos), len(neg)
    return round((suma_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg), 6)


def average_precision(scores: Sequence[float], positives: Sequence[bool]) -> float | None:
    """Precisión media: área bajo la curva precisión-recall.

    Más informativa que la ROC cuando la clase positiva es minoritaria. Se
    evalúa en cada puntuación distinta, no en cada observación, de modo que un
    grupo de empates aporta un único punto a la curva.
    """
    if not scores or not any(positives):
        return None

    ordenados = sorted(zip(scores, positives, strict=True), key=lambda par: -par[0])
    total_pos = sum(positives)

    area = 0.0
    recall_previo = 0.0
    tp = fp = 0
    i = 0
    while i < len(ordenados):
        j = i
        while j + 1 < len(ordenados) and ordenados[j + 1][0] == ordenados[i][0]:
            j += 1
        for k in range(i, j + 1):
            if ordenados[k][1]:
                tp += 1
            else:
                fp += 1
        recall = tp / total_pos
        precision = tp / (tp + fp)
        area += (recall - recall_previo) * precision
        recall_previo = recall
        i = j + 1

    return round(area, 6)


def threshold_sweep(
    scores: Sequence[float], positives: Sequence[bool], pasos: int = 21
) -> list[dict[str, float]]:
    """Tasas de error a lo largo del rango de umbrales."""
    if not scores:
        return []
    n_pos = sum(positives)
    n_neg = len(positives) - n_pos
    puntos: list[dict[str, float]] = []
    for i in range(pasos):
        umbral = i / (pasos - 1)
        fp = sum(1 for s, p in zip(scores, positives, strict=True) if not p and s >= umbral)
        fn = sum(1 for s, p in zip(scores, positives, strict=True) if p and s < umbral)
        puntos.append(
            {
                "threshold": round(umbral, 4),
                "false_positive_rate": round(fp / n_neg, 6) if n_neg else None,
                "false_negative_rate": round(fn / n_pos, 6) if n_pos else None,
            }
        )
    return puntos


def operating_point(
    scores: Sequence[float], positives: Sequence[bool], max_false_positive_rate: float
) -> dict[str, float] | None:
    """Umbral más permisivo que respeta un techo de falsos positivos.

    Es la forma habitual de fijar el punto de operación de un guardrail: se
    declara cuánta fricción se tolera sobre tráfico benigno y se mide qué
    sensibilidad se obtiene a cambio.
    """
    if not scores:
        return None
    candidatos = [
        punto
        for punto in threshold_sweep(scores, positives, pasos=101)
        if punto["false_positive_rate"] is not None
        and punto["false_positive_rate"] <= max_false_positive_rate
    ]
    if not candidatos:
        return None
    elegido = min(candidatos, key=lambda punto: punto["threshold"])
    return {"target_false_positive_rate": max_false_positive_rate, **elegido}


@dataclass
class Report:
    """Resultado completo de una evaluación."""

    observations: list[Observation] = field(default_factory=list)
    cost_model: CostModel = field(default_factory=CostModel)

    def add(self, observation: Observation) -> None:
        self.observations.append(observation)

    # -- derivados ---------------------------------------------------------

    @property
    def effective_n(self) -> int:
        """Número de unidades semánticas distintas.

        Si ninguna observación declara ``group_id``, cada fila es su propia
        unidad y coincide con el total.
        """
        grupos = {o.group_id for o in self.observations if o.group_id}
        return len(grupos) if grupos else len(self.observations)

    def as_dict(self) -> dict[str, Any]:
        obs = self.observations
        total = len(obs)
        if total == 0:
            return {"total": 0}

        m = _confusion(obs)
        allow = [o for o in obs if o.expected_decision == taxonomy.ALLOW]
        block = [o for o in obs if o.expected_decision == taxonomy.BLOCK]
        invalidas = sum(1 for o in obs if o.parse_failed)

        fp_rate = proportion(m.false_positive, len(allow))
        fn_rate = proportion(m.false_negative, len(block))
        tp, tn, fp, fn = m.true_positive, m.true_negative, m.false_positive, m.false_negative

        precision = proportion(tp, tp + fp)
        recall = proportion(tp, tp + fn)
        pv, rv = precision["value"], recall["value"]
        f1 = round(2 * pv * rv / (pv + rv), 6) if pv and rv else 0.0

        # Exactitud por etiqueta, macro y micro.
        por_etiqueta: dict[str, dict[str, Any]] = {}
        for etiqueta in taxonomy.LABELS:
            filas = [o for o in obs if o.expected_label == etiqueta]
            if filas:
                por_etiqueta[etiqueta] = proportion(sum(1 for o in filas if o.label_ok), len(filas))
        macro = (
            round(sum(v["value"] for v in por_etiqueta.values()) / len(por_etiqueta), 6)
            if por_etiqueta
            else None
        )

        confusion_etiqueta = Counter(f"{o.expected_label}->{o.predicted_label}" for o in obs)
        latencias = [o.latency_ms for o in obs if o.latency_ms is not None]

        costo_total = self.cost_model.total(tp, tn, fp, fn)

        con_score = [o for o in obs if o.score is not None]
        bloque_score: dict[str, Any] | None = None
        if con_score:
            valores = [o.score for o in con_score]
            positivos = [o.expected_decision == taxonomy.BLOCK for o in con_score]
            bloque_score = {
                "scored": len(con_score),
                "roc_auc": roc_auc(valores, positivos),
                "average_precision": average_precision(valores, positivos),
                "threshold_sweep": threshold_sweep(valores, positivos),
                "operating_points": {
                    f"fpr<={objetivo}": operating_point(valores, positivos, objetivo)
                    for objetivo in (0.01, 0.02, 0.05, 0.10)
                },
            }

        return {
            "total": total,
            "effective_n": self.effective_n,
            "confusion_matrix": m.as_dict(),
            "decision": {
                "accuracy": proportion(tp + tn, total),
                "balanced_accuracy": (
                    round(((recall["value"] or 0) + (proportion(tn, tn + fp)["value"] or 0)) / 2, 6)
                    if allow and block
                    else None
                ),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mcc": _mcc(m),
                "false_positive_rate": fp_rate,
                "false_negative_rate": fn_rate,
            },
            "primary_label": {
                "accuracy_micro": proportion(sum(1 for o in obs if o.label_ok), total),
                "accuracy_macro": macro,
                "per_label": por_etiqueta,
                "confusion": dict(sorted(confusion_etiqueta.items())),
            },
            "parse_errors": {
                "count": invalidas,
                "parse_error_rate": proportion(invalidas, total),
            },
            "cost": {
                "model": self.cost_model.as_dict(),
                "total": round(costo_total, 6),
                "per_observation": round(costo_total / total, 6),
                "expected_per_request": self.cost_model.expected_per_request(
                    fp_rate["value"], fn_rate["value"]
                ),
            },
            "latency_ms": _percentiles(latencias),
            "score": bloque_score,
        }


def summarize(observaciones: Sequence[Observation], cost_model: CostModel | None = None) -> dict[str, Any]:
    """Atajo: construye un informe y lo serializa."""
    reporte = Report(list(observaciones), cost_model or CostModel())
    return reporte.as_dict()


__all__ = [
    "INVALID",
    "ConfusionMatrix",
    "CostModel",
    "Observation",
    "Report",
    "average_precision",
    "clopper_pearson",
    "observation_from",
    "operating_point",
    "proportion",
    "roc_auc",
    "summarize",
    "threshold_sweep",
]
