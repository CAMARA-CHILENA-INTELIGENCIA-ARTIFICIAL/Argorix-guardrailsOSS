"""Cobertura y reproducibilidad del conjunto de evaluación."""

from __future__ import annotations

import csv

import pytest

from guardrails import taxonomy
from guardrails.evaluation import golden_set


@pytest.fixture(scope="module")
def filas() -> list[dict[str, str]]:
    return golden_set.build_rows()


def test_es_determinista() -> None:
    """Dos generaciones producen exactamente el mismo archivo."""
    primera = golden_set.build_rows()
    segunda = golden_set.build_rows()
    assert [f["id"] for f in primera] == [f["id"] for f in segunda]
    assert [f["text"] for f in primera] == [f["text"] for f in segunda]


def test_no_hay_textos_repetidos(filas: list[dict[str, str]]) -> None:
    textos = [f["text"] for f in filas]
    assert len(textos) == len(set(textos))


def test_las_seis_categorias_benignas_estan_representadas(filas: list[dict[str, str]]) -> None:
    """El generador recorre las seis categorías, sin tope que corte antes."""
    declaradas = set(golden_set.SAFE_TOPICS)
    presentes = {f["category"] for f in filas if f["expected_decision"] == taxonomy.ALLOW}
    assert presentes == declaradas
    assert len(declaradas) == 6


def test_se_usan_los_diez_contextos(filas: list[dict[str, str]]) -> None:
    usados = {
        f["text"].rsplit(", ", 1)[-1].rstrip(".")
        for f in filas
        if f["variant"].startswith("plain_") and f["expected_decision"] == taxonomy.ALLOW
    }
    assert usados == set(golden_set.SAFE_CONTEXTS)


def test_las_ocho_clases_del_esquema_estan_representadas(filas: list[dict[str, str]]) -> None:
    """Las ocho, incluida POLITICS."""
    presentes = {f["expected_primary_label"] for f in filas}
    assert presentes == set(taxonomy.LABELS)


def test_politics_designa_manipulacion_no_informacion_civica(filas: list[dict[str, str]]) -> None:
    """Explicar una elección es SAFE; fabricar propaganda es POLITICS."""
    civicas = [f for f in filas if f["category"] == "politica_civica"]
    manipulacion = [f for f in filas if f["expected_primary_label"] == "POLITICS"]

    assert civicas and all(f["expected_decision"] == taxonomy.ALLOW for f in civicas)
    assert manipulacion and all(f["expected_decision"] == taxonomy.BLOCK for f in manipulacion)


def test_incluye_variantes_ofuscadas(filas: list[dict[str, str]]) -> None:
    """Sin ellas, la normalización de la ruta de producción no se ejercita."""
    ofuscadas = [f for f in filas if f["variant"].startswith("obfuscated")]
    assert len(ofuscadas) > 200

    tipos = {f["variant"] for f in ofuscadas}
    assert tipos == {"obfuscated_leet", "obfuscated_spaced", "obfuscated_punctuated"}

    # Y deben activar realmente la normalización del runtime.
    from guardrails.serving import backend

    activan = sum(1 for f in ofuscadas if backend.normalize_for_guardrail(f["text"])["score"] > 0)
    assert activan > 150, f"sólo {activan} de {len(ofuscadas)} activarían la segunda inferencia"


def test_incluye_prompts_largos(filas: list[dict[str, str]]) -> None:
    """El tráfico real es varias veces más largo que un prompt de plantilla."""
    largos = [f for f in filas if f["length_bucket"] == "largo"]
    assert len(largos) >= 100
    assert max(len(f["text"]) for f in filas) > 500


def test_cada_fila_declara_su_grupo(filas: list[dict[str, str]]) -> None:
    """El group_id permite no tratar como independientes filas de una misma base."""
    assert all(f["group_id"] for f in filas)
    grupos = {f["group_id"] for f in filas}
    assert 100 < len(grupos) < len(filas)


def test_la_decision_se_deriva_de_la_taxonomia(filas: list[dict[str, str]]) -> None:
    for fila in filas:
        esperada = taxonomy.decision_for_label(fila["expected_primary_label"])
        assert fila["expected_decision"] == esperada


def test_el_csv_declara_group_id(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    destino = tmp_path / "golden.csv"
    monkeypatch.setattr("sys.argv", ["golden_set", "--output", str(destino)])
    golden_set.main()

    with destino.open(encoding="utf-8") as handle:
        lector = csv.DictReader(handle)
        assert "group_id" in (lector.fieldnames or [])
        assert next(lector)["group_id"]
    assert destino.with_suffix(".summary.json").is_file()
