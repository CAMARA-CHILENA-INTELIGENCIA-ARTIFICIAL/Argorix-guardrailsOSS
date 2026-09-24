"""Partición de datasets en train / validation / test.

La partición se calcula sobre la **clave de grupo**, no sobre el identificador
de cada fila. Una variante léxica comparte grupo con el texto del que deriva, de
modo que ambas caen en la misma partición por construcción; calcularla por fila
las repartiría de forma independiente y dejaría el original en ``train`` con su
paráfrasis en ``test``.

Las proporciones y la sal son configurables. Cambiar la sal rota el reparto sin
tocar los datos, que es lo que permite estimar cuánta varianza aporta.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

TRAIN = "train"
VALIDATION = "validation"
TEST = "test"


@dataclass(frozen=True)
class SplitConfig:
    """Proporciones y sal de la partición.

    ``salt`` permite rotar el reparto sin cambiar los datos, que es lo que hace
    falta para estimar cuánta varianza aporta la partición.
    """

    train: int = 80
    validation: int = 10
    test: int = 10
    salt: str = ""

    def __post_init__(self) -> None:
        total = self.train + self.validation + self.test
        if total != 100:
            raise ValueError(f"las proporciones deben sumar 100, suman {total}")
        if min(self.train, self.validation, self.test) < 0:
            raise ValueError("las proporciones no pueden ser negativas")

    def as_dict(self) -> dict[str, object]:
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
            "salt": self.salt,
        }


DEFAULT = SplitConfig()


def source_group(record: Mapping[str, object]) -> str:
    """Clave de grupo derivada del origen de un registro.

    La comparten una traducción y todas las mutaciones que salen de ella, porque
    proceden de la misma fila del mismo dataset y del mismo rol de texto.
    """
    return "src:{}|{}|{}".format(
        record.get("source_dataset", ""),
        record.get("source_row", ""),
        record.get("text_role", ""),
    )


def group_key(record: Mapping[str, object]) -> str:
    """Clave que agrupa a un registro con todas sus derivaciones.

    Se usa el campo ``group_id`` cuando está presente: las mutaciones lo heredan
    de su origen, de modo que caen en la misma partición por construcción. Si no
    está, se deriva del origen.
    """
    explicito = str(record.get("group_id") or "").strip()
    return explicito or source_group(record)


def bucket(key: str, salt: str = "") -> int:
    """Cubo determinista en ``[0, 100)`` para una clave de grupo."""
    raw = f"{salt}:{key}" if salt else key
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8], 16) % 100


def assign(key: str, config: SplitConfig = DEFAULT) -> str:
    """Partición que corresponde a una clave de grupo."""
    valor = bucket(key, config.salt)
    if valor < config.train:
        return TRAIN
    if valor < config.train + config.validation:
        return VALIDATION
    return TEST


def assign_record(record: Mapping[str, object], config: SplitConfig = DEFAULT) -> str:
    """Partición de un registro, a partir de su clave de grupo."""
    return assign(group_key(record), config)


__all__ = [
    "DEFAULT",
    "TEST",
    "TRAIN",
    "VALIDATION",
    "SplitConfig",
    "assign",
    "assign_record",
    "bucket",
    "group_key",
    "source_group",
]
