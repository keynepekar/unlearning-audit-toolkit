"""Read the metadata that gives an audit export its experimental meaning."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Self

ROLES = ("before", "candidate", "reference")


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class Manifest:
    title: str
    loss_definition: str
    models: dict[str, str]
    cohorts: dict[str, str]
    provenance: dict[str, str]
    schema_version: int = 1

    @classmethod
    def from_dict(cls, document: object) -> Self:
        fields = {"schema_version", "title", "loss_definition", "models", "cohorts", "provenance"}
        if not isinstance(document, dict) or document.keys() != fields:
            raise ValueError("manifest must contain: " + ", ".join(sorted(fields)))
        if type(document["schema_version"]) is not int or document["schema_version"] != 1:
            raise ValueError("unsupported manifest schema_version (expected 1)")

        mappings: dict[str, dict[str, str]] = {}
        for field in ("models", "cohorts", "provenance"):
            value = document[field]
            if not isinstance(value, dict) or not value:
                raise ValueError(f"{field} must be a non-empty object")
            mappings[field] = {
                _text(key, f"{field} key"): _text(item, f"{field}.{key}")
                for key, item in value.items()
            }
            if len(mappings[field]) != len(value):
                raise ValueError(f"{field} has duplicate keys after trimming")

        if mappings["models"].keys() != set(ROLES):
            raise ValueError("models must assign before, candidate and reference")
        if len(set(mappings["models"].values())) != len(ROLES):
            raise ValueError("each model role must have a different name")
        if "forget" not in mappings["cohorts"] or "holdout" not in mappings["cohorts"]:
            raise ValueError("cohorts must describe forget and holdout")
        if mappings["cohorts"].keys() - {"forget", "holdout", "utility"}:
            raise ValueError("unsupported cohort in manifest")

        return cls(
            title=_text(document["title"], "title"),
            loss_definition=_text(document["loss_definition"], "loss_definition"),
            models=mappings["models"],
            cohorts=mappings["cohorts"],
            provenance=mappings["provenance"],
        )

    def check_alignment(self, model_names: set[str], cohort_names: set[str]) -> None:
        if set(self.models.values()) != model_names:
            raise ValueError("manifest model names do not match the CSV")
        if self.cohorts.keys() != cohort_names:
            raise ValueError("manifest cohort names do not match the CSV")


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key!r}")
        result[key] = value
    return result


def load_manifest(path: Path) -> Manifest:
    with path.open(encoding="utf-8-sig") as stream:
        return Manifest.from_dict(json.load(stream, object_pairs_hook=_unique_fields))
