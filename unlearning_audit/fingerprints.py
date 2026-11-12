"""Fingerprint normalized audit inputs using a versioned encoding."""

import json
import re
from hashlib import sha256

from .manifest import ROLES, Manifest
from .validation import LossExport

SCHEME = "unlearning-audit-sha256-v1"
DIGEST_FIELDS = ("split_sha256", "losses_sha256", "manifest_sha256")


def _json_record(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def fingerprint_inputs(export: LossExport, manifest: Manifest) -> dict[str, str]:
    """Hash assignments, numeric losses and metadata without storing sample IDs."""
    split = sha256(_json_record([SCHEME, "split"]))
    losses = sha256(_json_record([SCHEME, "losses"]))
    for cohort, sample_ids in sorted(export.sample_ids.items()):
        indices = sorted(range(len(sample_ids)), key=sample_ids.__getitem__)
        for index in indices:
            split.update(_json_record([cohort, sample_ids[index]]))
        for role in ROLES:
            values = export.losses[manifest.models[role]][cohort]
            for index in indices:
                loss = float(values[index]) or 0.0
                losses.update(_json_record([role, cohort, sample_ids[index], loss.hex()]))
    metadata = sha256(_json_record([SCHEME, "manifest", manifest.as_dict()]))
    return {
        "scheme": SCHEME,
        "split_sha256": split.hexdigest(),
        "losses_sha256": losses.hexdigest(),
        "manifest_sha256": metadata.hexdigest(),
    }


def validate_fingerprints(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or value.keys() != {"scheme", *DIGEST_FIELDS}:
        raise ValueError("invalid fingerprint fields")
    if value["scheme"] != SCHEME:
        raise ValueError("unsupported fingerprint scheme")
    for field in DIGEST_FIELDS:
        digest = value[field]
        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError(f"invalid {field}")
    return value
