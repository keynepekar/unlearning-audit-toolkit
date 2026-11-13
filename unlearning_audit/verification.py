"""Replay an audit against the exact normalized inputs it declares."""

from .fingerprints import DIGEST_FIELDS, fingerprint_inputs
from .manifest import Manifest
from .metrics import compare_losses
from .report import validate_metrics
from .validation import LossExport


def verify_comparison(document: object, export: LossExport, manifest: Manifest) -> None:
    saved, _ = validate_metrics(document)
    recorded = saved.get("fingerprints")
    if recorded is None:
        raise ValueError("metrics have no fingerprints; compare again with --fingerprints")
    manifest.check_alignment(set(export.losses), set(export.sample_ids))
    expected = fingerprint_inputs(export, manifest)
    mismatches = [field for field in DIGEST_FIELDS if expected[field] != recorded[field]]
    if mismatches:
        raise ValueError("input fingerprints differ: " + ", ".join(mismatches))

    targets = tuple(point["target_fpr"] for point in saved["models"]["before"]["operating_points"])
    replayed = compare_losses(
        export, manifest, targets=targets, include_fingerprints=True,
        include_loss_gaps="loss_gaps" in saved, **saved["bootstrap"],
    )
    if saved != replayed:
        raise ValueError("saved metrics differ from replayed metrics")
