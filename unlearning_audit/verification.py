"""Replay an audit against the exact normalized inputs it declares."""

from .fingerprints import DIGEST_FIELDS, fingerprint_inputs
from .groups import align_groups, group_digest
from .manifest import Manifest
from .metrics import compare_losses
from .report import validate_metrics
from .validation import LossExport


def verify_comparison(
    document: object, export: LossExport, manifest: Manifest, *,
    group_assignments: dict[str, str] | None = None,
) -> None:
    saved, _ = validate_metrics(document)
    recorded = saved.get("fingerprints")
    if recorded is None:
        raise ValueError("metrics have no fingerprints; compare again with --fingerprints")
    manifest.check_alignment(set(export.losses), set(export.sample_ids))
    expected = fingerprint_inputs(export, manifest)
    mismatches = [field for field in DIGEST_FIELDS if expected[field] != recorded[field]]
    if mismatches:
        raise ValueError("input fingerprints differ: " + ", ".join(mismatches))
    grouping = saved.get("grouping")
    if (grouping is None) != (group_assignments is None):
        raise ValueError("provide group assignments exactly when the saved audit used groups")
    if group_assignments is not None:
        if group_digest(export, align_groups(export, group_assignments)) != grouping["groups_sha256"]:
            raise ValueError("input fingerprints differ: groups_sha256")

    targets = tuple(point["target_fpr"] for point in saved["models"]["before"]["operating_points"])
    replayed = compare_losses(
        export, manifest, targets=targets, include_fingerprints=True,
        include_loss_gaps="loss_gaps" in saved, **saved["bootstrap"],
        group_assignments=group_assignments,
    )
    if saved != replayed:
        raise ValueError("saved metrics differ from replayed metrics")
