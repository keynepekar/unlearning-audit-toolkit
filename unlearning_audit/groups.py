"""Align independent sampling groups with the ROC cohorts."""

import csv
import json
from hashlib import sha256
from pathlib import Path

from .validation import LossExport

GroupIndices = dict[str, tuple[tuple[int, ...], ...]]
ROC_COHORTS = ("forget", "holdout")


def load_groups(path: Path) -> dict[str, str]:
    assignments = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        if next(reader, None) != ["sample_id", "group_id"]:
            raise ValueError("expected groups header: sample_id,group_id")
        for row in reader:
            if len(row) != 2:
                raise ValueError(f"groups line {reader.line_num}: expected two fields")
            sample, group = (field.strip() for field in row)
            if not sample or not group or sample in assignments:
                raise ValueError(f"groups line {reader.line_num}: blank ID or duplicate sample")
            assignments[sample] = group
    return assignments


def align_groups(export: LossExport, assignments: dict[str, str]) -> GroupIndices:
    expected = {sample for cohort in ROC_COHORTS for sample in export.sample_ids[cohort]}
    if assignments.keys() != expected:
        raise ValueError("groups must assign every forget and holdout sample exactly once and no other samples")
    result = {}
    seen = {}
    for cohort in ROC_COHORTS:
        groups: dict[str, list[int]] = {}
        for index, sample in enumerate(export.sample_ids[cohort]):
            group = assignments[sample]
            if not isinstance(group, str) or not group.strip():
                raise ValueError("group IDs must be non-empty strings")
            group = group.strip()
            if seen.setdefault(group, cohort) != cohort:
                raise ValueError("groups must not span forget and holdout cohorts")
            groups.setdefault(group, []).append(index)
        if len(groups) < 2:
            raise ValueError(f"group bootstrap needs at least two groups in {cohort}")
        # Canonical order depends on members, so relabeling a group preserves replay.
        result[cohort] = tuple(sorted(tuple(indices) for indices in groups.values()))
    return result


def group_digest(export: LossExport, groups: GroupIndices) -> str:
    members = {
        cohort: [[export.sample_ids[cohort][index] for index in group] for group in groups[cohort]]
        for cohort in ROC_COHORTS
    }
    record = json.dumps(["unlearning-audit-groups-v1", members], separators=(",", ":"), ensure_ascii=True)
    return sha256(record.encode("ascii")).hexdigest()
