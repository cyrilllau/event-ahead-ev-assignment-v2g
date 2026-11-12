"""Separate immutable legacy evidence from the currently executing source.

Historical hashes always resolve to retained historical bytes. Current package,
validator and test hashes are recorded separately, never substituted for them.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .config import get_paths

LEGACY_RELEASE = Path("/home/Shixin/ev-assignment/releases/v54-q1-20260905T003328Z-6827f937")
LEGACY_CASE_NAME = "IEEE33_paper_aligned_case_data_v1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_runner_root():
    root = get_paths().frozen_source_root
    for relative in ("runner", "15_experiment_runner", f"{LEGACY_CASE_NAME}/15_experiment_runner"):
        candidate = root / relative
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"No immutable legacy runner under {root}")


def frozen_source_file(relative):
    name = Path(relative)
    if name.is_absolute() or ".." in name.parts:
        raise ValueError("Accepted source names must be relative to the frozen runner")
    return frozen_runner_root() / name


def frozen_control_root():
    root = get_paths().frozen_source_root
    for relative in ("control", "14_experiment_specs", f"{LEGACY_CASE_NAME}/14_experiment_specs"):
        candidate = root / relative
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"No immutable legacy control files under {root}")


def legacy_transfer_manifest():
    root = get_paths().frozen_source_root
    for candidate in (root / "TRANSFER_MANIFEST_SHA256.csv", root.parent / "TRANSFER_MANIFEST_SHA256.csv"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Frozen transfer manifest not found under {root}")


def verify_accepted_sources(source_hashes):
    for relative, expected in source_hashes.items():
        path = frozen_source_file(relative)
        if sha256(path) != expected:
            raise ValueError(f"Immutable accepted source hash mismatch: {path}")
    return len(source_hashes)


def frozen_source_paths():
    return sorted(frozen_runner_root().rglob("*.py"))


def current_source_hashes():
    root = get_paths().project_root
    candidates = sorted((root / "src").rglob("*.py")) + sorted((root / "tests").rglob("*.py"))
    candidates += sorted((root / "scripts/audits").glob("*.py"))
    return {str(path.relative_to(root)): sha256(path) for path in candidates}


def verify_scientific_kernel_identity():
    """Verify original kernels or an exact, recorded post-migration revision.

    Historical evidence retains its original bytes. Authorized later changes
    must bind both original and current SHA; no filename-only exception exists.
    """
    current = Path(__file__).resolve().parent
    revision_path = get_paths().project_root / 'provenance/source_revisions/s04_oos_access.json'
    revisions = json.loads(revision_path.read_text())['files'] if revision_path.exists() else {}
    allowed = {'wind_scenarios.py','load_scenarios.py','line_outages.py','road_disruptions.py','task_replay.py'}
    if set(revisions) - allowed:
        raise ValueError('Unapproved kernel in S04 gate revision registry')
    count = 0
    for original in sorted((frozen_runner_root() / "evcs_experiments").glob("*.py")):
        active = current / original.name
        old, current_sha = sha256(original), sha256(active)
        revision = revisions.get(original.name)
        if revision is not None:
            if revision.get("original_sha256") != old or revision.get("current_sha256") != current_sha:
                raise ValueError(f"Invalid scientific kernel revision hashes: {original.name}")
        elif current_sha != old:
            raise ValueError(f"Unregistered scientific kernel revision: {original.name}")
        count += 1
    return count


def historical_path(value):
    """Resolve an existing retained path or an explicit old-campaign mirror.

    Missing large local artifacts remain missing; no hash check is skipped.
    Absolute paths outside the known release are not silently remapped.
    """
    path = Path(value).expanduser()
    if path.exists():
        return path.resolve()
    paths = get_paths()
    try:
        relative = path.relative_to(LEGACY_RELEASE)
    except ValueError:
        return path
    parts = relative.parts
    candidates = []
    if parts[:1] == ("experiment_campaign",):
        rest = Path(*parts[1:])
        candidates.append(paths.campaign_root / rest)
        if rest.parts[:1] == ("acceptance",):
            candidates.append(paths.project_root / "provenance" / rest)
        if rest.parts[:1] == ("increments",) and len(rest.parts) >= 3:
            increment, stamp = rest.parts[1:3]
            directory = increment.lower().replace("-", "_") + "_results"
            if increment == "S01-I08-preflight":
                directory = "s01_i08_preflight_results"
            candidates.append(paths.project_root / "artifacts/legacy_transfer" / directory / stamp / Path(*rest.parts[3:]))
        if rest.parts[:1] == ("deliverables",):
            candidates.append(paths.project_root / "artifacts/legacy_transfer/campaign_deliverables" / Path(*rest.parts[1:]))
    elif parts[:1] == (LEGACY_CASE_NAME,):
        rest = Path(*parts[1:])
        if rest.parts[:1] == ("15_experiment_runner",):
            candidates.append(frozen_runner_root() / Path(*rest.parts[1:]))
        elif rest.parts[:1] == ("14_experiment_specs",):
            candidates.append(frozen_control_root() / Path(*rest.parts[1:]))
        else:
            candidates.append(paths.data_root / rest)
    elif parts == ("TRANSFER_MANIFEST_SHA256.csv",):
        candidates.append(legacy_transfer_manifest())
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return path


def acceptance_for(evidence, increment):
    evidence = Path(evidence).resolve()
    paths = get_paths()
    candidates = [paths.project_root / "provenance/acceptance/S01" / f"{increment}-{evidence.name}" / "acceptance.json",
                  paths.campaign_root / "acceptance/S01" / f"{increment}-{evidence.name}" / "acceptance.json"]
    for ancestor in evidence.parents:
        candidates.append(ancestor / "acceptance/S01" / f"{increment}-{evidence.name}" / "acceptance.json")
    for candidate in candidates:
        if candidate.exists():
            record = json.loads(candidate.read_text())
            if record.get("increment") != f"S01-{increment}" or record.get("status") != "PASS_INCREMENT":
                raise ValueError(f"Upstream increment is not accepted: {candidate}")
            if Path(record["evidence"]).name != evidence.name:
                raise ValueError("Acceptance does not identify the requested evidence")
            return candidate, record
    raise FileNotFoundError(f"Cannot locate accepted {increment} for {evidence}")


def legacy_campaign_root():
    root = get_paths().frozen_source_root
    for candidate in (root / "experiment_campaign", root.parent / "experiment_campaign"):
        if candidate.is_dir():
            return candidate
    return LEGACY_RELEASE / "experiment_campaign"


def source_provenance():
    return {"frozen_source_root": str(get_paths().frozen_source_root),
            "current_source_hashes": current_source_hashes(),
            "source_hash_policy": "source_hashes identify frozen S01 bytes; current_source_hashes identify this execution's migrated package and tools"}
