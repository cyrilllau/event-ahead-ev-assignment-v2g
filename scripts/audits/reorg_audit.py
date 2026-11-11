#!/usr/bin/env python3
"""Read-only, fail-closed migration audit for the accepted IEEE33 v54-Q1 S01.

No experiment package is imported and no acceptance/state is rewritten.
Old evidence references retain their original absolute paths and hash formats.
An optional output must be a new file outside the immutable legacy release.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import tomllib
import zipfile

import numpy as np


BASE_MANIFEST_SHA = "245a940345d6a01dc7eddaba3d0048ed88350eb637dd75a78b851354477cc461"
TRANSFER_SHA = "9d1d5fc1b12b4269c9f5427d44a2bffceda4411e3e036c808d4a42502e63e4b3"
ACCEPTANCE_SHA = "dc3328d2fdf3fede8813a36493850b574cc21506ef95200926728bf0d06651be"
PACK_MANIFEST_SHA = "dcb614a1a9e5607afc484d8860b3b4f20cfbcb6a2cecd901b87c46a9274674a6"
TARGETS = ("helene_2024", "ian_2022", "ida_2021", "idalia_2023", "laura_2020",
           "michael_2018", "milton_2024", "zeta_2020")
CAMPAIGN = "ieee33-v54-q1-full-experiments"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()


def joint_array_hash(array):
    # I08 JointContext's format: do not substitute the bootstrap format.
    a = np.ascontiguousarray(array)
    h = hashlib.sha256(str((a.shape, a.dtype.str)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def bootstrap_array_hash(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256(json.dumps({"shape": a.shape, "dtype": str(a.dtype)},
                                 sort_keys=True).encode())
    h.update(a.tobytes())
    return h.hexdigest()


class Audit:
    def __init__(self, legacy):
        self.legacy = legacy.resolve()
        self.hashes = {}
        self.checked = {}
        self.bytes_hashed = 0
        self.array_hash_checks = 0
        self.reference_checks = 0

    def old(self, path):
        path = Path(path)
        require(path.is_absolute(), f"Old reference must be absolute: {path}")
        resolved = path.resolve()
        require(resolved.is_relative_to(self.legacy), f"Old reference escaped frozen release: {path}")
        require(path.exists(), f"Missing old evidence: {path}")
        return resolved

    def sha(self, path):
        path = Path(path).resolve()
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
        if path in self.hashes:
            require(self.hashes[path][0] == signature, f"File changed during audit: {path}")
            return self.hashes[path][1]
        h = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                h.update(block)
        after = path.stat()
        require(signature == (after.st_size, after.st_mtime_ns, after.st_ino),
                f"File changed while hashing: {path}")
        self.hashes[path] = (signature, h.hexdigest())
        self.bytes_hashed += stat.st_size
        return h.hexdigest()

    def check(self, path, expected, size=None):
        if isinstance(expected, dict):
            size = expected.get("bytes", expected.get("size_bytes", size))
            expected = expected["sha256"]
        require(isinstance(expected, str) and len(expected) == 64, f"Invalid SHA256 for {path}")
        path = Path(path).resolve()
        require(self.sha(path) == expected, f"SHA256 mismatch: {path}")
        if size is not None:
            require(path.stat().st_size == int(size), f"Byte count mismatch: {path}")
        self.checked[str(path)] = expected
        return path

    def ref(self, value):
        path = self.old(value["path"])
        self.reference_checks += 1
        return self.check(path, value["sha256"], value.get("bytes"))

    def refs(self, value):
        """Validate explicit path/hash references, without rewriting any path."""
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                self.ref(value)
            for key, item in value.items():
                if (isinstance(key, str) and key.startswith("/") and
                    isinstance(item, str) and len(item) == 64):
                    self.check(self.old(key), item)
                self.refs(item)
        elif isinstance(value, list):
            for item in value:
                self.refs(item)

    def relative_manifest(self, root, manifest):
        root = Path(root).resolve()
        for name, expected in read_json(manifest).items():
            p = (root / name).resolve()
            require(not Path(name).is_absolute() and p.is_relative_to(root),
                    f"Relative artifact escaped its evidence root: {name}")
            if isinstance(expected, dict) and "path" in expected:
                require(self.ref(expected) == p, f"Manifest points to wrong artifact: {name}")
            self.check(p, expected)


def baseline(audit, root):
    manifest = audit.check(root / "MANIFEST_SHA256.csv", BASE_MANIFEST_SHA)
    with manifest.open(newline="", encoding="utf-8") as source:
        records = list(csv.DictReader(source))
    require(len(records) == 74 and len({r["relative_path"] for r in records}) == 74,
            "Expected 74 unique frozen baseline entries")
    for row in records:
        p = (root / row["relative_path"]).resolve()
        require(p.is_relative_to(root.resolve()), "Baseline manifest escaped dataset")
        audit.check(p, row["sha256"], row["size_bytes"])
    return 74


def audit_run(args):
    start = time.monotonic()
    legacy = args.legacy_root.resolve()
    a = Audit(legacy)
    old_case = legacy / "IEEE33_paper_aligned_case_data_v1"
    old_campaign = legacy / "experiment_campaign"
    baseline(a, old_case)
    a.check(legacy / "TRANSFER_MANIFEST_SHA256.csv", TRANSFER_SHA)
    new_data = args.data_root.resolve() if args.data_root else (
        args.new_root.resolve() / "data/cases/ieee33-v54-q1" if args.new_root else old_case)
    if new_data != old_case:
        baseline(a, new_data)

    acceptance_path = old_campaign / "acceptance/S01/20260905T163804Z/acceptance.json"
    a.check(acceptance_path, ACCEPTANCE_SHA)
    acceptance = read_json(acceptance_path)
    require(acceptance["status"] == "PASS" and acceptance["targets"] == 8 and
            acceptance["scenarios"] == 1920 and acceptance["npz_partitions"] == 16,
            "Unexpected accepted S01 scope")
    a.refs(acceptance)
    pack = a.old(acceptance["deliverables"])
    require(pack == old_campaign / "deliverables/S01/20260905T163046Z", "Wrong S01 pack")
    a.check(a.old(acceptance["deliverables_manifest"]), PACK_MANIFEST_SHA)
    require(acceptance["deliverables_manifest_sha256"] == PACK_MANIFEST_SHA, "Wrong pack anchor")
    a.relative_manifest(pack, pack / "artifact_manifest.json")
    a.check(a.old(acceptance["independent_delivery_audit_original"]),
            acceptance["independent_delivery_audit_sha256"])
    a.check(acceptance_path.parent / "scientific_review.md", acceptance["scientific_review_sha256"])
    history = acceptance["historical_input_evidence"]
    a.check(a.old(history["S00_acceptance"]), history["S00_acceptance_sha256"])
    for name, sha in acceptance["source_implementation_snapshot"].items():
        a.check(old_case / "15_experiment_runner" / name, sha)

    contract = read_json(pack / "generator_contract.json")
    generation = read_json(pack / "generation_audit.json")
    index = read_json(pack / "task_scenario_replay_index.json")
    for payload in (contract, generation, index):
        a.refs(payload)
    require(contract["target_order"] == list(TARGETS), "Target order changed")
    require(contract["development_only"] and contract["formal_OOS_locked"] and
            not contract["formal_outcomes_generated"], "Development/formal scope changed")
    require(len(generation["gates"]) == 6, "Missing S01 gate evidence")
    require(len(contract["accepted_increment_ancestry"]) == 8, "Missing increment ancestry")
    require(a.ref(generation["state_snapshot"]) == pack / "state_snapshot.json",
            "Pack state snapshot is not immutable pack file")
    cards = {}
    manifest_entries = 0
    for row in contract["accepted_increment_ancestry"]:
        increment = row["increment"]
        card_path = a.ref(row["acceptance"])
        card = read_json(card_path)
        require(card["increment"] == increment and card["status"] == "PASS_INCREMENT",
                f"Increment not accepted: {increment}")
        evidence = a.old(card["evidence"])
        require(evidence == a.old(row["evidence_directory"]), "Evidence ancestry changed")
        a.refs(card)
        for manifest_ref in row["evidence_manifests"]:
            manifest = a.ref(manifest_ref)
            require(manifest.parent == evidence, "Manifest belongs to another increment")
            a.relative_manifest(evidence, manifest)
            manifest_entries += len(read_json(manifest))
        for name, expected in card.get("raw_artifacts", {}).items():
            a.check(evidence / name, expected)
        for name, expected in card.get("source_hashes", {}).items():
            a.check(old_case / "15_experiment_runner" / name, expected)
        if "review_sha256" in card:
            a.check(a.old(card["review_file"]), card["review_sha256"])
        for key in ("independent_audit", "review_file"):
            if key in card:
                a.old(card[key])
        cards[increment] = (card, evidence)
    require(set(cards) == {f"S01-I{i:02d}" for i in range(1, 9)}, "Wrong increment IDs")
    for name, digest in contract["accepted_source_hashes"].items():
        a.check(old_case / "15_experiment_runner" / name, digest)
    i08, source = cards["S01-I08"]
    upstream = read_json(source / "upstream_validation.json")
    inputs = upstream["input_hashes"]
    require(len(inputs) == 29 and inputs == contract["input_hashes"], "Input identities changed")
    for name, digest in inputs.items():
        a.check(old_case / name, digest)
        a.check(new_data / name, digest)
    for path_key in ("last_acceptance", "route_acceptance"):
        a.check(a.old(upstream[path_key]), upstream[path_key + "_sha256"])
    for key, digest in i08["independent_audits"].items():
        payload = read_json(a.check(a.old(key), digest))
        require(str(payload.get("status", "")).startswith("PASS"), f"Audit is not PASS: {key}")

    tasks = {}
    task_total = 0
    for row in contract["frozen_task_sources"]:
        event = row["target_event"]
        require(event not in tasks and event in TARGETS, "Duplicate/unknown frozen H")
        bank = read_json(a.ref(row["frozen_tasks"]))
        ti = read_json(a.ref(row["task_index"]))
        require(bank["certified"] and bank["status"] == "COMPLETE", "H not certified")
        task_hash = canonical_hash({"dependency_sha256": bank["dependency_sha256"],
                                   "config": bank["config"], "certified": True, "tasks": bank["tasks"]})
        require(task_hash == bank["task_sha256"] == ti["task_sha256"] == row["task_sha256"],
                f"Frozen H semantic hash mismatch: {event}")
        require(ti["target_event"] == event and
                ti["task_ids"] == [t["task_id"] for t in bank["tasks"]], "H IDs/order changed")
        tasks[event] = ti
        task_total += len(ti["task_ids"])
    require(set(tasks) == set(TARGETS) and task_total == 18720, "Wrong complete H bank")

    require(len(index["archives"]) == 16, "Expected 16 NPZ partitions")
    seen = set()
    scenario_total = 0
    task_scenario_total = 0
    npz_member_count = 0
    npz_uncompressed_bytes = 0
    for row in index["archives"]:
        event, split = row["target_event"], row["split"]
        key = (event, split)
        require(key not in seen and event in TARGETS and split in ("planning", "calibration"),
                "Duplicate/unexpected NPZ partition")
        seen.add(key)
        n = {"planning": 160, "calibration": 80}[split]
        ti = tasks[event]
        h = len(ti["task_ids"])
        require(row["scenario_count"] == n and row["task_count"] == h and
                row["task_sha256"] == ti["task_sha256"] and row["development"] and row["regime"] == "R1",
                "Partition scope/schema mismatch")
        npz = a.ref(row["file"])
        require(npz == source / f"{event}.{split}.npz", "NPZ source path mismatch")
        ledger = read_json(a.ref(row["scenario_ledger"]))["scenarios"]
        require(len(ledger) == n and [r["scenario_id"] for r in ledger] == row["scenario_ids"],
                "Scenario index and source ledger disagree")
        for item in ledger:
            require(item["target_event"] == event and item["split"] == split and
                    item["task_sha256"] == ti["task_sha256"], "Scenario belongs to another H/target")
            require(canonical_hash({k: v for k, v in item.items() if k != "scenario_sha256"}) ==
                    item["scenario_sha256"], "Scenario provenance hash mismatch")
        with zipfile.ZipFile(npz) as zf:
            require(len(zf.namelist()) == len(set(zf.namelist())) and
                    set(zf.namelist()) == {name + ".npy" for name in row["array_schema"]},
                    "NPZ members changed")
            for name, schema in row["array_schema"].items():
                with zf.open(name + ".npy") as member:
                    version = np.lib.format.read_magic(member)
                    require(version in ((1, 0), (2, 0)), f"Unexpected NPY header version: {version}")
                    read_header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                                   else np.lib.format.read_array_header_2_0)
                    shape, fortran, dtype = read_header(member)
                require(list(shape) == schema["shape"] and dtype.str == schema["dtype"] and
                        fortran == schema["fortran_order"], "NPY header differs from immutable index")
            npz_uncompressed_bytes += sum(i.file_size for i in zf.infolist())
        with np.load(npz, allow_pickle=False) as arrays:
            for name, schema in row["array_schema"].items():
                data = arrays[name]  # Read every member; enforces full ZIP CRC and NPY decoding.
                npz_member_count += 1
                require(list(data.shape) == schema["shape"] and data.dtype.str == schema["dtype"],
                        f"NPZ shape/dtype mismatch: {event}/{split}/{name}")
                if name == "scenario_ids":
                    require(data.tolist() == row["scenario_ids"], "NPZ scenario IDs differ")
                elif name == "task_ids":
                    require(data.tolist() == ti["task_ids"], "NPZ H IDs differ")
                elif name == "times_h":
                    require(np.array_equal(data, np.arange(-1., 24., .25)), "NPZ time axis differs")
                else:
                    for j, item in enumerate(ledger):
                        require(joint_array_hash(data[j]) == item["array_hashes"][name],
                                f"Array identity mismatch: {event}/{split}/{j}/{name}")
                        a.array_hash_checks += 1
                del data
        scenario_total += n
        task_scenario_total += n * h
    require(scenario_total == 1920 and task_scenario_total == 4492800, "Incomplete replay bank")
    for row in contract["bootstrap_identities"]:
        event = row["target_event"]
        meta = read_json(source / f"{event}.bootstrap.json")
        with np.load(source / f"{event}.bootstrap.npz", allow_pickle=False) as bank:
            require(bootstrap_array_hash(bank["indices"]) == meta["index_sha256"] == row["index_sha256"],
                    "Bootstrap identity mismatch (distinct archived hash format)")
        require(meta["stream_identity"] == row["stream_identity"], "Bootstrap stream changed")
    static = a.ref(index["static_axes"])
    with np.load(static, allow_pickle=False) as archive:
        for name in archive.files:
            _ = archive[name]

    state_check = {"status": "NOT_REQUESTED"}
    if args.new_campaign:
        campaign = args.new_campaign.resolve()
        require(campaign.is_dir() and not campaign.is_relative_to(legacy),
                "New campaign must exist outside frozen release")
        state_path = args.state.resolve() if args.state else campaign / "state.json"
        require(state_path.is_relative_to(campaign), "Authoritative state escaped new campaign")
        state = read_json(state_path)
        require(state["campaign"] == CAMPAIGN, "Wrong campaign identity")
        require(Path(state["server_campaign_root"]).resolve() == campaign, "State points to wrong campaign")
        require(Path(state["server_case_root"]).resolve() == new_data.resolve(), "State points to wrong dataset")
        require(state["specs"]["S01"]["status"] == "PASS" and
                Path(state["specs"]["S01"]["acceptance"]).resolve() == acceptance_path,
                "New state does not inherit the verified S01 acceptance")
        require(not state.get("active_jobs"), "Migration audit requires no active jobs")
        require(args.new_root is not None, "New campaign audit requires --new-root")
        project = args.new_root.resolve()
        require(Path(state["server_project_root"]).resolve() == project and
                Path(state["server_spec_root"]).resolve() == project / "docs/experiments" and
                Path(state["frozen_source_root"]).resolve() == old_case,
                "New state source/spec roots differ from migration request")
        for field, expected_path in (("active_source_manifest", project / "RELEASE_MANIFEST.json"),
                                     ("active_config", project / "configs/powerai.toml")):
            ref = state[field]
            require(Path(ref["path"]).resolve() == expected_path, f"Wrong {field} location")
            a.check(expected_path, ref["sha256"])
        manifest = read_json(project / "RELEASE_MANIFEST.json")
        source_names = [row["path"] for row in manifest["files"]]
        require(len(source_names) == len(set(source_names)), "Duplicate active source manifest entries")
        for row in manifest["files"]:
            path = (project / row["path"]).resolve()
            require(not Path(row["path"]).is_absolute() and path.is_relative_to(project),
                    "Active source manifest escaped project")
            a.check(path, row["sha256"], row["size_bytes"])
        config_paths = tomllib.loads((project / "configs/powerai.toml").read_text())["paths"]
        for key, expected in {"data_root": new_data.resolve(), "spec_root": project / "docs/experiments",
                              "campaign_root": campaign, "frozen_source_root": old_case}.items():
            configured = Path(config_paths[key]).expanduser()
            if not configured.is_absolute():
                configured = project / configured
            require(configured.resolve() == expected, f"Active config has wrong {key}")
        for name, expected_hash in state["legacy_control_sha256"].items():
            require(name in ("state.json", "NEXT_ACTION.md", "decisions.jsonl"), "Unexpected legacy control key")
            a.check(campaign / "migration" / ("legacy_" + name), expected_hash)
            a.check(old_case / "14_experiment_specs" / name, expected_hash)
        state_check = {"status": "PASS", "path": str(state_path), "sha256": a.sha(state_path),
                       "current_spec": state.get("current_spec"), "active_source_files": len(source_names),
                       "active_config_sha256": state["active_config"]["sha256"]}
    # Detect changes to previously verified source/evidence during the audit window.
    for path in list(a.hashes):
        a.sha(path)
    return {"status": "PASS_MIGRATION_EVIDENCE_AUDIT", "legacy_root": str(legacy),
            "new_root": str(args.new_root.resolve()) if args.new_root else None,
            "data_root": str(new_data), "new_campaign": str(args.new_campaign.resolve()) if args.new_campaign else None,
            "baseline_files_per_copy": 74, "frozen_input_hashes": 29,
            "accepted_increments": 8, "increment_manifest_entries": manifest_entries,
            "targets": 8, "tasks": task_total, "npz_partitions": 16,
            "scenarios": scenario_total, "task_scenario_rows": task_scenario_total,
            "npz_members_read": npz_member_count, "npz_uncompressed_bytes_read": npz_uncompressed_bytes,
            "joint_array_hash_checks": a.array_hash_checks, "bootstrap_index_hash_checks": 8,
            "reference_checks": a.reference_checks, "unique_verified_files": len(a.checked),
            "bytes_hashed": a.bytes_hashed, "acceptance_sha256": ACCEPTANCE_SHA,
            "pack_manifest_sha256": PACK_MANIFEST_SHA, "state": state_check,
            "scope": "Relocation integrity and readable unchanged evidence; no experiment rerun or scientific re-acceptance",
            "files_or_states_modified": False, "seconds": time.monotonic() - start}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--new-root", type=Path)
    parser.add_argument("--data-root", type=Path, help="Override new-root/data/cases/ieee33-v54-q1")
    parser.add_argument("--new-campaign", type=Path)
    parser.add_argument("--state", type=Path, help="Authoritative new campaign state; defaults to new-campaign/state.json")
    parser.add_argument("--output", type=Path, help="Optional NEW audit JSON outside the legacy release")
    args = parser.parse_args()
    try:
        require(args.state is None or args.new_campaign is not None,
                "--state requires --new-campaign")
        if args.output:
            require(not args.output.resolve().is_relative_to(args.legacy_root.resolve()),
                    "Audit output cannot modify legacy release")
            require(not args.output.exists(), "Audit output already exists; will not overwrite")
        result = audit_run(args)
        code = 0
    except Exception as exc:
        result = {"status": "FAIL_MIGRATION_EVIDENCE_AUDIT", "error_type": type(exc).__name__, "error": str(exc)}
        code = 1
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    print(text, end="")
    if args.output and not args.output.exists() and not args.output.resolve().is_relative_to(args.legacy_root.resolve()):
        with args.output.open("x", encoding="utf-8") as target:
            target.write(text)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
