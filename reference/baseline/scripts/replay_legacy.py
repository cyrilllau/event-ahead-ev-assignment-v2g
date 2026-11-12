#!/usr/bin/env python3
"""Copy the archived case layout into an explicit scratch directory, then replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from evcs_experiments.config import get_paths
from evcs_experiments.evidence import frozen_control_root, frozen_runner_root


CASE_DIRECTORY = "IEEE33_paper_aligned_case_data_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(destination: Path, paths) -> tuple[Path, dict]:
    """Create independent copies; refuse any destination inside the checkout."""
    destination = destination.expanduser().resolve()
    if destination == paths.project_root or paths.project_root in destination.parents:
        raise ValueError("The destination must be outside the project checkout")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("The destination must be a new or empty directory")
    sources = (
        (paths.data_root, Path(".")),
        (paths.project_root / "archive/11_readiness_audit", Path("11_readiness_audit")),
        (paths.project_root / "archive/12_reactive_support_correction", Path("12_reactive_support_correction")),
        (frozen_runner_root(), Path("15_experiment_runner")),
        (frozen_control_root(), Path("14_experiment_specs")),
    )
    records = []
    for source, relative_destination in sources:
        source = source.resolve()
        if not source.is_dir():
            raise ValueError(f"Missing legacy source directory: {source}")
        if destination == source or source in destination.parents or destination in source.parents:
            raise ValueError("The scratch destination must not overlap a source directory")
        for item in sorted(source.rglob("*")):
            if item.is_symlink():
                raise ValueError(f"Legacy input symlinks are unsupported: {item}")
            if item.is_file() and "__pycache__" not in item.parts and item.suffix != ".pyc":
                records.append({"source": str(item),
                    "case_relative_path": (relative_destination / item.relative_to(source)).as_posix(),
                    "size_bytes": item.stat().st_size, "sha256": sha256(item)})
    names = [record["case_relative_path"] for record in records]
    if len(names) != len(set(names)):
        raise ValueError("Source trees contain overlapping reconstructed paths")
    case_root = destination / CASE_DIRECTORY
    case_root.mkdir(parents=True)
    for record in records:
        target = case_root / record["case_relative_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record["source"], target)
        if sha256(target) != record["sha256"]:
            raise RuntimeError(f"Copy verification failed: {target}")
    manifest = {"schema_version": 1, "purpose": "legacy compatibility replay in independent copies",
        "project_root": str(paths.project_root), "case_root": str(case_root),
        "specification_source": "immutable legacy control files; historical contract bytes retained",
        "files": records, "copy_sha256_verified": True, "execution": None}
    verify_sources(manifest)
    write_manifest(destination, manifest)
    return case_root, manifest


def verify_sources(manifest: dict) -> bool:
    changed = [record["source"] for record in manifest["files"]
               if not Path(record["source"]).is_file()
               or sha256(Path(record["source"])) != record["sha256"]]
    manifest["source_files_unchanged"] = not changed
    manifest["changed_source_files"] = changed
    return not changed


def write_manifest(destination: Path, manifest: dict) -> None:
    (destination / "legacy_replay_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Project TOML/JSON configuration")
    parser.add_argument("--destination", required=True, type=Path,
                        help="New or empty scratch directory outside the checkout")
    parser.add_argument("--run", metavar="CASE_RELATIVE_SCRIPT",
                        help="Optional archived Python script within the copied case")
    parser.add_argument("--timeout-seconds", type=float, default=600,
                        help="Explicit replay time limit (default: 600 seconds)")
    parser.add_argument("script_args", nargs=argparse.REMAINDER,
                        help="Arguments passed literally after --")
    args = parser.parse_args(argv)
    if not 0 < args.timeout_seconds < float("inf"):
        parser.error("--timeout-seconds must be finite and positive")
    script_relative = None
    if args.run:
        script_relative = Path(args.run)
        if script_relative.is_absolute() or ".." in script_relative.parts or script_relative.suffix != ".py":
            parser.error("--run must be a case-relative Python path without '..'")
    elif args.script_args:
        parser.error("Script arguments require --run")
    if args.config is not None:
        selected = Path(args.config).expanduser()
        if not selected.is_absolute():
            selected = get_paths(config={}).project_root / selected
        os.environ["EVCS_CONFIG"] = str(selected.resolve())
    try:
        case_root, manifest = prepare(args.destination, get_paths())
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"{error}\n")
    destination = case_root.parent
    returncode = 0
    if script_relative is not None:
        script = case_root / script_relative
        if not script.is_file():
            parser.exit(2, f"Reconstructed script does not exist: {script}\n")
        script_args = args.script_args[1:] if args.script_args[:1] == ["--"] else args.script_args
        command = [sys.executable, str(script), *script_args]
        started = time.monotonic()
        execution = {"argv": command, "cwd": str(case_root), "timeout_seconds": args.timeout_seconds,
                     "stdout": str(destination / "stdout.log"), "stderr": str(destination / "stderr.log")}
        manifest["execution"] = execution
        write_manifest(destination, manifest)
        with (destination / "stdout.log").open("wb") as stdout, (destination / "stderr.log").open("wb") as stderr:
            try:
                result = subprocess.run(command, cwd=case_root, stdout=stdout, stderr=stderr,
                                        timeout=args.timeout_seconds, check=False)
                returncode = result.returncode
                execution["state"] = "SUCCEEDED" if returncode == 0 else "FAILED"
            except subprocess.TimeoutExpired:
                returncode = 124
                execution["state"] = "TIMED_OUT"
            except OSError as error:
                returncode = 2
                execution.update(state="LAUNCH_FAILED", error=str(error))
        execution.update(returncode=returncode, elapsed_seconds=time.monotonic() - started)
    unchanged = verify_sources(manifest)
    write_manifest(destination, manifest)
    print(json.dumps({"case_root": str(case_root),
        "manifest": str(destination / "legacy_replay_manifest.json"),
        "source_files_unchanged": unchanged, "returncode": returncode}, ensure_ascii=False))
    return returncode if unchanged else 3


if __name__ == "__main__":
    raise SystemExit(main())
