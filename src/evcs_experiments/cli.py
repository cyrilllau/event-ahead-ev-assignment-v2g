"""Thin command dispatch; no solver, simulation, or scientific formulas here."""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
from pathlib import Path
import runpy
import sys

from .config import get_paths

VALIDATIONS = {name: "validate_s01_" + name for name in
               ("inputs", "wind", "lines", "roads", "load", "routes", "replay", "joint")}
VALIDATIONS["pack"] = "export_s01_pack"
VALIDATIONS["first-stage"] = "validate_s02_first_stage"
VALIDATIONS["recourse"] = "validate_s02_recourse"
VALIDATIONS["physical-pack"] = "validate_s02_completion"
VALIDATIONS["probability"] = "validate_s03_probability"
VALIDATIONS["extensive"] = "validate_s03_extensive"
VALIDATIONS["ccg"] = "validate_s03_ccg"
VALIDATIONS["scale-profile"] = "profile_s03_scale"
VALIDATIONS["scale-ccg"] = "validate_s03_scale_ccg"
VALIDATIONS["evaluation-foundation"] = "validate_s04_foundation"
VALIDATIONS["evaluation-integration"] = "validate_s04_integration"
AUDITS = {"s01-" + name: "audit_s01_" + name.replace("-", "_") + ".py"
          for name in ("i03", "i04", "i05", "i06", "i07", "i08-chain", "i08-features", "i08-moments", "pack")}
AUDITS.update({"reorg": "reorg_audit.py", "migration": "reorg_audit.py"})
AUDITS["s02-i01"] = "audit_s02_i01.py"
AUDITS["s02-i02"] = "audit_s02_i02.py"
AUDITS["s02-i03"] = "audit_s02_i03.py"
AUDITS["s03-i01"] = "audit_s03_i01.py"
AUDITS["s03-i02"] = "audit_s03_i02.py"
AUDITS["s03-i03"] = "audit_s03_i03.py"


def validate_manifest(arguments):
    parser = argparse.ArgumentParser(prog="evcs validate manifest")
    parser.add_argument("--case-root", type=Path, default=get_paths().data_root)
    args = parser.parse_args(arguments)
    from .evidence import sha256
    root = args.case_root.resolve()
    manifest = root / "MANIFEST_SHA256.csv"
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Frozen data manifest is empty")
    seen = set()
    for row in rows:
        name = row.get("path", row.get("relative_path"))
        if not name or name in seen or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Invalid or duplicate frozen manifest path")
        seen.add(name)
        path = root / name
        expected = row.get("sha256", row.get("SHA256"))
        if sha256(path) != expected:
            raise ValueError(f"Frozen data hash mismatch: {name}")
        byte_count = row.get("size_bytes", row.get("bytes"))
        if byte_count is not None and path.stat().st_size != int(byte_count):
            raise ValueError(f"Frozen data size mismatch: {name}")
    print(json.dumps({"status": "PASS_FROZEN_DATA_MANIFEST", "files_checked": len(rows),
                      "data_root": str(root), "manifest_sha256": sha256(manifest)}))
    return 0


def _call(module, arguments):
    previous = sys.argv
    try:
        sys.argv = [module.__name__, *arguments]
        return module.main() or 0
    finally:
        sys.argv = previous


def main(argv=None):
    parser = argparse.ArgumentParser(prog="evcs", description=__doc__)
    parser.add_argument("--config", help="TOML/JSON configuration; defaults to EVCS_CONFIG or configs/local.toml")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Run a named explicit validation")
    validate.add_argument("target", choices=[*VALIDATIONS, "manifest"])
    validate.add_argument("arguments", nargs=argparse.REMAINDER)
    audit = commands.add_parser("audit", help="Run an independent audit with explicit inputs")
    audit.add_argument("target", choices=AUDITS)
    audit.add_argument("arguments", nargs=argparse.REMAINDER)
    job = commands.add_parser("job", help="Delegate to the campaign-owned job controller")
    job.add_argument("arguments", nargs=argparse.REMAINDER)
    commands.add_parser("status", help="Read configured campaign state and job identities")
    args = parser.parse_args(argv)
    if args.config is not None:
        # Resolve once against the project root. Children inherit the same
        # explicit configuration even when their working directory differs.
        selected = Path(args.config).expanduser()
        if not selected.is_absolute():
            selected = get_paths(config={}).project_root / selected
        os.environ["EVCS_CONFIG"] = str(selected.resolve())
    paths = get_paths()
    try:
        if args.command == "validate":
            if args.target == "manifest":
                return validate_manifest(args.arguments)
            from .evidence import verify_scientific_kernel_identity
            verify_scientific_kernel_identity()
            arguments = list(args.arguments)
            if "--case-root" not in arguments and "--help" not in arguments and "-h" not in arguments:
                arguments += ["--case-root", str(paths.data_root)]
            if args.target == "pack" and "--campaign-root" not in arguments and "--help" not in arguments:
                arguments += ["--campaign-root", str(paths.campaign_root)]
            module = importlib.import_module("evcs_experiments.validation." + VALIDATIONS[args.target])
            return _call(module, arguments)
        if args.command == "audit":
            script = paths.project_root / "scripts/audits" / AUDITS[args.target]
            previous = sys.argv
            try:
                sys.argv = [str(script), *args.arguments]
                runpy.run_path(str(script), run_name="__main__")
                return 0
            except SystemExit as result:
                return result.code or 0
            finally:
                sys.argv = previous
        from .runtime import jobctl
        if args.command == "job":
            return _call(jobctl, args.arguments)
        state_path = paths.campaign_root / "state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else None
        jobs_root = paths.campaign_root / "jobs"
        jobs = jobctl.status(jobs_root) if jobs_root.exists() else {"outroot": str(jobs_root), "jobs": []}
        print(json.dumps({"paths": paths.as_dict(), "state_path": str(state_path), "state": state, "jobs": jobs},
                         indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
