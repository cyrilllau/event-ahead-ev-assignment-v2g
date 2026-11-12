"""Local historical L08 parity; explicit manifests, bounded slices, no SSH/OOS.

--spec points to a private JSON with inputs/input_manifest/evidence_root/
evidence_manifest/evidence_manifest_sha and cases. Each case has method, point,
point_key, point_sha, records (DRO/ARO), or audit/names (DET). All paths in cases
must occur in the hashed evidence manifest. Outputs must be outside the checkout.
Resume with the exact same command/source; --reaudit uses saved LP witnesses only.
This checker neither creates planning runs nor changes their original budgets.
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ev_assignment.domain.planning import Control
from ev_assignment.validation.l08 import audit_plan
from ev_assignment.validation.parity import decode_first, interval_parity, saved_dispatch
from ev_assignment_experiments.datasets.l08 import L08Dataset, R41
from ev_assignment_experiments.workflows.probability import source_snapshot


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_new(path, value):
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("Immutable validation record changed: " + str(path))
    else:
        path.write_bytes(data)


class Evidence:
    def __init__(self, root, manifest, expected):
        self.root = Path(root).resolve()
        raw = Path(manifest).read_bytes()
        if sha(raw) != expected:
            raise ValueError("Historical evidence manifest changed")
        self.refs = json.loads(raw)

    def path(self, name):
        name = name.removeprefix("/home/Shixin/ev-assignment/")
        rel = Path(name)
        p = self.root / rel
        if rel.is_absolute() or ".." in rel.parts or not p.resolve().is_relative_to(self.root):
            raise ValueError("Evidence path escapes explicit root")
        ref = self.refs[name]
        raw = p.read_bytes()
        if len(raw) != ref["bytes"] or sha(raw) != ref["sha256"]:
            raise ValueError("Historical evidence changed: " + name)
        return p

    def read(self, name):
        return json.loads(self.path(name).read_text())

    def npz(self, name):
        with np.load(self.path(name), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}


def locked_records(records, point_sha):
    if len(records) != 400 or [r["sid"] for r in records] != list(range(400)):
        raise ValueError("Historical records must contain ordered complete support400")
    for row in records:
        if not row.get("universal_zero") and row.get("point_sha") != point_sha:
            raise ValueError("Mixed historical plan")
        interval_parity(row["qL"], row["qU_audited"], row["qL"], row["qU_audited"])
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=float, default=55)
    parser.add_argument("--reaudit", action="store_true")
    args = parser.parse_args()
    if not 0 < args.seconds <= 60:
        raise ValueError("Validation slices must be <=60 seconds")
    out = Path(args.output).resolve()
    # Find repository boundaries by Git markers, never infer an input root.
    for parent in (out, *out.parents):
        if (parent / ".git").exists():
            raise ValueError("Validation evidence must be outside Git")
    out.mkdir(parents=True, exist_ok=True)
    spec_bytes = Path(args.spec).read_bytes()
    spec = json.loads(spec_bytes)
    evidence = Evidence(
        spec["evidence_root"], spec["evidence_manifest"], spec["evidence_manifest_sha"]
    )
    source = source_snapshot()
    run = dict(
        scope="historical_fixed_plan_parity",
        spec_sha=sha(spec_bytes),
        source_sha=sha(source),
        checker_sha=sha(Path(__file__).read_bytes()),
        input_manifest=spec["input_manifest"],
        oos_opened=False,
        global_mip_optimized=False,
        native_LB_reproved=False,
    )
    write_new(out / "RUN.json", run)
    if not (out / "source.zip").exists():
        (out / "source.zip").write_bytes(source)
    elif (out / "source.zip").read_bytes() != source:
        raise ValueError("Stored executing source changed")
    d = L08Dataset(spec["inputs"], spec["input_manifest"])
    names = d.inputs.read(R41 + "/identity/FIRST_NAMES.json")
    control = Control(args.seconds)
    processed = 0
    summary = []
    for case in spec["cases"]:
        method = case["method"]
        problem = d.problem(method, include_ambiguity=False)
        original = evidence.npz(case["point"])[case["point_key"]]
        if sha(original.tobytes()) != case["point_sha"]:
            raise ValueError("Historical point bytes changed")
        named = None
        if method == "DET":
            audit = evidence.read(case["audit"])
            if audit["science"] != d.input_identity:
                raise ValueError("Historical DET science identity mismatch")
            old_names = evidence.read(case["names"])
            if len(old_names) != len(original) or len(set(old_names)) != len(old_names):
                raise ValueError("Historical variable names changed")
            named = dict(zip(old_names, original))
            entries = decode_first(problem, [named[n] for n in names])
            if entries != audit["physical"]["plans"]:
                raise ValueError("DET saved audit and actual point differ")
            records = [dict(sid="forecast44", **audit["physical"])]
        else:
            entries = decode_first(problem, original)
            payload = evidence.read(case["records"])
            records = locked_records(
                payload["records"] if isinstance(payload, dict) else payload, case["point_sha"]
            )
        directory = out / method
        directory.mkdir(exist_ok=True)
        for i, old in enumerate(records):
            path = directory / f"{i:04d}.json"
            sid = "forecast44" if method == "DET" else f"train/{i:04d}"
            witness_file = directory / f"{i:04d}.npz"
            if path.exists() and not args.reaudit:
                summary.append(json.loads(path.read_text()))
                continue
            readback = directory / f"{i:04d}.readback.json"
            if args.reaudit and readback.exists():
                checked = json.loads(readback.read_text())
                if checked["cell_sha"] != sha(path.read_bytes()) or checked["witness_sha"] != sha(
                    witness_file.read_bytes()
                ):
                    raise ValueError("Reaudited evidence changed")
                summary.append(json.loads(path.read_text()))
                continue
            if control.remaining < 2:
                print(json.dumps(dict(status="SLICE_COMPLETE", processed=processed)), flush=True)
                return
            started = time.monotonic()
            try:
                single = replace(problem, scenario_ids=(sid,))
                raw = None
                if method == "DET":
                    raw = saved_dispatch(single, entries, sid, named_values=named)
                elif old.get("universal_zero"):
                    # Old analytic zero is a claim; the new LP independently checks it.
                    proof = old["zero_proof"]
                    if sha(evidence.path(proof["path"]).read_bytes()) != proof["sha256"]:
                        raise ValueError("Historical zero proof changed")
                else:
                    z = evidence.npz(old["primal"])
                    if not np.array_equal(z["x"], original):
                        raise ValueError("Historical witness uses a different plan")
                    for ref in old["sources"]:
                        if sha(evidence.path(ref["path"]).read_bytes()) != ref["sha256"]:
                            raise ValueError("Historical record reference changed")
                    raw = saved_dispatch(single, entries, sid, compact_y=z["y"])
                provider = None
                previous = None
                if args.reaudit:
                    previous = json.loads(path.read_text())
                    if (
                        not previous["passed"]
                        or sha(witness_file.read_bytes()) != previous["witness_sha"]
                    ):
                        raise ValueError("Saved numerical witness changed or failed")
                    with np.load(witness_file, allow_pickle=False) as z:
                        witness = {k: z[k] for k in z.files}

                    def provider(_):
                        return dict(previous["new"], **witness)

                result = audit_plan(
                    single, entries, control, retain_witness=True, witness_provider=provider
                ).checks
                row = result["scenarios"][0]
                keys = ("y", "z", "lambda_ub", "mu_eq", "lower", "upper")
                w = {k: np.asarray(row.pop(k)) for k in keys}
                if not args.reaudit:
                    np.savez_compressed(witness_file, **w)
                if method == "DET":
                    # Original audit is an incumbent, not a fixed-LP dual certificate.
                    difference = result["first_cost"] + row["primal"] - old["objective"]
                    parity = dict(
                        passed=abs(difference) <= 1e-7,
                        total_objective_delta=difference,
                        old_evidence="primal_incumbent_only",
                    )
                    if abs(raw["first_cost"] + raw["primal"] - old["objective"]) > 1e-7:
                        raise ValueError("DET old witness objective mismatch")
                else:
                    parity = interval_parity(old["qL"], old["qU_audited"], row["qL"], row["qU"])
                    if raw is not None and abs(raw["primal"] - old["qU_audited"]) > 1e-7:
                        raise ValueError("Historical witness and stored objective mismatch")
                if not parity["passed"]:
                    raise ValueError("Fixed-plan objective differs: " + json.dumps(parity))
                record = dict(
                    passed=True,
                    method=method,
                    sid=sid,
                    first_cost=result["first_cost"],
                    new=row,
                    old_dispatch=raw,
                    parity=parity,
                    witness_sha=sha(witness_file.read_bytes()),
                    elapsed_seconds=time.monotonic() - started,
                )
                if args.reaudit:
                    if abs(previous["new"]["primal"] - row["primal"]) > 1e-8:
                        raise ValueError("Independent readback changed objective")
                    readback = directory / f"{i:04d}.readback.json"
                    write_new(
                        readback,
                        dict(
                            passed=True,
                            cell_sha=sha(path.read_bytes()),
                            witness_sha=record["witness_sha"],
                            solver_calls=0,
                        ),
                    )
                else:
                    write_new(path, record)
                summary.append(record)
                processed += 1
            except Exception as exc:
                failure = directory / f"{i:04d}.failure-{time.time_ns()}.json"
                write_new(
                    failure,
                    dict(
                        passed=False,
                        method=method,
                        sid=sid,
                        exception=repr(exc),
                        elapsed_seconds=time.monotonic() - started,
                    ),
                )
                raise
    print(
        json.dumps(
            dict(
                status="COMPLETE",
                cells=len(summary),
                reaudited=args.reaudit,
                passed=all(r["passed"] for r in summary),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
