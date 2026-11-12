"""Final charged numerical/identity review and preregistered OOS statistics."""
from pathlib import Path
import hashlib
import json
import sys
import time
import traceback

from ev_assignment.domain.planning import Control
from ev_assignment_experiments.analysis.diagnostic import reference
from round2_worker import dump


def read(path):
    return json.loads(Path(path).read_text())


def compare(a, b, relation):
    aa, bb = a.get("optimization", {}), b.get("optimization", {})
    if any(v is None for v in [aa.get("LB"), aa.get("UB"), bb.get("LB"), bb.get("UB")]):
        return dict(status="INCOMPLETE_BOUNDS", relation=relation)
    if relation == "equal":
        contradicted = max(aa["LB"], bb["LB"]) > min(aa["UB"], bb["UB"]) + 1e-5
    else:
        contradicted = aa["LB"] > bb["UB"] + 1e-5
    return dict(status="CONTRADICTORY_INTERVALS" if contradicted else "COMPATIBLE_CERTIFIED_INTERVALS",
                relation=relation, a=[aa["LB"], aa["UB"]], b=[bb["LB"], bb["UB"]],
                does_not_claim_equal_incumbents=True)


def main():
    out, seconds, root = Path(sys.argv[1]), float(sys.argv[2]), Path(sys.argv[3])
    cc, begun = Control(max(.01, seconds - 5)), time.monotonic()
    result = dict(status="FINAL_EVIDENCE_REVIEW_INCOMPLETE", passed=False, comparisons=[], runs=[])
    try:
        freeze = read(root / "FREEZE.json")
        runs = {}
        for job in freeze["jobs"]:
            cc.check()
            if job["stage"] == "VREPORT":
                continue
            path = root / job["stage"] / job["id"] / "RESULT.json"
            r = read(path) if path.exists() else {}
            runs[job["id"]] = r
            s = r.get("optimization", {})
            record = dict(id=job["id"], stage=job["stage"], status=r.get("status", "NO_RESULT"),
                          passed=r.get("passed", False), result=reference(path) if path.exists() else None)
            if s.get("passed"):
                gap = s["UB"] - s["LB"]
                audits = s.get("original_audit", [])
                matching = [a["sid"] for a in audits] == s["ordered_support"]
                maxres = max(float(a["residual"]) for a in audits)
                if (gap < 0 or gap > max(1., .01 * max(1., abs(s["UB"]))) or not matching
                        or maxres > 1e-9 or not s.get("original_expansion_audited")):
                    raise ValueError(f"Stored acceptance lacks complete matching original evidence: {job['id']}")
                record.update(checked_gap=gap, checked_original_max_residual=maxres, audited_support=len(audits))
            result["runs"].append(record)
        for seed in freeze["validation_seeds"]:
            names = freeze["core_run_ids"][str(seed)]
            p3 = runs[names["P3"]]
            for other in ["P2", "P3-noadvance"]:
                result["comparisons"].append(dict(seed=seed, methods=["P3", other], **compare(p3, runs[names[other]], "less_equal")))
            baseline = runs[freeze["individual_run_ids"][str(seed)]]
            a, b = p3.get("optimization", {}), baseline.get("optimization", {})
            if a and b:
                if (a["science_identity"] != b["science_identity"] or a["ambiguity_identity"] != b["ambiguity_identity"]
                        or a["first_stage_weights"] != b["first_stage_weights"]
                        or a["first_stage_restriction"] != b["first_stage_restriction"]):
                    raise ValueError("Count/individual algorithm comparator is a different scientific problem")
                row = dict(seed=seed, methods=["P3-count", "P3-individual"], **compare(p3, baseline, "equal"),
                           both_acceptance_passed=bool(a.get("passed") and b.get("passed")))
                row["observed_time_ratio_individual_over_count"] = b["elapsed_seconds"] / a["elapsed_seconds"] if row["both_acceptance_passed"] else None
                result["comparisons"].append(row)
        if any(c["status"] == "CONTRADICTORY_INTERVALS" for c in result["comparisons"]):
            raise ValueError("Same-problem/equivalent-domain bound contradiction requires mathematical review")
        from ev_assignment_experiments.analysis.structured_statistics import summarize_paired
        result["oos_statistics"] = {}
        for seed in freeze["validation_seeds"]:
            oos = runs[freeze["oos_run_ids"][str(seed)]]
            stats = summarize_paired(oos.get("records", []), control=cc, seed=910000 + seed,
                                     repetitions=500)
            dump(out / f"OOS_STATS_{seed}.json", stats)
            result["oos_statistics"][str(seed)] = reference(out / f"OOS_STATS_{seed}.json")
        hashes = {}
        mutable_coordination = {"CASE_DESIGN_STATUS.md", "CASE_SEARCH_LEDGER.jsonl", "CAMPAIGN_PROGRESS.jsonl", "CAMPAIGN.log"}
        for path in sorted(root.rglob("*")):
            cc.check()
            if not path.is_file() or out in path.parents or (path.parent == root and path.name in mutable_coordination):
                continue
            h = hashlib.sha256()
            with path.open("rb") as f:
                while chunk := f.read(8 * 1024 * 1024):
                    cc.check()
                    h.update(chunk)
            hashes[str(path.relative_to(root))] = dict(bytes=path.stat().st_size, sha256=h.hexdigest())
        dump(out / "FROZEN_EVIDENCE_HASHES.json", hashes)
        result.update(status="EVIDENCE_RECONCILED_WITH_INCOMPLETE_RUNS_RETAINED", passed=True,
                      hash_count=len(hashes), all_runs_passed=all(r["passed"] for r in result["runs"]),
                      hash_exclusions=dict(current_report_run=True, mutable_root_coordination=sorted(mutable_coordination)))
    except Exception as exc:
        result.update(status="FINAL_REVIEW_BUDGET_INCOMPLETE" if isinstance(exc, (TimeoutError, RuntimeError)) else
                      "IMPLEMENTATION_OR_IDENTITY_ERROR", error=f"{type(exc).__name__}: {exc}")
        (out / "ERROR.txt").write_text(traceback.format_exc())
    result["elapsed_seconds"] = time.monotonic() - begun
    dump(out / "RESULT.json", result)


if __name__ == "__main__":
    main()
