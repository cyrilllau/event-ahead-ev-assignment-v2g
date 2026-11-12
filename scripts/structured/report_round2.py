"""Read settled Round2 evidence and format four external reports, without solves.

Usage: python report_round2.py ROOT [--stopped 'reason'] [--display-root LOCAL_ROOT]

ROOT is the actual execution-host Round2 directory, not an offline mirror.
The default requires both ledgers idle, the campaign completion marker, and
matching WAIT receipts or verified dead-claim recovery chains for all frozen
jobs. Recovery settlement never implies scientific acceptance. --stopped permits an explicit
incomplete snapshot; it neither terminates nor declares dead an active worker.
No model, witness, statistic, bootstrap, or probability certificate is recomputed.
Existing reports are not overwritten, except ROOT/CASE_DESIGN_STATUS.md, whose
previous bytes and reference are preserved inside the new results JSON.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shlex
import time
import zipfile

from ev_assignment_experiments.analysis.diagnostic import read, reference
from ev_assignment_experiments.runtime.ledger import BudgetLedger


DEV_RUNS = (("R0", "decompose-r04"), ("R0", "direct-r05"),
            ("R1", "fixed64-r05"), ("R2", "fixed400-r05"),
            ("R5", "evidence400-r06"), ("R5", "evidence400-r07"))
METHODS = ("DET", "SAA", "ARO", "P3", "P2", "P3-noadvance")
OUTPUTS = ("CASE_DESIGN_ROUND2_REPORT_CN.md", "CASE_DESIGN_ROUND2_RESULTS.json",
           "CASE_DESIGN_STATUS.md", "REPRODUCE_CN.md")
OLD_COST = 211.08199704252183
RECOVERY_REASON = "dead_owner_unknown_overhead_conservatively_charged_full_reservation"
# Exact historical artifacts reviewed for this pair only; never infer a domain
# from two missing fields or extend this adapter to later/other runs.
LEGACY_R0 = {
    "decompose-r04": dict(release="r04", mode="decompose", attempt="624c95c7abf74a1d9c6d4b4684aab898",
        claim="86a044b102b2e5bfa1e4127aae32861bb367a85125713abb98c6188fbfc055d1",
        result="3f326294e6cb0a7c9bacd4bfcbf8c5bec3aad01c4f980130b0ac1aafcd017abc",
        wait="c7539bf6811c65d403e88a935d69047c41184f25eb7db2f380280682e644eec5",
        worker="5e746809f76569e1360c377e94a46084e67787e646fce1343793992e233a1e7f",
        wheel="718f2ae436ff0d7b1a309b02525189ef8d75675c55f46f4c467cf3af80c608ab",
        solver="ev_assignment/methods/catalogue.py",
        solver_sha="85a87eae36810b9f8f0cd5bab6aab04509bbc0bb9c3c608fb1f35c88fb3b8867",
        worker_lines=[[33, 47], [78, 84]], solver_lines=[[32, 47], [79, 84], [148, 150]]),
    "direct-r05": dict(release="r05", mode="direct", attempt="d9d91a95ce2043a290860c827c262c97",
        claim="0310e1dd2dbe4a3c08a3cb49782b3710b8a892b8174cdd75fe177d972060fd1b",
        result="906a834bc71eb45f2625a85064be2a89e0473cd4c3ffce21f010ef924cac39aa",
        wait="3ce86bd4fc9d9dfb998b73816793341efcd949bfd0d0d01ef01255cda60d6616",
        worker="dbed3189ff8da503cc0d52d6b307f4c8802fd4bd441b9e1cc855ee195b0bf42c",
        wheel="581e2a57d424faf4d9d546bbfd55c6bffa3e16304fa9ef002e1149b4785609fa",
        solver="ev_assignment/methods/catalogue_direct.py",
        solver_sha="d382fc388ee1062f1b89de0417ce24a6e3c9ea33b346fb39e508c245c419ce18",
        worker_lines=[[76, 93], [127, 133]], solver_lines=[[55, 79], [85, 104], [241, 243], [285, 301]]),
}
_LEGACY_RESTRICTION_TOKEN = object()


class Evidence:
    """Parse JSON/text only. May hash CASE bytes/ZIP source, never deserialize data or execute it."""

    def __init__(self):
        self.refs = {}

    def ref(self, path):
        path = Path(path)
        if not path.is_file():
            return None
        ref = reference(path)
        self.refs[str(path)] = ref
        return ref

    def json(self, path, *, optional=False):
        if optional and not Path(path).is_file():
            return {}
        self.ref(path)
        return read(path)

    def bound_json(self, ref):
        if not isinstance(ref, dict) or not ref.get("path"):
            raise ValueError("Missing evidence reference")
        actual = self.ref(ref["path"])
        if actual is None or any(actual[k] != ref.get(k) for k in ("sha256", "bytes")):
            raise ValueError(f"Evidence reference hash/size mismatch: {ref['path']}")
        return read(ref["path"])


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def max_residual(value):
    """Scalar frozen schema, or explicit row/bound components; invalid leaves poison the result."""
    if isinstance(value, dict):
        if not ({"row", "bound"} <= value.keys() or {"row", "bounds"} <= value.keys()):
            return None
        values = [max_residual(v) for v in value.values()]
        return max(values) if values and all(v is not None for v in values) else None
    return float(value) if numeric(value) and value >= 0 else None


def sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def hashed_identity(value):
    return isinstance(value, dict) and sha256(value.get("sha256"))


def unique_support(value):
    return isinstance(value, list) and bool(value) and all(sha256(v) for v in value) and len(set(value)) == len(value)


def metadata_digest(value):
    """Frozen JSON identity encoding, without importing a scientific generator."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def ledger_summary(ledger, *, development):
    snap = ledger.read()  # validates the existing append-only hash chain
    definition = snap["definition"]
    expected_limit = 7200 if development else 28800
    if definition["limit_seconds"] != expected_limit:
        raise ValueError(f"Unexpected frozen budget limit: {definition}")
    claims = {e["payload"]["attempt"]: e["payload"] for e in snap["events"]
              if e["payload"]["kind"] == "claim"}
    children = {e["payload"]["attempt"]: e["payload"]["identity"] for e in snap["events"]
                if e["payload"]["kind"] == "child"}
    charges, stages = [], {}
    new_wrapper_wall = 0.
    new_conservative_charge = 0.
    for index, event in enumerate(snap["events"]):
        p = event["payload"]
        if p["kind"] not in ("settlement", "external_charge"):
            continue
        claim = claims.get(p.get("attempt"), {})
        run = claim.get("run", p.get("run", {}))
        stage = run.get("stage", "UNSPECIFIED")
        seconds = p["charged_seconds"]
        stages[stage] = stages.get(stage, 0.) + seconds
        new_round = not development or stage in {f"R{i}" for i in range(6)}
        if new_round and p["kind"] == "settlement" and p.get("actual_parent_wait"):
            new_wrapper_wall += seconds
        if new_round and p["kind"] == "settlement" and p.get("reason") == RECOVERY_REASON:
            new_conservative_charge += seconds
        charges.append(dict(kind=p["kind"], attempt=p.get("attempt"), run=run,
                            charged_seconds=seconds, actual_parent_wait=p.get("actual_parent_wait"),
                            receipt=p.get("receipt"), returncode=p.get("returncode"),
                            observed_epoch=p.get("observed_epoch"), evidence=p.get("evidence"),
                            reason=p.get("reason"), claim_payload=claim,
                            child_identity=children.get(p.get("attempt")), event_payload=p,
                            ledger_event=reference(ledger.root / "events" / f"{index:08d}.json")))
    return dict(group=ledger.group, ledger_root=str(ledger.root), definition=definition,
        ledger_store=str(ledger.store_root),
        chain_head=snap["previous"], event_count=len(snap["events"]),
        spent_seconds=snap["spent_seconds"], remaining_seconds=snap["remaining_seconds"],
        limit_seconds=expected_limit, within_limit=snap["spent_seconds"] <= expected_limit,
        historical_round1_seconds=OLD_COST if development else 0.,
        round2_charged_seconds=snap["spent_seconds"] - (OLD_COST if development else 0.),
        inherited_cost_already_in_total=True, new_actual_wrapper_wall_seconds=new_wrapper_wall,
        new_conservative_reservation_charge_seconds=new_conservative_charge,
        active=snap["active"], stage_charged_seconds=stages, charges=charges)


def verified_recovery(ev, folder, claim, settlement, recovery):
    """Bind an OOM recovery to the already verified ledger, never fabricate WAIT."""
    if ev.ref(folder / "WAIT.json") is not None or ev.ref(folder / "RESULT.json") is not None:
        raise ValueError(f"Recovery cannot replace an existing WAIT/RESULT: {folder}")
    if (recovery.get("schema") != "round2-dead-claim-recovery-v1"
            or recovery.get("status") != "RESOURCE_INTERRUPTED_OOM"
            or recovery.get("attempt") != claim.get("attempt")
            or recovery.get("actual_parent_wait") is not False
            or recovery.get("reported_passed", False) is not False
            or recovery.get("science_recomputed", False) is not False):
        raise ValueError(f"Unsupported or inconsistent recovery record: {folder}")
    if (recovery.get("claim") != ev.ref(folder / "CLAIM.json")
            or ev.bound_json(recovery["claim"]) != claim
            or settlement.get("claim_payload") != claim
            or recovery.get("progress") != ev.ref(folder / "PROGRESS.json")
            or recovery.get("settlement_event") != settlement.get("ledger_event")):
        raise ValueError(f"Recovery claim/progress/event identity mismatch: {folder}")
    event = ev.bound_json(recovery["settlement_event"])
    payload = event.get("payload", {})
    if (payload != settlement.get("event_payload") or payload.get("kind") != "settlement"
            or payload.get("attempt") != claim["attempt"]
            or payload.get("reason") != RECOVERY_REASON
            or payload.get("actual_parent_wait") is not False
            or "receipt" in payload or "returncode" in payload
            or not numeric(recovery.get("charged_seconds")) or recovery["charged_seconds"] <= 0
            or recovery["charged_seconds"] != payload.get("charged_seconds")
            or recovery["charged_seconds"] != claim.get("reserved_seconds")):
        raise ValueError(f"Recovery is not the full-reservation ledger settlement: {folder}")
    observation = ev.bound_json(recovery["evidence"])
    identities = observation.get("identities", {})
    for role, expected in (("owner", claim.get("owner")), ("child", settlement.get("child_identity"))):
        observed = identities.get(role, {})
        if not expected or observed.get("identity") != expected or observed.get("matches") is not False:
            raise ValueError(f"Recovery lacks matching dead {role} observation: {folder}")
    unit = dict(line.split("=", 1) for line in observation.get("unit", "").splitlines() if "=" in line)
    if (observation.get("progress") != recovery["progress"]
            or observation.get("wait_exists") is not False or observation.get("result_exists") is not False
            or unit.get("Result") != "oom-kill" or unit.get("ActiveState") != "failed"
            or "systemd-oomd" not in observation.get("journal", "")):
        raise ValueError(f"Recovery lacks bound OOM/no-finalization observation: {folder}")
    progress = ev.bound_json(recovery["progress"])
    summary = dict(reference=ev.ref(folder / "RECOVERY.json"), record=recovery,
        identity_chain_verified=True, observation_reference=recovery["evidence"],
        observation=dict(observed_utc=observation.get("observed_utc"), diagnosis=observation.get("diagnosis"),
            identities=identities, unit=unit,
            campaign_unit_memory_peak_bytes=int(unit["MemoryPeak"]) if unit.get("MemoryPeak", "").isdigit() else None,
            campaign_unit_memory_limit_bytes=int(unit["MemoryMax"]) if unit.get("MemoryMax", "").isdigit() else None,
            memory_scope="whole campaign systemd unit, NOT per-run peak"),
        worker_finalization_proven=False)
    return summary, progress


def run_record(ev, root, stage, run_id, budget):
    folder = root / stage / run_id
    optimization_stage = stage in ("R0", "R1", "R2", "VCORE", "VIND", "VABL", "VSTRESS")
    raw = ev.json(folder / "RESULT.json", optional=True)
    claim = ev.json(folder / "CLAIM.json", optional=True)
    wait = ev.json(folder / "WAIT.json", optional=True)
    receipt = ev.ref(folder / "WAIT.json")
    result_ref = ev.ref(folder / "RESULT.json")
    recovery_record = ev.json(folder / "RECOVERY.json", optional=True)
    settlements = [c for c in budget["charges"] if c["kind"] == "settlement"
                   and c["attempt"] == claim.get("attempt")]
    settled = bool(claim and wait and len(settlements) == 1
                   and settlements[0]["actual_parent_wait"] is True
                   and settlements[0].get("claim_payload") == claim
                   and type(wait.get("returncode")) is int
                   and type(settlements[0].get("returncode")) is int
                   and wait["returncode"] == settlements[0]["returncode"]
                   and settlements[0]["receipt"] == receipt
                   and claim["run"].get("stage") == stage
                   and claim["run"].get("run_id") == run_id)
    if claim and wait and not settled:
        raise ValueError(f"Run receipt is not bound to this ledger: {folder}")
    recovery, progress = None, None
    if recovery_record:
        if (not claim or len(settlements) != 1 or claim["run"].get("stage") != stage
                or claim["run"].get("run_id") != run_id):
            raise ValueError(f"Recovery has no unique matching ledger settlement: {folder}")
        recovery, progress = verified_recovery(ev, folder, claim, settlements[0], recovery_record)
        settled = True
    optimization = raw.get("optimization", {})
    nonoptimization_progress = {}
    partial = False
    if progress is not None or (not optimization and (folder / "PROGRESS.json").is_file()
            and (not raw or optimization_stage)):
        progress = progress if progress is not None else ev.json(folder / "PROGRESS.json")
        if optimization_stage:
            optimization = progress.get("optimization", progress)
        else:
            nonoptimization_progress = progress
        partial = True
    if settled and not recovery and wait["returncode"] != 0 and (
            raw.get("passed") is True or optimization.get("passed") is True):
        raise ValueError(f"Nonzero process returncode conflicts with passed=true; review required: {folder}")
    lb, ub = optimization.get("LB"), optimization.get("UB")
    gap = ub - lb if numeric(lb) and numeric(ub) else None
    gap_threshold = max(1., .01 * max(1., abs(ub))) if numeric(ub) else None
    gap_passed = 0 <= gap <= gap_threshold if gap is not None else None
    saved_audits = optimization.get("original_audit", [])
    audits = saved_audits if isinstance(saved_audits, list) else []
    residuals = [max_residual(a.get("residual")) if isinstance(a, dict) else None for a in audits]
    residual = max(residuals) if residuals and all(r is not None for r in residuals) else None
    support = optimization.get("ordered_support", [])
    count = optimization.get("full_support", raw.get("optimization_support", raw.get("support")))
    complete_original = bool(optimization.get("original_expansion_audited") is True and unique_support(support)
        and type(count) is int and count > 0 and len(audits) == count
        and all(isinstance(a, dict) for a in audits) and [a.get("sid") for a in audits] == support
        and residual is not None and residual <= 1e-9)
    accepted = bool(not partial and raw.get("passed") is True and optimization.get("passed") is True
        and settled and wait.get("returncode") == 0 and complete_original and gap_passed is True)
    attempted = bool(claim.get("attempt") and claim["run"].get("stage") == stage
                     and claim["run"].get("run_id") == run_id)
    optimization_job = bool(optimization) or stage in ("R0", "R1", "R2", "VCORE", "VIND", "VABL", "VSTRESS")
    job_accepted = accepted if optimization_job else bool(raw.get("passed") is True and settled and wait.get("returncode") == 0)
    timing = optimization.get("timing", {})
    audit_started = bool(audits or optimization.get("audit_completed", 0)
                         or (numeric(timing.get("audit")) and timing["audit"] > 0))
    if not optimization_job:
        phase = "NOT_APPLICABLE_NON_OPTIMIZATION_JOB"
    elif complete_original:
        phase = "ORIGINAL_AUDIT_COMPLETE"
    elif audit_started:
        phase = "ORIGINAL_AUDIT_INCOMPLETE"
    elif optimization.get("current_evaluation"):
        phase = "LAST_RECORDED_RECOURSE_EVALUATION"
    else:
        phase = "NOT_RECORDED"
    if not optimization_job:
        completion = "NOT_APPLICABLE_NON_OPTIMIZATION_JOB"
    elif accepted:
        completion = "ACCEPTED_GAP_AND_ORIGINAL_AUDIT"
    elif gap_passed and not complete_original:
        completion = "GAP_PASSED_AUDIT_INCOMPLETE" if audit_started else "GAP_PASSED_AUDIT_NOT_ESTABLISHED"
    elif gap_passed:
        completion = "GAP_AND_AUDIT_PASS_ACCEPTANCE_NOT_RECORDED"
    elif gap is None:
        completion = "NO_COMPLETE_BOUND_INTERVAL"
    elif gap < 0:
        completion = "INCOMPATIBLE_BOUND_INTERVAL"
    else:
        completion = "GAP_NOT_PASSED_AUDIT_COMPLETE" if complete_original else "GAP_NOT_PASSED_AUDIT_INCOMPLETE"
    recorded_parts = {k: v for k, v in timing.items() if numeric(v) and v >= 0}
    deployment = dict(raw.get("deployment", {}))
    packages = deployment.get("packages", [])
    if packages:
        deployment.setdefault("selected_stations", sorted({p["c"] for p in packages}))
        deployment.setdefault("effective_selected_types", len({p["g"] for p in packages}))
        deployment.setdefault("total_advance_vehicle_hours",
                              sum((64 - p["d"]) / 4 * p["count"] for p in packages))
    for name in ("ERROR.txt", "WORKER.log", "EXECUTION_CONFIG.json"):
        ev.ref(folder / name)
    return dict(id=run_id, stage=stage, directory=str(folder),
        status=recovery_record["status"] if recovery else raw.get("status", "PARTIAL_PROGRESS_ONLY" if partial else "NO_RESULT"),
        reported_passed=False if recovery else raw.get("passed") is True, acceptance_supported=accepted,
        reported_passed_raw=raw.get("passed"),
        invalid_boolean_flags=[label for label, doc, field in (("RESULT.passed", raw, "passed"),
            ("optimization.passed", optimization, "passed"),
            ("optimization.original_expansion_audited", optimization, "original_expansion_audited"))
            if field in doc and type(doc[field]) is not bool],
        attempted=attempted, job_acceptance_supported=job_accepted,
        optimization_job=optimization_job,
        bounds_audit_applicability="APPLICABLE" if optimization_job else "NOT_APPLICABLE",
        gap_passed=gap_passed, gap_threshold=gap_threshold, audit_complete=complete_original,
        completion_class=completion, last_saved_phase=phase,
        last_saved_phase_basis="saved audit rows/counter/timing, otherwise last current_evaluation; not a live phase observation",
        saved_current_evaluation=optimization.get("current_evaluation"),
        saved_audit_completed=optimization.get("audit_completed"),
        gap_first_passed_seconds=optimization.get("gap_first_passed_seconds"),
        largest_recorded_timing_part=max(recorded_parts, key=recorded_parts.get) if recorded_parts else None,
        settled=settled, progress_only=partial, returncode=wait.get("returncode"),
        nonoptimization_progress=nonoptimization_progress,
        result=result_ref, claim=claim, wait=receipt, recovery=recovery,
        progress=ev.ref(folder / "PROGRESS.json") if partial else None,
        worker_finalization_proven=bool(raw and wait and settled and not recovery),
        charged_seconds=settlements[0]["charged_seconds"] if settled else None,
        actual_parent_wait=settlements[0]["actual_parent_wait"] if settled else None,
        accounting_basis="conservative reservation" if recovery else "observed parent wait" if settled else "unsettled",
        wrapper_seconds=settlements[0]["charged_seconds"] if settled and settlements[0]["actual_parent_wait"] is True else None,
        worker_seconds=raw.get("total_seconds", raw.get("elapsed_seconds")),
        controller_seconds=optimization.get("elapsed_seconds"), timing_seconds=timing,
        preparation_timing_seconds=raw.get("timing", {
            "build_and_probability": raw.get("build_and_probability_seconds")}),
        LB=lb, UB=ub, gap=gap, stored_gap=optimization.get("gap"),
        relative_gap=gap / max(1., abs(ub)) if gap is not None else None,
        support=count, original_audit_count=len(audits), original_max_residual=residual,
        invalid_original_residual_count=sum(r is None for r in residuals),
        original_audit_list_valid=isinstance(saved_audits, list), ordered_support_unique=unique_support(support),
        original_complete=complete_original, deployment=deployment, service=raw.get("service", {}),
        method=raw.get("method", optimization.get("method")), seed=raw.get("seed"),
        representation=optimization.get("representation", raw.get("representation")),
        native_status=optimization.get("native_status"),
        native_status_scope="last saved master status; not all recourse/oracle calls" if raw.get("mode") != "direct" else "last direct extensive MIP status",
        last_master_native_status=optimization.get("native_status") if raw.get("mode") != "direct" else None,
        optimization_reason=optimization.get("reason"), raw_error=raw.get("error"),
        error=raw.get("error", optimization.get("reason")),
        lower_evidence=optimization.get("lower_evidence"), upper_evidence=optimization.get("upper_evidence"),
        model_sizes=raw.get("model_sizes", {}), master_size=optimization.get("master_size", optimization.get("model_size")),
        physical_screen={k: v for k, v in raw.get("physical_screen", {}).items()
                         if k not in ("healthy_grid_audit", "scenarios")},
        stress_protocol=raw.get("protocol"), soc_report=raw.get("soc_report"),
        counters={k: optimization.get(k) for k in ("iterations", "cuts", "nodes", "prepass_rounds",
                  "recourse_calls", "recourse_attempts", "probability_calls", "probability_attempts")},
        mode=raw.get("mode"), original_frozen_case=raw.get("original_frozen_case"),
        raw=raw, optimization=optimization)


def _legacy_restriction_subject(record):
    return {k: record.get(k) for k in ("id", "stage", "directory", "claim", "result", "raw", "optimization",
        "settled", "wait", "returncode", "actual_parent_wait", "effective_first_stage_restriction", "restriction_source_review")}


def bind_legacy_r0(ev, root, record, budget):
    """Hash-bound all-packages inference for the two reviewed R0 artifacts only.

    Read/hash files and ZIP members; never import wheel code, unpickle CASE,
    rebuild a model, alter saved records, or generalize missing-field semantics.
    """
    if record["stage"] != "R0" or record["id"] not in LEGACY_R0:
        return None
    spec = LEGACY_R0[record["id"]]
    folder, release = root / "R0" / record["id"], root / "releases" / spec["release"]

    def pinned(path, expected):
        ref = ev.ref(path)
        if not ref or ref["sha256"] != expected:
            raise ValueError(f"legacy-r0-all-packages-v1 artifact hash mismatch: {path}")
        return ref

    refs = {key: pinned(folder / name, spec[key]) for key, name in
            (("claim", "CLAIM.json"), ("result", "RESULT.json"), ("wait", "WAIT.json"))}
    claim, raw, wait = (ev.bound_json(refs[k]) for k in ("claim", "result", "wait"))
    opt, science = raw["optimization"], raw["optimization"]["science_identity"]
    settlements = [c for c in budget["charges"] if c["kind"] == "settlement" and c["attempt"] == spec["attempt"]]
    if (record["directory"] != str(folder) or record["settled"] is not True
            or record["claim"] != claim or record["raw"] != raw or record["optimization"] != opt
            or record["result"] != refs["result"] or record["wait"] != refs["wait"]
            or claim.get("attempt") != spec["attempt"] or len(settlements) != 1
            or settlements[0].get("claim_payload") != claim or settlements[0].get("receipt") != refs["wait"]
            or settlements[0].get("actual_parent_wait") is not True
            or type(wait.get("returncode")) is not int or wait["returncode"] != 0
            or type(settlements[0].get("returncode")) is not int or settlements[0]["returncode"] != 0
            or claim["run"].get("stage") != "R0" or claim["run"].get("run_id") != record["id"]
            or any(k in claim["run"] for k in ("argv", "release", "freeze"))
            or "first_stage_restriction" in opt or "first_stage_restriction" in science):
        raise ValueError("legacy-r0-all-packages-v1 requires the exact original settlement/schema")
    refs["worker"] = pinned(release / "round2_worker.py", spec["worker"])
    refs["wrapper"] = pinned(release / "stage.py", "270f9237005c4f4404236422f74365ae0cf4b88e78e241a5dc9214201a49cd57")
    refs["wheel"] = pinned(release / "ev_assignment-0.2.0.dev0-py3-none-any.whl", spec["wheel"])
    refs["original_case"] = pinned(root.parent / "A2/r03/C11/CASE.pkl",
        "f81a6b131a43ba1f826f3d9337fbb0ad867cda5fa8cc9c5bff8bee94ffbc2648")
    refs["case"] = pinned(folder / "CASE.pkl", "b124de49a936b1a750b0a07a91962b26564a2dc4c4fc040e3ada22839fe79000")
    refs["probability"] = pinned(folder / "PROBABILITY_DOMAIN.json",
        "d15820bd961f6cd48fe68914e5a58d28945515f8c23ff92972a603a23e12600e")
    domain = ev.bound_json(refs["probability"])
    if (claim["run"]["source"] != raw.get("source") or raw["source"] != refs["worker"]
            or claim["run"]["wrapper"] != refs["wrapper"] or raw.get("original_frozen_case") != refs["original_case"]
            or raw.get("case") != refs["case"] or raw.get("mode") != spec["mode"]
            or (raw.get("N"), raw.get("G"), raw.get("packages"), raw.get("support")) != (6, 6, 72, 8)
            or raw.get("case_identity") != science or science.get("sha256") != "bdbbcd8db41d3867e11d5ef49e4cd40476f9322f17db249afff0159491bf8d72"
            or science.get("catalogue") != "bb638cf9235825fd35fcfa9a475af5c9b2ae463104c5dbeea265cfb0afb6a2d4"
            or opt.get("ambiguity_identity") != "7199dac03516ce75d8a618095b5a18b0e9695f53694dbe831dad8643deca3991"
            or domain.get("identity") != opt["ambiguity_identity"] or domain.get("feature_rows") != 3
            or opt.get("first_stage_weights") != [1, 1, 1, 1] or opt.get("representation") != "count"
            or not unique_support(opt.get("ordered_support")) or len(opt["ordered_support"]) != 8
            or opt["ordered_support"] != science.get("scenarios")):
        raise ValueError("legacy-r0-all-packages-v1 case/probability/source identity mismatch")
    sources = [dict(reference=refs["worker"], lines=spec["worker_lines"],
        finding="Load the exact CASE and pass it unchanged to the selected solver; no departure restriction argument")]
    with zipfile.ZipFile(refs["wheel"]["path"]) as archive:
        for member, expected, lines, finding in (
                (spec["solver"], spec["solver_sha"], spec["solver_lines"],
                 "Pass all case packages to build_first; preserve its complete rows and bounds in the model"),
                ("ev_assignment/l08/catalogue.py", "dfaef697609945e8df900ee828824ebc65f4ee7b7b36f534ca33d9aa282dd297",
                 [[58, 59], [77, 98], [105, 130]], "Enumerate all declared packages; only declared capacity, contract and port constraints")):
            if archive.namelist().count(member) != 1:
                raise ValueError(f"Ambiguous/missing legacy wheel member: {member}")
            content = archive.read(member)
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f"Legacy wheel member hash mismatch: {member}")
            sources.append(dict(archive=refs["wheel"], member=member, sha256=expected,
                                bytes=len(content), lines=lines, finding=finding))
    pinned(Path(refs["wheel"]["path"]), spec["wheel"])
    release_ref = ev.ref(release / "RELEASE.json")
    if spec["release"] == "r04":
        if release_ref is not None:
            raise ValueError("Reviewed r04 schema had no RELEASE.json")
    else:
        release_ref = pinned(release / "RELEASE.json", "7ddb1d880bce122d6b7b96395d892e572c0aee85b4ea54dc31eb3c45c166340d")
        manifest = ev.bound_json(release_ref)
        if any(manifest.get("files", {}).get(name) != refs[key]["sha256"] for name, key in
               (("round2_worker.py", "worker"), ("stage.py", "wrapper"), (Path(refs["wheel"]["path"]).name, "wheel"))):
            raise ValueError("Legacy r05 release manifest does not bind reviewed artifacts")
    proof = dict(schema="legacy-r0-all-packages-v1", status="HASH_BOUND_SOURCE_REVIEW", run=record["id"],
        inferred_restriction="all_packages", artifacts=refs, source_review=sources,
        settlement_event=settlements[0].get("ledger_event"), release_manifest=release_ref,
        original_field_missing=True, science_identity_restriction_missing=True,
        claim_fields_missing=["argv", "release", "freeze"], r04_release_manifest_missing=spec["release"] == "r04",
        scope="Exact reviewed N6/G6/S8 count pair and all72 declared catalogue packages only; no old-full-model equivalence, execution-time module attestation or new scientific certificate")
    record["effective_first_stage_restriction"] = "all_packages"
    record["restriction_source_review"] = proof
    record["_legacy_restriction_binding"] = (_LEGACY_RESTRICTION_TOKEN, metadata_digest(_legacy_restriction_subject(record)))
    return proof


def effective_restriction(record):
    """A serialized 'bound' flag alone cannot activate the legacy inference."""
    opt = record["optimization"]
    if "first_stage_restriction" in opt:
        value = opt["first_stage_restriction"]
        return value if value in ("all_packages", "latest_departure_only") else None
    binding = record.get("_legacy_restriction_binding")
    if (record["stage"] == "R0" and record["id"] in LEGACY_R0 and binding
            and binding[0] is _LEGACY_RESTRICTION_TOKEN
            and binding[1] == metadata_digest(_legacy_restriction_subject(record))
            and record.get("effective_first_stage_restriction") == "all_packages"):
        return "all_packages"
    return None


def bind_frozen_job(ev, validation, freeze, release, job, record):
    """Standalone copy of the controller's pure expected-identity construction."""
    if not record["claim"]:
        return None
    root = Path(freeze["source_release"]["path"]).parent
    folder = validation / job["stage"] / job["id"]
    source, wrapper = ev.ref(root / job["worker"]), ev.ref(root / "stage.py")
    config, design = ev.ref(folder / "EXECUTION_CONFIG.json"), ev.ref(validation / "DESIGN_CONFIG.json")
    if (not source or not wrapper or not config or not design
            or source["sha256"] != release.get("files", {}).get(job["worker"])
            or wrapper["sha256"] != release.get("files", {}).get("stage.py")
            or (config["sha256"], config["bytes"]) != (design["sha256"], design["bytes"])):
        raise ValueError(f"Frozen source/wrapper/config hash mismatch: {job['id']}")
    expected = dict(stage=job["stage"], run_id=job["id"], source=source,
        argv=[str(root / ".venv/bin/python"), str(root / job["worker"]), *job["args"]],
        release=freeze["source_release"], freeze=ev.ref(validation / "FREEZE.json"), wrapper=wrapper, config=config)
    if record["claim"].get("run") != expected:
        raise ValueError(f"CLAIM differs from expected frozen job identity: {job['id']}")
    if "source" in record["raw"] and record["raw"]["source"] != source:
        raise ValueError(f"RESULT worker source differs from frozen job: {job['id']}")
    return dict(status="FROZEN_EXECUTION_METADATA_BOUND", expected=expected,
                scope="source/argv/wrapper/config/release/FREEZE; no scientific execution")


def bind_prepared_support(ev, validation, freeze, job, record, preparation):
    """Bind saved validation SID metadata to frozen V0; never open CASE/NPZ data."""
    if job["worker"] != "validation_worker.py" or not record["optimization"]:
        return None
    case_path, method, seed, representation, prepass = job["args"]
    seed = int(seed)
    if preparation["job_acceptance_supported"] is not True:
        raise ValueError("Optimization metadata requires an accepted frozen V0")
    entries = [c for c in preparation["raw"].get("cases", []) if c.get("split") == "validation" and c.get("seed") == seed]
    manifest_path = Path(preparation["directory"]) / f"validation_{seed}_MANIFEST.json"
    prepared = ev.json(manifest_path)
    if (len(entries) != 1 or entries[0] != prepared
            or preparation["raw"].get("freeze") != ev.ref(validation / "FREEZE.json")
            or prepared.get("case", {}).get("path") != case_path):
        raise ValueError(f"Frozen V0 manifest/case mismatch: {job['id']}")
    identity, ids, weights = prepared.get("identity", {}), prepared.get("ordered_ids"), prepared.get("weights")
    if (not hashed_identity(identity) or identity["sha256"] != metadata_digest({k: v for k, v in identity.items() if k != "sha256"})
            or not unique_support(ids) or len(ids) != 400 or identity.get("scenarios") != ids
            or identity.get("weights") != weights or not isinstance(weights, list) or len(weights) != 400
            or not all(numeric(w) and w >= 0 for w in weights) or not sha256(identity.get("forecast"))
            or identity.get("split") != "validation" or identity.get("seed") != seed
            or identity.get("fleet_size") != 30 or identity.get("groups") != 6):
        raise ValueError(f"Invalid prepared validation support identity: {job['id']}")
    raw, opt = record["raw"], record["optimization"]
    folder = validation / job["stage"] / job["id"]
    support = ev.json(folder / "SUPPORT_MANIFEST.json")
    expected = dict(case_identity=identity["sha256"], split="validation", seed=seed,
        ordered_ids=ids, weights=weights, forecast_id=identity["forecast"], N=30, G=6, S=400,
        catalogue_sha256=identity["catalogue"])
    if support != dict(expected, sha256=metadata_digest(expected)):
        raise ValueError(f"Worker support manifest differs from V0: {job['id']}")
    for field, value in (("base_case_identity", identity), ("frozen_case", prepared["case"]),
                         ("method", method), ("seed", seed), ("representation", representation),
                         ("prepass", prepass == "prepass")):
        if field in raw and (raw[field] != value or (field == "prepass" and type(raw[field]) is not bool)):
            raise ValueError(f"RESULT {field} differs from frozen input: {job['id']}")
    model = ev.json(folder / "MODEL_VIEW.json")
    expected_ids = [identity["forecast"]] if method == "DET" else ids
    restriction = "latest_departure_only" if method == "P3-noadvance" else "all_packages"
    if (not hashed_identity(model) or model["sha256"] != metadata_digest({k: v for k, v in model.items() if k != "sha256"})
            or opt.get("science_identity") != model or raw.get("model_identity", model) != model
            or model.get("parent_case_identity") != identity["sha256"] or model.get("requested_method") != method
            or model.get("source") != identity.get("source") or model.get("catalogue") != identity.get("catalogue")
            or model.get("ordered_optimization_ids") != expected_ids or model.get("scenarios") != expected_ids
            or opt.get("ordered_support") != expected_ids or opt.get("full_support") != len(expected_ids)
            or model.get("first_stage_restriction") != restriction or opt.get("first_stage_restriction") != restriction
            or opt.get("representation") != representation):
        raise ValueError(f"Model/support/restriction differs from frozen method: {job['id']}")
    domain_ref = ev.ref(folder / "PROBABILITY_DOMAIN.json")
    if raw.get("probability_domain", domain_ref) != domain_ref:
        raise ValueError(f"Probability metadata reference mismatch: {job['id']}")
    domain = ev.bound_json(domain_ref)
    # Frozen finite_domain stores fixed probabilities as identity rows (DET=1, SAA=400).
    rows = {"DET": 1, "SAA": 400, "ARO": 0, "P3": 3, "P2": 2, "P3-noadvance": 3}[method]
    if (not sha256(domain.get("identity")) or domain["identity"] != opt.get("ambiguity_identity")
            or domain["identity"] != model.get("probability_identity") or domain.get("ordered_support") != expected_ids
            or domain.get("moment_rows") != rows or raw.get("moment_rows", rows) != rows):
        raise ValueError(f"Probability identity/declared dimensions mismatch: {job['id']}")
    record["probability_metadata"] = domain
    return dict(status="V0_ORDERED_SUPPORT_METADATA_BOUND", expected_support_count=len(expected_ids),
        forecast_only=method == "DET", preparation_result=preparation["result"],
        prepared_manifest=ev.ref(manifest_path), worker_manifest=ev.ref(folder / "SUPPORT_MANIFEST.json"),
        model_view=ev.ref(folder / "MODEL_VIEW.json"), probability_domain=domain_ref,
        scope="Saved SID/weight/case/hash/method metadata only; no CASE/NPZ read or physics/probability recomputation")


def declared_variant_matches(a, b, variant):
    aa, bb = a["optimization"], b["optimization"]
    left, right = aa.get("science_identity", {}), bb.get("science_identity", {})
    allowed = {"sha256", "method", "requested_method", "probability_identity"}
    if variant == "noadvance":
        allowed |= {"first_stage_restriction", "noadvance_rule"}
    if (left.get("schema") != "fixed_load_validation_subproblem_v1" or left.get("requested_method") != "P3"
            or left.get("method") != "P3-DRO" or right.get("method") != ("P2-DRO" if variant == "P2" else "P3-DRO/no-advance")
            or right.get("requested_method") != ("P2" if variant == "P2" else "P3-noadvance")
            or {k: v for k, v in left.items() if k not in allowed} != {k: v for k, v in right.items() if k not in allowed}):
        return False
    if variant == "noadvance":
        return (aa.get("first_stage_restriction") == left.get("first_stage_restriction") == "all_packages"
            and bb.get("first_stage_restriction") == right.get("first_stage_restriction") == "latest_departure_only"
            and left.get("noadvance_rule") is None
            and right.get("noadvance_rule") == "departure_choice == 2 (d=64); no other constraints changed"
            and aa.get("ambiguity_identity") == bb.get("ambiguity_identity") == left.get("probability_identity") == right.get("probability_identity"))
    pa, pb = a.get("probability_metadata", {}), b.get("probability_metadata", {})
    scope = pa.get("feature_scope", {})
    bounds_valid = all(isinstance(p.get(k), list) and len(p[k]) == n
        and all(numeric(v) for v in p[k]) for p, n in ((pa, 3), (pb, 2)) for k in ("lower", "upper"))
    return (aa.get("first_stage_restriction") == bb.get("first_stage_restriction")
        == left.get("first_stage_restriction") == right.get("first_stage_restriction") == "all_packages"
        and pa.get("identity") == aa.get("ambiguity_identity") == left.get("probability_identity")
        and pb.get("identity") == bb.get("ambiguity_identity") == right.get("probability_identity")
        and pa.get("moment_rows") == pa.get("feature_rows") == 3 and pb.get("moment_rows") == pb.get("feature_rows") == 2
        and bool(scope) and scope == pb.get("feature_scope") and scope.get("joint_row_indices") == [2]
        and scope.get("declared_row_count") == 3 and bounds_valid
        and scope.get("marginal_rule") == "remove only joint_row_indices=[2]; preserve support and all other rows/bounds"
        and all(0 <= lo <= hi <= 1 for p in (pa, pb) for lo, hi in zip(p["lower"], p["upper"], strict=True))
        and all(pa[k][:2] == pb[k] for k in ("lower", "upper"))
        and isinstance(pa.get("rule"), dict) and bool(pa["rule"]) and pa["rule"] == pb.get("rule"))


def compare(a, b, *, variant="equal"):
    """Arithmetic on stored intervals only; no new optimality/physics proof."""
    aa, bb = a["optimization"], b["optimization"]
    restrictions = [effective_restriction(r) for r in (a, b)]
    shared = (bool(aa and bb) and all(hashed_identity(o.get("science_identity")) and sha256(o.get("ambiguity_identity"))
              and unique_support(o.get("ordered_support")) and o.get("first_stage_weights") == [1, 1, 1, 1]
              and all(numeric(w) for w in o["first_stage_weights"]) for o in (aa, bb))
              and aa.get("ordered_support") == bb.get("ordered_support"))
    if variant == "equal":
        shared = shared and aa.get("science_identity") == bb.get("science_identity")
        shared = shared and aa.get("ambiguity_identity") == bb.get("ambiguity_identity")
        shared = shared and restrictions[0] in ("all_packages", "latest_departure_only")
        shared = shared and restrictions[0] == restrictions[1]
    else:
        left, right = a["raw"].get("base_case_identity", {}), b["raw"].get("base_case_identity", {})
        if variant not in ("P2", "noadvance"):
            raise ValueError("Unknown comparison variant")
        shared = (shared and hashed_identity(left) and left == right
            and aa["science_identity"].get("parent_case_identity") == bb["science_identity"].get("parent_case_identity") == left["sha256"]
            and declared_variant_matches(a, b, variant))
    interval = None
    if (shared and a["original_complete"] is True and b["original_complete"] is True
            and all(numeric(r[k]) for r in (a, b) for k in ("LB", "UB"))):
        interval = [a["LB"] - b["UB"], a["UB"] - b["LB"]]
    contradicted = bool(interval and (interval[0] > 1e-5 or
                         (variant == "equal" and interval[1] < -1e-5)))
    both = bool(shared and a["acceptance_supported"] is True and b["acceptance_supported"] is True)
    return dict(a=a["id"], b=b["id"], variant=variant,
        identity_matched=bool(shared), both_accepted=both,
        effective_first_stage_restrictions=restrictions,
        source_review_restriction_applied=["first_stage_restriction" not in r["optimization"] and effective_restriction(r) is not None for r in (a, b)],
        identity_check_scope="Nonempty SHA identities, unique ordered support, explicit restriction or exact hash-bound legacy R0 source review; variants compare declared metadata only, not NPZ coefficients",
        missing_restriction_metadata=any(o.get("first_stage_restriction") not in ("all_packages", "latest_departure_only") for o in (aa, bb)),
        difference_a_minus_b=interval,
        status="CONTRADICTORY_INTERVALS" if contradicted else
               "INTERVALS_AVAILABLE" if interval else "UNRESOLVED_OR_IDENTITY_MISMATCH",
        controller_time_ratio_b_over_a=(b["controller_seconds"] / a["controller_seconds"]
            if variant == "equal" and both and not contradicted and numeric(a["controller_seconds"])
            and numeric(b["controller_seconds"]) and a["controller_seconds"] > 0 else None),
        scope="same optimization problem" if variant == "equal" else
              "same frozen case/support/fees; explicitly changed ambiguity or departure domain")


def public_run(record):
    return {k: v for k, v in record.items() if k not in ("raw", "optimization", "probability_metadata", "_legacy_restriction_binding")}


def job_coverage(records):
    """A matching claim is an attempt; settlement and scientific acceptance differ."""
    return dict(expected=len(records), attempted=sum(r["attempted"] for r in records),
                settled=sum(r["settled"] for r in records),
                accepted=sum(r["job_acceptance_supported"] for r in records))


def validation_findings(core, mechanisms, core_run_ids):
    """Count each accepted evidence scope separately, without altering any gate."""
    p3 = [r for r in core if r["id"] in {names["P3"] for names in core_run_ids.values()}]
    accepted = [r for r in p3 if r["acceptance_supported"] and r["support"] == 400]
    aro = [r for r in core if r["id"] in {names["ARO"] for names in core_run_ids.values()}]
    groups = {}
    for method in ("P2", "P3-noadvance"):
        rows = [p for p in mechanisms if p["methods"] == ["P3", method]]
        available = [p for p in rows if p["both_accepted"] and p["status"] == "INTERVALS_AVAILABLE"]
        groups[method] = dict(expected=len(rows), accepted_intervals=len(available),
            accepted_interval_seeds=[p["seed"] for p in available],
            strictly_negative_beyond_1e_5=sum(p["difference_a_minus_b"][1] < -1e-5 for p in available),
            unresolved=len(rows) - len(available),
            contradictory=sum(p["status"] == "CONTRADICTORY_INTERVALS" for p in rows))
    return dict(p3_full400=dict(expected=len(p3), accepted=len(accepted),
            accepted_seeds=[r["seed"] for r in accepted], runs=[r["id"] for r in accepted],
            deployments=[dict(seed=r["seed"], run=r["id"],
                selected_vehicles=r["deployment"].get("selected_vehicles"),
                contract_kwh=r["deployment"].get("contract_kwh"),
                selected_stations=r["deployment"].get("selected_stations")) for r in accepted]),
        mechanism_comparisons=groups,
        aro=dict(expected=len(aro), accepted=sum(r["acceptance_supported"] for r in aro)),
        interval_scope="Separately counted same-case comparisons, not statistically independent replications or OOS superiority; numeric evidence only",
        strict_separation_reporting_tolerance=1e-5)


def algorithm_timing_findings(pairs, records):
    """Describe saved matched-run timings; the prioritization rule is not an acceptance gate."""
    by_id = {r["id"]: r for r in records}
    rows = []
    for pair in pairs:
        a, b = by_id[pair["a"]], by_id[pair["b"]]
        ratio = pair["controller_time_ratio_b_over_a"]
        reserved = [r["claim"].get("reserved_seconds") for r in (a, b)]
        same_budget = all(numeric(v) and v > 0 for v in reserved) and reserved[0] == reserved[1]
        usable = ratio is not None and same_budget
        wrapper_ratio = (b["wrapper_seconds"] / a["wrapper_seconds"]
            if usable and all(numeric(r["wrapper_seconds"]) and r["wrapper_seconds"] > 0 for r in (a, b)) else None)
        at, bt = a["timing_seconds"], b["timing_seconds"]
        keys = ("build", "audit", "recourse", "master", "probability", "certificate_and_io")
        timing_complete = all(numeric(at.get(k)) and at[k] >= 0 for k in keys)
        candidate = bool(usable and timing_complete and numeric(bt.get("audit")) and bt["audit"] >= 0
            and at["audit"] > bt["audit"]
            and at["build"] + at["audit"] >= sum(at[k] for k in keys[2:]))
        rows.append(dict(seed=pair["seed"], count=a["id"], individual=b["id"],
            matched_and_both_accepted=pair["identity_matched"] and pair["both_accepted"],
            identity_matched=pair["identity_matched"],
            count_accepted=a["acceptance_supported"], individual_accepted=b["acceptance_supported"],
            count_status=a["status"], individual_status=b["status"],
            individual_completion_class=b["completion_class"],
            same_reserved_budget=same_budget, reserved_seconds_count_individual=reserved,
            usable_timing_comparison=usable,
            actual_gap_count_individual=[a["gap"], b["gap"]],
            gap_threshold_count_individual=[a["gap_threshold"], b["gap_threshold"]],
            same_gap_acceptance_rule="max(1, 0.01 * max(1, abs(UB))); actual gaps need not match",
            controller_seconds_count_individual=[a["controller_seconds"], b["controller_seconds"]],
            build_seconds_count_individual=[at.get("build"), bt.get("build")],
            audit_seconds_count_individual=[at.get("audit"), bt.get("audit")],
            controller_ratio_individual_over_count=ratio if usable else None,
            observed_wrapper_ratio_individual_over_count=wrapper_ratio,
            count_controller_faster_observed=ratio > 1 if usable else None,
            count_end_to_end_faster_observed=wrapper_ratio > 1 if wrapper_ratio is not None else None,
            audit_mapping_reuse_candidate=candidate))
    candidates = [r["seed"] for r in rows if r["audit_mapping_reuse_candidate"]]
    return dict(pairs=rows, audit_mapping_reuse_candidate_seeds=candidates,
        prioritization_rule="Accepted same-problem/equal-reservation runs only: count audit exceeds individual audit, and count build+audit >= other recorded phase times combined. Engineering heuristic, not a scientific acceptance criterion or causal profiling result.",
        construction_time_within_audit_not_separately_measured=True,
        controller_includes_build_and_audit=True,
        timing_scope="Single observed runs at the same gap rule, not necessarily the same achieved gap or policy; no controlled-load speedup or runtime CI")


def next_improvement(timing):
    seeds = timing["audit_mapping_reuse_candidate_seeds"]
    if seeds:
        return (f"优先工程假说（seed {', '.join(map(str, seeds))} 的同问题已接受计时支持排查）："
            "在独立版本实现直接count→原逐车审计向量映射，复用经身份校验确实不变的索引/稀疏结构，"
            "减少中间逐车catalogue模板的重复构建。该建议尚未实现或验证提速；audit内部构建占比未单独计量，"
            "不能把计时差当作因果证明。仍须逐场景覆盖全部400支持、未选套餐Big-M、原matrix.assemble全部行/界及目标审计，"
            "保留原容差和验收门。高服务规模设计与丰富moments仍是独立论文缺口，不由这项工程优化解决。")
    return ("当前已接受的同问题配对计时未满足审计映射复用的优先排查条件，不据此认定构建瓶颈；"
        "端到端时间结论仍只按匹配对照的已保存wrapper计时给出。"
        "下一项实例设计建议仍是容量受限的多区域/多时段服务需求，并保留健康网可行性与原限额；"
        "高服务规模和丰富moments的科学证据仍缺。这里只提出后续假说，不授权新运行。")


def certificate_limitations(critic):
    """Disclose the read-only review without changing stored acceptance or values."""
    return dict(
        status="KNOWN_LOCAL_ROUNDING_DEFECT_PENDING_INDEPENDENT_VERSION_FIX",
        critic=critic, critic_available=critic is not None,
        review_scope="Read-only source review and analytical counterexample; not frozen-release byte-parity verification or validation/OOS replay",
        confirmed_issue="recourse.audit evaluates the valid alpha+beta*x cut with ordinary rounding to qL; the scalar can exceed exact Q under the analytical counterexample conditions",
        affected_fields=["recourse.qL", "risk.risk_L", "fixed_policy_risk_lower_endpoints",
                         "OOS.mean_cost_interval[0]", "OOS.cvar95_cost_interval[0]"],
        affected_lower_endpoints_strictly_downward_certified=False,
        qL_occurrence_in_frozen_cases_established=False,
        global_master_LB_UB_failure_proven=False,
        structural_global_bound_blocker_found_in_review=False,
        master_bound_scope="Master LB and stored alpha/beta cuts do not directly use the rounded qL scalar; qL affects probability separation, whose added probabilities still require domain membership checks",
        risk_upper_scope="Risk upper plane is shifted to dominate every qU with floating-point guards/checks; not a claim of exact-rational dominance or exact primal feasibility",
        numerical_upper_evidence="NUMERICALLY_AUDITED_ONLY",
        end_to_end_strict_rational_certificate=False,
        interpretation="qL/risk_L and qL-derived OOS endpoints are numerical quantities, not rigorously enclosing lower bounds; this local defect has not established failure of the frozen master LB or full-support risk UB",
        disposition="Record for a separately versioned directed-rounding repair and independent checks; do not modify frozen code/criteria, overwrite evidence, recompute runs, or silently upgrade certificate claims",
    )


def verified_controller_recovery(ev, validation, freeze, runs):
    """Verify a separately registered orchestration repair; never execute it."""
    path = validation / "CONTROLLER_RECOVERY_MANIFEST.json"
    manifest = ev.json(path, optional=True)
    if not manifest:
        return None
    expected_python = str(Path(freeze["source_release"]["path"]).parent / ".venv/bin/python")
    recoveries = [r["recovery"]["reference"] for r in runs if r["recovery"]]
    if (manifest.get("schema") != "round2-controller-recovery-v1"
            or manifest.get("compute_source_unchanged") is not True
            or manifest.get("frozen_jobs_unchanged") is not True
            or manifest.get("worker_release") != freeze["source_release"]
            or manifest.get("freeze") != ev.ref(validation / "FREEZE.json")
            or manifest.get("recovery") not in recoveries
            or manifest.get("python") != expected_python):
        raise ValueError("Controller recovery does not preserve frozen compute/job identities")
    source = manifest.get("source", {})
    if (not source.get("path") or not Path(source["path"]).is_absolute()
            or ev.ref(source["path"]) != source):
        raise ValueError("Controller recovery source hash/path/size mismatch")
    ev.bound_json(manifest["worker_release"])
    ev.bound_json(manifest["freeze"])
    ev.bound_json(manifest["recovery"])
    if (not isinstance(manifest.get("unit"), str) or not manifest["unit"]
            or not isinstance(manifest.get("resources"), dict)
            or not manifest.get("launch_utc")):
        raise ValueError("Controller recovery lacks launch/unit/resource provenance")
    if datetime.fromisoformat(manifest["launch_utc"]).tzinfo is None:
        raise ValueError("Controller recovery launch time must include a timezone")
    return dict(reference=ev.ref(path), contents=manifest, identity_chain_verified=True,
                scope="orchestration-only recovery; frozen science release and jobs unchanged")


def oos_evidence(record):
    raw = record["raw"] or record.get("nonoptimization_progress", {})
    return dict(id=record["id"], status=record["status"], passed=record["reported_passed"],
        common_completed=raw.get("common_completed", 0), expected=raw.get("expected", 1000),
        missing_methods=raw.get("missing_methods"), summaries=raw.get("summaries", {}),
        inference_scope=raw.get("inference_scope", "PARTIAL_SAVED_PROGRESS_ONLY" if record["progress_only"] else None),
        result=record["result"], progress=record["progress"], progress_only=record["progress_only"])


def oos_complete_metadata(record):
    """Check the saved six-method completion claim, without recomputing metrics."""
    raw = record["raw"]
    rows = raw.get("records", [])
    valid = (record["job_acceptance_supported"] is True and record["progress_only"] is False
        and raw.get("common_completed") == raw.get("expected") == 1000
        and raw.get("missing_methods") == [] and set(raw.get("methods", {})) == set(METHODS)
        and isinstance(rows, list) and len(rows) == 1000
        and all(isinstance(r, dict) and set(r.get("methods", {})) == set(METHODS) for r in rows))
    if raw.get("passed") is True and not valid:
        raise ValueError(f"OOS passed flag contradicts saved completion metadata: {record['id']}")
    return valid


def bind_review_results(review, runs):
    """A review and its statistics may only describe the same saved RESULT bytes."""
    current = {r["id"]: r for r in runs if r["stage"] != "VREPORT"}
    rows = review["raw"].get("runs", [])
    seen = set()
    for row in rows:
        name = row.get("id")
        if name not in current or name in seen or row.get("result") != current[name]["result"]:
            raise ValueError(f"VREPORT result identity differs from current evidence: {name}")
        seen.add(name)
    if review["reported_passed"] and seen != set(current):
        raise ValueError("Completed VREPORT omitted or duplicated frozen runs")
    return seen


def collect(root, stopped=None):
    """Read and bind evidence. Returns a report payload; does not write files."""
    ev = Evidence()
    config = ev.json(root / "DESIGN_CONFIG.json")
    validation = root / "validation"
    vconfig = ev.json(validation / "DESIGN_CONFIG.json")
    freeze = ev.json(validation / "FREEZE.json")
    if config.get("inherited_charged_seconds") != OLD_COST:
        raise ValueError("Historical development cost changed")
    ledgers = [BudgetLedger(c["ledger_store"], c["budget_group"]) for c in (config, vconfig)]
    dev_budget, val_budget = [ledger_summary(ledger, development=i == 0)
                              for i, ledger in enumerate(ledgers)]
    if (dev_budget["active"] or val_budget["active"]) and not stopped:
        raise RuntimeError("Active/unsettled ledger: refuse report; --stopped REASON only permits a labelled snapshot")
    if dev_budget["spent_seconds"] < OLD_COST:
        raise ValueError("Development ledger lost inherited charges")
    for ledger in ledgers:
        ev.ref(ledger.root / "definition.json")
        for path in sorted((ledger.root / "events").glob("*.json")):
            ev.ref(path)
    release = ev.bound_json(freeze["source_release"])
    jobs = freeze["jobs"]
    if len(jobs) != 30 or len({j["id"] for j in jobs}) != 30:
        raise ValueError("Expected the frozen 30 distinct validation jobs")
    dev = [run_record(ev, root, stage, rid, dev_budget) for stage, rid in DEV_RUNS]
    for record in dev:
        bind_legacy_r0(ev, root, record, dev_budget)
    runs = [run_record(ev, validation, j["stage"], j["id"], val_budget) for j in jobs]
    additional = []
    for base, budget, known, pattern in ((root, dev_budget, {(r["stage"], r["id"]) for r in dev}, "R*/*/RESULT.json"),
                                         (validation, val_budget, {(r["stage"], r["id"]) for r in runs}, "V*/*/RESULT.json")):
        for path in sorted(base.glob(pattern)):
            stage, rid = path.parent.parent.name, path.parent.name
            if (stage, rid) not in known:
                additional.append(run_record(ev, base, stage, rid, budget))
    by_id = {r["id"]: r for r in runs}
    marker = ev.json(validation / "CAMPAIGN_COMPLETE.json", optional=True)
    all_settled = all(r["settled"] for r in runs)
    complete = bool(marker.get("status") == "ALL_FROZEN_JOBS_SETTLED_REVIEW_ACCEPTANCE_SEPARATELY"
                    and all_settled and not val_budget["active"] and not dev_budget["active"])
    if not complete and not stopped:
        raise RuntimeError("Campaign incomplete: no final report without --stopped REASON")
    preparation = next(r for r in runs if r["stage"] == "V0")
    for job, r in zip(jobs, runs, strict=True):
        r["frozen_execution_binding"] = bind_frozen_job(ev, validation, freeze, release, job, r)
        r["frozen_support_binding"] = bind_prepared_support(ev, validation, freeze, job, r, preparation)
    orchestration_recovery = verified_controller_recovery(ev, validation, freeze, runs)
    vreport = next(r for r in runs if r["stage"] == "VREPORT")
    reviewed_ids = bind_review_results(vreport, runs)
    stats = {}
    for seed in freeze["validation_seeds"]:
        ref = vreport["raw"].get("oos_statistics", {}).get(str(seed))
        summary = ev.bound_json(ref) if ref else None
        oos_run = by_id[freeze["oos_run_ids"][str(seed)]]
        if summary and (oos_run["id"] not in reviewed_ids or not oos_run["result"]
                or summary.get("observed_n") != oos_run["raw"].get("common_completed")):
            raise ValueError(f"Statistics lack matching reviewed OOS result: seed{seed}")
        if summary and "formal_ci" in summary and type(summary["formal_ci"]) is not bool:
            raise ValueError(f"Stored formal_ci is not a boolean: seed{seed}")
        if summary and (summary.get("seed") != 910000 + seed or summary.get("expected_n") != 1000
                        or summary.get("replicates") != 500 or summary.get("fixture_override")):
            raise ValueError(f"Stored statistics do not match preregistration: seed{seed}")
        if summary and summary.get("formal_ci") and (summary.get("observed_n") != 1000
                or summary.get("completed_replicates") != 500 or summary.get("missing_method_rows")
                or not oos_complete_metadata(oos_run)):
            raise ValueError(f"Stored CI reports incomplete coverage: seed{seed}")
        stats[str(seed)] = dict(reference=ref, summary=summary,
                               status="AVAILABLE" if ref else "NOT_AVAILABLE_NOT_RECOMPUTED")
    core = [by_id[freeze["core_run_ids"][str(seed)][method]]
            for seed in freeze["validation_seeds"] for method in METHODS]
    pairs, mechanisms, ablations = [], [], []
    for seed in freeze["validation_seeds"]:
        p3 = by_id[freeze["core_run_ids"][str(seed)]["P3"]]
        indiv = by_id[freeze["individual_run_ids"][str(seed)]]
        pairs.append(dict(seed=seed, **compare(p3, indiv)))
        for method, variant in (("P2", "P2"), ("P3-noadvance", "noadvance")):
            other = by_id[freeze["core_run_ids"][str(seed)][method]]
            mechanisms.append(dict(seed=seed, methods=["P3", method], **compare(p3, other, variant=variant)))
    for r in runs:
        if r["stage"] == "VABL":
            seed = r["seed"] or int(next(j for j in jobs if j["id"] == r["id"])["args"][2])
            baseline = by_id[freeze["core_run_ids"][str(seed)]["P3"]]
            ablations.append(dict(seed=seed, **compare(baseline, r)))
    r2, evidence = dev[3], dev[5]["raw"]
    r2["service"] = dict(status="SAVED_R5_OPTIMIZED_SERVICE", metrics=evidence.get("optimized_service", {}))
    physical = r2["raw"].get("physical_screen", {})
    healthy = physical.get("healthy_grid_audit", {})
    healthy_complete = bool(physical.get("passed") is True and healthy.get("passed") is True
        and healthy.get("scenario_count") == 401 and healthy.get("unchanged_demand") is True
        and healthy.get("all_grid_lines_healthy") is True and healthy.get("zero_ev_injection") is True
        and healthy.get("shedding_fixed_zero") is True)
    r0 = compare(dev[0], dev[1])
    entry = ev.json(root / "DEVELOPMENT_ACCEPTANCE.json", optional=True)
    old = {}
    for name in ("CASE_DESIGN_REPORT_CN.md", "CASE_DESIGN_RESULTS.json", "CASE_DESIGN_STATUS.md",
                 "FINAL_BUDGET.json", "CASE_SEARCH_LEDGER.jsonl"):
        old[name] = ev.ref(root.parent / name)
    session = ev.json(root / "ROUND2_SESSION.json")
    now = datetime.now(timezone.utc)
    wall = (now - datetime.fromisoformat(session["started_utc"])).total_seconds()
    wrappers = dev_budget["new_actual_wrapper_wall_seconds"] + val_budget["new_actual_wrapper_wall_seconds"]
    prior_status = root / "CASE_DESIGN_STATUS.md"
    releases = {p.parent.name: dict(reference=ev.ref(p), metadata=ev.json(p))
                for p in sorted((root / "releases").glob("*/RELEASE.json"))}
    for path in (root / "ROUND2_PROTOCOL_CN.md", root / "CASE_SEARCH_LEDGER.jsonl",
                 validation / "CASE_SEARCH_LEDGER.jsonl", root / "DEVELOPMENT_CRITIC_CN.md",
                 root / "PRE_VALIDATION_REVIEW_CN.md", root / "FINAL_EVIDENCE_CRITIC_PRELIM_CN.md",
                 vreport_path(vreport) / "FROZEN_EVIDENCE_HASHES.json"):
        ev.ref(path)
    limitations = certificate_limitations(ev.ref(root / "IMPLEMENTATION_CRITIC_CN.md"))
    all_core_pass = all(r["acceptance_supported"] for r in core)
    findings = validation_findings(core, mechanisms, freeze["core_run_ids"])
    timing_findings = algorithm_timing_findings(pairs, runs)
    mechanism_counts = list(findings["mechanism_comparisons"].values())
    mechanism_status = ("MECHANISM_INTERVALS_REQUIRE_REVIEW" if any(g["contradictory"] for g in mechanism_counts) else
        "SCOPED_GLOBAL_COMPARISONS_AVAILABLE" if all(g["accepted_intervals"] == g["expected"] for g in mechanism_counts) else
        "SOME_SCOPED_GLOBAL_COMPARISONS_AVAILABLE" if any(g["accepted_intervals"] for g in mechanism_counts) else
        "NO_ACCEPTED_MECHANISM_INTERVALS")
    oos = [by_id[freeze["oos_run_ids"][str(seed)]] for seed in freeze["validation_seeds"]]
    formal_stats = all(s["summary"] and s["summary"].get("formal_ci") is True for s in stats.values())
    oos_completion_checks = [oos_complete_metadata(r) for r in oos]
    complete_oos = all(oos_completion_checks)
    contradiction = any(p["status"] == "CONTRADICTORY_INTERVALS" for p in [r0, *pairs, *mechanisms, *ablations])
    statuses = dict(
        adapter_status="SAME_CASE_INTERVAL_AND_ORIGINAL_AUDIT_PASS" if r0["both_accepted"] and r0["status"] != "CONTRADICTORY_INTERVALS" else "INCOMPLETE_OR_REVIEW_REQUIRED",
        fixed_load_full400_status="PASS" if r2["acceptance_supported"] and r2["support"] == 400 else "NOT_ACCEPTED",
        physical_design_status="FIXED_N6_LOAD_HEALTHY_ZERO_EV_ZERO_SHED_PASS" if healthy_complete else "NOT_ESTABLISHED",
        mechanism_status=mechanism_status,
        moment_scope="P3_THREE_MOMENTS_NOT_ORIGINAL_322311",
        algorithm_comparison_status="THREE_MATCHED_PAIRS_ACCEPTED" if all(p["both_accepted"] for p in pairs) and not contradiction else "MATCHED_COMPARISONS_INCOMPLETE_OR_REVIEW_REQUIRED",
        validation_status="COMPLETE_CORE_AND_PAIRED_OOS_EVIDENCE" if complete and all_core_pass and complete_oos and formal_stats and vreport["reported_passed"] and not contradiction and dev_budget["within_limit"] and val_budget["within_limit"] else
                          "SETTLED_WITH_LIMITATIONS" if complete else "EXPLICIT_STOPPED_INCOMPLETE_SNAPSHOT")
    references = [dict(name=p.get("policy", {}).get("name"),
                       first_cost=p.get("policy", {}).get("first_stage_cost"),
                       total=p.get("total"), recourse_risk=p.get("recourse_risk"))
                  for p in evidence.get("policies", [])]
    result = dict(schema="round2-read-only-report-v1", generated_utc=now.isoformat(), root=str(root),
        **statuses, stopped_reason=stopped, campaign_complete=complete, frozen_job_count=len(jobs),
        frozen_job_coverage=job_coverage(runs),
        all_frozen_jobs_attempted=all(r["attempted"] for r in runs),
        all_frozen_jobs_accepted=all(r["job_acceptance_supported"] for r in runs),
        all_frozen_jobs_settled=all_settled, all_frozen_jobs_reported_passed=all(r["reported_passed"] for r in runs),
        report_is_new_scientific_certificate=False, contradiction_requires_review=contradiction,
        known_certificate_limitations=limitations,
        validation_findings=findings, algorithm_timing_evidence=timing_findings,
        recovered_failed_jobs=[r["id"] for r in runs if r["recovery"]],
        orchestration_recovery_manifest=orchestration_recovery,
        development=[public_run(r) for r in dev], validation=[public_run(r) for r in runs],
        additional_runs_not_substituted_for_frozen_jobs=[public_run(r) for r in additional],
        core_run_order=[r["id"] for r in core], algorithm_pairs=pairs, no_prepass_comparisons=ablations,
        mechanism_intervals=mechanisms, R0_comparison=r0, reference_policy_evaluations=references,
        reference_policies_are_global_optima=False, oos_statistics=stats,
        oos=[oos_evidence(r) for r in oos],
        VREPORT=dict(result=vreport["result"], status=vreport["status"], passed=vreport["reported_passed"],
                     comparisons=vreport["raw"].get("comparisons", [])),
        budget=dict(development=dev_budget, validation=val_budget, units="wall seconds",
            software_work=dict(session_wall_seconds=wall, new_actual_wrapper_wall_seconds=wrappers,
                session_wall_minus_new_wrapper_wall_seconds=wall - wrappers, exact_active_coding_seconds=None,
                note="Wall difference includes reading/coding/deployment/waiting and unmeasured interrupted-worker time; coding can overlap wrappers. Conservative reservation debits are not observed wall time. Not additive CPU/active labor time; old211.081997 is not subtracted again.")),
        runtime_environment_scope=dict(existing_healthy_DET_task_colocated_during_part_of_campaign=True,
            legacy_finalization_observation=ev.json(root / "observations/LEGACY_FINALIZATION_20261112T2042Z.json", optional=True),
            legacy_saved_audit_observation=ev.json(root / "observations/LEGACY_AUDIT_20261112T2042Z.json", optional=True),
            legacy_scope="Read-only legacy finalization metadata; no adoption, restart, new solve or independent recertification",
            dedicated_server_claim=False, cpu_pinning_claim=False,
            timing_ratios="Observed single same-case/budget runs; background load is not experimentally controlled; no universal speedup or runtime CI claim"),
        memory_scope=dict(R2_per_run_MemoryPeak_bytes=None,
            R2_reason="Original unit garbage-collected; MemoryPeak unavailable",
            per_run_peaks_not_inferred=True,
            campaign_MemoryPeak_scope="Whole campaign only if separately recorded; never apportioned to individual runs"),
        freeze=dict(reference=ev.ref(validation / "FREEZE.json"), contents=freeze),
        release_at_validation_freeze=release, releases=releases, development_entry=entry,
        preserved_round1=old, R3_R4=dict(status=entry.get("physical_revision", "NOT_RECORDED"),
            reason=entry.get("skip_reason"), allowed_limited_scope=True),
        physical_summary=dict(passed=physical.get("passed"), healthy_passed=healthy.get("passed"),
            healthy_scenarios=healthy.get("scenario_count"), details_source=r2["result"],
            healthy_zero_ev_zero_shed_metadata_complete=healthy_complete,
            healthy_max_residual=healthy.get("max_residual"),
            physical_load_fleet_size=r2["raw"].get("case_identity", {}).get("physical_load_fleet_size")),
        probability_features=r2["raw"].get("probability_features", {}), probability_domains=evidence.get("domains", {}),
        reporter_source=ev.ref(Path(__file__).resolve()),
        previous_round2_status=dict(reference=ev.ref(prior_status), text=prior_status.read_text() if prior_status.exists() else None),
        scope_limits=["Original network/energy kernels retained; new type/SOC/catalogue/weather/load distribution assumptions.",
            "R0 direct extensive and decomposed solvers share physical catalogue/recourse kernels; their assembly and bound comparison is not a fully independent physical implementation audit.",
            "Catalogue integer equivalence is internal to the declared restricted catalogue, not lossless equivalence to the old task model or LP relaxation.",
            "P3 has three calibrated moments; 400 scenarios do not restore the original 322311 rows.",
            "DET optimizes the forecast singleton; other methods optimize 400 scenarios. Their training objectives are not a common ranking.",
            "Lower bounds are native or finite-box numerical certificates; upper values combine numerically audited recourse with risk-dual certificates, not exact-rational feasibility proofs.",
            "Known local rounding defect: qL/risk_L and qL-derived OOS lower endpoints are not strictly downward certified; no frozen master LB/UB failure has been established by this review. Pending independent-version repair.",
            "Healthy zero-EV/zero-shedding audits use fixed N6 demand. Optional N30 capacity is not large service demand.",
            "Stress G changes physical types/SOC and seed, and its 600s whole-worker caps differ from 1000s core caps; it is not a pure representation scaling experiment. Timeout alone cannot identify a type-count effect.",
            "R04/r05 historical oracle settings were not all capped at60; later releases cap calls. Do not relabel old execution.",
            "Bootstrap eligibility/CI are copied from VREPORT; descriptive OOS summaries are copied from saved OOS RESULT files. Conditional on fixed training policies; no probability-set or simultaneous coverage claim."],
        paper_gaps=["高服务规模：按各seed实际选车/合同/站点及放电证据判断；扩充候选池或full400通过不等于容量受限的多区域服务已得到验证。",
                    "丰富moments：P3仍只有三行，不恢复原322311行；这需要独立科学版本与证据，不能由表示或审计工程优化替代。"],
        next_improvement=next_improvement(timing_findings))
    result["evidence"] = list(ev.refs.values())
    # A read-only snapshot must not silently mix ledger states during assembly.
    for ledger, budget in zip(ledgers, (dev_budget, val_budget), strict=True):
        if ledger.read()["previous"] != budget["chain_head"]:
            raise RuntimeError("Ledger changed while report was being assembled; retry after settlement")
    return result


def vreport_path(record):
    return Path(record["directory"])


def fmt(value):
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.9g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def table(headers, rows):
    def clean(value):
        return fmt(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(clean(v) for v in row) + " |" for row in rows])


def evidence_link(ref, label="证据", *, execution_root=None, display_root=None):
    if not ref:
        return "未记录"
    path = Path(ref["path"])
    if display_root is not None:
        # Includes preserved Round1 siblings. This is presentation only; JSON
        # references and all reproduction commands retain execution-host paths.
        try:
            path = Path(display_root).parent / path.relative_to(Path(execution_root).parent)
        except ValueError:
            pass
    return f"[{label}](<{path}>)"


def run_table(rows):
    return table(["Run", "原状态 / 完成分类", "job accepted", "gap_passed / audit_complete / acceptance_supported",
                  "LB / UB ($)", "gap ($)", "原审计/支持", "原残差", "observed wrapper / final worker / saved controller (s)",
                  "charged(s) / accounting basis"],
        [[r["id"], f"{r['status']} / {r['completion_class']}" if r["optimization_job"] else f"{r['status']} / N/A（非优化作业）",
          r["job_acceptance_supported"],
          [r["gap_passed"], r["audit_complete"], r["acceptance_supported"]] if r["optimization_job"] else "N/A",
          [r["LB"], r["UB"]] if r["optimization_job"] else "N/A", r["gap"] if r["optimization_job"] else "N/A",
          f"{r['original_audit_count']}/{r['support']}" if r["optimization_job"] else "N/A",
          r["original_max_residual"] if r["optimization_job"] else "N/A",
          [r["wrapper_seconds"], r["worker_seconds"], r["controller_seconds"]],
          [r["charged_seconds"], r["accounting_basis"]]] for r in rows])


def findings_prose(findings):
    p3, aro = findings["p3_full400"], findings["aro"]
    mechanisms = []
    for method, group in findings["mechanism_comparisons"].items():
        mechanisms.append(f"P3−{method}有{group['accepted_intervals']}/{group['expected']}组双方接受且身份匹配的值差区间，"
            f"其中{group['strictly_negative_beyond_1e_5']}组在保存的数值界证据下与零分离（上端小于−1e-5）；"
            f"{group['unresolved']}组未确立（含{group['contradictory']}组需矛盾复核）")
    deployments = []
    for item in p3["deployments"]:
        stations = item["selected_stations"]
        deployments.append(f"seed{item['seed']}选{fmt(item['selected_vehicles'])}车、合同{fmt(item['contract_kwh'])}kWh、"
            f"{len(stations) if isinstance(stations, list) else '未记录'}站")
    return dict(p3=f"验证P3已有{p3['accepted']}/{p3['expected']}组full400通过项目gap门及全部原逐车审计"
                f"（seed：{fmt(p3['accepted_seeds'])}）。这是数值审计接受，不是严格有理证明。",
        mechanisms="；".join(mechanisms) + "。这些比较分别计数，不要求其他核心方法全部通过，也不表示统计独立或OOS优势。",
        deployment="；".join(deployments) + "。合同能量不等于实际放电量。" if deployments else "尚无已接受P3部署证据。",
        aro=f"ARO完整接受{aro['accepted']}/{aro['expected']}组；未接受的ARO及其他作业继续保留原状态，不能据此抹去已完成的P3和机制证据。")


def timing_prose(timing):
    usable = [r for r in timing["pairs"] if r["usable_timing_comparison"]]
    wrappers = [r for r in usable if r["count_end_to_end_faster_observed"] is not None]
    matched = [r for r in timing["pairs"] if r["identity_matched"] and r["same_reserved_budget"]]
    return (f"同问题、同预留预算的{len(matched)}组中，count完整接受{sum(r['count_accepted'] for r in matched)}组，"
        f"individual完整接受{sum(r['individual_accepted'] for r in matched)}组；其余资源中断与gap/审计超时分列，失败不生成成功时间比。"
        f"{len(usable)}/{len(timing['pairs'])}组同问题、同预留预算且双方接受的计时可比；"
        f"count含构建/审计的controller总时间较短{sum(r['count_controller_faster_observed'] for r in usable)}组、"
        f"未较短{sum(not r['count_controller_faster_observed'] for r in usable)}组。"
        f"有实际wrapper计时的{len(wrappers)}组中，count端到端较短{sum(r['count_end_to_end_faster_observed'] for r in wrappers)}组。"
        "实际gap和所选政策可不同，采用同一max(1美元,1%×max(1,|UB|))规则；不称为相同终止gap的受控速度实验。")


def certificate_section(data, *, display_root=None):
    limitation = data["known_certificate_limitations"]
    critic = evidence_link(limitation["critic"], "IMPLEMENTATION_CRITIC_CN.md",
                           execution_root=data["root"], display_root=display_root)
    return "\n\n".join([
        "### 已知局部证书舍入缺陷（待独立版本修复）",
        "只读审查：" + critic + "。该审查是源码与解析反例审查，不是冻结release逐字节一致性认证，也未重放validation/OOS。",
        "已确认：有效仿射割的普通舍入标量求值`qL`存在略高于真实Q的解析反例。因此`qL`、由其导出的`risk_L`及固定政策/OOS区间的qL下端，均不能称为严格向下证书或保证包围真实最优值的下界。解析反例有相应平台精度条件，不等于已在本冻结算例中观测到该偏差。",
        "影响范围须区分：主master LB及存储的alpha/beta割不直接使用该qL标量；概率分离虽使用qL目标，加入的概率行仍须通过完整域成员检查。风险UB的对偶平面依据全部qU上移并检查支配。此次审查未发现全局界的结构性blocker，也未证明冻结主LB/UB已失效；这同样不能反过来宣称所有数值证书均严格有效。",
        "保留`NUMERICALLY_AUDITED_ONLY`证据边界：原primal残差门、native MIP bound及有浮点保护的风险对偶，不组成端到端严格有理证明。OOS的qL/qU端点表仍按原值列出，但不是严格数值包围区间；bootstrap抽样CI与这些数值端点是不同口径。",
        "处置：作为待独立版本修复并验证的局部舍入缺陷记录；不修改冻结源码/标准，不调整原验收状态，不重算或覆盖既有证据。",
    ])


def render_report(data, *, display_root=None):
    def link(ref, label="证据"):
        return evidence_link(ref, label, execution_root=data["root"], display_root=display_root)

    def phase_time(record, key):
        return record["timing_seconds"].get(key) if record["optimization_job"] else "N/A"

    by_id = {r["id"]: r for r in data["validation"]}
    core = [by_id[k] for k in data["core_run_order"]]
    r2 = data["development"][3]
    budget = data["budget"]
    service = r2["service"].get("metrics", {})
    findings = findings_prose(data["validation_findings"])
    questions = [
        ("新分解器是否实际运行、正确性对照是否通过？", data["adapter_status"] + "；R0比较合法区间与原矩阵审计，不把项目gap通过写成两个精确最优值证明。"),
        ("固定负荷30车/full400是否得到完整证书？", findings["p3"] + f" 开发R2：{data['fixed_load_full400_status']}；LB={fmt(r2['LB'])}，UB={fmt(r2['UB'])}，gap={fmt(r2['gap'])}；原审计{r2['original_audit_count']}/400。"),
        ("实际选用和放电车辆有多少，规模压力是否充分？", findings["deployment"] + f" 开发R2选用{fmt(r2['deployment'].get('selected_vehicles'))}车；名义平均放电车数{fmt(service.get('discharging_vehicles', {}).get('nominal_weighted'))}，AC放电{fmt(service.get('discharge_AC_kwh', {}).get('nominal_weighted'))}kWh。候选池扩充不代表大服务规模。"),
        ("容量约束修订是否执行、是否通过健康网和完整优化？", f"{data['R3_R4']['status']}；{data['R3_R4']['reason']}。固定需求健康网：{data['physical_design_status']}。"),
        ("全局政策是否支持提前部署/联合信息机制？", findings["mechanisms"] + "固定三政策排序不能替代全局结果。"),
        ("算法速度结论基于哪一概率集、哪一同问题对照？", timing_prose(data["algorithm_timing_evidence"]) + "仅P3三矩、同case/support/fees的计数与逐车表示。"),
        ("开发总账及验证总账分别用了多少？", f"开发{budget['development']['spent_seconds']!r}/7200秒（已含旧{OLD_COST!r}）；验证{budget['validation']['spent_seconds']!r}/28800秒。"),
        ("是否可以进入论文主实验，依据及剩余限制是什么？", findings["aro"] + "可引用已验收的受限合成诊断；原丰富矩模型、高服务规模、普遍加速和完整六方法OOS证据分别判断。整轮状态：" + data["validation_status"] + "。论文采用仍需研究者判断。"),
    ]
    parts = ["# Round 2 实例设计与验证报告", "\n".join(f"{i}. **{q}** {a}" for i, (q, a) in enumerate(questions, 1)),
             f"报告时间：{data['generated_utc']}。显式停止原因：{fmt(data['stopped_reason'])}。campaign完成={data['campaign_complete']}；本文件只整理已有证据，不新增科学证书。Markdown链接显示根={fmt(display_root)}（仅展示映射，证据仍为实际执行路径）。",
             "冻结job覆盖：" + fmt(data["frozen_job_coverage"]) + "。attempted以匹配CLAIM定义（不保证已调用native求解）；settled表示账本结算；accepted按各job自己的门判断。全部尝试/全部结算不等于全部接受，VREPORT本身通过也不改变失败的科学Run。",
             "结算可来自真实WAIT，或已核验的RECOVERY→CLAIM/PROGRESS/死亡观测/账本settlement身份链。`RESOURCE_INTERRUPTED_OOM`是已保守结算的失败，不伪造RESULT/WAIT或worker完整收尾；原PROGRESS中的界与审计只作为最后保存快照。",
             "## 科学身份与解释范围",
             "四项费用乘数均须为[1,1,1,1]。N30/G6/72套餐/96时段；物理需求固定在N6水平。P3仅L/R/LR三行，P2删除联合行；**不是恢复原322311行ST-DRO**。DET是forecast支持1，其他核心方法是400支持；SAA使用原分层权重，ARO为完整simplex。CAL2000逐坐标Wilson区间没有同时覆盖保证。",
             "新类型与初始SOC、有限套餐、共同工程天气、线路20/23短恢复与corridor1封锁、负荷及预测均是新合成假设。保留原P/Q、电压、线路、能量、到达/资格内核不等于保持原分布或完整原任务域。整数聚合等价只在声明的同类套餐域内成立，不主张原LP松弛等价或对旧模型无损。",
             "健康网核对：" + fmt(data["physical_summary"]) + "。同负荷、线路全健康、EV注入0、削减0；不靠提高线路限额掩盖过载。",
             "数值口径：原行容差1e-9、目标分项1e-5、gap阈值max(1美元,1%×max(1,|UB|))。LB来自native bound或有限盒证书；UB依赖数值审计原recourse及风险对偶，**不是精确有理数物理可行性证明**。固定政策风险下界不是全局LB。",
             certificate_section(data, display_root=display_root),
             "**gap闭合、原矩阵审计完成、最终接受分开报告。** gap_passed只核对已保存LB/UB达到项目阈值；audit_complete还要求完整匹配支持和1e-9原残差；acceptance_supported保留正式接受与结算条件，不要求每个native子调用都返回OPTIMAL。TIME_LIMIT可能发生在原审计而非优化搜索：`GAP_PASSED_AUDIT_INCOMPLETE`不是gap未收敛，但仍未达到完整验收。原status/native_status不改写。",
             "非优化作业（准备、固定政策评价、OOS与汇总）的全局LB/UB、gap及优化原审计列为N/A；其job状态与自身验收单列，不因这些字段不适用而记为界失败。",
             "报告额外核对完整CLAIM与账本payload、整数退出码、冻结source/argv/wrapper/config身份，以及V0中原始SID/权重和方法支持。仅核验metadata及字节哈希，不反序列化CASE/NPZ或重算模型。非有限、负值、缺失分量及非法残差不能被其他0分量抵消；异常布尔标志不当作接受。",
             "审计覆盖采用保存的original_audit列表长度；audit_completed是可能滞后的progress计数，单独保留而不替代实际行数。即使gap已闭合，未完成原审计的政策仍不满足冻结OOS准入；缺任一方法时全部六方法的正式CI保持NOT_ELIGIBLE，不事后放宽。",
             "若预算停止的最终RESULT未包含optimization，报告从PROGRESS读取最后保存的界及审计，标记progress_only=true并保留双方文件哈希；它不是完整优化收尾或最终审计覆盖，不能支持接受。",
             "## 开发全局结果与失败历史", run_table(data["development"]),
             "额外Run（若存在只列历史，不替换冻结job）：", run_table(data["additional_runs_not_substituted_for_frozen_jobs"]),
             "R0区间对照：" + fmt(data["R0_comparison"]),
             "R0直接有限场景模型与分解器共享物理catalogue/recourse内核；该对照验证不同优化组装路径、界与原逐车展开的一致性，不构成两套完全独立物理实现的交叉验证。",
             "R0的legacy-r0-all-packages-v1仅适用于decompose-r04/direct-r05：确切CLAIM/RESULT/CASE/概率域、worker、wheel及ZIP成员哈希核对后，由已审阅源码推导全72套餐域。original first_stage_restriction缺失仍保留，effective字段不回填原记录；旧CLAIM缺argv/release/freeze，r04无RELEASE.json也明确保留。任何身份不符拒绝适配，不推广到其他Run，不声称执行时模块取证或新科学证明。",
             table(["R0 Run", "源码文件或wheel成员", "审阅行（含首尾）", "SHA256", "域证据"],
                   [[r["id"], link(s.get("reference", s.get("archive")), s.get("member", "round2_worker.py")),
                     s["lines"], s.get("sha256", s.get("reference", {}).get("sha256")), s["finding"]]
                    for r in data["development"] for s in r.get("restriction_source_review", {}).get("source_review", [])]),
             "R5/evidence400-r06的float residual元数据读取错误发生在新增参考LP之前；r07修复后的结果单列，旧失败及费用保留。Round1负荷缩放设计门失败原报告不改写。R3/R4是允许跳过的有限范围，不记为容量修订PASS。",
             table(["Run", "错误/原因", "证据"], [[r["id"], r["error"], link(r["result"])] for r in data["development"] if not r["reported_passed"]]),
             "## 三参考政策：固定政策评价，不是全局排名",
             "本表沿用保存的数值端点；qL/risk_L相关下端受上述局部舍入缺陷限制，不称为严格政策下界，与主master LB分开。",
             table(["政策", "F", "DET", "SAA", "ARO", "P3", "P2"],
                   [[r["name"], r["first_cost"], *[(r["total"] or {}).get(k) for k in ("DET", "SAA", "ARO", "P3-DRO", "P2-DRO")]] for r in data["reference_policy_evaluations"]]),
             "保留全局优化发现的第四种政策。实际DET部署见核心表（已观察到的两车策略也不应被三参考表覆盖）；这些评价规则不同，不能把各自训练目标直接作共同OOS排名。",
             "## 18个核心全局结果（含超时/缺失）", run_table(core),
             table(["Run", "最后保存master native status", "optimization reason（原文）", "顶层错误"],
                   [[r["id"], r["last_master_native_status"], r["optimization_reason"], r["raw_error"]] for r in core]),
             "master native status只描述最后保存的master，不代表全部recourse调用OPTIMAL；终止原因中的oracle状态独立保留。原有完整支持整数候选UB不会因最后一轮未完成而改称不存在，但原审计不完整仍不满足冻结OOS门。",
             table(["Run", "选车", "合同(kWh)", "提前(车·小时，相对16:00)", "站点/有效类型", "平均放电车/AC kWh", "服务口径"],
                [[r["id"], r["deployment"].get("selected_vehicles"), r["deployment"].get("contract_kwh"),
                  r["deployment"].get("total_advance_vehicle_hours"),
                  [r["deployment"].get("selected_stations"), r["deployment"].get("effective_selected_types")],
                  [r["service"].get("metrics", {}).get(k, {}).get("nominal_weighted") for k in ("discharging_vehicles", "discharge_AC_kwh")],
                  r["service"].get("status")] for r in core]),
             "提前小时只描述相对16:00的部署时点；原费用30*(r+1-d)/4不是单独的提前小时费。DET服务统计是forecast，其他是TRAIN400，正式共同政策比较使用后面的OOS。",
             "## 同问题算法与消融", run_table([r for r in data["validation"] if r["stage"] in ("VIND", "VABL")]),
             table(["seed", "count/individual", "身份一致/双方接受", "值差区间(count−individual)", "controller秒比individual/count"],
                   [[p["seed"], [p["a"], p["b"]], [p["identity_matched"], p["both_accepted"]], p["difference_a_minus_b"], p["controller_time_ratio_b_over_a"]] for p in data["algorithm_pairs"]]),
             table(["seed", "同预算且双方接受可比", "实际gap(count/individual)", "gap门限(count/individual)",
                    "controller总秒(count/individual)", "build秒(count/individual)", "audit秒(count/individual)",
                    "实测wrapper秒比individual/count", "count端到端较短"],
                   [[p["seed"], p["usable_timing_comparison"], p["actual_gap_count_individual"],
                     p["gap_threshold_count_individual"], p["controller_seconds_count_individual"],
                     p["build_seconds_count_individual"], p["audit_seconds_count_individual"],
                     p["observed_wrapper_ratio_individual_over_count"], p["count_end_to_end_faster_observed"]]
                    for p in data["algorithm_timing_evidence"]["pairs"]]),
             timing_prose(data["algorithm_timing_evidence"]),
             table(["seed", "prepass/no-prepass", "状态", "值差区间", "秒比no-prepass/prepass"],
                   [[p["seed"], [p["a"], p["b"]], p["status"], p["difference_a_minus_b"], p["controller_time_ratio_b_over_a"]] for p in data["no_prepass_comparisons"]]),
             "时间比是同问题、同预算条件下单次运行的观测比；旧DET曾在同机并行，最新只读WAIT记录其于18:54 UTC正常结束（返回码0），范围仅FORECAST44_DET_ONLY。该旧结果不属于本轮计算或独立复核。没有CPU绑核或独占服务器保证，后台负载随时间变化且未受实验控制，不能据此给普遍加速结论或受控负载的时间CI。两方未接受、身份不一致或缺计时则null。不能用旧322311行问题作分母归因类型聚合。",
             "原任务完成观察（与本轮科学结果分开）：" + fmt(data["runtime_environment_scope"]),
             "## P3/P2与无提前的配对值差",
             table(["seed", "P3−对照", "身份范围", "区间($)", "状态"],
                   [[p["seed"], p["methods"], p["scope"], p["difference_a_minus_b"], p["status"]] for p in data["mechanism_intervals"]]),
             "仅对相同seed/物理case/支持/费用给区间；P2故意改变概率域，无提前故意限制第一阶段域，它们不是同一优化问题的算法对照。区间跨0时未辨识严格值差，P3较小的嵌套风险目标不自动证明OOS更好。",
             "## G=3/15/30压力：保留严格失败", run_table([r for r in data["validation"] if r["stage"] == "VSTRESS"]),
             table(["Run", "顶层错误", "physical_screen.errors / pending_checks / status", "证据"],
                   [[r["id"], r["error"], {k: r["physical_screen"].get(k) for k in ("errors", "pending_checks", "status")},
                     link(r["result"])] for r in data["validation"] if r["stage"] == "VSTRESS"]),
             "这些运行改变物理类型/SOC，stress seed2101与验证三seed不同；G6参考也不是同seed另一次严格控制试验。压力测试每项600秒包含实例准备，核心P3每项1000秒且共用V0准备；预算和准备口径也不同。不可删除失败或称为纯表示缩放曲线，超时不能单独归因类型数。",
             "## 三组配对OOS与已保存的bootstrap区间",
             "只有完整1000样本及全部六方法时给正式CI。共同前缀仅描述，不把缺失填0。500次四层配对bootstrap，保留层样本数/质量，seed=910000+validationseed；条件于已冻结训练政策，不包含训练不确定性或概率集覆盖，也没有多重比较校正。成本采用F+qU，保存的qL/qU数值端点另列；qL端未获严格向下认证，不能将端点表当作严格包围区间。"]
    for seed, item in data["oos_statistics"].items():
        run = next(r for r in data["oos"] if r["id"] == data["freeze"]["contents"]["oos_run_ids"][seed])
        parts += [f"### seed {seed}：{run['status']}，{run['common_completed']}/{run['expected']}",
            "缺失方法：" + fmt(run["missing_methods"]) + "；口径：" + fmt(run["inference_scope"]),
            table(["方法", "均值成本 / 保存端点(非严格包围)", "CVaR95 / 保存端点(非严格包围)", "关键ENS均值(kWh)"],
                  [[m, [s.get("mean_cost"), s.get("mean_cost_interval")],
                    [s.get("cvar95_cost"), s.get("cvar95_cost_interval")], s.get("critical_ENS_kwh")]
                   for m, s in run["summaries"].items()])]
        stats = item["summary"] or {}
        parts += ["统计状态：" + stats.get("status", item["status"]) + "；" + link(item["reference"], "已保存统计"),
            table(["P3−对照", "指标", "点差", "percentile95% CI"],
                  [[method, metric, answer.get("difference"), answer.get("ci95")]
                   for method, metrics in stats.get("comparisons", {}).items() for metric, answer in metrics.items()])]
    parts += ["## 计时、模型大小与累计预算",
        table(["Run", "最后保存阶段 / controller累计(s)", "准备计时(s)", "build(s)", "master / direct native(s)",
               "recourse(s)", "probability(s)", "certificate+IO(s)", "原audit(s)", "最大已记录阶段"],
              [[r["id"], [r["last_saved_phase"], r["controller_seconds"]] if r["optimization_job"] else f"N/A；job={r['status']}",
                r["preparation_timing_seconds"], phase_time(r, "build"), [phase_time(r, "master"), phase_time(r, "native")],
                phase_time(r, "recourse"), phase_time(r, "probability"),
                phase_time(r, "certificate_and_io"), phase_time(r, "audit"),
                r["largest_recorded_timing_part"] if r["optimization_job"] else "N/A"]
               for r in [*data["development"], *data["validation"]]]),
        "阶段判断只依据保存的审计行/计数/计时及current_evaluation，不是新观察。最后快照可能落后；进行中audit计时若尚未写回，即使显示0也不表示审计没有耗时。最大已记录项只比较已有计时，不当作完整实际瓶颈证明。首次gap闭合时刻未单独保存则null，不用总elapsed减audit时间反推。",
        table(["Run", "阶段计时(s，原字段)", "实际master大小", "calls/attempts"],
              [[r["id"], r["timing_seconds"], r["master_size"], r["counters"]]
               for r in [*data["development"], *data["validation"]] if r["controller_seconds"] is not None]),
        "controller计时包含构建/证书/审计等；recourse项不等于纯native求解时间。wrapper、worker、controller是嵌套口径，不可相加。早期r04/r05失败调用及中断审计计时可能不完整；总账仍计整个wrapper。原始/聚合实际rows/vars/W/T nnz详见JSON各Run model_sizes及原证据；不外推缺失大小。",
        "内存口径：R2 unit已被GC，其MemoryPeak不可取得，记null而非0。若另有campaign MemoryPeak，它是整个campaign的峰值，不是逐Run峰值；不按运行时长分摊，也不编造任何per-run peak。",
        "### 资源中断与保守结算",
        table(["Run", "恢复身份链", "扣费(s)/依据", "observed wrapper(s)", "最后保存审计/支持", "campaign unit MemoryPeak/MemoryMax(bytes)", "观测"],
              [[r["id"], link(r["recovery"]["reference"], "RECOVERY"), [r["charged_seconds"], r["accounting_basis"]],
                r["wrapper_seconds"], f"{r['original_audit_count']}/{r['support']}",
                [r["recovery"]["observation"]["campaign_unit_memory_peak_bytes"],
                 r["recovery"]["observation"]["campaign_unit_memory_limit_bytes"]],
                link(r["recovery"]["observation_reference"], "OBSERVATION")]
               for r in [*data["development"], *data["validation"]] if r["recovery"]]),
        "保守结算全额预留并非测得的wrapper墙钟：actual_parent_wait=false，observed wrapper秒数为null，charged_seconds单列。OOM证据来自绑定的unit/journal与owner/child死亡观测，不把campaign峰值当单Run峰值；峰值低于MemoryMax也不能单凭它排除systemd-oomd或认定精确压力原因。此失败不解释为数学非收敛，亦不提升科学验收。",
        table(["预算", "累计(s)", "上限(s)", "余额(s)", "活动claim", "分阶段累计(s)"],
              [[name, b["spent_seconds"], b["limit_seconds"], b["remaining_seconds"], b["active"], b["stage_charged_seconds"]]
               for name, b in (("development", budget["development"]), ("validation", budget["validation"]))]),
        "软件/阅读墙钟口径：" + fmt(budget["software_work"]),
        "旧211.08199704252183秒已包含在开发总账，不再次相加；失败、重试、构建、审计、外部扣账保留。精确active coding时间为null；session减新wrapper墙钟不是可加的人工工时，存在并行重叠。",
        "## 完整冻结job账与证据", table(["stage/id", "状态", "attempted / settled / job accepted", "reported passed", "退出码", "RESULT或RECOVERY"],
              [[r["stage"] + "/" + r["id"], r["status"], [r["attempted"], r["settled"], r["job_acceptance_supported"]],
                r["reported_passed"], r["returncode"], link(r["result"] or (r["recovery"] or {}).get("reference"))] for r in data["validation"]]),
        "VREPORT的passed表示证据汇总完成，**不是所有实验通过**。状态字段、原失败、所有时间和未决值均保留。",
        "编排恢复登记（不替换冻结worker）：" + fmt(data["orchestration_recovery_manifest"]),
        "冻结源码：" + fmt(data["release_at_validation_freeze"]) + "。报告脚本：" + fmt(data["reporter_source"]),
        table(["证据", "SHA256", "bytes"], [[link(r, Path(r["path"]).name), r["sha256"], r["bytes"]] for r in data["evidence"]]),
        "## 下一步的单项改进与论文缺口", data["next_improvement"],
        "工程优先排查依据（非验收门、非因果证明）：" + fmt(data["algorithm_timing_evidence"]["prioritization_rule"]),
        *data["paper_gaps"],
        "本报告不启动任何新任务，不批准发布，也不改写原Round1结论、冻结配置或源码。复现及安全恢复见REPRODUCE_CN.md。"]
    return "\n\n".join(parts) + "\n"


def render_reproduce(data):
    root = Path(data["root"])
    release = Path(data["freeze"]["contents"]["source_release"]["path"]).parent
    controller = [str(release / ".venv/bin/python"), str(release / "validation_campaign.py"),
                  str(root / "validation"), str(release)]
    recovery_manifest = data.get("orchestration_recovery_manifest")
    if recovery_manifest:
        manifest = recovery_manifest["contents"]
        controller = [manifest["python"], manifest["source"]["path"],
                      str(root / "validation"), str(release)]
    parts = ["# Round2 复现与恢复说明", "以下命令仅供定位/复现，不由报告执行。已结算任务不得重复计算；不得重建/清零账本或修改原release。离线副本不能代替执行主机的ledger。",
        "## 只读定位", f"根：`{root}`；配置：`{root / 'DESIGN_CONFIG.json'}`与`{root / 'validation/DESIGN_CONFIG.json'}`。",
        "两个预算组：" + fmt({k: {f: data["budget"][k][f] for f in ("group", "ledger_root", "chain_head")} for k in ("development", "validation")}),
        "先读取实际ledger active、CLAIM/CHILD/WAIT与hash链。活动或未知任务不当作已死；报告脚本不恢复/接管任何PID。",
        "## 按已登记编排入口、同一冻结worker与账本恢复（仅在确认无活动任务且有合法未执行job时）",
        ("当前命令来自已核验的CONTROLLER_RECOVERY_MANIFEST：" + fmt(recovery_manifest)
         + "。仅更换编排入口，Python及worker release/FREEZE保持原身份；恢复Run保留为失败结算，不重算。"
         if recovery_manifest else "未登记新的编排恢复manifest；以下保留原r08控制器入口。"),
        "```bash\n" + shlex.join(controller) + "\n```",
        "该控制器按冻结30-job列表读取已完成Run的argv/release/FREEZE/配置及settlement receipt；已有未结算目录会拒绝重提。全部已结算时复用结果，不重复求解；实现错误必须先按既有规则处理，不能换seed/费用/模型绕过。",
        ("禁止在已恢复的controller仍活动时再执行此命令；身份未知仍须先核对。"
         if recovery_manifest else
         "若存在无WAIT的RECOVERY结算，上述原r08控制器会拒绝该目录：它不是可直接执行的恢复命令。报告识别恢复链不等于控制器已支持恢复；恢复剩余job须使用另行验证的调度恢复入口，保留同一科学release、冻结列表与账本，不重跑中断job、不伪造WAIT/RESULT。"),
        "## 历史R0/R1/R2命令与版本（记录用途，禁止重复提交同Run）"]
    for r in data["development"]:
        identity = r["claim"].get("run", {})
        argv = identity.get("argv")
        parts += [f"### {r['stage']}/{r['id']}", "源码/原命令证据：" + fmt(identity)]
        if argv:
            command = [argv[0], identity["wrapper"]["path"], str(root), r["stage"], r["id"],
                       argv[1], argv[0], *argv[2:]]
            parts += ["完整wrapper调用（来自已保存argv）：", "```bash\n" + shlex.join(command) + "\n```"]
        else:
            # Early wrappers saved only argv hashes. Do not invent an 'exact'
            # command, input path or time allocation that was never recorded.
            parts += ["早期CLAIM未保存明文argv；CHILD只保存argv_sha256，因此完整原命令不可由本记录精确恢复。保留源码/配置/hash/Run目录，以下仅是接口模板，不冒充实际执行命令：",
                "```text\nPYTHON stage.py ROOT STAGE RUN_ID WORKER PYTHON MODE INPUT [SUPPORT]\n```"]
            source = identity.get("source", {}).get("path")
            wrapper = identity.get("wrapper", {}).get("path")
            mode = r.get("mode")
            original_input = (r.get("original_frozen_case") or {}).get("path")
            if mode == "fixed":
                original_input = next((j["args"][0] for j in data["freeze"]["contents"]["jobs"]
                                       if j["stage"] == "V0"), None)
            if source and wrapper and mode in ("decompose", "direct", "fixed") and original_input:
                python = str(Path(source).parent / ".venv/bin/python")
                rebuilt = [python, wrapper, str(root), r["stage"], r["id"], source, python, mode, original_input]
                if mode == "fixed":
                    rebuilt.append(str(r["support"]))
                parts += ["完整接口重建（源码版本/记录mode/support及输入身份确定；旧Python路径按release惯例推定，fixed输入根取冻结V0。不是明文历史argv证明；不执行）：",
                          "```bash\n" + shlex.join(rebuilt) + "\n```"]
    parts += ["## 重新格式化报告（不会重算科学结果）",
        "```bash\n" + shlex.join([str(release / ".venv/bin/python"), data["reporter_source"]["path"], str(root)]) + "\n```",
        "默认拒绝活动账本或未完成campaign；显式`--stopped '原因'`仅生成标注的未完成快照，不意味着活动进程已停止。三个新报告文件采用排他创建，已有输出拒绝覆盖；仅Round2状态页可更新，其旧内容封存在JSON及status-history中的按哈希命名副本。",
        "本报告读取RESULT/PROGRESS、账本、FREEZE、RELEASE与VREPORT统计；R0历史适配只哈希CASE字节并读取ZIP源码成员，不反序列化CASE、不执行wheel源码，不调用模型/优化器/原矩阵/概率LP/bootstrap。新的科学重跑需要另行授权与新Run，同时保留当前累计账本、失败和冻结来源。"]
    return "\n\n".join(parts) + "\n"


def prepare_status_archive(data, root):
    """Keep the historical status reference resolvable after updating the live page."""
    prior = data["previous_round2_status"]
    old = prior["reference"]
    if old is None:
        return None
    source = Path(old["path"])
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != old["sha256"] or len(raw) != old["bytes"]:
        raise ValueError("Previous status changed during report assembly")
    target = root / "status-history" / f"{digest}.md"
    if target.exists() and target.read_bytes() != raw:
        raise ValueError("Existing status archive bytes differ")
    archived = dict(path=str(target), bytes=len(raw), sha256=digest)
    prior.update(reference=archived, original_location=str(source), text=raw.decode("utf-8"))
    data["evidence"] = [archived if r == old else r for r in data["evidence"]]
    return source, target, raw


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--stopped", metavar="REASON")
    parser.add_argument("--display-root", type=Path, help="Absolute local mirror root for Markdown links only")
    args = parser.parse_args(argv)
    root = args.root.resolve(strict=True)
    if args.stopped is not None and not args.stopped.strip():
        parser.error("--stopped requires a nonempty explicit reason")
    if args.display_root is not None and not args.display_root.is_absolute():
        parser.error("--display-root must be absolute; it need not exist on this host")
    for name in OUTPUTS:
        if name != "CASE_DESIGN_STATUS.md" and (root / name).exists():
            raise FileExistsError(f"Preserve existing report: {root / name}")
    begun = time.monotonic()
    data = collect(root, args.stopped)
    old_status = prepare_status_archive(data, root)
    data["markdown_display_root"] = str(args.display_root) if args.display_root else None
    report, reproduce = render_report(data, display_root=args.display_root), render_reproduce(data)
    data["report_formatting_seconds"] = time.monotonic() - begun
    data["report_formatting_accounting"] = "pure reading/formatting, no science recomputation; no ledger mutation by reporter"
    status = "# Round2状态\n\n" + "\n\n".join(f"{k}: {data[k]}" for k in (
        "adapter_status", "fixed_load_full400_status", "physical_design_status", "mechanism_status",
        "moment_scope", "algorithm_comparison_status", "validation_status"))
    prose = findings_prose(data["validation_findings"])
    status += "\n\n" + "\n\n".join(prose[k] for k in ("p3", "mechanisms", "aro"))
    status += f"\n\n显式停止原因：{fmt(args.stopped)}；campaign完成：{data['campaign_complete']}。活动claim原样记录在JSON，未推定已死。\n\n"
    status += "冻结job覆盖（expected/attempted/settled/accepted分别统计）：" + fmt(data["frozen_job_coverage"]) + "。全部尝试不等于全部接受。\n\n"
    status += "开发/验证累计秒：" + fmt([data["budget"][k]["spent_seconds"] for k in ("development", "validation")])
    status += "；上限7200/28800，开发含旧211.08199704252183。\n\n详见CASE_DESIGN_ROUND2_REPORT_CN.md、CASE_DESIGN_ROUND2_RESULTS.json及REPRODUCE_CN.md。原Round1报告/账本未改写。\n"
    contents = {OUTPUTS[0]: report, OUTPUTS[1]: json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                OUTPUTS[2]: status, OUTPUTS[3]: reproduce}
    # Recheck both chains after potentially slow rendering, before any output.
    for key in ("development", "validation"):
        b = data["budget"][key]
        current = BudgetLedger(b["ledger_store"], b["group"]).read()
        if current["previous"] != b["chain_head"]:
            raise RuntimeError("Ledger changed before report publication")
    if old_status:
        source, target, raw = old_status
        if source.read_bytes() != raw:
            raise RuntimeError("Live status changed before report publication")
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            with target.open("xb") as stream:
                stream.write(raw)
        elif target.read_bytes() != raw:
            raise RuntimeError("Historical status archive changed")
    for name in (OUTPUTS[0], OUTPUTS[1], OUTPUTS[3]):
        with (root / name).open("x") as stream:
            stream.write(contents[name])
    (root / OUTPUTS[2]).write_text(status)
    print(json.dumps(dict(status=data["validation_status"], outputs=[str(root / n) for n in OUTPUTS]), ensure_ascii=False))


if __name__ == "__main__":
    main()
