"""Offline, solver-free scientific readback of sealed attempt evidence."""

import json
import numpy as np
from ev_assignment.domain.planning import Control
from ev_assignment.validation.l08 import audit_plan, check_risk, check_first_vector
from ev_assignment_experiments.tracking.store import Ref
from .l08 import recipe_for, problem_for
from .probability import utcnow


def reaudit(store, ref, *, register=False):
    attempt = store.read(ref)["payload"]
    if attempt.get("schema_version") != 2:
        from .probability import reaudit as probability

        return probability(store, ref)
    store.closure([ref])
    if attempt["execution"] != "exited":
        raise ValueError("Failed attempt has no accepted scientific outcome")
    run_ref = Ref(**attempt["run"])
    run = store.read(run_ref)["payload"]
    result = json.loads(store.object(Ref(**attempt["result"])))
    if not result.get("plan") or not result.get("audit") or not result["audit"]["passed"]:
        raise ValueError("Attempt has no complete physical evidence")
    recipe = recipe_for(store, run)
    problem = problem_for(store, run, recipe)
    if result["method"] != recipe.method or result["plan"]["problem"] != problem.identity:
        raise ValueError("Result changed method or problem identity")
    check_first_vector(problem, result["plan"]["entries"], result["plan"]["first_stage"])
    rows = result["audit"]["checks"]["scenarios"]
    if [r["sid"] for r in rows] != list(problem.scenario_ids):
        raise ValueError("Incomplete/misordered physical evidence")
    by = {r["sid"]: r for r in rows}

    def witness(sid):
        value = json.loads(store.object(Ref(**by[sid]["witness"])))
        if value["kind"] != "physical_evidence" or value["evidence"]["sid"] != sid:
            raise ValueError("Physical witness identity changed")
        return value["evidence"]

    checked = audit_plan(
        problem, result["plan"]["entries"], Control(3600), witness_provider=witness
    )
    upper_costs = [r["qU"] for r in checked.checks["scenarios"]]
    risk_upper = (
        check_risk(problem.ambiguity, upper_costs, result["audit"]["checks"]["risk"])
        if recipe.method == "DRO"
        else max(upper_costs)
    )
    expected = checked.checks["first_cost"] + risk_upper
    upper = result["upper"]
    if upper is None or not np.isfinite(upper["value"]) or upper["value"] + 1e-8 < expected:
        raise ValueError("Result upper bound is not supported by physical evidence")
    scope = dict(
        problem=problem.identity,
        method=recipe.method,
        scenarios=list(problem.scenario_ids),
        integer_first_stage=True,
    )
    for bound in (result["lower"], upper):
        if bound is not None and bound["scope"] != scope:
            raise ValueError("Bound scope mismatch")
    if result["lower"] is not None and result["lower"]["value"] > upper["value"]:
        raise ValueError("Negative external gap")
    settlements = [
        (r, v["payload"])
        for r, v in store.records("events")
        if v["payload"].get("kind") == "parent_settlement"
        and v["payload"].get("attempt") == ref.as_dict()
    ]
    settlement = None
    if settlements:
        if len(settlements) != 1:
            raise ValueError("Ambiguous parent settlement")
        sr, event = settlements[0]
        ledger = json.loads(store.object(Ref(**event["ledger"])))
        wait = json.loads(store.object(Ref(**event["wait"])))
        charges = [
            v["payload"]
            for v in ledger["events"]
            if v["payload"]["kind"] == "settlement"
            and v["payload"]["attempt"] == attempt["attempt_id"]
        ]
        if (
            len(charges) != 1
            or not charges[0]["actual_parent_wait"]
            or not wait["actual_parent_wait"]
        ):
            raise ValueError("Missing real parent settlement")
        if charges[0]["charged_seconds"] + 1e-6 < attempt["elapsed_seconds"]:
            raise ValueError("Parent settlement omitted worker elapsed cost")
        settlement = dict(
            reference=sr.as_dict(),
            charged_seconds=charges[0]["charged_seconds"],
            cumulative_seconds=ledger["spent_seconds"],
        )
    report = dict(
        passed=True,
        method=recipe.method,
        problem=problem.identity,
        scope=run["scope"],
        physics=checked.checks,
        solver_calls=0,
        upper_verified=expected,
        lower_level=result["lower"]["level"] if result["lower"] else None,
        native_lower_reproved=False,
        execution_settlement=settlement,
        accepted_for_comparison=False,
    )
    if register:
        audit = store.put(
            "audits",
            dict(study=recipe.study, attempt=ref.as_dict(), report=report, observed_at=utcnow()),
            [ref],
        )
        report["audit_ref"] = audit.as_dict()
    return report
