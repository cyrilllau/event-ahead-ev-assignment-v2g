"""Bounded, paginated factual handoff generated from immutable research records."""

import json
from .store import Store, sha256
from ev_assignment_experiments.workflows.probability import utcnow


def bounded(value, depth=0):
    """Bound nested observations and human notes while retaining the record link."""
    if isinstance(value, str):
        return (
            value
            if len(value) <= 800
            else value[:800] + f"… [{len(value) - 800} characters omitted]"
        )
    if isinstance(value, list):
        if depth >= 5:
            return {"items_omitted": len(value)}
        items = [bounded(v, depth + 1) for v in value[:10]]
        return items + ([{"items_omitted": len(value) - 10}] if len(value) > 10 else [])
    if isinstance(value, dict):
        keys = (["ref"] if "ref" in value else []) + [k for k in value if k != "ref"]
        if depth >= 5:
            return {"ref": value.get("ref"), "fields_omitted": len(value)}
        result = {k: bounded(value[k], depth + 1) for k in keys[:32]}
        if len(keys) > 32:
            result["fields_omitted"] = len(keys) - 32
        return result
    return value


def context(store: Store, study: str, limit: int = 10, *, offset=0, refresh=False) -> dict:
    if not 1 <= limit <= 100 or offset < 0:
        raise ValueError("Invalid pagination")

    def records(kind):
        return [
            dict(record["payload"], ref=ref.as_dict())
            for ref, record in store.records(kind)
            if record["payload"].get("study") == study
        ]

    raw = records("attempts")
    rows = [
        dict(
            id=p["ref"]["id"],
            **{
                k: p.get(k)
                for k in (
                    "observed_at",
                    "run",
                    "execution",
                    "scientific",
                    "convergence",
                    "decision",
                    "scope",
                    "audit",
                    "error_type",
                    "checkpoint",
                    "native_status",
                )
            },
        )
        for p in raw
    ]
    rows.sort(key=lambda p: (p["observed_at"] or "", p["id"]), reverse=True)
    settled = {r["run"]["id"] for r in rows if r.get("run")}
    unresolved = []
    for path in sorted((store.root / "claims").glob("*")):
        claim = json.loads(path.read_text())
        if claim["study"] == study and claim["run"]["id"] not in settled:
            unresolved.append(
                dict(claim, execution="unknown", reason="claim_without_sealed_outcome")
            )
    observations = records("observations")
    current = store.root / "runtime/studies" / sha256(study.encode()) / "CURRENT.json"
    if refresh and current.exists():
        from ev_assignment_experiments.runtime.supervisor import status

        claim = json.loads(current.read_text())
        if claim.get("kind") == "paired_evaluation":
            from ev_assignment_experiments.runtime.identity import matches
            from ev_assignment_experiments.runtime.ledger import BudgetLedger

            state = BudgetLedger(store.root, claim["budget_group"]).read()
            value = dict(
                kind="paired_evaluation",
                run=claim["run"],
                owner_live=matches(claim["owner"]),
                unsettled=state["active"] is not None,
                spent_seconds=state["spent_seconds"],
            )
        else:
            value = status(store, claim["attempt"])
        ref = store.put(
            "observations",
            dict(study=study, observed_at=utcnow(), origin="local_process_query", value=value),
        )
        observations.append(dict(store.read(ref)["payload"], ref=ref.as_dict()))
    if current.exists():
        claim = json.loads(current.read_text())
        from ev_assignment_experiments.runtime.ledger import BudgetLedger

        budget = BudgetLedger(store.root, claim["budget_group"]).read()
        if budget["active"]:
            unresolved.append(
                dict(
                    claim,
                    execution="unknown",
                    reason="unsettled_cumulative_claim",
                    spent_seconds=budget["spent_seconds"],
                )
            )
    decisions = records("decisions")
    superseded = {
        json.dumps(p["supersedes"], sort_keys=True) for p in decisions if p.get("supersedes")
    }
    active = [p for p in decisions if json.dumps(p["ref"], sort_keys=True) not in superseded]
    active.sort(key=lambda p: (p.get("observed_at", ""), p["ref"]["id"]), reverse=True)
    adopted = [p for p in active if p.get("outcome") in {"adopted", "approximate"}]
    actions = [p for p in active if p.get("kind") == "next_action"]
    studies = records("studies")
    runs = records("runs")
    evaluation_audits = [r for r in records("audits") if r.get("kind") == "paired_evaluation_audit"]
    finished_evaluations = {r["run"]["id"] for r in evaluation_audits if r.get("passed")}
    pending_runs = [r for r in runs if r["ref"]["id"] not in settled | finished_evaluations]
    observations.sort(key=lambda p: p.get("observed_at", ""), reverse=True)
    failures = [r for r in rows if r["execution"] == "failed" or r["scientific"] == "fail"]
    result = dict(
        schema_version=2,
        study=study,
        observation_mode="local_sealed_records",
        generated_at=utcnow(),
        server_live_status="not_observed",
        total=len(rows),
        omitted=max(0, len(rows) - len(rows[offset : offset + limit])),
        offset=offset,
        next_offset=offset + limit if offset + limit < len(rows) else None,
        attempts=rows[offset : offset + limit],
        planned_runs=[bounded(r) for r in pending_runs[offset : offset + limit]],
        planned_runs_omitted=max(0, len(pending_runs) - len(pending_runs[offset : offset + limit])),
        evaluation_audits=[bounded(r) for r in evaluation_audits[-limit:]],
        evaluation_audits_omitted=max(0, len(evaluation_audits) - limit),
        objectives=studies[:limit],
        objectives_omitted=max(0, len(studies) - limit),
        unsettled_claims=unresolved[:limit],
        unsettled_claims_omitted=max(0, len(unresolved) - limit),
        adopted_results=adopted[:limit],
        adopted_results_omitted=max(0, len(adopted) - limit),
        failed_routes=failures[:limit],
        failed_routes_omitted=max(0, len(failures) - limit),
        observations=observations[:limit],
        observations_omitted=max(0, len(observations) - limit),
        next_action=actions[0] if actions else None,
        limitations=[
            "Observations describe their recorded time; absence is not process death",
            "Only explicit recorded decisions constitute adoption",
        ],
    )
    for key in ("objectives", "adopted_results", "observations"):
        result[key] = [bounded(v) for v in result[key]]
    result["next_action"] = bounded(result["next_action"])
    result["markdown"] = markdown(result)
    return result


def markdown(value):
    lines = [
        f"# {value['study']}",
        f"Observed local records: {value['generated_at']}",
        f"Server live status: {value['server_live_status']}",
    ]
    for p in value["objectives"]:
        lines.extend(
            [
                f"- Goal: {p.get('objective', '')} / protocol: {p.get('protocol', '')}",
                f"  Evidence: studies/{p['ref']['id']}",
            ]
        )
    for p in value.get("planned_runs", []):
        lines.append(
            f"- runs/{p['ref']['id']}: planned {p.get('kind', 'planning')}; inspect attempts/evaluation audits"
        )
    for p in value.get("evaluation_audits", []):
        lines.append(
            f"- audits/{p['ref']['id']}: paired evaluation {p.get('cells')}/{p.get('required_cells')}; passed={p.get('passed')}"
        )
    for p in value["attempts"]:
        lines.append(
            f"- attempts/{p['id']}: execution={p['execution']}; science={p['scientific']}; convergence={p['convergence']}"
        )
    for p in value["adopted_results"]:
        lines.append(f"- decisions/{p['ref']['id']}: {p['outcome']} — {p['reason']}")
    if value["next_action"]:
        p = value["next_action"]
        lines.append(f"Next: {p['action']} (decisions/{p['ref']['id']})")
    else:
        lines.append("Next: no recorded decision; inspect the protocol and unresolved evidence.")
    lines.append(f"Omitted attempts: {value['omitted']}; next offset: {value['next_offset']}")
    return "\n".join(lines) + "\n"
