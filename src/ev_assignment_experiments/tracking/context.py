"""Bounded factual handoff; observations are never represented as live queries."""

import json

from .store import Store


def context(store: Store, study: str, limit: int = 10) -> dict:
    if not 1 <= limit <= 100:
        raise ValueError("limit must be in [1,100]")
    rows = []
    for ref, record in store.records("attempts"):
        p = record["payload"]
        if p.get("study") == study:
            rows.append(
                {
                    "id": ref.id,
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
                        )
                    },
                }
            )
    settled = {r["run"]["id"] for r in rows if r.get("run")}
    unresolved = []
    for path in sorted((store.root / "claims").glob("*")):
        claim = json.loads(path.read_text())
        if claim["study"] == study and claim["run"]["id"] not in settled:
            unresolved.append(
                {**claim, "execution": "unknown", "reason": "claim_without_sealed_outcome"}
            )
    rows.sort(key=lambda p: (p["observed_at"] or "", p["id"]), reverse=True)
    return {
        "schema_version": 1,
        "study": study,
        "observation_mode": "local_sealed_records",
        "server_live_status": "not_observed",
        "total": len(rows),
        "omitted": max(0, len(rows) - limit),
        "attempts": rows[:limit],
        "unsettled_claims": unresolved[:limit],
        "unsettled_claims_omitted": max(0, len(unresolved) - limit),
        "adopted_results": [],
        "next_action": "Read implementation status before formal L08 work",
        "limitations": [
            "No automatic research adoption",
            "Formal L08 dispatch not admitted",
            "No live SSH refresh or checkpoint recovery in this development slice",
        ],
    }
