"""Audited, source-complete offline bundles, prepared locally before public selection."""

import csv
import html
import json
from pathlib import Path
from ev_assignment_experiments.tracking.store import Ref, Store, encode, sha256
from ev_assignment_experiments.workflows.audit import reaudit
from ev_assignment_experiments.runtime import release


def select(store, study, title, items, evaluations=()):
    if not items or not title.strip():
        raise ValueError("Explicit publication items required")
    labels = set()
    refs = []
    for item in items:
        if set(item) != {"label", "attempt", "decision", "generator"} or item["label"] in labels:
            raise ValueError("Publication item schema/label mismatch")
        if item["generator"] not in {"objective_table", "procurement_profile"}:
            raise ValueError("Unknown reproducible generator")
        labels.add(item["label"])
        a = Ref(**item["attempt"])
        d = Ref(**item["decision"])
        decision = store.read(d)["payload"]
        attempt = store.read(a)["payload"]
        if (
            decision.get("outcome") not in {"adopted", "approximate"}
            or decision["attempt"] != a.as_dict()
            or attempt["study"] != study
        ):
            raise ValueError("Item lacks an explicit accepted decision for this study")
        if not reaudit(store, a)["passed"]:
            raise ValueError("Publication scientific audit failed")
        refs.extend([a, d])
    from ev_assignment_experiments.workflows.evaluation import audit as evaluation_audit, describe

    selected_attempts = {Ref(**i["attempt"]) for i in items}
    for item in evaluations:
        if (
            set(item) != {"label", "run", "generator"}
            or item["label"] in labels
            or item["generator"] != "paired_metrics"
        ):
            raise ValueError("Invalid paired publication item")
        labels.add(item["label"])
        ev = Ref(**item["run"])
        run, locked, _ = describe(store, ev)
        if (
            run["study"] != study
            or {Ref(**e["attempt"]) for e in locked["plans"].values()} != selected_attempts
        ):
            raise ValueError("Comparison must bind the exact selected plans/study")
        report = evaluation_audit(store, ev, register=True)
        if not report["passed"]:
            raise ValueError("Common evaluation is incomplete or failed")
        refs.extend([ev, Ref(**report["audit_ref"])])
    return store.put(
        "publications",
        dict(
            study=study,
            title=title,
            items=items,
            evaluations=list(evaluations),
            public_repository="separate-from-private-development",
            published=False,
        ),
        refs,
    )


def build(store: Store, attempts: list[Ref], destination, *, publication=None) -> dict:
    if not attempts or len(set(attempts)) != len(attempts):
        raise ValueError("Choose a nonempty, unique list of attempts explicitly")
    scopes = set()
    sources = set()
    for ref in attempts:
        if ref.kind != "attempts":
            raise ValueError("Publication must select attempts")
        payload = store.read(ref)["payload"]
        if payload["execution"] != "exited" or not reaudit(store, ref)["passed"]:
            raise ValueError("Selected result failed independent audit")
        scopes.add(payload["scope"])
        run = store.read(Ref(**payload["run"]))["payload"]
        sources.add(Ref(**run["source"]))
    from ev_assignment_experiments.workflows.probability import source_snapshot

    generator_source = store.put_object(source_snapshot())
    sources.add(generator_source)
    roots = list(attempts) + [generator_source]
    for ref in attempts:
        payload = store.read(ref)["payload"]
        if payload.get("schema_version") == 2:
            settlements = [
                r
                for r, v in store.records("events")
                if v["payload"].get("kind") == "parent_settlement"
                and v["payload"].get("attempt") == ref.as_dict()
            ]
            if len(settlements) != 1:
                raise ValueError("Final archive requires one exact parent settlement")
            roots.extend(settlements)
    if publication:
        selected = store.read(publication)["payload"]["items"]
        if {Ref(**i["attempt"]) for i in selected} != set(attempts):
            raise ValueError("Selection changed")
        roots.append(publication)
        for e in store.read(publication)["payload"].get("evaluations", []):
            erun = store.read(Ref(**e["run"]))["payload"]
            sources.add(Ref(**erun["source"]))
    destination = Path(destination).resolve()
    if destination == store.root or destination.is_relative_to(store.root):
        raise ValueError("Bundle must be separate from the source archive")
    destination.mkdir(parents=True, exist_ok=False)
    receipt = store.sync_to(Store(destination / "archive"), roots)
    releases = {}
    for source in sorted(sources):
        releases[source.id] = release.build(store.object(source), destination / "code" / source.id)
    manifest = dict(
        schema_version=2,
        scope=sorted(scopes)[0] if len(scopes) == 1 else sorted(scopes),
        paper_facing=(
            publication is not None
            and scopes == {"frozen_l08"}
            and bool(store.read(publication)["payload"].get("evaluations"))
            and all(
                store.read(Ref(**e["run"]))["payload"]["scope"] == "oos"
                for e in store.read(publication)["payload"].get("evaluations", [])
            )
        ),
        publication=publication.as_dict() if publication else None,
        selected_attempts=[r.as_dict() for r in attempts],
        roots=[r.as_dict() for r in roots],
        source_releases={k: v["source_sha256"] for k, v in releases.items()},
        generator_source=generator_source.as_dict(),
        transfer=receipt,
        training_objectives_are_not_common_evaluation=True,
    )
    (destination / "manifest.json").write_bytes(encode(manifest))
    (destination / "README.md").write_text("""# Offline reproduction bundle

This bundle contains only the explicitly selected results and their complete evidence closure.
Native lower bounds, numerical certificates, convergence and research decisions remain distinct.
Training objectives for DET/DRO/ARO must not be treated as a common test metric.

Common comparison: tables/evaluation-<run>.csv, figures/paired-cost-<run>.svg/.pdf.
Error bars are one paired standard error, not confidence intervals. Procurement plots
use actual selected contracts/windows. All evaluation samples and primal/dual witnesses
are in the archive; `figures` validates them before drawing. Resume evaluation in <=60s
slices with the same evaluation Run and budget; never restart planning to redraw.

Use Python 3.12. For figures/verify install `manifest.generator_source`; for full planning
install the selected Run source; for evaluation install its Run source. Exact releases
are under `code/<source SHA>`:
`python -m pip install -r code/<SHA>/requirements.lock code/<SHA>`.
No PowerAI path or service is needed. The original environment is in `environment.json`.

- Verify and regenerate tables: `python reproduce.py figures` (no optimization/license).
- Verify evidence: `python reproduce.py verify` (no optimization/license).
- Evaluation: `python reproduce.py evaluation --evaluation RUN_SHA --store /absolute/new/store --budget-seconds 60` recomputes paired costs from archived fixed plans; resume longer batches with `ev-assignment evaluate run` and their same Run/budget.
- Smoke: `python reproduce.py smoke --store /absolute/new/store` (Gurobi 12.0.3 and valid license).
- Full: use the archived experiment definition with `ev-assignment plan` and `job launch`
  in a newly named study/budget, explicitly bind archived inputs, and use the original
  science and resource policy. This is new computation, never a continuation of a legacy task.

`tables/objectives.csv` and its SVG table are generated from sealed values, not copied images.
`manifest.json` identifies each selected attempt and source. Publication selection does not
change repository visibility or upload anything. Large archives belong in versioned release assets.
""")
    (destination / "reproduce.py").write_text("""from pathlib import Path
import argparse, json
from ev_assignment_experiments.publication.bundle import verify, render, smoke, full, evaluate_again
p=argparse.ArgumentParser();p.add_argument('mode',choices=['verify','figures','smoke','full','evaluation']);p.add_argument('--store');p.add_argument('--attempt');p.add_argument('--evaluation');p.add_argument('--budget-seconds',type=float)
a=p.parse_args();root=Path(__file__).resolve().parent
if a.mode=='evaluation': print(json.dumps(evaluate_again(root,a.store,a.evaluation,a.budget_seconds),indent=2))
elif a.mode=='full': print(json.dumps(full(root,a.store,a.attempt,a.budget_seconds),indent=2))
elif a.mode=='smoke': print(json.dumps(smoke(root,a.store),indent=2))
else:
 r=verify(root)
 if not r['passed']: raise SystemExit(1)
 if a.mode=='figures': render(root)
 print(json.dumps(r,indent=2))
""")
    render(destination)
    return manifest


def verify(destination) -> dict:
    destination = Path(destination)
    manifest = json.loads((destination / "manifest.json").read_text())
    store = Store(destination / "archive")
    roots = [Ref(**r) for r in manifest.get("roots", manifest["selected_attempts"])]
    closure = store.closure(roots)
    if sha256(encode([r.as_dict() for r in closure])) != manifest["transfer"]["closure_sha256"]:
        raise ValueError("Bundle closure changed")
    for source in manifest.get("source_releases", {}):
        if release.verify(destination / "code" / source)["source_sha256"] != source:
            raise ValueError("Bundle release differs from selected source")
    reports = [reaudit(store, Ref(**r)) for r in manifest["selected_attempts"]]
    evaluation_reports = []
    if manifest.get("publication"):
        from ev_assignment_experiments.workflows.evaluation import audit as evaluation_audit

        publication = store.read(Ref(**manifest["publication"]))["payload"]
        evaluation_reports = [
            evaluation_audit(store, Ref(**i["run"])) for i in publication.get("evaluations", [])
        ]
    return dict(
        passed=all(r["passed"] for r in reports + evaluation_reports),
        evaluation_reports=evaluation_reports,
        reports=reports,
        paper_facing=manifest["paper_facing"],
        scope=manifest["scope"],
        solver_calls=0,
    )


def render(destination):
    destination = Path(destination)
    m = json.loads((destination / "manifest.json").read_text())
    store = Store(destination / "archive")
    if m.get("generator_source"):
        import io
        import zipfile
        from ev_assignment_experiments.workflows.probability import source_snapshot

        def python_files(raw):
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                return {n: sha256(archive.read(n)) for n in archive.namelist() if n.endswith(".py")}

        if python_files(source_snapshot()) != python_files(
            store.object(Ref(**m["generator_source"]))
        ):
            raise ValueError("Install the exact generator_source release to redraw this bundle")
    rows = []
    for r in m["selected_attempts"]:
        a = store.read(Ref(**r))["payload"]
        raw = json.loads(store.object(Ref(**a.get("result", a.get("raw")))))
        if a.get("schema_version") == 2:
            row = [
                r["id"],
                raw["method"],
                raw["lower"]["value"] if raw["lower"] else "",
                raw["upper"]["value"] if raw["upper"] else "",
                raw["native_status"],
                a["convergence"],
            ]
        else:
            row = [
                r["id"],
                "probability",
                raw["primal_objective_usd"],
                raw["primal_objective_usd"],
                raw["status"],
                a["convergence"],
            ]
        rows.append(row)
    folder = destination / "tables"
    folder.mkdir(exist_ok=True)
    header = ["attempt", "method", "lower", "upper", "native_status", "convergence"]
    with (folder / "objectives.csv").open("w", newline="") as h:
        writer = csv.writer(h)
        writer.writerow(header)
        writer.writerows(rows)
    lines = [" | ".join(map(str, row)) for row in [header] + rows]
    body = "".join(
        f'<text x="12" y="{28 + i * 26}" font-size="12">{html.escape(line)}</text>'
        for i, line in enumerate(lines)
    )
    (folder / "objectives.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="{40 + 26 * len(lines)}"><rect width="100%" height="100%" fill="white"/>{body}</svg>'
    )
    from ev_assignment_experiments.publication.figures import render_science

    science = render_science(store, destination, m)
    return dict(
        rows=len(rows), source_attempts=m["selected_attempts"], solver_calls=0, science=science
    )


def smoke(destination, storage):
    if not storage or not Path(storage).is_absolute():
        raise ValueError("Explicit fresh external store required")
    if Path(storage).exists():
        raise ValueError("Smoke uses a fresh store, never an existing budget")
    from ev_assignment_experiments.config.run import RunPlan
    from ev_assignment_experiments.workflows.l08 import plan
    from ev_assignment_experiments.runtime.supervisor import launch

    store = Store(storage)
    recipe = RunPlan.parse(
        dict(
            schema_version=2,
            study="reproduction-smoke",
            title="L08 small physical smoke",
            method="DET",
            input_kind="synthetic_l08",
            budget_group="reproduction-smoke",
            budget_seconds=60,
            solver=dict(seconds=30, absolute_gap=1e-5),
        )
    )
    return launch(store, plan(store, recipe))


def full(destination, storage, attempt_id, budget_seconds):
    """Explicit fresh reproduction, never a resume or free reset of an existing task."""
    import io
    import zipfile
    from dataclasses import replace
    from ev_assignment_experiments.workflows.l08 import recipe_for, materialize_inputs, plan
    from ev_assignment_experiments.workflows.probability import source_snapshot
    from ev_assignment_experiments.runtime.supervisor import launch

    if not storage or not Path(storage).is_absolute() or Path(storage).exists():
        raise ValueError("Full reproduction needs an explicit fresh absolute store")
    if budget_seconds is None or budget_seconds <= 0:
        raise ValueError("Explicit new reproduction budget required")
    root = Path(destination)
    manifest = json.loads((root / "manifest.json").read_text())
    ref = Ref("attempts", attempt_id)
    if ref.as_dict() not in manifest["selected_attempts"]:
        raise ValueError("Select an archived attempt")
    old = Store(root / "archive")
    attempt = old.read(ref)["payload"]
    run = old.read(Ref(**attempt["run"]))["payload"]
    if run.get("schema_version") != 2:
        raise ValueError("Full planning reproduction requires an L08 recipe")

    def code(raw):
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            return {n: sha256(z.read(n)) for n in z.namelist() if n.endswith(".py")}

    if code(source_snapshot()) != code(old.object(Ref(**run["source"]))):
        raise ValueError("Install the exact source release selected for this result")
    recipe = recipe_for(old, run)
    if budget_seconds < recipe.options.seconds:
        raise ValueError("Budget is smaller than original per-attempt resource policy")
    input_root = (
        str(materialize_inputs(old, Ref(**run["inputs"])))
        if recipe.input_kind == "frozen_l08"
        else None
    )
    recipe = replace(
        recipe,
        study="reproduction-" + attempt_id[:16],
        budget_group="explicit-reproduction-" + attempt_id,
        budget_seconds=budget_seconds,
        inherited_seconds=0,
        inherited_evidence=(),
        input_root=input_root,
    )
    new = Store(storage)
    return launch(new, plan(new, recipe))


def evaluate_again(destination, storage, evaluation_id, budget_seconds):
    """Recompute common evaluation from archived fixed plans, never re-optimize plans."""
    from ev_assignment_experiments.workflows import evaluation
    from ev_assignment_experiments.workflows.probability import source_snapshot
    import io
    import zipfile

    if not storage or not Path(storage).is_absolute() or Path(storage).exists():
        raise ValueError("Evaluation reproduction requires a fresh absolute store")
    if budget_seconds is None or not 0 < budget_seconds <= 60:
        raise ValueError(
            "Explicit bounded evaluation budget (<=60s) required; resume by Run for longer work"
        )
    root = Path(destination)
    manifest = json.loads((root / "manifest.json").read_text())
    old = Store(root / "archive")
    if not manifest.get("publication"):
        raise ValueError("No selected common evaluation")
    pub = old.read(Ref(**manifest["publication"]))["payload"]
    ref = Ref("runs", evaluation_id)
    if ref.as_dict() not in [i["run"] for i in pub.get("evaluations", [])]:
        raise ValueError("Evaluation not selected for this publication")
    run, locked, spec = evaluation.describe(old, ref)

    def code(raw):
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            return {n: sha256(z.read(n)) for n in z.namelist() if n.endswith(".py")}

    if code(source_snapshot()) != code(old.object(Ref(**run["source"]))):
        raise ValueError("Install the exact evaluation source before reproducing")
    new = Store(storage)
    old.sync_to(new, [Ref(**manifest["publication"])])
    planned = evaluation.prepare(
        new,
        study=run["study"],
        decisions=[Ref(**p["decision"]) for p in locked["plans"].values()],
        manifest=spec,
        budget_group="explicit-evaluation-reproduction-" + evaluation_id,
        budget_seconds=budget_seconds,
    )
    return dict(
        run=planned.as_dict(),
        report=evaluation.execute(new, planned, input_root=storage, seconds=budget_seconds),
        optimization_calls=0,
        original_plans_unchanged=True,
    )
