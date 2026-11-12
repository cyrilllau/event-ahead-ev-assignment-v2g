"""Frozen L08 runs and sealed numerical evidence, independent of compute location."""

from __future__ import annotations
from dataclasses import asdict, replace
import io
import json
import os
import shutil
import time
import traceback
import uuid
from pathlib import Path

import numpy as np
from scipy import sparse

from ev_assignment.domain.planning import Control
from ev_assignment.methods.l08 import solve
from ev_assignment.l08.recourse import RecourseTemplate, build_template
from ev_assignment_experiments.config.run import RunPlan
from ev_assignment_experiments.datasets import synthetic
from ev_assignment_experiments.datasets.l08 import FrozenInputs, L08Dataset
from ev_assignment_experiments.runtime.ledger import BudgetLedger, publish
from ev_assignment_experiments.tracking.store import Ref, Store, encode, sha256
from .probability import source_snapshot, source_revision, utcnow


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def atomic_pointer(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with tmp.open("xb") as h:
        h.write(encode(value))
        h.flush()
        os.fsync(h.fileno())
    os.replace(tmp, path)


def plan(store: Store, recipe: RunPlan):
    cost_refs = [Ref(**r) for r in recipe.inherited_evidence]
    store.closure(cost_refs)
    source = store.put_object(source_snapshot())
    study = store.put(
        "studies",
        dict(
            study=recipe.study,
            title=recipe.title,
            protocol="L08_FULL_DWELL_FORECAST44_V1",
            objective=recipe.hypothesis,
            baseline=recipe.baseline,
            scope="development" if recipe.input_kind == "synthetic_l08" else "L08_FROZEN400",
        ),
    )
    definition = recipe.as_dict()
    definition.pop("input_root", None)
    if recipe.input_kind == "synthetic_l08":
        inputs = store.put("inputs", dict(kind="synthetic_l08", definition=synthetic.definition()))
    else:
        dataset = FrozenInputs(recipe.input_root, recipe.input_manifest)
        dataset.verify()
        files = {}
        for name in sorted(dataset.entries):
            files[name] = store.put_file(dataset.path(name)).as_dict()
        manifest = store.put_file(dataset.root / "import-manifest.json")
        inputs = store.put(
            "inputs",
            dict(kind="frozen_l08", manifest=manifest.as_dict(), files=files),
            [manifest] + [Ref(**r) for r in files.values()],
        )
    experiment = store.put("experiments", dict(definition, study_ref=study.as_dict()), [study])
    run = store.put(
        "runs",
        dict(
            schema_version=2,
            study=recipe.study,
            scope=recipe.input_kind,
            source=source.as_dict(),
            code_revision=source_revision(),
            experiment=experiment.as_dict(),
            inputs=inputs.as_dict(),
            budget_group=recipe.budget_group,
        ),
        [source, experiment, inputs] + cost_refs,
    )
    BudgetLedger(store.root, recipe.budget_group).create(
        recipe.study,
        recipe.budget_seconds,
        inherited=recipe.inherited_seconds,
        evidence=recipe.inherited_evidence,
    )
    return run


def recipe_for(store, run):
    value = store.read(Ref(**run["experiment"]))["payload"].copy()
    value.pop("study_ref", None)
    if value["input_kind"] == "frozen_l08":
        value["input_root"] = "/bound-at-execution"
    return RunPlan.parse(value)


def materialize_inputs(store, ref):
    value = store.read(ref)["payload"]
    if value["kind"] != "frozen_l08":
        raise ValueError("Expected frozen inputs")
    root = store.root / "cache" / "inputs" / ref.id
    for name, r in {**value["files"], "import-manifest.json": value["manifest"]}.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Input path escape")
        target = root / relative
        obj = Ref(**r)
        if not target.resolve().is_relative_to(root.resolve()):
            raise ValueError("Input symlink escape")
        from ev_assignment_experiments.tracking.transfer import file_digest

        if target.exists():
            if target.is_symlink() or file_digest(target) != obj.id:
                raise ValueError("Materialized input changed")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
            shutil.copyfile(store.path(obj), temporary)
            if file_digest(temporary) != obj.id:
                raise ValueError("Input copy failed digest")
            os.link(temporary, target)
            temporary.unlink()
    return root


def problem_for(store, run, recipe, *, for_evaluation=False):
    ref = Ref(**run["inputs"])
    value = store.read(ref)["payload"]
    if value["kind"] == "synthetic_l08":
        return synthetic.problem(recipe.method, value["definition"])
    return L08Dataset(materialize_inputs(store, ref), recipe.input_manifest).problem(
        recipe.method, include_ambiguity=not for_evaluation
    )


def template_cache(store, problem, source):
    folder = store.root / "cache" / "templates" / source / problem.identity

    def provider(sid, env):
        key = sha256(sid.encode())
        pointer = folder / (key + ".json")
        if pointer.exists():
            ref = Ref(**json.loads(pointer.read_text()))
            raw = store.object(ref)
            with np.load(io.BytesIO(raw), allow_pickle=False) as z:
                meta = json.loads(str(z["metadata"]))
                if meta["problem"] != problem.identity or meta["sid"] != sid:
                    raise ValueError("Template cache identity mismatch")
                mats = {
                    n: sparse.csr_matrix(
                        (z[n + "_data"], z[n + "_indices"], z[n + "_indptr"]),
                        shape=tuple(z[n + "_shape"]),
                    )
                    for n in ("W", "T")
                }
                data = {k: z["a_" + k] for k in meta["data_keys"]}
                return RecourseTemplate(
                    mats["W"],
                    mats["T"],
                    data,
                    tuple(meta["names"]),
                    tuple(meta["proof"]),
                    meta["qbar"],
                )
        template = build_template(problem, sid, env=env)
        data = {"a_" + k: v for k, v in template.data.items()}
        for n in ("W", "T"):
            m = getattr(template, n)
            data.update(
                {
                    n + "_data": m.data,
                    n + "_indices": m.indices,
                    n + "_indptr": m.indptr,
                    n + "_shape": np.array(m.shape),
                }
            )
        data["metadata"] = np.array(
            json.dumps(
                plain(
                    dict(
                        problem=problem.identity,
                        sid=sid,
                        names=template.names,
                        proof=template.bound_proof,
                        qbar=template.qbar,
                        data_keys=list(template.data),
                    )
                ),
                allow_nan=False,
            )
        )
        stream = io.BytesIO()
        np.savez_compressed(stream, **data)
        ref = store.put_object(stream.getvalue())
        try:
            publish(pointer, ref.as_dict())
        except FileExistsError:
            if json.loads(pointer.read_text()) != ref.as_dict():
                raise ValueError("Concurrent template differs")
        return template

    return provider


def execute(store, run_ref, attempt_id, seconds, *, resume=None, stop=lambda: False):
    run = store.read(run_ref)["payload"]
    recipe = recipe_for(store, run)
    if sha256(source_snapshot()) != run["source"]["id"]:
        raise ValueError("Source/environment changed: create a related run with inherited budget")
    area = store.root / "runtime" / "attempts" / attempt_id
    if not area.exists():
        raise ValueError("Supervisor admission is required")
    begun = time.monotonic()
    refs = [run_ref]
    checkpoint = None
    previous = None
    if resume:
        resumed = store.read(resume)["payload"]
        if resumed["run"] != run_ref.as_dict() or resumed["source"] != run["source"]:
            raise ValueError("Checkpoint must bind the exact run and source")
        checkpoint = json.loads(store.object(Ref(**resumed["state"])))
        previous = resume.as_dict()
        refs.append(resume)
    pending_evidence = []

    def event(value):
        nonlocal previous
        kind = value["kind"]
        if kind == "checkpoint":
            raw = store.put_object(encode(plain(value["state"])))
            record = store.put(
                "checkpoints",
                dict(
                    study=recipe.study,
                    run=run_ref.as_dict(),
                    source=run["source"],
                    attempt_id=attempt_id,
                    state=raw.as_dict(),
                    observed_at=utcnow(),
                    previous=previous,
                ),
                [run_ref, raw] + pending_evidence + ([Ref(**previous)] if previous else []),
            )
            refs.append(record)
            pending_evidence.clear()
            previous = record.as_dict()
            atomic_pointer(area / "CHECKPOINT.json", record.as_dict())
            return record.as_dict()
        raw = store.put_object(encode(plain(value)))
        refs.append(raw)
        row = store.put(
            "events",
            dict(
                study=recipe.study,
                run=run_ref.as_dict(),
                attempt_id=attempt_id,
                observed_at=utcnow(),
                kind=kind,
                evidence=raw.as_dict(),
            ),
            [run_ref, raw],
        )
        refs.append(row)
        pending_evidence.append(row)
        return raw.as_dict()

    result = None
    error = None
    status = "exited"
    scientific = "unchecked"
    convergence = "unknown"
    try:
        problem = problem_for(store, run, recipe)
        options = replace(recipe.options, seconds=min(seconds, recipe.options.seconds))
        kwargs = dict(
            control=Control(options.seconds, stop=stop, event=event), checkpoint=checkpoint
        )
        if recipe.method != "DET":
            kwargs["template_provider"] = template_cache(store, problem, run["source"]["id"])
        result = solve(problem, recipe.method, options, **kwargs)
        scientific = "pass" if result.audit and result.audit.passed else "unchecked"
        convergence = "met" if result.gap_passed else "unmet"
        raw = store.put_object(encode(plain(asdict(result))))
        refs.append(raw)
    except BaseException as exc:
        status = "cancelled" if isinstance(exc, (TimeoutError, InterruptedError)) else "failed"
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        raw = store.put_object(encode(error))
        refs.append(raw)
    payload = dict(
        schema_version=2,
        study=recipe.study,
        scope=run["scope"],
        run=run_ref.as_dict(),
        source=run["source"],
        attempt_id=attempt_id,
        observed_at=utcnow(),
        execution=status,
        scientific=scientific,
        convergence=convergence,
        decision="pending",
        result=raw.as_dict(),
        checkpoint=previous,
        elapsed_seconds=time.monotonic() - begun,
        error=error,
        accepted_for_comparison=False,
        native_status=result.native_status if result else None,
    )
    record = store.put("attempts", payload, refs)
    publish(
        area / "END.json", dict(attempt=record.as_dict(), returncode=0 if status == "exited" else 2)
    )
    return record
