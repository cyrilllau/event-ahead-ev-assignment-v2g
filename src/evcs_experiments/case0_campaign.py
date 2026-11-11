"""Independent, resumable Case0 campaign control; no Case1 path mutations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import os
import signal
import subprocess
import sys
import traceback

from .config import get_paths
from .case0_profiles import LEGACY, EXPANDED, CAMPAIGNS, TARGETS, validate_config, context_options
from .runtime.jobctl import atomic_json, utcnow

METHODS = ("EA_DRO", "OR_DRO", "SAA", "ARO")
CAMPAIGN_NAME = "ieee33-v54-q1-case0"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def campaign_root():
    root = get_paths().campaign_root
    if root.name not in CAMPAIGNS.values():
        raise ValueError("Case0 requires its own explicitly configured campaign directory")
    return root


def sources():
    paths = get_paths()
    files = [*sorted((paths.project_root / "src/evcs_experiments").rglob("*.py")),
             paths.project_root / "requirements.lock", paths.spec_root / "CASE0_SPEC_CN.md"]
    if paths.campaign_root.name == CAMPAIGNS[EXPANDED]:
        files.append(paths.spec_root / "CASE0_EXPANDED_SPEC_CN.md")
    return {str(p.relative_to(paths.project_root)): digest(p) for p in files}


def verify_sources(config):
    if config["source_sha256"] != sources():
        raise ValueError("Case0 source changed since configuration lock; record and review a new revision")
    if digest(get_paths().data_root / "MANIFEST_SHA256.csv") != config["data_manifest_sha256"]:
        raise ValueError("Case0 baseline identity changed")


def active_config(root):
    """Read the immutable original lock or its explicitly reviewed source revision."""
    state=read(root/'state.json') if (root/'state.json').exists() else {}
    ref=state.get('active_config_reference')
    if ref is None:
        config=read(root/'config.lock.json');validate_config(config,root);return config
    if state.get('representation_revision'):
        from .case0_memory_repair import validate_representation
        return validate_representation(root,state)[0]
    if state.get('memory_repair'):
        from .case0_memory_repair import validate_transition
        return validate_transition(root,state)[0]
    if state.get('operational_interruption'):
        from .case0_interruption import validate_transition
        return validate_transition(root,state)[0]
    path=Path(ref['path']).resolve()
    if not path.is_relative_to((root/'revisions').resolve()) or digest(path)!=ref['sha256'] or path.stat().st_size!=ref['size_bytes']:
        raise ValueError('Active Case0 source revision changed')
    transition_ref=state.get('numerical_refinement')
    if not transition_ref or digest(transition_ref['path'])!=transition_ref['sha256']:
        raise ValueError('Numerical source transition changed')
    transition=read(transition_ref['path'])
    original=read(root/'config.lock.json');revised=read(path)
    if transition['active_config_reference']!=ref or transition['original_config_reference']['sha256']!=digest(root/'config.lock.json'):
        raise ValueError('Original/source config transition changed')
    if {k:v for k,v in revised.items() if k not in ('source_sha256','solver_feasibility_tolerance')}!={k:v for k,v in original.items() if k!='source_sha256'} or revised.get('solver_feasibility_tolerance')!=1e-9:
        raise ValueError('Scientific configuration changed during numerical refinement')
    validate_config(revised,root)
    return revised


def configuration_attempt(root,state,key):
    spent=float(state.get('configuration_spent_seconds',{}).get(key,0.))
    if not 0<=spent<1800:raise ValueError('Original configuration budget exhausted or invalid')
    base=root/'planning'/key
    folder=Path(state.get('configuration_attempt_roots',{}).get(key,str(base)))
    if folder==base/'compact-metadata-1':
        from .case0_memory_repair import validate_representation
        _,transition=validate_representation(root,state)
        if key!='size0/EA_DRO' or str(folder)!=transition['new_attempt_directory'] or spent!=transition['prior_configuration_seconds'] or state.get('configuration_master_spent_seconds',{}).get(key)!=transition['prior_master_seconds']:
            raise ValueError('Representation carryover ledger missing/changed')
        return folder,spent
    if folder==base/'memory-repair-1':
        from .case0_memory_repair import validate_transition
        _,transition=validate_transition(root,state)
        if key!='size0/EA_DRO' or str(folder)!=transition['new_attempt_directory'] or spent!=transition['prior_configuration_seconds'] or state.get('configuration_master_spent_seconds',{}).get(key)!=transition['prior_master_seconds']:
            raise ValueError('Memory repair carryover ledger missing/changed')
        return folder,spent
    if folder==base/'continuation-1':
        from .case0_interruption import validate_transition
        _,transition=validate_transition(root,state)
        if key!='size0/EA_DRO' or str(folder)!=transition['new_attempt_directory'] or spent!=transition['prior_configuration_seconds'] or state.get('configuration_master_spent_seconds',{}).get(key)!=transition['prior_master_seconds']:
            raise ValueError('Interruption carryover ledger missing/changed')
        return folder,spent
    if folder not in (base,base/'refinement-1'):raise ValueError('Foreign Case0 attempt directory')
    if folder==base/'refinement-1':
        ref=state.get('numerical_refinement')
        if not ref or digest(ref['path'])!=ref['sha256']:raise ValueError('Refinement transition missing/changed')
        transition=read(ref['path'])
        if key!='size1/OR_DRO' or spent!=transition['prior_configuration_seconds'] or str(folder)!=transition['new_attempt_directory'] or state.get('configuration_master_spent_seconds',{}).get(key)!=transition['prior_master_seconds']:
            raise ValueError('Refinement carryover ledger missing/changed')
    return folder,spent


def decision(root, event, **details):
    with (root / "decisions.jsonl").open("a") as handle:
        handle.write(json.dumps(dict(at_utc=utcnow(), event=event, **details), allow_nan=False) + "\n")


def save_state(root, state):
    state["updated_at_utc"] = utcnow()
    atomic_json(root / "state.json", state)
    (root / "NEXT_ACTION.md").write_text(state["next_action"] + "\n")


def prepare(backend, profile=LEGACY):
    root = campaign_root()
    if profile not in CAMPAIGNS or root.name != CAMPAIGNS[profile]:
        raise ValueError('Case0 profile and campaign directory differ')
    if profile == EXPANDED and backend != 'gurobi':
        raise ValueError('Expanded Case0 locks Gurobi for the complete group')
    root.mkdir(parents=True, exist_ok=True)
    if (root / "config.lock.json").exists():
        config = active_config(root)
        verify_sources(config)
        if config["backend"] != backend or config.get("profile",LEGACY) != profile:
            raise ValueError("Cannot change a locked Case0 backend")
        return read(root / "state.json")
    config = dict(kind="CASE0_DEVELOPMENT_PROTOCOL_V1", target_event="michael_2018",
        created_at_utc=utcnow(), campaign=CAMPAIGN_NAME, backend=backend,
        source_sha256=sources(), data_manifest_sha256=digest(get_paths().data_root / "MANIFEST_SHA256.csv"),
        horizon_hours=24, network="IEEE33", planning_methods=list(METHODS),
        sizes=[dict(fleet_per_depot=3, planning_variants=2), dict(fleet_per_depot=2, planning_variants=1)],
        calibration_per_base=4, evaluation_per_base=8, base_weather_count=8,
        variant_start=10000, debug_variant_start=20000, development=True,
        configuration_seconds=1800, master_seconds=1200, final_reserve_seconds=600,
        total_compute_seconds=21600, gap_target=.005, evaluation_item_count=768,
        cpu_cores=1, memory_max_bytes=8*1024**3, notification="meaningful_changes_only",
        scope="Independent small-scale model validation, not formal Case1 or C&CG performance")
    if profile == EXPANDED:
        config.update(profile=EXPANDED,campaign=CAMPAIGNS[EXPANDED],target_event=TARGETS[EXPANDED],
            sizes=[dict(fleet_per_depot=12,planning_variants=1)],variant_start=30000,
            debug_variant_start=40000,solver_feasibility_tolerance=1e-9,
            scope='Independent expanded development study; not formal Case1')
    validate_config(config,root)
    atomic_json(root / "config.lock.json", config)
    state = dict(kind="CASE0_STATE_V1", status="PREPARED", size_index=0,
        completed={}, active_configuration=None, compute_seconds=0., evaluation_opened=False,
        next_action="Independently review Case0 inputs, solver and isolation, then launch plan batch; no OOS yet.")
    save_state(root, state)
    decision(root, "CASE0_PROTOCOL_LOCKED", config_sha256=digest(root / "config.lock.json"))
    return state


def plan_one(method, started):
    from .case0_isolation import verify_current_cgroup
    verify_current_cgroup()
    from .case0_inputs import Case0Context
    from .case0_planning import run_case0_planning
    root = campaign_root()
    config = active_config(root)
    state = read(root / "state.json")
    key = f"size{state['size_index']}/{method}"
    if method not in METHODS or state["active_configuration"] != key:
        raise ValueError("Configuration child is not the recorded active method")
    folder,prior_spent = configuration_attempt(root,state,key)
    ctx = Case0Context(folder / "inputs", case_root=get_paths().data_root,
        lead_hours=0 if method == "OR_DRO" else 1, **context_options(config), **config["sizes"][state["size_index"]])
    probability_set = ctx.probability_set if method.endswith("DRO") else None
    scenes = [ctx.scenario(i) for i in range(len(ctx.planning_ids))]
    if config.get('solver_compact_recourses') is True:
        from .compact_recourses import compact_recourse
        recourses = [compact_recourse(ctx.recourse(i)) for i in range(len(ctx.planning_ids))]
    else:
        recourses = [ctx.recourse(i) for i in range(len(ctx.planning_ids))]
    import numpy as np
    return run_case0_planning(ctx.first, recourses, ctx.planning_ids, probability_set,
        case=ctx.case, tasks=ctx.tasks, scenes=scenes, method_id=method,
        output_dir=folder / "solve", backend=config["backend"], deadline=started+1800-prior_spent,
        configuration_started_at=started-prior_spent,
        master_seconds=config["master_seconds"]-state.get("configuration_master_spent_seconds",{}).get(key,0.),
        **({"zero_presolve":True} if config.get("solver_zero_presolve") is True else {}),
        **({"compact_descriptive_metadata":True} if config.get("solver_compact_recourses") is True else {}),
        **({"solver_feasibility_tolerance":config["solver_feasibility_tolerance"]} if "solver_feasibility_tolerance" in config else {}),
        empirical_probabilities=np.full(len(scenes), 1/len(scenes)) if method=="SAA" else None)[1]


def bounded_configuration(method, started, folder, *, prior_spent=0.):
    """A hard wall deadline covers imports, calibration, assembly, solve and audits."""
    with (folder / "stdout.log").open("xb") as stdout, (folder / "stderr.log").open("xb") as stderr:
        child = subprocess.Popen([sys.executable, "-m", "evcs_experiments.case0_campaign", "_plan-one", "--method", method,
            "--started", str(started)], stdout=stdout, stderr=stderr, process_group=0)
        try:
            code = child.wait(timeout=max(.001, 1800-prior_spent-(time.monotonic()-started)))
        except subprocess.TimeoutExpired:
            try: os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            child.wait()
            raise TimeoutError("Case0 complete configuration reached its hard 1800-second limit")
    if code != 0:
        raise RuntimeError(f"Case0 configuration child exited {code}; inspect its preserved stderr")
    return read(folder / "solve/result.json")


def plan_batch(seconds=7180):
    job_started = time.monotonic()
    # The cgroup assertion precedes imports of numerical libraries.
    from .case0_isolation import verify_current_cgroup
    isolation = verify_current_cgroup()
    root = campaign_root()
    config = active_config(root)
    verify_sources(config)
    state = read(root / "state.json")
    if state["evaluation_opened"] or state["active_configuration"]:
        raise ValueError("Evaluation already opened or prior configuration requires settlement")
    if state["status"] not in ("PREPARED", "PLANNING", "FALLBACK_PREPARED", "REFINEMENT_PREPARED", "INTERRUPTION_PREPARED", "MEMORY_REPAIR_PREPARED", "REPRESENTATION_PREPARED"):
        raise ValueError("State is not admitted for planning")
    parent_deadline = job_started + min(float(seconds), 7180.)
    accounted_at = job_started
    def charge():
        nonlocal accounted_at
        now = time.monotonic()
        state["compute_seconds"] += now-accounted_at
        accounted_at = now
    size_index = state["size_index"]
    size = config["sizes"][size_index]
    for method in METHODS:
        key = f"size{size_index}/{method}"
        if key in state["completed"]:
            saved=state["completed"][key];path=Path(saved["path"])
            if digest(path)!=saved["sha256"]:raise ValueError('Completed Case0 plan bytes changed')
            result=read(path)
            if result['method_id']!=method or result['backend']!=config['backend'] or result['numerical_valid'] is not True or result['gap_passed'] is not True:
                raise ValueError('Completed Case0 plan identity/numerics changed')
            continue
        charge()
        remaining = min(parent_deadline-time.monotonic(), config["total_compute_seconds"]-state["compute_seconds"])
        folder,prior_spent=configuration_attempt(root,state,key)
        configuration_remaining=config["configuration_seconds"]-prior_spent
        if remaining < configuration_remaining + 10:
            if config["total_compute_seconds"]-state["compute_seconds"] < configuration_remaining + 10:
                state["status"] = "BUDGET_EXHAUSTED"
                state["next_action"] = "The remaining Case0 budget cannot admit another complete planning configuration. Deliver supported results and a diagnosis; no budget reset."
            else:
                state["next_action"] = "Continue the remaining locked planning methods in a new bounded Case0 job."
            save_state(root, state)
            return state
        started = time.monotonic()
        folder.mkdir(parents=True, exist_ok=False)
        if folder.name in ('continuation-1','memory-repair-1','compact-metadata-1'):
            if folder.name=='compact-metadata-1':
                from .case0_memory_repair import validate_representation as validate_transition
            elif folder.name=='memory-repair-1':
                from .case0_memory_repair import validate_transition
            else:
                from .case0_interruption import validate_transition
            _,transition=validate_transition(root,state)
            (folder/'inputs').symlink_to(transition['original_input_directory'],target_is_directory=True)
        state.update(status="PLANNING", active_configuration=key,
            next_action="Healthy Case0 planning job: inspect once and leave it running; no Case1 changes.")
        save_state(root, state)
        atomic_json(folder / "start.json", dict(at_utc=utcnow(), isolation=isolation,
            config_sha256=digest(Path(state["active_config_reference"]["path"]) if "active_config_reference" in state else root / "config.lock.json"), method=method, prior_configuration_seconds=prior_spent, configuration_remaining_seconds=configuration_remaining, **size))
        try:
            result = bounded_configuration(method, started, folder,prior_spent=prior_spent) if prior_spent else bounded_configuration(method, started, folder)
            state["completed"][key] = dict(path=str(folder / "solve/result.json"),
                sha256=digest(folder / "solve/result.json"), gap=result["gap"],
                numerical_valid=result["numerical_valid"], gap_passed=result["gap_passed"])
            if not result["gap_passed"]:
                state["status"] = "FALLBACK_REQUIRED" if size_index+1<len(config["sizes"]) else "DIAGNOSIS_REQUIRED"
                state["next_action"] = "Review valid time-limited result; use the single declared smaller-group fallback before any OOS." if size_index+1<len(config["sizes"]) else "Fixed study size did not meet the gap target; report a supported diagnosis, no further size tuning."
        except Exception as error:
            atomic_json(folder / "failure.json", dict(type=type(error).__name__, error=str(error), traceback=traceback.format_exc()))
            state.update(status="FAILURE_REQUIRES_REVIEW",
                next_action=("Review exact failure and ledger; fix a code defect or admit the one declared resource/evidence fallback. Do not blindly retry." if len(config["sizes"])>1 else "Review exact failure and ledger at the locked expanded size; no size fallback or budget reset. Do not blindly retry."))
        finally:
            elapsed = time.monotonic()-started
            charge()
            state["active_configuration"] = None
            atomic_json(folder / "accounting.json", dict(configuration_wall_seconds=elapsed,
                prior_configuration_seconds=prior_spent,cumulative_configuration_seconds=prior_spent+elapsed,
                cumulative_case0_compute_seconds=state["compute_seconds"], at_utc=utcnow()))
            state.setdefault("configuration_spent_seconds",{})[key]=prior_spent+elapsed
            save_state(root, state)
        if state["status"] != "PLANNING":
            return state
    state.update(status="PLANNING_REVIEW_REQUIRED", next_action=
        "All four Case0 plans have numerical evidence. Independently verify inputs, original arrays, bounds and physical checks; then lock all plans and evaluator before OOS.")
    save_state(root, state)
    return state


def fallback(reason):
    root = campaign_root()
    config = active_config(root)
    verify_sources(config)
    state = read(root / "state.json")
    if (len(config["sizes"]) != 2 or state["size_index"] != 0 or state["evaluation_opened"] or state["active_configuration"]
            or state["status"] not in ("FALLBACK_REQUIRED", "FAILURE_REQUIRES_REVIEW")):
        raise ValueError("The single pre-evaluation fallback is not available")
    if not reason.strip():
        raise ValueError("Fallback requires a recorded evidence-based reason")
    decision(root, "DECLARED_GROUP_FALLBACK", reason=reason, old_size=0, new_size=1,
             compute_seconds_preserved=state["compute_seconds"])
    state.update(size_index=1, status="FALLBACK_PREPARED", next_action=
        "Run all four methods at the declared eight-vehicle/eight-support fallback; preserve previous costs/evidence.")
    save_state(root, state)
    return state


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config")
    sub = p.add_subparsers(dest="action", required=True)
    q = sub.add_parser("prepare"); q.add_argument("--backend", choices=("highs", "gurobi"), required=True)
    q.add_argument("--profile",choices=(LEGACY,EXPANDED),default=LEGACY)
    q = sub.add_parser("plan"); q.add_argument("--seconds", type=float, default=7180)
    q = sub.add_parser("_plan-one"); q.add_argument("--method", choices=METHODS, required=True); q.add_argument("--started", type=float, required=True)
    q = sub.add_parser("numerical-repair"); q.add_argument("--request",required=True); q.add_argument("--expected-sha256",required=True)
    sub.add_parser("status")
    q = sub.add_parser("fallback"); q.add_argument("--reason", required=True)
    args = p.parse_args(argv)
    if args.config:
        import os
        os.environ["EVCS_CONFIG"] = str(Path(args.config).resolve())
    if args.action == "prepare": value = prepare(args.backend,args.profile)
    elif args.action == "plan": value = plan_batch(args.seconds)
    elif args.action == "numerical-repair":
        from .case0_refinement import prepare_numerical_refinement
        value = prepare_numerical_refinement(args.request,args.expected_sha256)
    elif args.action == "fallback": value = fallback(args.reason)
    elif args.action == "_plan-one": value = plan_one(args.method, args.started)
    else: value = read(campaign_root() / "state.json")
    print(json.dumps(value, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
