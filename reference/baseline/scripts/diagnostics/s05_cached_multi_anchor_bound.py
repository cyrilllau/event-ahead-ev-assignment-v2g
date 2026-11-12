"""C1-A repaired cached-model auxiliary bound; never a C&CG resume.

Candidate cuts receive an independent original-array audit before the solve.
The independently audited auxiliary bound remains provisional until external
terminal parent settlement. No old driver gate or original solver LB is changed.
"""
from decimal import Decimal
import math

KIND = 'S05_CACHED_MULTI_ANCHOR_BOUND_ONLY_V1'
OPENING = Decimal('5633.96054736330092')
REMAINING = Decimal('1566.03945263669908')
FIXED_UB = 227762.10627548306
PREVIOUS_LB = 24135.73087520793
SUPPORT_COUNT = 160
PLAN_ID = 'mechanism.michael_2018.EA_DRO.baseline'
PLAN_SHA256 = '33179f0050e20f734a9f1706e26a88e393e0c2ba79de75b46a2070d25857c1d7'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def budget_guard(request, settlement, previous_review):
    from evcs_experiments.cached_bound_model import budget_guard as admit_cached
    return admit_cached(request, settlement, previous_review)


def solver_grant(request, deadline, now):
    """Actual source/derivation/assembly time consumes the same absolute parent."""
    need(all(type(x) in (int, float) and math.isfinite(x) for x in (deadline, now)),
         'Invalid absolute deadline')
    grant = min(request['solver_seconds'], deadline - now - request['post_master_reserve_seconds'])
    need(grant > 0, 'Preparation exhausted master admission cutoff')
    return grant


def bound_result(native_lower_bound, *, audited):
    """No new incumbent UB, no original-native-LB rewrite, no normal resume."""
    need(audited is True, 'Original-array cut and native-bound audit required')
    need(type(native_lower_bound) in (int, float) and math.isfinite(native_lower_bound),
         'Missing finite native auxiliary lower bound')
    need(native_lower_bound <= FIXED_UB + 1e-4, 'Auxiliary bound contradicts accepted UB')
    lower = max(PREVIOUS_LB, native_lower_bound)
    return dict(kind=KIND, plan_id=PLAN_ID, plan_sha256=PLAN_SHA256,
        upper_bound_usd=FIXED_UB, previous_lower_bound_usd=PREVIOUS_LB,
        auxiliary_native_lower_bound_usd=native_lower_bound, lower_bound_usd=lower,
        relative_gap=max(0., FIXED_UB-lower)/max(1., abs(FIXED_UB)),
        original_native_lower_bound_modified=False, new_candidate_UB_verified=False,
        recourse_solver_calls=0, probability_solver_calls=0, master_milp_calls=1,
        normal_ccg_checkpoint_compatible=False, full_CCG_converged=False,
        scientific_acceptance=False, formal_OOS=False, lockable=False,
        requires_terminal_parent_settlement=True, cumulative_seconds_provisional=True)


def settle_parent_seconds(settlement, actual_parent_seconds_decimal):
    """Arithmetic for external terminal settlement; never use nested phase timers."""
    actual = Decimal(actual_parent_seconds_decimal)
    need(actual.is_finite() and actual >= 0, 'Invalid terminal parent elapsed')
    prior = Decimal(settlement['cumulative_accounted_seconds_decimal'])
    need(prior == OPENING, 'Changed predecessor balance')
    total = prior + actual
    need(total <= Decimal(7200), 'Original cumulative cap exceeded')
    return dict(kind='S05_MULTI_ANCHOR_BOUND_PARENT_ACCOUNTING_V1',
        opening_cumulative_seconds_decimal=str(prior),
        terminal_parent_seconds_decimal=str(actual),
        cumulative_accounted_seconds_decimal=str(total),
        remaining_seconds_decimal=str(Decimal(7200)-total),
        cumulative_cap_seconds=7200, parent_charged_once=True,
        nested_seconds_added=False, cumulative_seconds_provisional=False)


def execute(request_path, expected):
    """One supervised audited bound attempt, pending terminal parent settlement.

    Atom cuts and assembled coefficients are independently audited before the
    MILP; native bound is independently checked afterward in the same deadline.
    This routine cannot issue terminal accounting or scientific acceptance.
    """
    import argparse
    import hashlib
    import json
    import os
    from pathlib import Path
    import sys
    import time
    from datetime import datetime, timezone
    import numpy as np
    from scipy.sparse import save_npz
    from evcs_experiments.config import get_paths
    from evcs_experiments.runtime.jobctl import observation
    from evcs_experiments.campaign_batch import verify_reference
    from evcs_experiments.evaluation_lock import file_reference
    from evcs_experiments.random_streams import canonical_hash
    from evcs_experiments.candidate_dual_adapter import CandidateDualContext, load_old_anchor, iter_candidate_cuts
    from evcs_experiments.candidate_dual_audit import audit_derived_atom, audit_candidate_axis
    from evcs_experiments.multi_anchor_epigraph import build_multi_anchor_epigraph_master
    from evcs_experiments.native_gurobi import solve_milp
    from evcs_experiments.multi_anchor_bound_audit import audit_multi_anchor_model, audit_native_auxiliary_bound
    from evcs_experiments.relaxation_bound import coefficient_sha256
    from evcs_experiments.cached_bound_model import load_cached_model
    from evcs_experiments.bounded_row_relaxation import relax_bounded_rows
    from evcs_experiments.bounded_row_relaxation_audit import audit_bounded_row_relaxation

    def read(path):
        return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    def write(path, value):
        with Path(path).open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False);stream.write('\n')
        return file_reference(path)
    began=time.monotonic();phase='parent_admission';phase_started=began;timings={}
    def enter(name):
        nonlocal phase,phase_started
        now=time.monotonic();timings[phase]=timings.get(phase,0.)+now-phase_started
        phase=name;phase_started=now
    rr=file_reference(request_path);need(rr['sha256']==expected,'Changed request')
    req=read(request_path);refs=req['references'];p=req['parent_job'];paths=get_paths()
    job=Path(p['directory']).resolve();root=paths.project_root.resolve();campaign=paths.campaign_root.resolve()
    need(job==campaign/'jobs'/p['job_id'] and os.environ.get('EVCS_JOB_ID')==p['job_id']
         and os.environ.get('EVCS_JOB_DIRECTORY')==str(job),'Unsupervised/wrong parent')
    manifest=read(job/'manifest.json');status=read(job/'status.json')
    need(manifest['argv']==[sys.executable,'-B',str(Path(__file__).resolve()),'--request',rr['path'],'--expected-request-sha256',expected]
         and manifest['cwd']==str(root) and manifest['outroot']==str(job.parent)
         and manifest['job_id']==p['job_id'] and manifest['spec_id']=='S05','Wrong parent command')
    need(manifest['threads']==p['threads']==1 and manifest['timeout_seconds']==p['timeout_seconds']==1000
         and manifest['grace_seconds']==p['grace_seconds']==5,'Wrong parent resource limits')
    verify_reference(p['controller_reference'])
    need(Path(p['controller_reference']['path'])==Path(manifest['controller_path']).resolve()==root/'src/evcs_experiments/runtime/jobctl.py','Changed controller')
    need(status['state']=='RUNNING' and os.getpid()==status['command_pid']
         and os.getsid(0)==status['supervisor_sid'] and observation(status)['supervisor_alive'],'Wrong live parent')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        need(os.environ.get(key)=='1','Wrong thread environment')
    age=(datetime.now(timezone.utc)-datetime.fromisoformat(manifest['created_at_utc'])).total_seconds()
    need(0<=age<990,'Expired parent');deadline=time.monotonic()+990-age
    # This reservation precedes large source/input reads and survives failures.
    attempt=campaign/'multi_anchor_bound_attempts'/canonical_hash(dict(settlement_sha256=refs['settlement']['sha256'],plan_id=req['plan_id']))
    attempt.mkdir(parents=True,exist_ok=False)
    write(attempt/'reservation.json',dict(request_reference=rr,parent_manifest_reference=file_reference(job/'manifest.json'),
        requires_terminal_parent_settlement=True,settlement=refs['settlement'],plan_id=req['plan_id']))
    out=Path(req['output_root']).resolve();need(out==campaign/'multi_anchor_bounds'/p['job_id'],'Wrong output root')
    out.mkdir(parents=True,exist_ok=False)
    try:
        enter('source_admission')
        need({'execution_policy','component_review'} <= set(refs),'Prospective policy/component review references required')
        for value in refs.values():verify_reference(value)
        opening=budget_guard(req,read(refs['settlement']['path']),read(refs['previous_result_review']['path']))
        need(req['adapter_references']['candidate_review']==refs['previous_result_review'],'Unbound candidate review')
        source=read(refs['source_manifest']['path']);need(Path(refs['source_manifest']['path']).parent==root,'Wrong source root')
        files={}
        for row in source['files']:
            path=(root/row['path']).resolve();need(path.is_relative_to(root) and str(path) not in files,'Source escape/duplicate')
            value=dict(path=str(path),sha256=row['sha256'],size_bytes=row['size_bytes']);verify_reference(value);files[str(path)]=value
            need(time.monotonic()<deadline-100,'Source admission cutoff')
        need(files[str(Path(__file__).resolve())]==refs['script'],'Unbound driver')
        for name,module in tuple(sys.modules.items()):
            if name=='evcs_experiments' or name.startswith('evcs_experiments.'):
                path=getattr(module,'__file__',None)
                if path is not None:need(file_reference(path)==files.get(str(Path(path).resolve())),'Mixed package source')
        need(Path(os.environ['EVCS_CONFIG']).resolve()==Path(refs['runtime_config']['path']),'Wrong runtime config')
        # Source admission covers the cached evidence loader, transformer and checker.
        for name in ('cached_bound_model.py','bounded_row_relaxation.py','bounded_row_relaxation_audit.py'):
            need(str(root/'src/evcs_experiments'/name) in files,'Missing bound source member')
        enter('cached_input_admission')
        original,model_audit,assembly,cache_admission=load_cached_model(req,deadline=deadline-100)
        write(out/'cached_admission.json',cache_admission)
        model_audit_ref=write(out/'original_model_audit.json',model_audit)
        witness=read(refs['bound_witnesses']['path'])
        need(witness['original_coefficient_sha256']==coefficient_sha256(original),'Bound witness original model differs')
        first=assembly['anchor_sets'][0]['construction']
        prefix=assembly['retained_prefix_rows'];columns=first['original_first_stage_columns']
        rows=(prefix,original['matrix'].shape[0])
        need(witness['prefix_rows']==prefix and witness['first_stage_columns']==columns
             and tuple(witness['new_cut_row_range_half_open'])==rows,'Witness range differs')
        bounds={int(k):v for k,v in witness['column_to_reserve_upper_row'].items()}
        kwargs=dict(allowed_rows=rows,allowed_columns=(0,columns),threshold=1e-13,
                    bound_witnesses=bounds,witness_rows=(0,first['first_stage_rows']),deadline=deadline-100)
        enter('outward_relaxation_and_independent_audit')
        model,relaxation=relax_bounded_rows(original,**kwargs)
        relaxation_audit=audit_bounded_row_relaxation(original,model,relaxation,**kwargs)
        write(out/'relaxation.json',relaxation);write(out/'relaxation_audit.json',relaxation_audit)
        expected_coefficient=coefficient_sha256(model)
        write(out/'assembly.json',assembly);save_npz(out/'matrix.npz',model['matrix'])
        np.savez_compressed(out/'model_arrays.npz',**{key:model[key] for key in
            ('c','lower_bounds','upper_bounds','row_lower','row_upper','integrality','objective_constant_usd')})
        grant=solver_grant(req,deadline,time.monotonic())
        write(out/'execution.json',dict(request_reference=rr,opening_cumulative_seconds_decimal=str(opening),
            absolute_internal_deadline=deadline,solver_grant_seconds=grant,post_master_reserve_seconds=100,
            requires_terminal_parent_settlement=True))
        enter('native_master')
        solved=solve_milp(model['c'],model['integrality'],model['lower_bounds'],model['upper_bounds'],
            model['matrix'],model['row_lower'],model['row_upper'],options=dict(time_limit=grant,mip_rel_gap=.005,threads=1),
            log_path=out/'native_solver.log')
        native_ref=write(out/'native_solver.json',solved.native_record)
        if solved.x is not None:np.save(out/'unverified_auxiliary_incumbent.npy',solved.x,allow_pickle=False)
        write(out/'native_return.json',dict(status=int(solved.status),message=str(solved.message),
            bound_without_constant_usd=solved.mip_dual_bound,objective_without_constant_usd=solved.fun,
            native_reference=native_ref,model_audit_reference=model_audit_ref,
            coefficient_sha256=expected_coefficient,request_reference=rr))
        enter('independent_native_bound_audit')
        bound_audit=audit_native_auxiliary_bound(model,model_audit,solved.native_record,
            native_status=solved.status,native_bound=solved.mip_dual_bound,incumbent=solved.x,
            expected_coefficient_sha256=expected_coefficient,fixed_upper_bound=FIXED_UB,deadline=deadline,relaxation_audit=relaxation_audit)
        bound_audit_ref=write(out/'native_bound_audit.json',bound_audit)
        lower=bound_audit['lower_bound_usd']
        result=bound_result(lower,audited=True)
        result.update(status='AUDITED_BOUND_PENDING_TERMINAL_PARENT_SETTLEMENT',request_reference=rr,
            solver_status=int(solved.status),native_message=str(solved.message),
            auxiliary_coefficient_sha256=expected_coefficient,
            model_audit_reference=model_audit_ref,native_bound_audit_reference=bound_audit_ref,
            native_solver_reference=native_ref,opening_cumulative_seconds_decimal=str(opening))
        enter('evidence_sealing')
        evidence_refs=[file_reference(path) for path in sorted(out.rglob('*')) if path.is_file()]
        evidence_ref=write(out/'numerical_evidence_manifest.json',dict(request_reference=rr,files=evidence_refs))
        result['numerical_evidence_manifest_reference']=evidence_ref
        result['phase_seconds']=dict(timings)
        result['attempt_elapsed_seconds']=time.monotonic()-began
        result_ref=write(out/'result.json',result)
        write(out/'artifact_manifest.json',dict(request_reference=rr,result_reference=result_ref,
            files=[file_reference(path) for path in sorted(out.rglob('*')) if path.is_file()]))
        need(time.monotonic()<deadline,'Evidence exceeded internal deadline')
    except Exception as exc:
        write(out/'failure.json',dict(error=type(exc).__name__,message=str(exc),request_reference=rr,
            requires_terminal_parent_settlement=True,scientific_acceptance=False,phase=phase,
            phase_seconds={**timings,phase:time.monotonic()-phase_started},attempt_elapsed_seconds=time.monotonic()-began))
        if not (out/'failure_artifact_manifest.json').exists():
            write(out/'failure_artifact_manifest.json',dict(request_reference=rr,
                files=[file_reference(path) for path in sorted(out.rglob('*')) if path.is_file()]))
        raise


if __name__=='__main__':
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--expected-request-sha256',required=True)
    args=parser.parse_args();execute(args.request,args.expected_request_sha256)
