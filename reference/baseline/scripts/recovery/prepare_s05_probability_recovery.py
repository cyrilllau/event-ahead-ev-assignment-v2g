"""Source-bound probability-only continuation of an independently audited failure.

This is a standalone bridge, NOT a C&CG/worker checkpoint importer. It may solve
the small probability primal/dual, but never assembles or solves a master or
recourse problem. Original evidence is read-only. A successful result is UB-only,
not a locked plan, a converged algorithm, an OOS permit, or a completed spec.

The complete entry path has local provenance-fixture tests. Formal execution
still requires the actual accepted audit and independently reviewed request;
terminal parent-job settlement is mandatory before any later budget can be used.
"""
from __future__ import annotations

import argparse
from contextvars import ContextVar
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

import numpy as np

from evcs_experiments.config import get_paths
from evcs_experiments.failure_carryover import ENGINE_FILES, verify_failure_carryover
from evcs_experiments.probability import ProbabilitySet, solve_worst_distribution
from evcs_experiments.probability_check import check_probability_solution
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.runtime.jobctl import observation

KIND = 'PROBABILITY_CERTIFICATE_REPAIR_FULL160_V1'
PROBABILITY_SOURCE_SHA256 = '39a33c1dab694a65113eeb9424c2eaf7e056d4e3478ec4d0f3c84b454754d68c'
AUDIT_STATUS = 'PASS_SAVED_RMP_PRIMAL_AND_160_LP_AUDIT'
REVIEW_STATUS = 'PASS_SAVED160_COMPONENT_REVIEW'
CAP_SECONDS = 7200.
_HASH_DEADLINE = ContextVar('recovery_hash_deadline', default=None)
_VERIFIED_REFS = ContextVar('recovery_verified_refs', default=None)
MAX_PROGRAM_SECONDS=120.


def parent_binding(request, request_ref):
    """Bind a directly supervised job; creation time gives a conservative wall."""
    p=request['parent_job'];paths=get_paths();directory=Path(p['directory']).resolve()
    need(directory==paths.campaign_root.resolve()/'jobs'/p['job_id'], 'Successor parent campaign/path differs')
    need(os.environ.get('EVCS_JOB_ID')==p['job_id'] and os.environ.get('EVCS_JOB_DIRECTORY')==str(directory), 'Successor parent environment differs')
    manifest_ref=reference(directory/'manifest.json');manifest=read(manifest_ref['path']);status=read(directory/'status.json')
    expected=[sys.executable,'-B',str(Path(__file__).resolve()),'--request',request_ref['path'],
              '--expected-request-sha256',request_ref['sha256'],'--action','run']
    need(manifest['job_id']==p['job_id'] and manifest['spec_id']=='S05' and manifest['argv']==expected
         and manifest['cwd']==str(paths.project_root.resolve()) and manifest['outroot']==str(directory.parent), 'Successor parent command/source differs')
    need(manifest['threads']==p['threads']==1 and manifest['timeout_seconds']==p['timeout_seconds']==request['wall_seconds']+10
         and manifest['grace_seconds']==p['grace_seconds']==5, 'Successor parent hard wall or reserve differs')
    need(verify(p['controller_reference'])==Path(manifest['controller_path']).resolve()
         ==paths.project_root/'src/evcs_experiments/runtime/jobctl.py', 'Unbound parent hard-wall controller')
    need(status['state']=='RUNNING' and os.getsid(0)==status['supervisor_sid']
         and os.getpid()==status['command_pid'] and observation(status)['supervisor_alive'], 'Bridge is not the live direct parent command')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        need(os.environ.get(key)=='1', 'Probability recovery thread environment differs')
    created=datetime.fromisoformat(manifest['created_at_utc'])
    age=(datetime.now(timezone.utc)-created).total_seconds()
    need(created.tzinfo is not None and 0<=age<manifest['timeout_seconds']-10, 'Expired or reversed parent clock')
    available=min(request['wall_seconds'],manifest['timeout_seconds']-10-age)
    return dict(job_id=p['job_id'],directory=str(directory),manifest_reference=manifest_ref,
                available_seconds=available,manifest_age_seconds=age,
                deadline_monotonic=time.monotonic()+available)


def need(ok, message):
    if not bool(ok):
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))


def reference(path):
    p = Path(path).resolve(strict=True)
    h = hashlib.sha256()
    with p.open('rb') as f:
        while chunk := f.read(4 * 1024 * 1024):
            deadline = _HASH_DEADLINE.get()
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('Recovery evidence hashing exhausted the shared wall budget')
            h.update(chunk)
    return dict(path=str(p), sha256=h.hexdigest(), size_bytes=p.stat().st_size)


def verify(ref):
    need(isinstance(ref, dict) and set(ref) == {'path', 'sha256', 'size_bytes'}, 'Malformed strong reference')
    cache = _VERIFIED_REFS.get()
    if cache is not None and cache.get(ref['path']) == ref:
        return Path(ref['path'])
    need(Path(ref['path']).is_absolute() and reference(ref['path']) == ref, 'Evidence bytes changed: ' + str(ref.get('path')))
    if cache is not None:
        cache[ref['path']] = ref
    return Path(ref['path'])


def load(ref):
    return read(verify(ref))


def write(path, value):
    # Small newly produced receipts must still seal after the input-hash
    # deadline. Their bytes are hashed in memory; no expensive input read occurs.
    payload=(json.dumps(value, indent=2, allow_nan=False)+'\n').encode()
    p=Path(path).resolve()
    with p.open('xb') as f:
        f.write(payload)
    return dict(path=str(p),sha256=hashlib.sha256(payload).hexdigest(),size_bytes=len(payload))


def release(ref):
    manifest = load(ref)
    root = Path(ref['path']).parent
    files = {}
    for row in manifest['files']:
        p = (root / row['path']).resolve()
        need(p.is_relative_to(root) and str(p) not in files, 'Release path escape/duplicate')
        item = dict(path=str(p), sha256=row['sha256'], size_bytes=row['size_bytes'])
        verify(item)
        files[str(p)] = item
    prefix = root / 'src/evcs_experiments'
    sources = {str(Path(p).relative_to(prefix)): r['sha256'] for p, r in files.items()
               if Path(p).is_relative_to(prefix) and p.endswith('.py')}
    need(set(sources) == {str(p.relative_to(prefix)) for p in prefix.rglob('*.py')}, 'Release Python inventory differs')
    need(set(ENGINE_FILES) <= set(sources), 'Incomplete scientific engine closure')
    return root, files, sources, canonical_hash({name: sources[name] for name in ENGINE_FILES})


def verify_source_transition(oldsrc, newsrc):
    need(set(oldsrc) == set(newsrc) and {n for n in oldsrc if oldsrc[n] != newsrc[n]} == {'probability.py'},
         'Unapproved scientific/adapter source changes')
    need(newsrc['probability.py'] == PROBABILITY_SOURCE_SHA256, 'Probability source is not the reviewed tolerance repair')


def verify_pending(state, qinput, ambiguity_input):
    """Pure identity gate; no proof is inferred from a solver's passed label."""
    need(state['status'] == 'PROBABILITY_FAILED' and state['trace'] == [] and state['next_iteration'] == 1,
         'Only the first probability-failed pending iteration is supported')
    need(all(state[k] is None for k in ('best', 'lower_bound_usd', 'upper_bound_usd')), 'Unexpected incumbent/LB/UB')
    pending = state['pending']
    ids = state['identity']['scenario_ids']
    q = np.asarray(pending['q_usd'], dtype=float)
    need(len(ids) == len(set(ids)) == 160 and q.shape == (160,) and np.isfinite(q).all(), 'Complete ordered 160 q required')
    need(pending['iteration'] == 1 and len(pending['recourse_certificates']) == 160, 'Incomplete pending certificates')
    need(qinput['support_ids'] == ids and qinput['q_usd'] == pending['q_usd'] and qinput['plan'] == pending['plan'],
         'Original probability q/plan/support differs')
    for i, cert in enumerate(pending['recourse_certificates']):
        need(cert['index'] == i and cert['scenario_id'] == ids[i] and cert['optimal_certificate_valid'] is True,
             'Pending certificate addressing differs')
    a = ProbabilitySet(tuple(ambiguity_input['support_ids']), tuple(ambiguity_input['feature_ids']),
                       np.asarray(ambiguity_input['features']), np.asarray(ambiguity_input['lower']),
                       np.asarray(ambiguity_input['upper']))
    need(list(a.support_ids) == ids and a.features.shape == (160, 5), 'Wrong probability axes')
    need(a.ambiguity_sha256 == ambiguity_input['ambiguity_sha256'] == state['identity']['effective_ambiguity_sha256'],
         'Ambiguity content differs')
    need(state['config']['separation_absolute_usd'] == 1e-4 and state['config']['separation_relative'] == 1e-8,
         'Original separation thresholds changed')
    rmp = pending['rmp']
    need(rmp['status'] == 'TIME_LIMIT' and rmp['optimal'] is False and rmp['lower_bound_usd'] is None
         and rmp['certificate']['solver_bound_usable'] is False, 'Original limited master must retain null LB')
    need(pending['last_incomplete_probability']['certified_optimal'] is False, 'Original probability was not a failed certificate')
    active = pending['active_indices']
    need(active == state['active_indices'] and len(active) == len(set(active)) and active
         and all(type(i) is int and 0 <= i < 160 for i in active), 'Original active support differs')
    return a, q


def separate(state, ambiguity, q, first_stage_cost, probability_upper_bound):
    """Use ONLY the original master dual; never substitute the new LP dual."""
    pending = state['pending']
    dual = pending['raw_rmp_dual']
    alpha = np.asarray(dual['alpha'], dtype=float)
    beta = np.asarray(dual['beta'], dtype=float)
    need(alpha.shape == beta.shape == (5,) and np.isfinite(alpha).all() and np.isfinite(beta).all()
         and math.isfinite(dual['theta0']), 'Malformed original master dual')
    residual = q - (dual['theta0'] + ambiguity.features @ (beta - alpha))
    need(np.isfinite(residual).all(), 'Nonfinite original master separation')
    active = pending['active_indices']
    omitted = [i for i in range(len(q)) if i not in active]
    included = float(np.max(residual[active], initial=0.))
    full = float(np.max(residual, initial=0.))
    tolerance = 1e-4 + 1e-8 * max(1., float(np.max(np.abs(q))))
    worst = omitted[int(np.argmax(residual[omitted]))] if omitted else None
    add = worst if worst is not None and residual[worst] > tolerance else None
    need(math.isfinite(first_stage_cost) and math.isfinite(probability_upper_bound), 'Nonfinite derived upper bound')
    valid = included <= tolerance
    return dict(status='VALID_INCUMBENT_BOUND_UNAVAILABLE' if valid else 'RMP_RESIDUAL_FAILED',
                lower_bound_usd=None, upper_bound_usd=float(first_stage_cost + probability_upper_bound),
                gap_relative=None, lockable=False, formal_OOS=False, scientific_acceptance=False,
                full_support_q_audited=True, original_master_status='TIME_LIMIT',
                original_rmp_dual=dual, residuals_usd=residual.tolist(),
                included_maximum_usd=included, full_support_maximum_usd=full,
                separation_tolerance_usd=tolerance, included_residual_passed=valid,
                original_active_indices=active, next_added_index=add if valid else None,
                next_active_indices=active + ([add] if valid and add is not None else []),
                requires_new_master_bound=True, normal_ccg_checkpoint_compatible=False)


def verify_audit(refs, state, history, origin_refs):
    audit = load(refs['full160_audit'])
    review = load(refs['independent_acceptance'])
    manifest = load(refs['audit_artifact_manifest'])
    inputs = load(refs['audit_input_references'])
    need(audit['status'] == AUDIT_STATUS and audit['completed_lp_count'] == audit['expected_lp_count'] == 160,
         'An actual complete full160 audit is required; diagnostic prefixes are inadmissible')
    need(audit['solver_calls'] == 0 and audit['formal_OOS'] is False and audit['probability_PASS'] is False
         and audit['original_LB'] is None and audit['original_engine_status'] == 'PROBABILITY_FAILED', 'Audit scope differs')
    expected = {**origin_refs, **{k: refs[k] for k in ('full160_audit', 'audit_artifact_manifest', 'audit_input_references')}}
    need(review['status'] == REVIEW_STATUS and all(review['references'].get(k) == v for k, v in expected.items())
         and review['problem_sha256'] == state['problem_sha256'] and review['complete_lp_count'] == 160
         and review['probability_PASS'] is False and review['formal_OOS'] is False, 'Independent full160 acceptance binding differs')
    for ref in review['references'].values():
        verify(ref)
    if 'problem_identity' in review:
        need(review['problem_identity'] == state['identity'], 'Independent acceptance scientific identity differs')
    artifacts = {}
    root = Path(refs['full160_audit']['path']).parent
    for row in manifest:
        p = verify(row)
        need(p.parent == root and p.name not in artifacts, 'Audit artifact location/duplicate')
        artifacts[p.name] = row
    need(artifacts.get('audit.json') == refs['full160_audit']
         and artifacts.get('input_references.json') == refs['audit_input_references'], 'Audit manifest scope differs')
    input_map = {r['path']: r for r in inputs}
    need(len(input_map) == len(inputs), 'Duplicate full-byte audit input references')
    rawroot=Path(origin_refs['raw_request']['path']).parent
    need({str(p.resolve()) for p in rawroot.rglob('*') if p.is_file()} ==
         {p for p in input_map if Path(p).is_relative_to(rawroot)}, 'Original raw inventory differs from accepted audit')
    # Refresh every original byte hash. Do not substitute inventory/size checks
    # for this pass, and do not rerun the already accepted KKT/physical audit.
    for ref in inputs:
        verify(ref)
    for ref in origin_refs.values():
        need(input_map.get(ref['path']) == ref, 'Origin was not included in full-byte audit')
    for row in history:
        need(input_map.get(row['path']) == row, 'A raw callback is absent from full-byte audit')
    pending = state['pending']
    for i, cert in enumerate(pending['recourse_certificates']):
        name = f'lp_{i:03d}.json'
        need(name in artifacts, 'Missing independent LP proof')
        lp = read(artifacts[name]['path'])
        need(lp['index'] == i and lp['scenario_id'] == cert['scenario_id'] and lp['q_usd'] == pending['q_usd'][i]
             and lp['template_sha256'] == cert['template_sha256'] and lp['physical_passed'] is True,
             'Independent LP proof does not certify the exact pending q')
        vals = [lp[k] for k in ('primal_residual', 'stationarity', 'dual_sign_violation', 'infinite_bound_dual_violation',
                                'duality_gap_usd', 'primal_objective_usd', 'dual_objective_usd')]
        need(all(math.isfinite(v) for v in vals) and 0 <= vals[0] <= 1e-7
             and all(0 <= v <= 1e-8 for v in vals[1:4])
             and 0 <= vals[4] <= 1e-4 + 1e-7 * max(1., abs(vals[5]), abs(vals[6])), 'Independent LP certificate threshold failed')
    rmp = read(artifacts['rmp.json']['path'])
    need(rmp == audit['rmp'] and rmp['status'] == 'TIME_LIMIT' and rmp['lower_bound_usd'] is None
         and rmp['solver_bound_usable'] is False, 'Audit master bound/status differs')
    return audit, rmp, read(artifacts['raw_rmp_dual_residuals.json']['path'])


def opening_budget(refs, worker, invocation):
    job = load(refs['job_result'])
    manifest = load(refs['job_manifest'])
    queue = load(invocation['request_reference'])
    need(job['complete'] is True and job['state'] == 'FAILED' and job['returncode'] != 0,
         'Predecessor must be a genuine terminal failure')
    need(not observation(job)['blocks_new_launch'], 'Predecessor process still blocks successor')
    need(invocation['parent_job']['manifest_reference'] == refs['job_manifest']
         and invocation['parent_job']['job_id'] == job['job_id'] == manifest['job_id']
         and invocation['item_request_reference'] == refs['worker_request'], 'Original parent/worker binding differs')
    need(len(queue['items']) == 1 and queue['items'][0]['request_reference'] == refs['worker_request']
         and queue['items'][0]['plan_id'] == worker['plan_id'] == invocation['plan_id'], 'Only one fixed predecessor plan is supported')
    need(invocation['previous_checkpoint'] is None and invocation['previous_result_reference'] is None,
         'Unsupported prior queue attempts must not reset budget')
    carry = verify_failure_carryover(worker['failure_carryover_reference'], worker, queue['queue_root'])
    need(invocation['failure_carryover_reference'] == worker['failure_carryover_reference']
         and invocation['cumulative_plan_seconds'] == invocation['opening_carryover_seconds'] == carry['accounted_seconds'],
         'Prior carryover must be charged exactly once')
    elapsed = job['elapsed_seconds']
    need(type(elapsed) in (float, int) and math.isfinite(elapsed) and elapsed > 0, 'Invalid terminal parent elapsed')
    opening = float(carry['accounted_seconds'] + elapsed)
    need(worker['budget']['cumulative_wall_seconds'] == invocation['plan_budget_seconds'] == CAP_SECONDS
         and 0 < opening < CAP_SECONDS, 'Original cumulative budget exhausted or changed')
    return dict(opening_cumulative_seconds=opening, prior_carryover_seconds=carry['accounted_seconds'],
                terminal_parent_seconds=elapsed, remaining_seconds=CAP_SECONDS-opening,
                cap_seconds=CAP_SECONDS, nested_driver_seconds_added=False,
                predecessor_job_result_reference=refs['job_result'],
                prior_carryover_reference=worker['failure_carryover_reference'])


def prepare(request_reference):
    """Verify the immutable request. This function has no write or solver calls."""
    req = load(request_reference)
    need(req['schema_version'] == 1 and req['kind'] == KIND and req['domain'] == 'formal'
         and req['spec_id'] == 'S05' and req['method_id'] == 'EA' and req['target_id'] == 'michael_2018', 'Unsupported recovery scope')
    refs = req['references']
    oldroot, _, oldsrc, oldengine = release(refs['old_source_manifest'])
    newroot, newfiles, newsrc, newengine = release(refs['new_source_manifest'])
    need(newroot == get_paths().project_root.resolve(), 'Recovery must use the pinned new installed package')
    verify_source_transition(oldsrc,newsrc)
    approval = load(refs['probability_source_approval'])
    need(approval == dict(status='PASS_PROBABILITY_SOURCE_CHANGE_REVIEW', old_source_manifest=refs['old_source_manifest'],
                         new_source_manifest=refs['new_source_manifest'], changed_sources={'probability.py': dict(old=oldsrc['probability.py'], new=newsrc['probability.py'])},
                         review_reference=refs['probability_review']), 'Exact probability source approval differs')
    verify(refs['probability_review'])
    need(verify(refs['recovery_script']) == Path(__file__).resolve()
         and newfiles.get(str(Path(__file__).resolve())) == refs['recovery_script'], 'Recovery script is not in the pinned release')
    for name, module in tuple(sys.modules.items()):
        if name == 'evcs_experiments' or name.startswith('evcs_experiments.'):
            filename = getattr(module, '__file__', None)
            if filename is not None:
                p = str(Path(filename).resolve())
                need(p in newfiles, 'Mixed or unbound loaded package source')
                verify(newfiles[p])
    cp = load(refs['checkpoint']); state = cp['state']; history = load(refs['history'])
    need(cp['sha256'] == canonical_hash(state) and state['problem_sha256'] == canonical_hash(state['identity'])
         and state['engine_sha256'] == oldengine, 'Original checkpoint semantic/source identity differs')
    rawreq = load(refs['raw_request']); worker = load(refs['worker_request']); inv = load(refs['invocation'])
    need(rawreq['problem_identity'] == state['identity'] and rawreq['domain'] == 'formal' and rawreq['formal_OOS'] is False,
         'Original raw identity differs')
    need(worker['plan_id'] == req['plan_id'] and worker['method_id'] == req['method_id'] and worker['target_id'] == req['target_id']
         and worker['domain'] == 'formal' and worker['release_reference'] == refs['old_source_manifest'], 'Original worker identity differs')
    origin = Path(worker['output_root']).resolve()
    need(refs['checkpoint']['path'] == str(origin/'driver/checkpoint.json')
         and refs['history']['path'] == str(origin/'raw/history.json')
         and refs['raw_request']['path'] == str(origin/'raw/request.json'), 'Wrong original evidence root')
    need(len(history) == 163 and [load(r)['event'] for r in history] == ['phase1', 'rmp'] + ['recourse']*160 + ['probability'],
         'Original callback inventory differs')
    probrow = load(history[-1]); probrefs = {Path(r['path']).name: r for r in probrow['artifacts']}
    for r in probrefs.values(): verify(r)
    qinput = read(probrefs['q_input.json']['path'])
    ainput = read(probrefs['probability_input.json']['path'])['effective_ambiguity']
    need(qinput['recourse_evidence'] == history[2:-1], 'Probability raw q provenance differs')
    need(read(probrefs['solver.json']['path']) == state['pending']['last_incomplete_probability'], 'Old probability failure changed')
    a, q = verify_pending(state, qinput, ainput)
    origins = {k: refs[k] for k in ('job_result', 'checkpoint', 'history', 'raw_request')}
    origins['source_manifest'] = refs['old_source_manifest']
    audit, rmp, saved_residual = verify_audit(refs, state, history, origins)
    check = separate(state, a, q, rmp['physical_first_stage_cost_usd'], 0.)
    need(saved_residual['support_ids'] == list(a.support_ids) and saved_residual['active_indices'] == state['active_indices']
         and saved_residual['residuals_usd'] == check['residuals_usd']
         and audit['included_dual_violation_usd'] == check['included_maximum_usd']
         and audit['full_support_dual_violation_usd'] == check['full_support_maximum_usd'], 'Original master residual audit differs')
    budget = opening_budget(refs, worker, inv)
    wall = req['wall_seconds']
    need(type(wall) in (int, float) and math.isfinite(wall) and 0 < wall <= min(MAX_PROGRAM_SECONDS, budget['remaining_seconds']-15), 'Invalid bounded recovery wall')
    out = Path(req['output_root']).resolve()
    jobpath = Path(refs['job_result']['path'])
    need(jobpath.name == 'result.json' and jobpath.parents[1].name == 'jobs', 'Unknown predecessor campaign root')
    campaign = jobpath.parents[2]
    need(out.is_relative_to(campaign/'recoveries') and not out.exists(), 'Fresh dedicated campaign recovery output required')
    protected = (oldroot, newroot, origin, get_paths().data_root.resolve(), Path(refs['full160_audit']['path']).parent)
    need(not any(out.is_relative_to(p) for p in protected), 'Recovery output overlaps protected evidence')
    return dict(request=req, request_reference=request_reference, state=state, ambiguity=a, q=q, budget=budget,
                first_stage_cost_usd=rmp['physical_first_stage_cost_usd'], origin_engine_sha256=oldengine,
                recovery_engine_sha256=newengine, output=out, campaign=campaign)


def claim_path(verified):
    return (verified['campaign']/'recovery_claims'/(canonical_hash(verified['request']['references']['job_result'])+'.json')).resolve()


def reserve_successor(verified):
    """Exclusive current-chain-tail claim. Never rewrites a previous claim."""
    p = claim_path(verified)
    p.parent.mkdir(parents=True, exist_ok=True)
    return write(p, dict(kind=KIND, request_reference=verified['request_reference'],
                         predecessor_job_result_reference=verified['request']['references']['job_result'],
                         successor_output_root=str(verified['output']), plan_id=verified['request']['plan_id']))


def verify_claim(verified, ref):
    need(verify(ref) == claim_path(verified), 'Wrong predecessor claim location')
    need(load(ref) == dict(kind=KIND, request_reference=verified['request_reference'],
                          predecessor_job_result_reference=verified['request']['references']['job_result'],
                          successor_output_root=str(verified['output']), plan_id=verified['request']['plan_id']), 'Predecessor already claimed by another successor')


def execute(request_reference, claim_reference=None):
    """Atomic verify/claim/solve; no repeated prepare on the normal run path.

The legacy carryover verifier is checked before/after, but has no chunk deadline
hook. Production use requires the explicitly outstanding hard-wall parent job.
"""
    started = time.monotonic()
    request = load(request_reference)
    need(request['kind']==KIND and request['domain']=='formal' and request['spec_id']=='S05', 'Unknown attempt scope')
    wall = request['wall_seconds']
    need(type(wall) in (int, float) and math.isfinite(wall) and 0 < wall <= MAX_PROGRAM_SECONDS, 'Invalid recovery wall')
    parent=parent_binding(request,request_reference)
    # Reserve this chain tail before expensive verification. A hard kill leaves
    # the reservation in place; retry requires a separately settled successor.
    jobref=request['references']['job_result'];verify(jobref)
    jobpath=Path(jobref['path'])
    need(jobpath.name=='result.json' and jobpath.parents[1].name=='jobs', 'Unknown attempt campaign')
    campaign=jobpath.parents[2];out=Path(request['output_root']).resolve()
    need(out.is_relative_to(campaign/'recoveries') and not out.exists(), 'Fresh recovery attempt output required')
    attempt_dir=campaign/'recovery_attempts'/canonical_hash(jobref)
    paths=get_paths()
    need(campaign == paths.campaign_root.resolve(), 'Attempt campaign differs from the configured authority')
    protected=(paths.project_root.resolve(),paths.data_root.resolve(),paths.frozen_source_root.resolve())
    need(not any(attempt_dir.is_relative_to(p) or out.is_relative_to(p) for p in protected), 'Attempt overlaps protected inputs/source')
    attempt_dir.mkdir(parents=True,exist_ok=False)
    write(attempt_dir/'reservation.json',dict(kind=KIND,request_reference=request_reference,
        predecessor_job_result_reference=jobref,output_root=str(out),automatic_retry_allowed=False,
        started_monotonic=started,parent_job=parent,final_parent_job_settlement_required=True))
    token = _HASH_DEADLINE.set(min(started+wall,parent['deadline_monotonic']))
    cache_token = _VERIFIED_REFS.set({})
    try:
        return _execute(request_reference, claim_reference, started,parent)
    except Exception as exc:
        write(attempt_dir/'failure.json',dict(status='TIME_LIMIT' if isinstance(exc,TimeoutError) else 'FAILED_VERIFICATION_OR_EXECUTION',
            error_type=type(exc).__name__,error=str(exc),elapsed_seconds=time.monotonic()-started,
            request_reference=request_reference,original_job_result_reference=jobref,
            parent_job=parent,
            cumulative_seconds_provisional=True,requires_final_parent_job_settlement=True,
            automatic_retry_allowed=False,scientific_acceptance=False,formal_OOS=False,lockable=False))
        raise
    finally:
        _VERIFIED_REFS.reset(cache_token)
        _HASH_DEADLINE.reset(token)


def _execute(request_reference, claim_reference, started,parent=None):
    """One bounded standalone execution; final parent-job accounting is separate."""
    verified = prepare(request_reference)
    req = verified['request']; out = verified['output']; budget = verified['budget']
    def remaining():
        left = min(req['wall_seconds'], budget['remaining_seconds']) - (time.monotonic() - started)
        if parent is not None:left=min(left,parent['deadline_monotonic']-time.monotonic())
        if left <= 0:
            raise TimeoutError('Probability recovery wall/cumulative budget exhausted')
        return left
    remaining()
    if claim_reference is None:
        claim_reference=reserve_successor(verified)
    else:
        verify_claim(verified, claim_reference)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = []
    probability = None
    try:
        artifacts.append(write(out/'origin.json', dict(kind=KIND, request_reference=request_reference,
            claim_reference=claim_reference, original_checkpoint_reference=req['references']['checkpoint'],
            original_engine_sha256=verified['origin_engine_sha256'], recovery_engine_sha256=verified['recovery_engine_sha256'],
            original_probability_status=verified['state']['pending']['last_incomplete_probability']['status'],
            source_refs={k:req['references'][k] for k in ('old_source_manifest','new_source_manifest','recovery_script')},
            reused_q_count=160, recourse_solver_calls=0, master_solver_calls=0, formal_OOS=False, **budget)))
        left = remaining() - 2.  # preserve explicit local export time; never borrow from plan cap
        if left <= 0: raise TimeoutError('Insufficient time to seal probability witnesses')
        probability = solve_worst_distribution(verified['ambiguity'], verified['q'],
            q_support_ids=verified['ambiguity'].support_ids, time_limit_seconds=left)
        raw = probability.as_dict()
        artifacts.append(write(out/'probability_solver.json', raw))
        independent = check_probability_solution(verified['ambiguity'], verified['q'], raw,
                                                 q_support_ids=verified['ambiguity'].support_ids)
        artifacts.append(write(out/'probability_independent_check.json', independent.as_dict()))
        remaining()
        if not probability.certified_optimal or not independent.passed:
            status = 'TIME_LIMIT' if probability.status == 'TIME_LIMIT' else 'PROBABILITY_CERTIFICATE_FAILED'
            derived = dict(status=status, lower_bound_usd=None, upper_bound_usd=None, gap_relative=None,
                           lockable=False, scientific_acceptance=False, formal_OOS=False)
        else:
            derived = separate(verified['state'], verified['ambiguity'], verified['q'],
                               verified['first_stage_cost_usd'], probability.upper_bound_usd)
            derived.update(plan=verified['state']['pending']['plan'], q_usd=verified['q'].tolist(),
                           support_ids=list(verified['ambiguity'].support_ids),
                           original_checkpoint_reference=req['references']['checkpoint'],
                           original_problem_sha256=verified['state']['problem_sha256'],
                           origin_engine_sha256=verified['origin_engine_sha256'],
                           recovery_engine_sha256=verified['recovery_engine_sha256'])
        artifacts.append(write(out/'derived_state.json', derived))
        remaining()
        status = derived['status']
    except Exception as exc:
        status = 'TIME_LIMIT' if isinstance(exc, TimeoutError) else 'RECOVERY_FAILED_REQUIRES_REVIEW'
        artifacts.append(write(out/'failure.json', dict(status=status, error_type=type(exc).__name__, error=str(exc),
            partial_outputs_retained=True, lockable=False, formal_OOS=False)))
    elapsed = time.monotonic()-started
    result = dict(kind=KIND, status=status, request_reference=request_reference, claim_reference=claim_reference,
        parent_job=parent,
        artifacts=artifacts, original_job_result_reference=req['references']['job_result'],
        original_checkpoint_reference=req['references']['checkpoint'],
        opening_budget=budget, attempt_elapsed_seconds=elapsed,
        cumulative_accounted_seconds=budget['opening_cumulative_seconds']+elapsed,
        cumulative_seconds_provisional=True, requires_final_parent_job_settlement=True,
        accounting_authority='opening carryover plus original terminal parent job plus successor terminal parent job once',
        scientific_acceptance=False, lockable=False, formal_OOS=False, lower_bound_usd=None, gap_relative=None,
        recourse_solver_calls=0, master_solver_calls=0, automatic_resume_supported=False)
    if elapsed >= req['wall_seconds'] or result['cumulative_accounted_seconds'] >= CAP_SECONDS or (parent is not None and time.monotonic()>=parent['deadline_monotonic']):
        result['status']='TIME_LIMIT'
    write(out/'result.json', result)
    # A sealing overrun must never leave a successful terminal record.
    if (time.monotonic()-started >= req['wall_seconds'] or (parent is not None and time.monotonic()>=parent['deadline_monotonic'])) and result['status'] != 'TIME_LIMIT':
        (out/'result.json').rename(out/'unaccepted_result.json')
        result['status']='TIME_LIMIT'
        write(out/'result.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request', type=Path, required=True)
    p.add_argument('--expected-request-sha256', required=True)
    p.add_argument('--action', choices=('verify','run'), default='verify')
    args = p.parse_args()
    ref = reference(args.request)
    need(ref['sha256'] == args.expected_request_sha256, 'Request SHA differs')
    if args.action == 'run':
        result = execute(ref)
        print(json.dumps(result, allow_nan=False))
        return 0 if result['status'] == 'VALID_INCUMBENT_BOUND_UNAVAILABLE' else 2
    verified = prepare(ref)
    print(json.dumps(dict(status='VERIFIED_ONLY_NO_SOLVER_CALLS', budget=verified['budget']), allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
