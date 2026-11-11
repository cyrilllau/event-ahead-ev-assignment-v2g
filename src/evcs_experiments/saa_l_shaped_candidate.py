"""Source-bound L-shaped candidates and no-solve joint numerical audits.

Trust enters through caller-retained whole-file references, including *all* old
and new master executions. Rehashing altered evidence is not authorization.
Formal physical evidence has an explicit separate ingress; it is not job admission.
Parent supervision, inherited expenses, sealing and formal worker remain external.
"""
import json
import time
from pathlib import Path
import numpy as np
from .campaign_batch import file_reference, verify_reference
from .config import get_paths
from .random_streams import canonical_hash
from . import saa_l_shaped as kernel
from . import saa_l_shaped_master_evidence as masters
from . import saa_fixed_plan_audit as full_q

CANDIDATE = 'SAA_L_SHAPED_CANDIDATE_DEVELOPMENT_V1'
AUDIT = 'SAA_L_SHAPED_JOINT_AUDIT_DEVELOPMENT_V1'
FORMAL_CANDIDATE = 'SAA_L_SHAPED_CANDIDATE_FORMAL_V1'
FORMAL_AUDIT = 'SAA_L_SHAPED_JOINT_AUDIT_FORMAL_V1'
EVIDENCE_KEYS = ('kernel_checkpoint_reference', 'master_execution_references',
                 'full_q_checkpoint_reference', 'full_q_request_reference',
                 'template_envelope_reference', 'source_references')


def _need(ok, message):
    if not bool(ok):
        raise ValueError(message)


def _read(reference):
    path = Path(reference['path'])
    _need(path.is_absolute() and path == path.resolve(), 'Canonical absolute evidence path required')
    verify_reference(reference)
    return json.loads(path.read_text())


def _write(path, value):
    path = Path(path)
    _need(path.is_absolute() and path == path.resolve(), 'Canonical absolute output path required')
    paths = get_paths()
    for protected in (paths.data_root, paths.frozen_source_root, paths.project_root / 'src'):
        _need(not path.is_relative_to(Path(protected).resolve()), 'Output overlaps frozen data or source')
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return file_reference(path)


def _sources(references):
    _need(isinstance(references, list) and references, 'Explicit actual sources required')
    _need(len({r['path'] for r in references}) == len(references), 'Duplicate source reference')
    for ref in references:
        path = Path(ref['path'])
        _need(path.is_absolute() and path == path.resolve(), 'Aliased source reference')
        verify_reference(ref)
    for path in (__file__, kernel.__file__, masters.__file__, full_q.__file__):
        _need(file_reference(path) in references, 'Actual joint candidate/audit source missing')


def _same_numeric(actual, expected, message):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    _need(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
          and np.allclose(a, b, rtol=0, atol=1e-8), message)


def _native_provenance(execution, reference):
    """Keep backend-native codes distinct from the kernel's 0/1 encoding."""
    prep = _read(execution['preparation_reference'])
    backend = prep['solver_backend']
    normalized = execution['native_result']['status']
    if backend == 'gurobipy':
        masters._verify_native_adapter(prep, execution)
        raw = _read(execution['native_adapter_reference'])['native_record']
        raw_status = raw['raw_model_status']
        name = raw['status']
        _need((raw_status in (2, 9) and normalized == (0 if raw_status == 2 else 1))
              or (raw_status == 1 and normalized == 1 and name == 'MODEL_LOADING_TIME_LIMIT'),
              'Gurobi and normalized native status differ')
    else:
        _need(backend == 'scipy.optimize.milp/HiGHS' and 'native_adapter_reference' not in execution,
              'Master backend and actual native provenance differ')
        raw_status = normalized
        name = 'OPTIMAL' if normalized == 0 else 'TIME_LIMIT'
    return dict(backend=backend, raw_model_status=raw_status,
        normalized_status=normalized, status_name=name, execution_reference=reference)


def _formal_native_guard(execution):
    """A license-free caller double can never supply formal solver evidence."""
    _need('native_adapter_reference' in execution, 'Formal native adapter evidence required')
    native = _read(execution['native_adapter_reference'])['native_record']
    _need(native.get('backend') == 'gurobipy' and native.get('gurobi_version') == [12, 0, 3]
          and native.get('synthetic_backend_test_double', False) is False,
          'Actual pinned Gurobi execution is required; test doubles cannot enter formal evidence')


def _domain_contract(domain):
    _need(domain in ('development', 'formal'), 'Explicit L-shaped domain required')
    if domain == 'formal':
        from .saa_l_shaped_formal import FORMAL_ALGORITHM_ID
        return (FORMAL_ALGORITHM_ID, FORMAL_CANDIDATE, FORMAL_AUDIT,
                'L_SHAPED_REAL_LP_EVENT_FORMAL_V1',
                'INDEPENDENT_SAA_SMALL_MASTER_AUDIT_FORMAL_V1')
    return (kernel.ALGORITHM_ID, CANDIDATE, AUDIT,
            'L_SHAPED_REAL_LP_EVENT_DEVELOPMENT_V1',
            'INDEPENDENT_SAA_SMALL_MASTER_AUDIT_DEVELOPMENT_V1')


def _problem_scope(problem, domain):
    if domain == 'formal':
        from .saa_l_shaped_formal import validate_formal_problem
        return validate_formal_problem(problem)
    _need(domain == 'development' and problem.domain == 'development'
          and problem.tasks.config.get('synthetic_validation_only') is True,
          'Only synthetic development candidates are supported')
    return None


def _bound_candidate(problem, plan_id, evidence, *, domain='development'):
    _need(isinstance(plan_id, str) and plan_id.strip() == plan_id and plan_id, 'Explicit plan_id required')
    physical = _problem_scope(problem, domain)
    if domain == 'formal':
        _need(_read(physical['worker_request_reference'])['plan_id'] == plan_id,
              'Formal candidate differs from original requested plan')
    algorithm, candidate_kind, _, producer, _ = _domain_contract(domain)
    _need(set(evidence) == set(EVIDENCE_KEYS), 'Explicit evidence binding fields differ')
    _sources(evidence['source_references'])
    envelope = _read(evidence['kernel_checkpoint_reference'])
    state = envelope['state']
    _need(envelope['sha256'] == canonical_hash(state), 'Kernel checksum changed')
    _need(state['identity'] == problem.identity and state['source_identity'] == kernel.source_identity()
          and state['algorithm_id'] == algorithm, 'Kernel problem or algorithm/source differs')
    _need(state['status'] in ('TIME_LIMIT', 'ITERATION_LIMIT', 'CONVERGED'), 'Active or failed kernel has no terminal candidate')
    for name in ('formal_admission', 'formal_OOS', 'scientific_acceptance'):
        _need(state[name] is False, 'Kernel admission scope differs')
    best = state['best']
    _need(best is not None, 'Candidate requires a complete best-plan sweep')
    n = len(problem.support_ids)
    _need(domain != 'formal' or (n == 160 and np.array_equal(
        np.asarray(problem.identity['empirical_probabilities']), np.full(n, 1/n))),
        'Formal SAA requires all 160 support scenarios and fixed uniform empirical weights')
    _need(len(best['atoms']) == n and len(best['q_usd']) == n, 'Incomplete best-plan support')
    plan = problem.plan(best['anchor'])
    pd = dict(x=plan.x.tolist(), r=plan.r.tolist(), task_sha256=plan.task_sha256, config_hash=plan.config_hash)
    _need(best['plan_sha256'] == canonical_hash(dict(x=pd['x'], r=pd['r'])), 'Kernel best-plan hash differs')
    _need(problem.first_stage.evaluate(plan).feasible, 'Candidate first-stage domain failed')
    fpre = float(problem.first_stage.c @ np.asarray(best['anchor']))
    _same_numeric(fpre, best['Fpre_usd'], 'Kernel best Fpre differs')
    _same_numeric([a['q_usd'] for a in best['atoms']], best['q_usd'], 'Kernel best atom Q differs')
    for atom in best['atoms']:
        _need(kernel._anchor_matches(atom, best['anchor']), 'Best atom belongs to a different plan')
    ub = float(fpre + np.mean(best['q_usd']))
    _same_numeric(ub, state['upper_bound_usd'], 'Kernel best upper bound differs')
    history = state['master_history']
    refs = evidence['master_execution_references']
    _need(isinstance(refs, list) and refs and len(refs) == len(history)
          and len({r['path'] for r in refs}) == len(refs), 'Every old and new master execution is required')
    _need(refs == [row.get('evidence_reference') for row in history], 'Master history/external reference order differs')
    lbs, statuses, native_provenance = [], [], []
    for record, ref in zip(history, refs):
        execution = _read(ref)
        if domain == 'formal':
            _formal_native_guard(execution)
        _need(execution['kind'] == masters.EXECUTION, 'Wrong master producer')
        _need(execution['master_record'] == {k: v for k, v in record.items() if k != 'evidence_reference'},
              'Kernel master differs from actual execution')
        lb = masters.native_bound_contribution(execution)
        if lb is not None:
            lbs.append(lb)
        statuses.append(record['solver_status'])
        native_provenance.append(_native_provenance(execution, ref))
    _need(domain != 'formal' or all(v['backend'] == 'gurobipy' for v in native_provenance),
          'Formal L-shaped candidates require every original Gurobi master')
    _need(lbs, 'Candidate requires at least one retained finite native global bound')
    lb = max(lbs)
    _same_numeric(lb, state['lower_bound_usd'], 'Kernel lower bound is not max of complete native history')
    _need(lb <= ub + 1e-4 + 1e-7 * max(1., abs(lb), abs(ub)), 'Contradictory candidate bounds')
    gap = max(0., ub - lb) / max(1., abs(ub))
    cfg = state['config']
    threshold = cfg['optimality_relative_gap']
    _need(type(threshold) in (int, float) and 0 <= threshold <= .005, 'Wrong convergence threshold')
    _need(0 <= cfg['mip_relative_gap'] <= .001 and 0 < cfg['master_time_limit_seconds'] <= 600,
          'Master configuration exceeds original limits')
    if state['status'] == 'CONVERGED':
        _need(gap <= threshold, 'CONVERGED requires full-model gap, not native master OPTIMAL')
    _need(state['iterations'] == len(state['trace']) and all(r['master'] in history for r in state['trace']),
          'Kernel trace/master history differs')
    sessions = state['sessions']
    _need(sessions and all(type(s['elapsed_seconds']) in (int, float) and np.isfinite(s['elapsed_seconds'])
          and s['elapsed_seconds'] >= 0 for s in sessions), 'Invalid kernel session accounting')
    _same_numeric(sum(s['elapsed_seconds'] for s in sessions), state['elapsed_seconds'], 'Kernel elapsed sessions differ')
    request = _read(evidence['full_q_request_reference'])
    cp = _read(evidence['full_q_checkpoint_reference'])
    env = _read(evidence['template_envelope_reference'])
    _need(request.get('kind') == 'SAA_SERIAL_FIXED_PLAN_Q_V1' and request['domain'] == domain
          and request.get('evidence_producer') == producer, 'Actual L-shaped LP events required')
    _need(request['plan'] == pd and request['plan_sha256'] == canonical_hash(pd), 'Full-Q belongs to another best plan')
    expected = dict(case_sha256=problem.case.case_sha256, task_sha256=problem.tasks.task_sha256,
        scenario_ids=list(problem.support_ids), scenario_sha256s=problem.identity['scenario_sha256s'],
        template_sha256s=problem.identity['template_sha256s'], constants_usd=problem.constants_usd.tolist())
    _need(all(request[k] == v for k, v in expected.items()), 'Full-Q physical/support identity differs')
    _need(isinstance(request.get('input_binding'), dict) and request['input_binding'], 'Explicit physical input binding required')
    _need(cp['request_reference'] == evidence['full_q_request_reference']
          and cp['status'] == 'COMPLETE_FIXED_Q_PENDING_INDEPENDENT_AUDIT'
          and cp['completed_indices'] == list(range(n)), 'Complete same-plan full Q required')
    _same_numeric(cp['q_usd'], best['q_usd'], 'Full-Q does not match kernel best sweep')
    _need(env['request_reference'] == evidence['full_q_request_reference']
          and env['first_stage_sha256'] == problem.first_stage.matrix_sha256, 'Template envelope differs')
    if domain == 'formal':
        _need(request['input_binding'] == physical['input_binding'], 'Formal Q physical input binding differs')
    result = dict(kind=candidate_kind, domain=domain, plan_id=plan_id, algorithm_id=algorithm,
        driver_status=state['status'], native_master_statuses=statuses,
        native_master_status_encoding='NORMALIZED_0_OPTIMAL_1_TIME_LIMIT',
        master_native_provenance=native_provenance, plan=pd, plan_sha256=canonical_hash(pd),
        problem_identity=problem.identity, first_stage_sha256=problem.first_stage.matrix_sha256,
        support_ids=list(problem.support_ids), input_binding=request['input_binding'],
        Fpre_usd=fpre, q_usd=best['q_usd'], lower_bound_usd=lb, upper_bound_usd=ub, relative_gap=gap, gap_relative=gap,
        native_master_status='OPTIMAL' if statuses[-1] == 0 else 'TIME_LIMIT',
        native_master_solver_status=statuses[-1], original_rmp=None,
        optimality_relative_gap=threshold, gap_passed=bool(gap <= threshold),
        convergence_claim=bool(state['status'] == 'CONVERGED' and gap <= threshold),
        kernel_elapsed_seconds=state['elapsed_seconds'], kernel_sessions=sessions, evidence=evidence,
        numerical_audit_passed=False, budget_settled=False, formal_admission=False,
        scientific_acceptance=False, formal_OOS=False, extensive_model_manifest=False)
    if domain == 'formal':
        result['formal_input_reference'] = problem.identity['formal_input_reference']
    return result


def write_development_candidate(problem, *, plan_id, kernel_checkpoint_reference,
        master_execution_references, full_q_checkpoint_reference, full_q_request_reference,
        template_envelope_reference, source_references, output_path):
    """Bind real retained evidence without granting numerical/canonical acceptance."""
    evidence = dict(kernel_checkpoint_reference=kernel_checkpoint_reference,
        master_execution_references=master_execution_references, full_q_checkpoint_reference=full_q_checkpoint_reference,
        full_q_request_reference=full_q_request_reference, template_envelope_reference=template_envelope_reference,
        source_references=source_references)
    return _write(output_path, _bound_candidate(problem, plan_id, evidence))


def _audit_candidate(problem, *, candidate_reference, expected_plan_id, expected_evidence,
        case, tasks, scene_loader, helper_reference, full_q_source_references, deadline, output_path=None,
        domain):
    """Recompute original full Q and all original-row master audits without solves.

    ``expected_evidence`` and ``candidate_reference`` must be retained by the
    trusted caller before this audit, never synthesized from an untrusted file.
    """
    started = time.monotonic()
    algorithm, _, audit_kind, _, _ = _domain_contract(domain)
    def alive():
        _need(type(deadline) in (int, float) and np.isfinite(deadline), 'Finite shared audit deadline required')
        if time.monotonic() >= deadline:
            raise TimeoutError('L-shaped joint audit deadline')
    alive()
    candidate = _read(candidate_reference)
    _need(candidate['evidence'] == expected_evidence and candidate['plan_id'] == expected_plan_id,
          'Candidate differs from caller-retained evidence/plan pins')
    recomputed = _bound_candidate(problem, expected_plan_id, expected_evidence, domain=domain)
    _need(candidate == recomputed, 'Candidate summary differs from original bound evidence')
    _need(case.case_sha256 == problem.case.case_sha256 and tasks.task_sha256 == problem.tasks.task_sha256
          and tasks.config_hash == problem.tasks.config_hash, 'Foreign physical audit context')
    qa = full_q.audit_saa_fixed_plan_q(checkpoint_reference=expected_evidence['full_q_checkpoint_reference'],
        expected_request_reference=expected_evidence['full_q_request_reference'],
        template_envelope_reference=expected_evidence['template_envelope_reference'],
        first_stage=problem.first_stage, case=case, tasks=tasks, scene_loader=scene_loader,
        empirical_probabilities=problem.identity['empirical_probabilities'], helper_reference=helper_reference,
        source_references=full_q_source_references, deadline=deadline)
    _need(qa['passed'] and qa['full_support_recourse_verified'], 'Original full-Q audit failed')
    _same_numeric(qa['q_usd'], candidate['q_usd'], 'Audited Q differs from candidate')
    _same_numeric(qa['first_stage_cost_usd'], candidate['Fpre_usd'], 'Audited Fpre differs from candidate')
    _same_numeric(qa['upper_bound_usd'], candidate['upper_bound_usd'], 'Audited full mean differs from candidate')
    ma = []
    for ref in expected_evidence['master_execution_references']:
        alive()
        ma.append(masters.audit_master_evidence(problem, ref, deadline=deadline))
    _need(all(a['passed'] for a in ma), 'Original small-master audit failed')
    for checked, native in zip(ma, candidate['master_native_provenance']):
        _need(dict(checked['native_provenance'], execution_reference=checked['execution_reference']) == native,
              'Audited raw backend/native status differs')
    finite_bounds = [a['native_lower_bound_usd'] for a in ma if a['contributes_lower_bound']]
    _need(finite_bounds, 'Candidate requires an audited finite native global bound')
    _same_numeric(max(finite_bounds), candidate['lower_bound_usd'], 'Audited maximum LB differs')
    alive()
    result = dict(kind=audit_kind, passed=True, candidate_reference=candidate_reference, expected_evidence=expected_evidence,
        plan_id=expected_plan_id, plan_sha256=candidate['plan_sha256'], problem_identity=problem.identity,
        first_stage_sha256=problem.first_stage.matrix_sha256, input_binding=candidate['input_binding'],
        support_ids=list(problem.support_ids), algorithm_id=algorithm,
        driver_status=candidate['driver_status'], native_master_statuses=[a['native_status'] for a in ma],
        native_master_status_encoding=candidate['native_master_status_encoding'],
        master_native_provenance=candidate['master_native_provenance'],
        lower_bound_usd=candidate['lower_bound_usd'], upper_bound_usd=qa['upper_bound_usd'],
        relative_gap=candidate['relative_gap'], gap_passed=candidate['gap_passed'], convergence_claim=candidate['convergence_claim'],
        full_q_audit=qa, master_audits=ma, full_q_source_references=full_q_source_references,
        helper_reference=helper_reference, solver_calls=0, probability_optimization_calls=0,
        native_solver_output_trusted=True, native_branch_and_bound_proof_replayed=False,
        budget_settled=False, formal_admission=False, scientific_acceptance=False, formal_OOS=False,
        elapsed_seconds=time.monotonic()-started)
    if domain == 'formal':
        result['formal_input_reference'] = candidate['formal_input_reference']
    if output_path is not None:
        result['audit_reference'] = _write(output_path, result)
    return result


def _reference_closure(value, seen=None):
    """Current candidate evidence only; no historical proof-role expansion."""
    seen = set() if seen is None else seen
    if isinstance(value, dict):
        if set(value) == {'path', 'sha256', 'size_bytes'}:
            key = (value['path'], value['sha256'], value['size_bytes'])
            if key in seen:
                return
            seen.add(key)
            path = Path(value['path'])
            _need(path.is_absolute() and path == path.resolve(), 'Aliased candidate evidence')
            verify_reference(value)
            if path.suffix == '.json':
                _reference_closure(_read(value), seen)
        else:
            for v in value.values():
                _reference_closure(v, seen)
    elif isinstance(value, list):
        for v in value:
            _reference_closure(v, seen)


def _validate_joint(joint, request, candidate_record, *, domain):
    """Metadata-only typed readback for the existing receipt/canonical adapter.

    The outer adapter must already have verified the caller-retained joint audit
    reference. This does not recreate its numerical acceptance or settle costs.
    """
    algorithm, candidate_kind, audit_kind, producer, master_audit_kind = _domain_contract(domain)
    _need(request.get('domain') == domain and request.get('method_id') == 'SAA'
          and request.get('model_input_mode') == 'l_shaped_' + domain
          and request.get('algorithm_id') == algorithm,
          'Explicit ' + domain + ' L-shaped request required')
    _need(joint.get('kind') == audit_kind and joint.get('passed') is True,
          'Typed L-shaped joint numerical audit required')
    _need(joint['expected_evidence'] == candidate_record['l_shaped_evidence'], 'Joint differs from sealed candidate producer evidence pins')
    evidence = candidate_record['l_shaped_evidence']
    _need(set(evidence) == set(EVIDENCE_KEYS), 'Wrong pinned evidence fields')
    _sources(evidence['source_references'])
    _reference_closure(joint)
    candidate = _read(joint['candidate_reference'])
    _need(candidate_record['algorithm_id'] == request['algorithm_id'] == algorithm
          and candidate_record['method_id'] == 'SAA', 'Candidate producer method/algorithm differs')
    if 'request_reference' in candidate_record:
        _need(_read(candidate_record['request_reference']) == request, 'Candidate producer request differs')
    _need(candidate_record['candidate'] == candidate and candidate['evidence'] == evidence,
          'Retained candidate differs from canonical input')
    _need(candidate_record['plan_id'] == candidate['plan_id'] == request['plan_id'] == joint['plan_id'],
          'Candidate/joint/request plan_id differs')
    for body in (candidate, joint):
        _need(body['formal_admission'] is False and body['scientific_acceptance'] is False
              and body['formal_OOS'] is False and body['budget_settled'] is False, 'Candidate admission scope changed')
    _need(candidate['kind'] == candidate_kind and candidate['domain'] == domain
          and candidate['extensive_model_manifest'] is False and candidate['numerical_audit_passed'] is False,
          'Wrong candidate domain or producer')
    state_envelope = _read(evidence['kernel_checkpoint_reference'])
    state = state_envelope['state']; identity = state['identity']
    _need(state_envelope['sha256'] == canonical_hash(state) and state['source_identity'] == kernel.source_identity(),
          'Kernel checkpoint or source changed')
    _need(identity == candidate['problem_identity'] == joint['problem_identity']
          and identity['algorithm_id'] == candidate['algorithm_id'] == joint['algorithm_id'] == algorithm
          and identity['first_stage_sha256'] == joint['first_stage_sha256'] == candidate['first_stage_sha256'],
          'Problem/first-stage/algorithm identity differs')
    status = state['status']
    _need(status in ('TIME_LIMIT', 'ITERATION_LIMIT', 'CONVERGED')
          and status == candidate['driver_status'] == joint['driver_status']
          == candidate_record['driver_status'] == candidate_record['raw_engine_status'],
          'Original kernel stop must remain distinct from native master status')
    _need(evidence['master_execution_references'] == [m.get('evidence_reference') for m in state['master_history']]
          and len(joint['master_audits']) == len(state['master_history']) > 0,
          'All old/new master history must be audited')
    lbs, codes, native_provenance = [], [], []
    for ref, checked, record in zip(evidence['master_execution_references'], joint['master_audits'], state['master_history']):
        actual = _read(ref); prep = _read(actual['preparation_reference'])
        if domain == 'formal':
            _formal_native_guard(actual)
        _need(checked['kind'] == master_audit_kind
              and checked['passed'] is True and checked['execution_reference'] == ref
              and checked['preparation_reference'] == actual['preparation_reference']
              and checked['problem_identity_sha256'] == canonical_hash(identity)
              and prep['problem_identity'] == identity
              and checked['source_references'] == masters._sources(), 'Master audit identity differs')
        _need(actual['kind'] == masters.EXECUTION and actual['master_record'] == {k: v for k, v in record.items() if k != 'evidence_reference'}
              and checked['coefficient_sha256'] == actual['submitted_coefficient_sha256'] == prep['coefficient_sha256'],
              'Audited master submission differs')
        lb = masters.native_bound_contribution(actual); code = actual['native_result']['status']
        _need(lb == checked['native_lower_bound_usd'] == record['lower_bound_usd']
              and code == checked['native_status'] == record['solver_status']
              and checked['contributes_lower_bound'] is (lb is not None)
              and checked['bound_disposition'] == ('FINITE_NATIVE_GLOBAL_BOUND' if lb is not None
                                                  else 'TIME_LIMIT_NO_FINITE_NATIVE_BOUND')
              and checked['solver_calls'] == 0 and checked['formal_admission'] is False,
              'Actual native master bound/status differs')
        if lb is not None:
            lbs.append(lb)
        codes.append(code)
        native = _native_provenance(actual, ref)
        _need(dict(checked['native_provenance'], execution_reference=checked['execution_reference']) == native,
              'Audited raw backend/native status differs')
        native_provenance.append(native)
    _need(domain != 'formal' or all(v['backend'] == 'gurobipy' for v in native_provenance),
          'Formal L-shaped readback requires every original Gurobi master')
    _need(lbs, 'Candidate requires an audited retained finite native global bound')
    fixed = joint['full_q_audit']; qreq = _read(evidence['full_q_request_reference'])
    qcp = _read(evidence['full_q_checkpoint_reference']); plan = candidate['plan']; digest = canonical_hash(plan)
    ids = identity['support_ids']; n = len(ids)
    _need(domain != 'formal' or n == 160, 'Formal readback requires all 160 support scenarios')
    _need(n > 0 and len(set(ids)) == n and candidate['support_ids'] == joint['support_ids'] == fixed['support_ids'] == qreq['scenario_ids'] == ids,
          'Full ordered support differs')
    _need(fixed['status'] == 'PASS_SAA_FIXED_PLAN_FULL_Q_AND_EMPIRICAL_LAW_AUDIT'
          and fixed['passed'] is True and fixed['full_support_recourse_verified'] is True
          and fixed['raw_auxiliary_kkt_arithmetic_recomputed'] is True and fixed['risk_certificate_kind'] == 'fixed_empirical'
          and fixed['solver_calls'] == fixed['probability_optimization_calls'] == 0,
          'Original full-Q numerical audit required')
    _need(fixed['request_reference'] == evidence['full_q_request_reference']
          and fixed['checkpoint_reference'] == evidence['full_q_checkpoint_reference']
          and fixed['template_envelope_reference'] == evidence['template_envelope_reference']
          and fixed['source_references'] == joint['full_q_source_references']
          and fixed['helper_reference'] == joint['helper_reference'], 'Full-Q audit bindings differ')
    _need(digest == candidate['plan_sha256'] == joint['plan_sha256'] == fixed['plan_sha256'] == qreq['plan_sha256']
          and qreq['plan'] == plan and qreq['domain'] == domain and qreq.get('evidence_producer') == producer
          and qreq['input_binding'] == candidate['input_binding'] == joint['input_binding'], 'Full-Q best plan/input differs')
    if domain == 'development':
        _need(qreq['input_binding'] == request['input_binding'], 'Development original input binding differs')
    _need(qreq['scenario_sha256s'] == identity['scenario_sha256s'] and qreq['template_sha256s'] == identity['template_sha256s']
          and qreq['constants_usd'] == identity['constants_usd'], 'Full-Q scenarios or constants differ')
    _need(qcp['request_reference'] == evidence['full_q_request_reference'] and qcp['completed_indices'] == list(range(n)),
          'Full-Q checkpoint incomplete')
    pi = np.asarray(fixed['empirical_probabilities']); q = np.asarray(fixed['q_usd'])
    _need(pi.shape == q.shape == (n,) and np.array_equal(pi, np.full(n, 1/n))
          and np.array_equal(pi, identity['empirical_probabilities']) and np.isfinite(q).all(), 'Original empirical law differs')
    _same_numeric(qcp['q_usd'], q, 'Q checkpoint differs from audit')
    _same_numeric(candidate['q_usd'], q, 'Candidate Q differs from audit')
    _same_numeric(state['best']['q_usd'], q, 'Kernel best Q differs from audit')
    _need(state['best']['plan_sha256'] == canonical_hash(dict(x=plan['x'], r=plan['r'])), 'Kernel best plan differs')
    _same_numeric(candidate['Fpre_usd'], fixed['first_stage_cost_usd'], 'Fpre differs')
    ub = float(fixed['first_stage_cost_usd'] + pi @ q); lb = max(lbs)
    _same_numeric(float(pi @ q), fixed['expected_recourse_usd'], 'Mean Q differs')
    _same_numeric(ub, fixed['upper_bound_usd'], 'Full-Q UB differs')
    gap = max(0., ub-lb)/max(1., abs(ub))
    _need(lb <= ub+1e-4+1e-7*max(1., abs(lb), abs(ub)), 'Joint bound contradiction')
    for body in (candidate, joint, state):
        _same_numeric(body['lower_bound_usd'], lb, 'Joint lower bound differs')
        _same_numeric(body['upper_bound_usd'], ub, 'Joint upper bound differs')
    _same_numeric(candidate['gap_relative'], gap, 'Candidate gap differs')
    for body in (candidate, joint):
        _same_numeric(body['relative_gap'], gap, 'Joint gap differs')
        _need(body['gap_passed'] is (gap <= candidate['optimality_relative_gap'])
              and body['convergence_claim'] is (status == 'CONVERGED' and gap <= candidate['optimality_relative_gap']),
              'Gap/convergence metadata differs')
    _need(0 <= candidate['optimality_relative_gap'] == state['config']['optimality_relative_gap'] <= .005
          and (status != 'CONVERGED' or gap <= candidate['optimality_relative_gap']), 'Convergence threshold differs')
    _need(codes == candidate['native_master_statuses'] == joint['native_master_statuses'], 'Native history status differs')
    _need(native_provenance == candidate['master_native_provenance'] == joint['master_native_provenance']
          and candidate['native_master_status_encoding'] == joint['native_master_status_encoding']
          == 'NORMALIZED_0_OPTIMAL_1_TIME_LIMIT', 'Raw/normalized native status provenance differs')
    native = 'OPTIMAL' if codes[-1] == 0 else 'TIME_LIMIT'
    _need(candidate['native_master_status'] == native and candidate['native_master_solver_status'] == codes[-1]
          and candidate.get('original_rmp') is None, 'Native master status or false CCG certificate differs')
    sources = request['planning_source_references']
    required_sources = evidence['source_references'] + joint['full_q_source_references'] + qreq['sources']
    required_sources += _read(evidence['template_envelope_reference'])['source_references']
    if domain == 'development':
        required_sources.append(joint['helper_reference'])
    for checked in joint['master_audits']:
        required_sources += list(checked['source_references'].values())
    _need(all(r in sources for r in required_sources), 'Original planning source closure differs')
    _need(joint['solver_calls'] == joint['probability_optimization_calls'] == 0, 'Joint audit unexpectedly solved')
    if domain == 'formal':
        _formal_physical_readback(request, candidate_record, candidate, joint, identity)
    return dict(plan_sha256=digest, lower_bound_usd=lb, upper_bound_usd=ub, gap_relative=gap,
        native_master_status=native, native_master_solver_status=codes[-1], raw_engine_status=status,
        algorithm_id=algorithm)


def _formal_physical_readback(request, candidate_record, candidate, joint, identity):
    """Bind output preflight to its original immutable request, without a hash cycle."""
    from .saa_l_shaped_formal import validate_formal_input_reference
    helper = joint['helper_reference']
    _need(helper == request['helper_reference'] and helper['sha256'] == full_q.HELPER_SHA256,
          'Formal original KKT helper binding differs')
    verify_reference(helper)
    ref = candidate['formal_input_reference']
    _need(ref == joint['formal_input_reference'] == identity['formal_input_reference'],
          'Formal physical preflight reference differs')
    request_ref = candidate_record.get('request_reference')
    _need(isinstance(request_ref, dict) and _read(request_ref) == request,
          'Formal candidate must retain the exact original worker request')
    checked = validate_formal_input_reference(ref, expected_request=request_ref, expected_identity=identity)
    _need(checked['worker_request_reference'] == request_ref,
          'Formal preflight belongs to another worker request')
    _need(checked['input_binding'] == candidate['input_binding'] == joint['input_binding'],
          'Formal readback physical binding differs')
    if 'input_binding' in request:
        _need(request['input_binding'] == checked['input_binding'], 'Explicit formal request input binding differs')
    original = _read(checked['worker_request_reference'])
    for field in ('plan_id', 'target_id', 'method_id', 'protocol_reference', 'config_reference',
                  'plan_input_references', 'bank_acceptance_reference', 'summary_reference',
                  'audit_reference', 'budget', 'planning_source_references'):
        _need(request.get(field) == original.get(field), 'Formal original worker binding differs: ' + field)


def audit_development_candidate(problem, **kwargs):
    """Audit synthetic development evidence with the unchanged numerical checks."""
    return _audit_candidate(problem, domain='development', **kwargs)


def write_formal_candidate(problem, *, plan_id, kernel_checkpoint_reference,
        master_execution_references, full_q_checkpoint_reference, full_q_request_reference,
        template_envelope_reference, source_references, output_path):
    """Export formal physical/Gurobi evidence; does not authorize a job or settle cost."""
    evidence = dict(kernel_checkpoint_reference=kernel_checkpoint_reference,
        master_execution_references=master_execution_references,
        full_q_checkpoint_reference=full_q_checkpoint_reference,
        full_q_request_reference=full_q_request_reference,
        template_envelope_reference=template_envelope_reference,
        source_references=source_references)
    return _write(output_path, _bound_candidate(problem, plan_id, evidence, domain='formal'))


def audit_formal_candidate(problem, **kwargs):
    """Run the original all-master and full-Q checks on validated formal inputs."""
    return _audit_candidate(problem, domain='formal', **kwargs)


def validate_development_joint(joint, request, candidate_record):
    return _validate_joint(joint, request, candidate_record, domain='development')


def validate_formal_joint(joint, request, candidate_record):
    """Read back a typed formal numerical result without granting worker admission."""
    return _validate_joint(joint, request, candidate_record, domain='formal')
