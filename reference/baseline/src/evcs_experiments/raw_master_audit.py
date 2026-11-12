"""Audit a saved selected RMP in original units, without any optimizer call.

A native MILP bound remains a solver-reported bound: this checks its provenance,
status, offset and consistency, not an independently replayed branch-and-bound
proof. Full-support recourse/probability and campaign budget audits are separate.
"""
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_matrix, hstack, load_npz

from .first_stage_check import check_first_stage
from .recourse_check import check_recourse
from .probability import ProbabilitySet
from .random_streams import canonical_hash
from .raw_plan_audit import _module
from .raw_template_audit import audit_template_envelope, document, need, read, record

ORDER = ('rav', 'p', 'q', 'short', 'shed', 'line_p', 'line_q', 'u', 'soc_dep')


def audit_selected_master(*, snapshot_reference, expected_request_reference,
        plan_dict, iteration, first_stage, recourse_loader, ambiguity, case,
        tasks, scene_loader, helper_source_reference, q_usd=None, deadline=None):
    """Check the selected iteration against explicitly source-bound physical inputs.

    Caller binds helper source, first stage, templates and ambiguity to reviewed
    sources. Optional q_usd must be the freshly independently audited full Q for
    this same plan, ordered like the request. No saved ``passed`` flag suffices.
    """
    started = time.monotonic()
    need(type(iteration) is int and iteration >= 1, 'Explicit positive iteration required')
    if deadline is not None:
        need(np.isfinite(deadline) and started < deadline <= started + 7190, 'Invalid master audit deadline')
    def remaining():
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('Shared master audit deadline reached')
    remaining()
    h = _module(helper_source_reference, '_selected_rmp_original_helper')
    envelope = audit_template_envelope(snapshot_reference, expected_request_reference=expected_request_reference)
    request = document(expected_request_reference); identity = request['problem_identity']
    snapshot = document(snapshot_reference); ids = identity['scenario_ids']; risk = identity['risk_model']
    need(risk in ('DRO', 'ARO'), 'Unsupported CCG master risk model')
    need(ambiguity.support_ids == tuple(ids) and
         ambiguity.ambiguity_sha256 == identity['effective_ambiguity_sha256'], 'Foreign effective ambiguity')
    hashes = (identity['lazy_input_binding']['scenario_sha256s'] if 'lazy_input_binding' in identity
              else identity['scenario_sha256s'])
    constants = np.asarray(identity['constants_usd'], dtype=float)
    need(constants.shape == (len(ids),) and np.isfinite(constants).all(), 'Invalid master scenario constants')
    plan = SimpleNamespace(x=np.asarray(plan_dict['x']), r=np.asarray(plan_dict['r']),
        task_sha256=plan_dict['task_sha256'], config_hash=plan_dict['config_hash'])
    need(plan.task_sha256 == tasks.task_sha256 == first_stage.task_sha256 and
         plan.config_hash == tasks.config_hash == first_stage.config_hash and
         case.case_sha256 == tasks.physical_case_sha256, 'Foreign master physical inputs')
    physical = check_first_stage(case, tasks, plan)
    need(physical['passed'], 'Master first-stage physical check failed')
    first_record = record(snapshot['first_stage'])
    fd = {Path(r['path']).name: r for r in first_record['artifacts']}
    fm = document(fd['identity.json']); fv = h.npz(read(fd['arrays.npz']))
    fa = load_npz(read(fd['matrix.npz'])).tocsr()
    need(first_stage.matrix_sha256 == fm['matrix_sha256'] == identity['first_stage_sha256'], 'Foreign master first stage')
    need(fa.shape == first_stage.A.shape and (fa - first_stage.A).nnz == 0, 'First-stage saved matrix differs')
    for key in ('row_lower', 'row_upper', 'lower_bounds', 'upper_bounds', 'c', 'integrality'):
        h.same(fv[key], getattr(first_stage, key), 'First-stage saved arrays differ: ' + key)
    candidates = []
    for ref in snapshot['events']:
        row = document(ref)
        if row['event'] == 'rmp' and row['iteration'] == iteration:
            candidates.append((ref, row))
    need(len(candidates) == 1, 'Selected iteration needs exactly one RMP')
    reference, event = candidates[0]; record(reference)
    need(event['status'] == 'RAW_EVIDENCE_RETAINED_PENDING_AUDIT', 'Failed raw master event')
    d = Path(reference['path']).parent
    model = h.read(d/'model.json'); solver = h.read(d/'solver.json'); solution = h.read(d/'solution.json')
    v = h.npz(d/'model_arrays.npz'); saved = h.npz(d/'solution.npz'); z = saved['vector']
    a = load_npz(d/'matrix.npz').tocsr()
    need(np.isfinite(a.data).all() and z.shape == v['c'].shape and np.isfinite(z).all(), 'Nonfinite/invalid RMP data')
    need(h.model_digest(model, a, v) == model['model_sha256'] == solution['model_sha256'] ==
         solver['certificate']['model_sha256'], 'Original RMP coefficient digest differs')
    active_ids = model['scenario_ids']
    need(active_ids and len(active_ids) == len(set(active_ids)) and all(s in ids for s in active_ids), 'Invalid RMP support')
    active = [ids.index(s) for s in active_ids]
    restricted = ProbabilitySet(tuple(active_ids), ambiguity.feature_ids, ambiguity.features[active], ambiguity.lower, ambiguity.upper)
    need(model['risk_model'] == risk and model['ambiguity_sha256'] == restricted.ambiguity_sha256 and
         model['first_stage_sha256'] == first_stage.matrix_sha256 and model['empirical_probabilities'] is None,
         'Foreign master model identity')
    need(model['task_sha256'] == tasks.task_sha256 and model['config_hash'] == tasks.config_hash and
         model['task_ids'] == list(tasks.task_ids) and model['periods'] == list(case.periods), 'Foreign master axes')
    h.same(model['scenario_cost_constants_usd'], constants[active], 'Master scenario constants differ')
    need(model['objective_constant_usd'] == v['objective_constant_usd'] == 0., 'Unexpected DRO/ARO objective offset')
    h.same(saved['x'], plan.x, 'RMP x differs'); h.same(saved['r'], plan.r, 'RMP reserve differs')
    for key in ('x', 'r', 'task_sha256', 'config_hash'):
        need(solution[key] == plan_dict[key], 'RMP saved plan differs')
    need(solver['solution'] == solution, 'RMP solver solution differs')
    nf = len(first_stage.c); H = len(tasks.task_ids); n = len(z)
    h.same(z[:nf], np.r_[plan.x, plan.r.ravel()], 'RMP vector uses another plan')
    residuals = h.rmp_residuals(a, v, z)
    sl = model['variable_slices']; expected_slices = dict(x=[0,H], r=[H,nf], first_stage=[0,nf], recourses={})
    templates = {row['scenario_id']: row for row in envelope['templates']}
    refs = h.read(d/'template_references.json')
    need(len(refs) == len(active), 'RMP template reference count differs')
    c = np.zeros(n); c[:nf] = first_stage.c
    lower = np.empty(n); upper = np.empty(n); lower[:nf] = first_stage.lower_bounds; upper[:nf] = first_stage.upper_bounds
    integer = np.zeros(n); integer[:nf] = first_stage.integrality
    cursor = nf; row_cursor = 0; expected_blocks = []; template_rows = []
    def block(matrix, lo, hi, scope, kind, sha):
        nonlocal row_cursor
        count = matrix.shape[0]; rs = slice(row_cursor, row_cursor+count)
        need(matrix.shape == (count,n) and a[rs].shape == matrix.shape and (a[rs]-matrix).nnz == 0,
             'Original RMP block coefficients differ: ' + kind)
        h.same(v['row_lower'][rs], lo, 'RMP lower row bounds differ: ' + kind)
        h.same(v['row_upper'][rs], hi, 'RMP upper row bounds differ: ' + kind)
        expected_blocks.append(dict(scope=scope,kind=kind,row_start=row_cursor,row_stop=row_cursor+count,source_sha256=sha))
        row_cursor += count
    block(hstack((fa, csr_matrix((fa.shape[0],n-nf))),format='csr'),fv['row_lower'],fv['row_upper'],
          'first_stage','first_stage',first_stage.matrix_sha256)
    for j, i in enumerate(active):
        remaining(); sid = ids[i]; t = recourse_loader(i)
        need(t.first_stage.matrix_sha256 == first_stage.matrix_sha256 and t.task_sha256 == tasks.task_sha256 and
             t.config_hash == tasks.config_hash and t.task_ids == tasks.task_ids and t.periods == case.periods and
             t.scenario_sha256 == hashes[i], 'Foreign RMP recourse inputs')
        need(refs[j] == templates[sid]['original_reference'] and t.template_sha256 == templates[sid]['template_sha256'] ==
             model['recourse_template_sha256s'][j] and t.scenario_sha256 == model['scenario_sha256s'][j], 'RMP recourse binding differs')
        stop = cursor + len(t.c); ys = slice(cursor,stop)
        expected_slices['recourses'][sid] = [cursor,stop]
        need(sl['recourses'][sid] == [cursor,stop], 'RMP recourse columns differ')
        lower[ys] = t.lower_bounds; upper[ys] = t.upper_bounds
        for sense in ('ub','eq'):
            aa,bx,br,b = (getattr(t,k+'_'+sense) for k in ('A','Bx','Br','b'))
            matrix = hstack((-bx,-br,csr_matrix((len(b),cursor-nf)),aa,csr_matrix((len(b),n-stop))),format='csr')
            block(matrix, b if sense == 'eq' else np.full(len(b),-np.inf),b,sid,'recourse_'+sense,t.template_sha256)
        rw = h.npz(d/f'witness_{i:03d}.npz')
        h.same(z[ys], np.concatenate([np.asarray(rw[k]).ravel() for k in ORDER]), 'RMP recourse vector differs')
        scene = scene_loader(i)
        need(scene.scenario_sha256 == hashes[i] and scene.physical_case_sha256 == case.case_sha256, 'Foreign master scene')
        check = check_recourse(case,tasks,scene,plan,SimpleNamespace(**rw),check_objective_bound=False)
        need(check['passed'], 'Original RMP recourse physics failed')
        cost = float(t.c@z[ys]+constants[i])
        h.close(check['costs']['Qop']+constants[i],cost,'RMP physical witness cost differs',1e-4+1e-7*abs(cost))
        h.close(solution['scenario_witness_costs_usd'][j],cost,'RMP reported witness cost differs',1e-4+1e-7*abs(cost))
        template_rows.append((sid,t,ys,i)); cursor = stop
    risk_start = cursor; lower[cursor:] = 0.; upper[cursor:] = np.inf
    c[cursor] = 1.; lower[cursor] = -np.inf
    if risk == 'DRO':
        f = len(ambiguity.feature_ids)
        for key,length in (('theta0',1),('alpha',f),('beta',f)):
            expected_slices[key]=[cursor,cursor+length]; cursor += length
        alpha = slice(*expected_slices['alpha']); beta = slice(*expected_slices['beta'])
        c[alpha] = -ambiguity.lower; c[beta] = ambiguity.upper
        dual = dict(theta0=float(z[risk_start]),alpha=z[alpha].tolist(),beta=z[beta].tolist(),threshold=None)
        need(solution['theta0'] == dual['theta0'] and solution['alpha'] == dual['alpha'] and
             solution['beta'] == dual['beta'] and solution['risk_threshold'] is None, 'RMP uncertainty dual differs')
        envelope_q = z[risk_start] + ambiguity.features@(z[beta]-z[alpha])
    else:
        expected_slices['risk_threshold']=[cursor,cursor+1]; cursor += 1
        dual = dict(theta0=None,alpha=None,beta=None,threshold=float(z[risk_start]))
        need(solution['risk_threshold'] == dual['threshold'] and all(solution[k] is None for k in ('theta0','alpha','beta')),
             'ARO raw threshold differs')
        envelope_q = np.full(len(ids),z[risk_start])
    need(cursor == n and sl == expected_slices, 'Unexpected RMP variable layout')
    for sid,t,ys,i in template_rows:
        risk_row = np.r_[-1.,ambiguity.features[i],-ambiguity.features[i]] if risk == 'DRO' else np.array([-1.])
        matrix = hstack((csr_matrix((1,ys.start)),csr_matrix(t.c.reshape(1,-1)),
            csr_matrix((1,risk_start-ys.stop)),csr_matrix(risk_row.reshape(1,-1))),format='csr')
        block(matrix,np.array([-np.inf]),np.array([-constants[i]]),sid,'risk_epigraph',t.template_sha256)
    need(row_cursor == a.shape[0] and model['block_map'] == expected_blocks, 'Unexpected master rows/block provenance')
    for key,expected in (('c',c),('lower_bounds',lower),('upper_bounds',upper),('integrality',integer)):
        h.same(v[key],expected,'RMP complete column data differs: '+key)
    objective = float(c@z); tol = 1e-4+1e-7*max(1.,abs(objective))
    h.close(solver['objective_usd'],objective,'RMP objective differs',tol)
    h.close(physical['costs']['Fpre_usd'],first_stage.c@z[:nf],'Master first-stage cost differs',
            1e-4+1e-7*max(1.,abs(float(physical['costs']['Fpre_usd']))))
    raw = solver['raw_solver']; code = raw['status']
    need(type(code) is int and code in (0,1) and solver['solver_status'] == code, 'RMP native status cannot supply selected incumbent')
    h.close(raw['raw_objective_without_constant_usd'],objective,'Raw native objective differs',tol)
    native = raw.get('native_interface')
    if native is not None and raw['solver'] == 'gurobipy/Gurobi':
        status_map = {2:'OPTIMAL',7:'ITERATION_LIMIT',8:'NODE_LIMIT',9:'TIME_LIMIT',10:'SOLUTION_LIMIT',
                      11:'INTERRUPTED',15:'USER_OBJ_LIMIT',16:'WORK_LIMIT',17:'MEM_LIMIT'}
        need(native['raw_model_status'] in status_map and native['status'] == status_map[native['raw_model_status']] and
             code == (0 if native['raw_model_status'] == 2 else 1), 'Native Gurobi status mapping differs')
        h.close(native['incumbent_without_constant_usd'],objective,'Native Gurobi incumbent differs',tol)
        need(native['solver_bound_without_constant_usd'] == raw['raw_lower_bound_without_constant_usd'], 'Native Gurobi bound differs')
        original_bound = native['raw_solver_bound_without_constant_usd']
        need(original_bound is None or (np.isfinite(original_bound) and original_bound <= objective+tol),
             'Native Gurobi original bound contradicts incumbent')
        need(original_bound == native['solver_bound_without_constant_usd'], 'Native Gurobi original bound was changed')
        need(type(native['solution_count']) is int and native['solution_count'] > 0, 'Native Gurobi lacks an incumbent')
        source = None if original_bound is None else ('MIP_ObjBound' if native['is_mip'] else 'optimal_LP_objective')
        need(native['bound_source'] == source and (native['is_mip'] or original_bound is None or code == 0),
             'Native Gurobi bound has unsupported source')
    bound = raw['raw_lower_bound_without_constant_usd']
    need(bound is None or np.isfinite(bound), 'Nonfinite raw master bound')
    usable = bound is not None and bound <= objective + tol
    accepted_bound = float(bound) if usable else None
    need(solver['lower_bound_usd'] == accepted_bound and solver['certificate']['solver_bound_usable'] is usable,
         'Promoted/changed native master bound')
    gap = raw['raw_mip_gap']; requested_gap = solver['certificate']['requested_mip_relative_gap']
    need(np.isfinite(requested_gap) and requested_gap >= 0, 'Invalid native master gap tolerance')
    optimal = bool(code == 0 and usable and abs(objective-bound) <= tol and
                   gap is not None and np.isfinite(gap) and 0 <= gap <= requested_gap)
    need(solver['optimal'] is optimal, 'Promoted master optimality')
    expected_status = ('OPTIMAL' if optimal else 'NUMERICAL_CERTIFICATE_FAILED') if code == 0 else (
        native['status'] if native is not None and raw['solver'] == 'gurobipy/Gurobi' else 'TIME_LIMIT')
    need(solver['status'] == expected_status, 'Promoted master status')
    q_residual = None
    if q_usd is not None:
        q = np.asarray(q_usd,dtype=float)
        need(q.shape == (len(ids),) and np.isfinite(q).all(), 'Invalid audited full-support Q')
        rr = q-envelope_q
        need(float(np.max(rr[active],initial=0.)) <= 1e-4+1e-7*max(1.,float(np.max(abs(q)))), 'Included RMP uncertainty witness violates audited Q')
        q_residual = dict(residuals_usd=rr.tolist(),included_maximum_usd=float(np.max(rr[active],initial=0.)),
                         full_support_maximum_usd=float(np.max(rr,initial=0.)))
    remaining()
    return dict(status='PASS_SELECTED_PLAN_RAW_MASTER_AUDIT',plan_sha256=canonical_hash(plan_dict),iteration=iteration,
        support_ids=ids,active_indices=active,rmp_event_reference=reference,model_sha256=model['model_sha256'],
        native_status=solver['status'],raw_solver_status=code,lower_bound_usd=accepted_bound,
        solver_bound_usable=usable,optimal=optimal,objective_usd=objective,
        first_stage_cost_usd=float(physical['costs']['Fpre_usd']),uncertainty_dual=dual,dual_residuals=q_residual,
        residuals=residuals,master_bound_audited=True,bound_scope='original_native_solver_report',
        branch_and_bound_proof_replayed=False,snapshot_reference=snapshot_reference,request_reference=expected_request_reference,
        helper_source_reference=helper_source_reference,probability_audited=False,scientific_acceptance=False,
        formal_OOS=False,solver_calls=0,elapsed_seconds=time.monotonic()-started)
