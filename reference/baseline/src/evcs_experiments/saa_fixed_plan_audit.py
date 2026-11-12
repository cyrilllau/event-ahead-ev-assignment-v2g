"""Original-matrix SAA full-Q and fixed empirical-law audit; no solver calls.

The template envelope binds existing original template records to the producer
request. Formal physical-source admission and master/budget/queue acceptance
remain separate. No CCG event or adversarial probability witness is fabricated.
"""
import importlib.util
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from . import first_stage_check, recourse_check, recourse_store
from .campaign_batch import file_reference, verify_reference
from .first_stage_check import check_first_stage
from .recourse_check import check_recourse
from .random_streams import canonical_hash

HELPER_SHA256='041613210cfd024382f270d8c709ccd0de4586c2a86758a7b20c15739854714d'


def _need(value,message):
    if not value:raise ValueError(message)


def _read(ref):
    verify_reference(ref)
    return json.loads(Path(ref['path']).read_text())


def _write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    return file_reference(path)


def bind_saa_template_envelope(*, request_reference, template_references,
        first_stage_sha256, source_references, output_path):
    """Bind already saved original numerics; this binding is not numerical PASS."""
    request=_read(request_reference);ids=request['scenario_ids']
    _need(len(template_references)==len(ids),'Template support missing')
    for ref in source_references:verify_reference(ref)
    _need(source_references,'Original compiler/source references required')
    rows=[]
    for i,ref in enumerate(template_references):
        row,refs=recourse_store._record(ref)
        _need(row['scenario_index']==i and row['scenario_id']==ids[i] and
              row['template_sha256']==request['template_sha256s'][i],'Original template order/identity differs')
        rows.append(ref)
    return _write(output_path,dict(kind='SAA_ORIGINAL_TEMPLATE_ENVELOPE_V1',
        request_reference=request_reference,first_stage_sha256=first_stage_sha256,
        template_references=rows,source_references=source_references,
        numerical_acceptance=False,formal_OOS=False))


def audit_saa_fixed_plan_q(*, checkpoint_reference, expected_request_reference,
        template_envelope_reference, first_stage, case, tasks, scene_loader,
        empirical_probabilities, helper_reference, source_references, deadline,
        output_path=None):
    started=time.monotonic()
    def tick():
        _need(type(deadline) in (int,float) and math.isfinite(deadline),'Invalid shared audit deadline')
        if time.monotonic()>=deadline:raise TimeoutError('SAA full-Q audit deadline expired')
    tick()
    for ref in source_references:verify_reference(ref)
    for path in (__file__,recourse_store.__file__,first_stage_check.__file__,recourse_check.__file__):
        _need(file_reference(path) in source_references,'Actual numerical audit source missing')
    verify_reference(helper_reference);_need(helper_reference['sha256']==HELPER_SHA256,'Changed original KKT helper')
    spec=importlib.util.spec_from_file_location('_saa_original_kkt',helper_reference['path'])
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    request=_read(expected_request_reference);cp=_read(checkpoint_reference)
    _need(request.get('kind')=='SAA_SERIAL_FIXED_PLAN_Q_V1','Wrong SAA fixed-Q request kind')
    _need(cp['request_reference']==expected_request_reference and
        cp['status']=='COMPLETE_FIXED_Q_PENDING_INDEPENDENT_AUDIT','Complete original fixed-Q checkpoint required')
    for ref in request['sources']:verify_reference(ref)
    env=_read(template_envelope_reference)
    _need(env['kind']=='SAA_ORIGINAL_TEMPLATE_ENVELOPE_V1' and env['request_reference']==expected_request_reference,
        'Foreign original template envelope')
    for ref in env['source_references']:verify_reference(ref)
    ids=request['scenario_ids'];n=len(ids);pi=np.asarray(empirical_probabilities,dtype=float)
    constants=np.asarray(request['constants_usd'],dtype=float)
    _need(n>0 and len(set(ids))==n and len(request['scenario_sha256s'])==n and len(request['template_sha256s'])==n
        and len(env['template_references'])==n and constants.shape==(n,) and np.isfinite(constants).all(),'Invalid ordered support/constants')
    _need(pi.shape==(n,) and np.array_equal(pi,np.full(n,1./n)), 'SAA empirical weights must be strictly uniform')
    _need(request['domain'] in ('development','formal') and (request['domain']!='formal' or n==160),'Wrong formal support count')
    _need(cp['completed_indices']==list(range(n)) and len(cp['q_usd'])==n and np.isfinite(cp['q_usd']).all(),'Incomplete Q')
    pd=request['plan'];digest=canonical_hash(pd)
    _need(digest==request['plan_sha256'] and request['case_sha256']==case.case_sha256 and request['task_sha256']==tasks.task_sha256,
        'Changed physical plan identity')
    plan=SimpleNamespace(x=np.asarray(pd['x']),r=np.asarray(pd['r']),task_sha256=pd['task_sha256'],config_hash=pd['config_hash'])
    recourse_store._verify_first_stage(first_stage)
    _need(first_stage.matrix_sha256==env['first_stage_sha256'],'First-stage source differs')
    physical_first=check_first_stage(case,tasks,plan);_need(physical_first['passed'],'Original first-stage physics failed')
    zfirst=np.r_[plan.x,plan.r.ravel()]
    fv={k:getattr(first_stage,k) for k in ('row_lower','row_upper','lower_bounds','upper_bounds','integrality','c')}
    helper.rmp_residuals(first_stage.A,fv,zfirst)
    fpre=float(first_stage.c@zfirst)
    helper.close(fpre,physical_first['costs']['Fpre_usd'],'First-stage physical objective differs',1e-4+1e-7*abs(fpre))
    selected={}
    _need(len({r['path'] for r in cp['attempts']})==len(cp['attempts']),'Duplicate attempt reference')
    for ref in cp['attempts']:
        tick();row=_read(ref)
        _need(row['request_reference']==expected_request_reference,'Foreign fixed-Q attempt')
        names=[Path(r['path']).name for r in row['artifacts']]
        _need(len(set(names))==len(names),'Duplicate raw artifact')
        for a in row['artifacts']:
            _need(Path(a['path']).parent==Path(ref['path']).parent,'Foreign raw artifact path');verify_reference(a)
        _need(type(row['completed']) is bool,'Nonboolean completion flag')
        if row['completed']:
            i=row['scenario_index'];_need(type(i) is int and 0<=i<n and i not in selected,'Duplicate/invalid completed atom')
            selected[i]=(row,{Path(r['path']).name:r for r in row['artifacts']})
    _need(set(selected)==set(range(n)),'Missing complete scenario evidence')
    results=[]
    for i in range(n):
        tick();row,refs=selected[i];original,template_refs=recourse_store._record(env['template_references'][i])
        _need(row['scenario_id']==ids[i] and row['scenario_sha256']==request['scenario_sha256s'][i]
            and row['plan_sha256']==digest and row['template_sha256']==request['template_sha256s'][i], 'Raw Q identity differs')
        _need(original['scenario_index']==i and original['scenario_id']==ids[i] and original['template_sha256']==row['template_sha256'],
            'Original template support differs')
        matrices,arrays=recourse_store._numeric(template_refs)
        meta=helper.metadata_header(template_refs['identity.json']['path']);scene=scene_loader(i)
        _need(scene.scenario_id==ids[i] and scene.scenario_sha256==request['scenario_sha256s'][i] and
              scene.development is (request['domain']=='development'),'Scene order/domain/hash differs')
        recourse_store._identity_check(meta,original,first_stage,case,tasks,scene,matrices,arrays)
        _need(set(refs)=={'solver.json','plan.json','fixed_rhs.npz','primal.npy','physical_check.json','dual_kkt.npz'},
            'Raw fixed-Q artifacts incomplete')
        _need(_read(refs['plan.json'])==pd,'Saved fixed plan changed')
        solver=_read(refs['solver.json'])
        _need(row['native_status']==solver['status']=='OPTIMAL','Conflicting native fixed-Q status')
        vector=np.load(refs['primal.npy']['path'],allow_pickle=False)
        with np.load(refs['fixed_rhs.npz']['path'],allow_pickle=False) as v:fixed={k:v[k] for k in v.files}
        with np.load(refs['dual_kkt.npz']['path'],allow_pickle=False) as v:dual={k:v[k] for k in v.files}
        _need(set(dual)=={'inequality','equality','lower','upper'} and set(fixed)=={'b_ub','b_eq'},'Raw dual/RHS schema differs')
        _need(vector.shape==arrays['c'].shape and np.isfinite(vector).all(),'Raw primal shape/finiteness differs')
        _need(solver['certificate']['template_sha256']==meta['template_sha256'],'Solver source template differs')
        # The producer saved raw duals only. Derive auxiliary arithmetic here;
        # audit_kkt separately checks feasibility/stationarity/gap from originals.
        lo,hi=arrays['lower_bounds'],arrays['upper_bounds'];fl,fu=np.isfinite(lo),np.isfinite(hi)
        bu=arrays['b_ub']+matrices['Bx_ub']@plan.x+matrices['Br_ub']@plan.r.ravel()
        be=arrays['b_eq']+matrices['Bx_eq']@plan.x+matrices['Br_eq']@plan.r.ravel()
        aux=dict(dual,stationarity=arrays['c']-matrices['A_ub'].T@dual['inequality']-matrices['A_eq'].T@dual['equality']-dual['lower']-dual['upper'],
            inequality_slack=bu-matrices['A_ub']@vector,equality_residual=matrices['A_eq']@vector-be,
            inequality_complementarity=dual['inequality']*(bu-matrices['A_ub']@vector),
            lower_complementarity=dual['lower'][fl]*(vector-lo)[fl],upper_complementarity=dual['upper'][fu]*(hi-vector)[fu])
        kkt=helper.audit_kkt(matrices,arrays,plan,vector,aux,solver,fixed)
        slices,shapes=meta['variable_slices'],meta['variable_shapes']
        solution={k:vector[slice(*slices[k])].reshape(shapes[k]) for k in shapes}
        physical=check_recourse(case,tasks,scene,plan,SimpleNamespace(**solution,task_sha256=tasks.task_sha256,config_hash=tasks.config_hash,scenario_sha256=scene.scenario_sha256),check_objective_bound=True)
        _need(physical['passed'],'Independent physical witness failed')
        costs=np.zeros_like(arrays['component_objectives']);dt=case.dt_h
        costs[0,slice(*slices['shed'])]=np.broadcast_to(case.c_shed_usd_per_kwh[None,:]*dt,solution['shed'].shape).ravel()
        costs[1,slice(*slices['short'])]=case.c_short_usd_per_kwh*dt
        costs[2,slice(*slices['p'])]=np.broadcast_to(tasks.c_v2g_usd_per_kwh[:,None]*dt,solution['p'].shape).ravel()
        costs[3,slice(*slices['p'])]=np.broadcast_to(tasks.c_deg_usd_per_kwh[:,None]*dt/tasks.eta_dis[:,None],solution['p'].shape).ravel()
        helper.same(costs,arrays['component_objectives'],'Original physical cost coefficients differ')
        helper.same(costs.sum(axis=0),arrays['c'],'Original physical objective differs')
        q=kkt['primal_objective_usd']+float(constants[i]);tol=1e-4+1e-7*abs(q)
        helper.close(physical['costs']['Qop'],kkt['primal_objective_usd'],'Physical Q differs',tol)
        helper.close(row['q_usd'],q,'Saved Q/constant differs',tol);helper.close(cp['q_usd'][i],q,'Checkpoint Q differs',tol)
        results.append(dict(index=i,scenario_id=ids[i],q_usd=q,template_sha256=meta['template_sha256'],physical_passed=True,**kkt))
    tick();q=[r['q_usd'] for r in results];expected=float(pi@q);upper=fpre+expected
    _need(math.isfinite(upper),'Nonfinite empirical UB')
    result=dict(status='PASS_SAA_FIXED_PLAN_FULL_Q_AND_EMPIRICAL_LAW_AUDIT',passed=True,plan_sha256=digest,
        request_reference=expected_request_reference,checkpoint_reference=checkpoint_reference,
        template_envelope_reference=template_envelope_reference,source_references=source_references,helper_reference=helper_reference,
        support_ids=ids,q_usd=q,empirical_probabilities=pi.tolist(),risk_certificate_kind='fixed_empirical',
        first_stage_cost_usd=fpre,expected_recourse_usd=expected,upper_bound_usd=upper,rows=results,
        full_support_recourse_verified=True,solver_calls=0,probability_optimization_calls=0,
        raw_auxiliary_kkt_arithmetic_recomputed=True,master_bound_audited=False,budget_settled=False,
        scientific_spec_pass=False,formal_OOS=False,elapsed_seconds=time.monotonic()-started)
    if output_path is not None:return _write(output_path,result)
    return result
