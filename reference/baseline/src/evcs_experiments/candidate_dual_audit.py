"""Independent saved-array audit of candidate affine cuts; calls no solver.

Trusted original context admission belongs to the reviewed controller. Explicit
expected identities must originate there, not from the candidate producer.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse import load_npz

from .campaign_batch import verify_reference
from .random_streams import canonical_hash
from .recourse_store import _semantic_sha

MATRICES=('A_ub','A_eq','Bx_ub','Bx_eq','Br_ub','Br_eq')


def _need(ok,message):
    if not bool(ok): raise ValueError(message)


def _same(a,b,message):
    _need(np.array_equal(a,b),message)


def _close(a,b,tol,message):
    a,b=np.asarray(a),np.asarray(b)
    _need(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
          and float(np.max(abs(a-b),initial=0.))<=tol,message)


def _read(ref):
    verify_reference(ref)
    return json.loads(Path(ref['path']).read_text())


def _inventory(record):
    refs=record['artifacts'];names=[Path(r['path']).name for r in refs]
    _need(len(names)==len(set(names)), 'Duplicate original artifact')
    for ref in refs: verify_reference(ref)
    return dict(zip(names,refs))


def _npz(ref):
    verify_reference(ref)
    with np.load(ref['path'],allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def _metadata(ref):
    verify_reference(ref)
    # Original exporter puts numerical identity before descriptive variable_map.
    with Path(ref['path']).open() as f:
        text=''
        while True:
            part=f.read(8192);text+=part
            _need(len(text)<=4*1024*1024,'Oversized/unknown original metadata header')
            if '\n  "variable_map":' in text:
                return json.loads(text.split('\n  "variable_map":',1)[0].rstrip().removesuffix(',')+'}')
            if not part:return json.loads(text)


def audit_candidate_atom(cut,gradient,provenance,*,expected_index,expected_scenario_id,
                         expected_template_sha256,expected_first_stage_sha256,expected_constant,
                         expected_plan_sha256,expected_q_usd,deadline):
    """Read original evidence and independently rederive one globally valid cut.

    provenance requires event_reference, template_reference, anchor_plan_reference.
    Expected values are supplied from independently admitted original context.
    Returns only one-atom numerical evidence, not a full bank/plan acceptance.
    """
    def alive():
        if not np.isfinite(deadline) or time.monotonic()>=deadline:raise TimeoutError('Candidate cut audit deadline')
    alive()
    _need(type(expected_index) is int and expected_index>=0,'Invalid original scenario index')
    _need(np.isfinite(expected_constant) and np.isfinite(expected_q_usd),'Nonfinite original constant/Q')
    event=_read(provenance['event_reference']);refs=_inventory(event)
    _need(event['event']=='recourse' and event['domain']=='formal'
          and event['scenario_index']==expected_index and event['scenario_id']==expected_scenario_id
          and event['plan_sha256']==expected_plan_sha256,'Original event scenario/anchor differs')
    pointer=_read(refs['template_reference.json'])
    _need(pointer==provenance['template_reference'],'Original template pointer differs')
    row=_read(pointer);tr=_inventory(row)
    _need(row['kind']=='recourse_template' and row['scenario_index']==expected_index
          and row['scenario_id']==expected_scenario_id and row['template_sha256']==expected_template_sha256,
          'Original template scenario/order differs')
    plan=_read(provenance['anchor_plan_reference'])
    _need(plan==_read(refs['plan.json']) and canonical_hash(plan)==expected_plan_sha256,'Original anchor plan differs')
    x=np.asarray(plan['x'],dtype=float);r=np.asarray(plan['r'],dtype=float).ravel();anchor=np.r_[x,r]
    _need(np.isfinite(anchor).all(),'Nonfinite anchor')
    arrays=_npz(tr['arrays.npz']);meta=_metadata(tr['identity.json'])
    matrices={key:load_npz(tr[key+'.npz']['path']).tocsr() for key in MATRICES}
    alive()
    _need(all(np.isfinite(a.data).all() for a in matrices.values()),'Nonfinite original coefficients')
    _need(meta['template_sha256']==expected_template_sha256
          and _semantic_sha(meta,matrices,arrays,expected_first_stage_sha256)==expected_template_sha256,
          'Original template semantic coefficients changed')
    _need(meta['task_sha256']==plan['task_sha256'] and meta['config_hash']==plan['config_hash'],
          'Original task/config anchor identity differs')
    _same(arrays['c'],arrays['component_objectives'].sum(axis=0),'Original cost components differ')
    base_ub,base_eq=arrays['b_ub'],arrays['b_eq']
    fixed=_npz(refs['fixed_rhs.npz']);dual=_npz(refs['dual_kkt.npz']);z=_npz(refs['primal_vector.npz'])['vector']
    solver=_read(refs['solver.json']);cert=solver['certificate']
    _need(solver['status']=='OPTIMAL' and solver['optimal'] is True and solver['solver_status']==0,
          'Original candidate LP not optimal')
    bu=base_ub+matrices['Bx_ub']@x+matrices['Br_ub']@r
    be=base_eq+matrices['Bx_eq']@x+matrices['Br_eq']@r
    _same(bu,fixed['b_ub'],'Original fixed RHS ub differs')
    _same(be,fixed['b_eq'],'Original fixed RHS eq differs')
    y,eq,lo,hi=[np.asarray(dual[k]) for k in ('inequality','equality','lower','upper')]
    n=len(arrays['c'])
    _need([a.shape for a in (y,eq,lo,hi,z)]==[(len(bu),),(len(be),),(n,),(n,),(n,)]
          and all(np.isfinite(a).all() for a in (y,eq,lo,hi,z)),'Invalid original primal/dual axes')
    for key,a in zip(('inequality','equality','lower','upper'),(y,eq,lo,hi)):
        _same(a,cert['dual_'+key+'_marginals'],'Raw dual differs from original solver')
    lb,ub=arrays['lower_bounds'],arrays['upper_bounds'];fl,fu=np.isfinite(lb),np.isfinite(ub)
    _need(not np.isnan(lb).any() and not np.isnan(ub).any() and np.all(lb<=ub)
          and not np.isposinf(lb).any() and not np.isneginf(ub).any(),'Invalid original recourse bounds')
    maximum=lambda a:float(np.max(a,initial=0.))
    slack=bu-matrices['A_ub']@z;residual=matrices['A_eq']@z-be
    stationarity=arrays['c']-matrices['A_ub'].T@y-matrices['A_eq'].T@eq-lo-hi
    primal=max(maximum(-slack),maximum(abs(residual)),maximum(lb-z),maximum(z-ub))
    station=maximum(abs(stationarity));sign=max(maximum(y),maximum(-lo),maximum(hi))
    inactive=max(maximum(abs(lo[~fl])),maximum(abs(hi[~fu])))
    primal_q=float(arrays['c']@z);dual_q=float(bu@y+be@eq+lb[fl]@lo[fl]+ub[fu]@hi[fu])
    tol=1e-4+1e-7*max(1.,abs(primal_q),abs(dual_q))
    _need(primal<=1e-7 and max(station,sign,inactive)<=1e-8 and abs(primal_q-dual_q)<=tol,
          'Original candidate primal/dual KKT failed')
    arithmetic=dict(stationarity=stationarity,inequality_slack=slack,equality_residual=residual,
        inequality_complementarity=y*slack,lower_complementarity=lo[fl]*(z-lb)[fl],upper_complementarity=hi[fu]*(ub-z)[fu])
    for key,value in arithmetic.items():_close(dual[key],value,1e-8,'Saved KKT arithmetic differs: '+key)
    for key in ('dual_objective_usd','dual_bound_usd'):_close(cert[key],dual_q,tol,'Saved native dual bound differs')
    _close(cert['primal_bound_usd'],primal_q,tol,'Saved native primal bound differs')
    _close(solver['objective_usd'],primal_q,tol,'Saved native objective differs')
    for value in (event['q_usd'],expected_q_usd):_close(value,primal_q+expected_constant,tol,'Original candidate Q/constant differs')
    # Independent minorant: base RHS only in intercept; fixed RHS never added twice.
    intercept=float(base_ub@y+base_eq@eq+lb[fl]@lo[fl]+ub[fu]@hi[fu]+expected_constant)
    calculated=np.r_[matrices['Bx_ub'].T@y+matrices['Bx_eq'].T@eq,
                     matrices['Br_ub'].T@y+matrices['Br_eq'].T@eq]
    _close(gradient,calculated,1e-8,'Derived cut gradient differs from original dual')
    _close(cut['intercept_usd'],intercept,1e-8,'Derived cut intercept differs from original base RHS')
    _need(cut['scenario_id']==expected_scenario_id and cut['template_sha256']==expected_template_sha256
          and cut['scenario_constant_usd']==expected_constant and cut['anchor_first_stage_sha256']==hashlib.sha256(anchor.astype('<f8').tobytes()).hexdigest(),
          'Derived cut original scenario/constant/anchor differs')
    _need(cut['passed'] is True and cut['status']=='PASS_NUMERICAL_RECOURSE_AFFINE_MINORANT'
          and cut['dual_tolerance']==1e-8 and cut['first_stage_columns']==len(anchor)
          and cut['x_columns']==len(x) and cut['reserve_columns']==len(r),
          'Derived cut numerical status/axes differ')
    for key,value in (('max_dual_stationarity_residual',station),('max_dual_sign_violation',sign),
                      ('max_infinite_bound_multiplier',inactive)):
        _close(cut[key],value,1e-8,'Derived cut numerical residual differs')
    anchor_value=float(intercept+calculated@anchor)
    anchor_tol=1e-4+1e-7*max(1.,abs(expected_q_usd),abs(anchor_value))
    _close(cut['anchor_affine_value_usd'],anchor_value,1e-8,'Derived anchor affine value differs')
    _close(cut['anchor_gap_usd'],expected_q_usd-anchor_value,1e-8,'Derived anchor gap differs')
    _need(cut['anchor_gap_tolerance_usd']==anchor_tol,'Derived anchor tolerance differs')
    _same(cut['gradient_sha256'],hashlib.sha256(np.asarray(gradient,dtype='<f8').tobytes()).hexdigest(),'Cut gradient identity differs')
    body=dict(cut);digest=body.pop('cut_sha256');_need(canonical_hash(body)==digest,'Cut record changed')
    _close(cut['anchor_q_usd'],expected_q_usd,tol,'Cut anchor Q differs')
    _close(intercept+calculated@anchor,expected_q_usd,tol,'Derived cut not tight at original candidate')
    alive()
    return dict(kind='INDEPENDENT_CANDIDATE_CUT_ATOM_V1',passed=True,index=expected_index,
        scenario_id=expected_scenario_id,plan_sha256=expected_plan_sha256,template_sha256=expected_template_sha256,
        scenario_constant_usd=float(expected_constant),q_usd=float(expected_q_usd),
        intercept_usd=intercept,gradient_sha256=cut['gradient_sha256'],cut_sha256=digest,
        source_references={k:provenance[k] for k in ('event_reference','template_reference','anchor_plan_reference')},
        original_primal_residual=primal,original_dual_stationarity=station,original_duality_gap=abs(primal_q-dual_q),
        solver_calls=0,physical_replay_repeated=False,scientific_acceptance=False)


def audit_candidate_axis(records,*,support_ids,probability_set,expected_ambiguity_sha256,expected_plan_sha256):
    ids=tuple(support_ids);records=list(records)
    _need(ids==tuple(probability_set.support_ids) and probability_set.ambiguity_sha256==expected_ambiguity_sha256,
          'Original ambiguity/support axis changed')
    _need(len(records)==len(ids) and len(set(ids))==len(ids)>0,'Incomplete candidate cut support')
    for i,(sid,record) in enumerate(zip(ids,records)):
        _need(record['kind']=='INDEPENDENT_CANDIDATE_CUT_ATOM_V1' and record['passed'] is True
              and record['index']==i and record['scenario_id']==sid and record['plan_sha256']==expected_plan_sha256,
              'Candidate cut order/anchor/audit differs')
    return dict(kind='INDEPENDENT_CANDIDATE_CUT_AXIS_V1',passed=True,support_ids=list(ids),
                ambiguity_sha256=expected_ambiguity_sha256,plan_sha256=expected_plan_sha256,
                complete_cut_count=len(records),solver_calls=0,scientific_acceptance=False)


def audit_derived_atom(context,index,result,*,deadline):
    """Thin bridge from separately admitted CandidateDualContext; no producer call."""
    cut,gradient,provenance=result
    _need(provenance['context_binding']==context.binding
          and provenance['context_binding_sha256']==context.binding['binding_sha256'],
          'Candidate atom context binding differs')
    return audit_candidate_atom(cut,gradient,provenance,
        expected_index=index,expected_scenario_id=context.ids[index],
        expected_template_sha256=context.template_hashes[index],
        expected_first_stage_sha256=context.identity['first_stage_sha256'],
        expected_constant=float(context.constants[index]),expected_plan_sha256=canonical_hash(context.plan),
        expected_q_usd=float(context.q[index]),deadline=deadline)
