"""Receipt adapter for an unchanged, source-bound SAA numerical joint audit.

No adversarial probability or restricted-master certificate is manufactured.
The independent numerical audit remains the authority; this adapter checks its
original evidence closure and exact field mapping before receipt settlement.
"""
import json
import math
from pathlib import Path
import numpy as np
from .evaluation_lock import _verify_reference, file_reference
from .random_streams import canonical_hash

STATUS='PASS_SAA_FORMAL_WORKER_NUMERICAL_AUDIT'

def _need(ok,message):
    if not bool(ok): raise ValueError(message)

def _read(ref):
    _verify_reference(ref)
    return json.loads(Path(ref['path']).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))

def _closure(value,seen=None):
    seen=set() if seen is None else seen
    if isinstance(value,dict):
        if set(value)=={'path','sha256','size_bytes'}:
            key=(value['path'],value['sha256'],value['size_bytes'])
            if key in seen:return
            seen.add(key);_verify_reference(value)
            if Path(value['path']).suffix=='.json':_closure(_read(value),seen)
        else:
            for v in value.values():_closure(v,seen)
    elif isinstance(value,list):
        for v in value:_closure(v,seen)

def _joint(request,candidate_record,joint):
    _need(request.get('method_id')=='SAA','Explicit SAA request required')
    _closure(joint)
    if request.get('model_input_mode') in ('l_shaped_development', 'l_shaped_formal'):
        from .saa_l_shaped_candidate import validate_development_joint, validate_formal_joint
        check = validate_formal_joint if request['model_input_mode'] == 'l_shaped_formal' else validate_development_joint
        return check(joint, request, candidate_record)
    _need(joint.get('status')=='PASS_SAA_CANDIDATE_NUMERICAL_AUDIT' and joint.get('passed') is True
          and joint.get('risk_certificate_kind')=='fixed_empirical'
          and joint.get('full_support_recourse_verified') is True,'Complete SAA joint numerical audit required')
    master,fixed=joint['master_audit'],joint['fixed_q_audit']
    _need(master['status']=='PASS_INDEPENDENT_SAA_MASTER_AUDIT' and master['incumbent_physical_audited'] is True
          and master['diagnostic_only'] is False and fixed['status']=='PASS_SAA_FIXED_PLAN_FULL_Q_AND_EMPIRICAL_LAW_AUDIT'
          and fixed['passed'] is True and fixed['full_support_recourse_verified'] is True
          and fixed['raw_auxiliary_kkt_arithmetic_recomputed'] is True
          and fixed['risk_certificate_kind']=='fixed_empirical' and fixed['probability_optimization_calls']==0,
          'SAA original master/full-Q evidence differs')
    candidate=candidate_record['candidate'];plan=candidate['plan'];digest=canonical_hash(plan)
    _need(candidate_record['plan_id']==request['plan_id'] and master['plan']==plan
          and all(p['plan_sha256']==digest for p in (candidate,joint,master,fixed)), 'SAA plan identity differs')
    manifest=_read(master['model_manifest']);native_result=_read(master['result_reference'])
    native=_read(native_result['raw_native_result']);qreq=_read(fixed['request_reference'])
    _need(manifest['risk_model']=='SAA' and manifest['domain']==request['domain']==qreq['domain'], 'SAA model domain differs')
    if 'model_manifest_reference' in request:
        _need(request['model_manifest_reference']==master['model_manifest'],'SAA requested model differs')
    _need(native_result['model_manifest']==master['model_manifest'] and native_result['primal'] is not None,
          'SAA original primal/model missing')
    ids=manifest['scenario_ids'];n=len(ids)
    _need(n>0 and len(set(ids))==n and fixed['support_ids']==qreq['scenario_ids']==ids
          and qreq['scenario_sha256s']==manifest['scenario_sha256s']
          and qreq['template_sha256s']==[x['template_sha256'] for x in manifest['scenarios']]
          and qreq['constants_usd']==manifest['scenario_cost_constants_usd'], 'SAA ordered support/constants differ')
    pi=np.asarray(fixed['empirical_probabilities']);q=np.asarray(fixed['q_usd'])
    _need(pi.shape==q.shape==(n,) and np.isfinite(q).all()
          and np.array_equal(pi,np.full(n,1./n)) and np.array_equal(pi,manifest['empirical_probabilities']),
          'SAA empirical law differs')
    _need(qreq['plan']==plan and qreq['plan_sha256']==digest and master['original_recourse_witness_count']==n,
          'SAA original plan/support witness differs')
    vector=np.load(native_result['primal']['path'],allow_pickle=False,mmap_mode='r')
    first=np.r_[np.asarray(plan['x']),np.asarray(plan['r']).ravel()]
    _need(vector.ndim==1 and np.isfinite(first).all() and np.array_equal(vector[:len(first)],first),
          'SAA saved primal first-stage plan differs')
    checkpoint=_read(fixed['checkpoint_reference'])
    _need(checkpoint['request_reference']==fixed['request_reference']
          and checkpoint['completed_indices']==list(range(n))
          and np.asarray(checkpoint['q_usd']).shape==(n,)
          and np.allclose(checkpoint['q_usd'],q,rtol=1e-7,atol=1e-4),
          'SAA original fixed-Q checkpoint differs')

    sources=request['planning_source_references']
    _need(bool(sources) and all(ref in sources for ref in joint['source_references'])
          and master['source_references']==fixed['source_references']==joint['source_references'],
          'SAA original source closure differs')
    from . import saa_candidate_audit,saa_master_audit,saa_fixed_plan_audit
    _need(all(file_reference(p) in joint['source_references'] for p in
              (saa_candidate_audit.__file__,saa_master_audit.__file__,saa_fixed_plan_audit.__file__)),
          'SAA actual numerical auditor source missing')
    status=joint['native_status'];code=joint['native_solver_status']
    _need((status,code) in (('OPTIMAL',0),('TIME_LIMIT',1)) and type(code) is int
          and master['native_status']==native_result['status']==status
          and master['native_solver_status']==native_result['solver_status']==native['status']==code,
          'SAA original native status differs')
    if request['domain']=='formal':
        _need(n==160 and native['backend']=='gurobi','Formal SAA support/backend differs')
    lb,ub,gap=(joint[k] for k in ('lower_bound_usd','upper_bound_usd','gap_relative'))
    _need(all(type(v) in (int,float) and math.isfinite(v) for v in (lb,ub,gap)),'Nonfinite SAA bound/gap')
    tolerance=1e-4+1e-7*max(1.,abs(ub))
    constant=manifest['objective_constant_usd'];rawlb=native['raw_lower_bound_without_constant_usd']
    _need(type(rawlb) in (int,float) and math.isfinite(rawlb)
          and constant==float(pi@np.asarray(manifest['scenario_cost_constants_usd']))
          and native_result['objective_constant_usd']==constant
          and abs(rawlb+constant-lb)<=tolerance and abs(native_result['lower_bound_usd']-lb)<=tolerance
          and abs(native_result['objective_usd']-master['extensive_incumbent_objective_usd'])<=tolerance,
          'SAA original native lower bound/constant differs')

    _need(lb==master['lower_bound_usd'] and ub==fixed['upper_bound_usd']
          and lb<=ub+tolerance and ub<=master['extensive_incumbent_objective_usd']+tolerance
          and abs(float(pi@q)-fixed['expected_recourse_usd'])<=tolerance
          and abs(fixed['first_stage_cost_usd']+float(pi@q)-ub)<=tolerance
          and math.isclose(gap,max(0.,ub-lb)/max(1.,abs(ub)),rel_tol=1e-10,abs_tol=1e-12),
          'SAA audited bounds/law arithmetic differs')
    _need(joint['normal_gap_met'] is (gap<=.005),'SAA normal gap flag differs')
    for key in ('lower_bound_usd','upper_bound_usd','gap_relative'):
        _need(candidate[key]==joint[key],'SAA candidate bound differs: '+key)
    _need(candidate['native_master_status']==candidate_record['raw_engine_status']==status
          and candidate['native_master_solver_status']==code
          and candidate_record['driver_status']==('CONVERGED' if gap<=.005 else 'TIME_LIMIT')
          and candidate.get('original_rmp') is None,'SAA native/convergence semantics differ')
    return dict(plan_sha256=digest,lower_bound_usd=lb,upper_bound_usd=ub,gap_relative=gap,
                native_master_status=status,native_master_solver_status=code,raw_engine_status=status)

def build_saa_record_bridge(request_reference,candidate_record_reference,candidate_audit_reference,output_path=None):
    request,candidate,joint=map(_read,(request_reference,candidate_record_reference,candidate_audit_reference))
    values=_joint(request,candidate,joint)
    answer=dict(status=STATUS,passed=True,full_support_verified=True,plan_id=request['plan_id'],
        request_reference=request_reference,candidate_record_reference=candidate_record_reference,
        candidate_audit_reference=candidate_audit_reference,risk_certificate_kind='fixed_empirical',
        scientific_acceptance=False,formal_OOS=False,**values)
    answer['bridge_sha256']=canonical_hash(answer)
    if output_path is None:return answer
    with Path(output_path).open('x') as f:json.dump(answer,f,indent=2,allow_nan=False);f.write('\n')
    return file_reference(output_path)

def validate_saa_record_bridge(audit,request,candidate_record):
    body=dict(audit);digest=body.pop('bridge_sha256')
    _need(canonical_hash(body)==digest and body.get('status')==STATUS and body.get('passed') is True
          and body.get('full_support_verified') is True and body.get('risk_certificate_kind')=='fixed_empirical',
          'SAA bridge changed')
    joint=_read(audit['candidate_audit_reference']);values=_joint(request,candidate_record,joint)
    _need(all(audit[k]==v for k,v in values.items()),'SAA bridge differs from original joint audit')
    return joint


def joint_physical_binding(joint, request, candidate_record):
    """Read physical identity only after checking the explicit evidence branch."""
    _joint(request, candidate_record, joint)
    if request.get('model_input_mode') in ('l_shaped_development', 'l_shaped_formal'):
        return joint['support_ids'], joint['input_binding']
    manifest = _read(joint['master_audit']['model_manifest'])
    return manifest['scenario_ids'], manifest['input_binding']
