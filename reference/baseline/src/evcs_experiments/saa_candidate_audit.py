"""Join original SAA master and independently minimized Q evidence, without a solve."""
import math
import time
from pathlib import Path
import numpy as np
from .campaign_batch import file_reference,verify_reference
from .random_streams import canonical_hash
from .saa_fixed_plan_audit import audit_saa_fixed_plan_q,_read,_need,_write
from .saa_master_audit import audit_saa_master


def audit_saa_candidate(*,model,recourses,case,tasks,scene_loader,master_result_reference,
        checkpoint_reference,fixed_q_request_reference,template_envelope_reference,
        helper_reference,source_references,deadline,output_path=None):
    """Preserve native state while joining its LB to the same plan's empirical UB.

    This component does not settle campaign time or admit a formal source/plan.
    The caller supplies accepted physical inputs and original template references.
    """
    _need(type(deadline) in (int,float) and math.isfinite(deadline) and time.monotonic()<deadline,
        'Expired or invalid SAA candidate audit deadline')
    for ref in source_references:verify_reference(ref)
    _need(file_reference(__file__) in source_references,'SAA candidate audit source missing')
    request=_read(fixed_q_request_reference);m=model.manifest
    _need(request['kind']=='SAA_SERIAL_FIXED_PLAN_Q_V1' and request['domain']==m['domain'],
          'SAA master/fixed-Q domain or kind differs')
    for key,other in [('scenario_ids','scenario_ids'),('scenario_sha256s','scenario_sha256s'),
                      ('template_sha256s',None),('constants_usd','scenario_cost_constants_usd')]:
        expected=[r['template_sha256'] for r in m['scenarios']] if other is None else m[other]
        _need(request[key]==expected,'SAA master/fixed-Q support or constants differ: '+key)
    master=audit_saa_master(model=model,recourses=recourses,case=case,tasks=tasks,scene_loader=scene_loader,
        result_reference=master_result_reference,source_references=source_references,deadline=deadline)
    _need(master['incumbent_physical_audited'] and master['native_solver_status'] in (0,1)
        and master['plan_sha256']==request['plan_sha256']==canonical_hash(request['plan']),
        'SAA master and fixed-Q must describe the same usable incumbent')
    lower=master['lower_bound_usd'];_need(type(lower) in (int,float) and math.isfinite(lower),'No usable native SAA lower bound')
    fixed=audit_saa_fixed_plan_q(checkpoint_reference=checkpoint_reference,expected_request_reference=fixed_q_request_reference,
        template_envelope_reference=template_envelope_reference,first_stage=model.first_stage,case=case,tasks=tasks,
        scene_loader=scene_loader,empirical_probabilities=m['empirical_probabilities'],helper_reference=helper_reference,
        source_references=source_references,deadline=deadline)
    _need(fixed['plan_sha256']==master['plan_sha256'],'SAA audited plan differs')
    upper=fixed['upper_bound_usd'];tol=1e-4+1e-7*max(1.,abs(upper))
    _need(lower<=upper+tol and upper<=master['extensive_incumbent_objective_usd']+tol,
        'SAA minimized-Q UB contradicts native LB or extensive feasible objective')
    gap=max(0.,upper-lower)/max(1.,abs(upper))
    _need(time.monotonic()<deadline,'SAA joined audit deadline exceeded')
    result=dict(status='PASS_SAA_CANDIDATE_NUMERICAL_AUDIT',passed=True,
        plan_sha256=master['plan_sha256'],risk_certificate_kind='fixed_empirical',
        lower_bound_usd=lower,upper_bound_usd=upper,gap_relative=gap,
        native_status=master['native_status'],native_solver_status=master['native_solver_status'],
        normal_gap_met=gap<=.005,master_audit=master,fixed_q_audit=fixed,
        extensive_incumbent_objective_usd=master['extensive_incumbent_objective_usd'],
        source_references=source_references,solver_calls=0,full_support_recourse_verified=True,
        budget_settled=False,scientific_spec_pass=False,formal_OOS=False)
    if output_path is not None:return _write(output_path,result)
    return result
