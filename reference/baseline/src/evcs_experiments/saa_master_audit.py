"""Independent SAA extensive witness/native-bound audit; no fixed-plan Q claim."""
import json
import math
from pathlib import Path
import time
import numpy as np
from .campaign_batch import verify_reference
from .evaluation_lock import file_reference
from .random_streams import canonical_hash
from . import saa_streaming, saa_streaming_audit, saa_streaming_solve, first_stage_check, recourse_check, recourse


def _need(value,message):
    if not bool(value):raise ValueError(message)


def _read(reference):
    verify_reference(reference)
    return json.loads(Path(reference['path']).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))


def _close(actual,expected,message):
    _need(type(actual) in (int,float) and math.isfinite(actual) and math.isfinite(expected)
        and abs(actual-expected)<=1e-4+1e-7*max(1.,abs(expected)),message)


def audit_saa_master(*,model,recourses,case,tasks,scene_loader,result_reference,
                     source_references,deadline,output_path=None):
    """Recheck exact assembly and saved extensive primal without solving again.

    A native branch-and-bound lower bound remains native evidence; its search
    proof is not replayed. The extensive feasible objective is not the new
    independently minimized full-Q UB. Formal admission remains external.
    """
    def check_time():
        _need(type(deadline) in (int,float) and math.isfinite(deadline),'Finite deadline required')
        if time.monotonic()>=deadline:raise TimeoutError('SAA master audit deadline')
    check_time()
    required=[__file__,saa_streaming.__file__,saa_streaming_audit.__file__,saa_streaming_solve.__file__,recourse.__file__,
              first_stage_check.__file__,recourse_check.__file__]
    for ref in source_references:verify_reference(ref)
    _need(all(file_reference(path) in source_references for path in required),'SAA master audit/producer/checkers source binding missing')
    result=_read(result_reference);manifest_ref=file_reference(model.root/'manifest.json')
    _need(result['model_manifest']==manifest_ref and result['algorithm_id']==model.manifest['algorithm_id'],
          'Native result model/algorithm differs')
    assembly=saa_streaming_audit.audit_streaming_saa(model,recourses,deadline=deadline)
    native=_read(result['raw_native_result']);status=native['status'];formal=model.manifest['domain']=='formal'
    _need(type(status) is int and status in (0,1,2,3,4) and result['solver_status']==status,'Native solver status differs')
    _need(native['backend'] in ('highs','gurobi') and (not formal or native['backend']=='gurobi'),'Native backend differs')
    _need(result['full_support_verified'] is False and result['scientific_spec_pass'] is False
        and result['formal_OOS'] is False,'Extensive result cannot claim full-Q or scientific acceptance')
    if formal:
        options=native['options']
        _need(type(options['threads']) is int and options['threads']==4 and options['mip_rel_gap']==.001
            and type(options['time_limit']) in (int,float) and 0<options['time_limit']<=600,
            'Frozen formal native solver options differ')
    if native['backend']=='gurobi':
        from . import native_gurobi
        _need(file_reference(native_gurobi.__file__) in source_references,'Native Gurobi source must be bound')
        interface=native['native_interface'];_need(isinstance(interface,dict),'Native interface missing')
        _need(interface.get('gurobi_version')==[12,0,3],'Native Gurobi version differs')
        native_status=interface['status'];raw_code=interface['raw_model_status']
        _need(type(raw_code) is int,'Native raw status code missing')
        _need(interface['valid_incumbent'] is (result['primal'] is not None),'Native incumbent metadata differs')
        if native_status=='MODEL_LOADING_TIME_LIMIT':
            _need(status==1 and result['status']=='TIME_LIMIT' and result['primal'] is None
                and native['raw_lower_bound_without_constant_usd'] is None
                and native['raw_objective_without_constant_usd'] is None,
                'Model-loading timeout cannot have a result/bound')
        else:
            names={1:'LOADED',2:'OPTIMAL',3:'INFEASIBLE',4:'INF_OR_UNBD',5:'UNBOUNDED',6:'CUTOFF',
                7:'ITERATION_LIMIT',8:'NODE_LIMIT',9:'TIME_LIMIT',10:'SOLUTION_LIMIT',11:'INTERRUPTED',
                12:'NUMERIC',13:'SUBOPTIMAL',14:'INPROGRESS',15:'USER_OBJ_LIMIT',16:'WORK_LIMIT',17:'MEM_LIMIT'}
            limits={7,8,9,10,11,15,16,17}
            _need(native_status==names.get(raw_code,'STATUS_'+str(raw_code)),'Native raw status name/code differs')
            failures=interface['certificate_failures']
            expected=4 if failures else 0 if raw_code==2 else 1 if raw_code in limits else 2 if raw_code==3 else 3 if raw_code==5 else 4
            _need(status==expected,'Native normalized solver status differs')
            label=native_status if status==1 else {0:'OPTIMAL',2:'INFEASIBLE',3:'UNBOUNDED',4:'SOLVER_ERROR'}[status]
            _need(result['status']==label,'Original Gurobi normalized status differs')
            if failures:
                _need(status==4 and result['primal'] is None and native['raw_lower_bound_without_constant_usd'] is None
                    and native['raw_objective_without_constant_usd'] is None,'Native failed certificate cannot supply a plan/bound')
        if native['raw_lower_bound_without_constant_usd'] is not None:
            _need(interface['is_mip'] is True and interface['solver_bound_consistent'] and interface['bound_source']=='MIP_ObjBound',
                'Unusable original native MIP bound')
            _close(native['raw_lower_bound_without_constant_usd'],interface['solver_bound_without_constant_usd'],'Native bound interface differs')
    else:
        native_status=result['status']
        _need(result['status']=={0:'OPTIMAL',1:'TIME_LIMIT',2:'INFEASIBLE',3:'UNBOUNDED',4:'SOLVER_ERROR'}[status],
              'Original HiGHS status differs')
    constant=model.objective_constant_usd;_close(result['objective_constant_usd'],constant,'Constant offset differs')
    rawlb=native['raw_lower_bound_without_constant_usd']
    lower=None if rawlb is None or status not in (0,1) else float(rawlb)+constant
    if lower is None:_need(result['lower_bound_usd'] is None,'Invalid native lower bound promoted')
    else:_close(result['lower_bound_usd'],lower,'Native lower bound offset differs')
    plan_dict=None;objective=None;physical=[];primal_ref=result['primal']
    if primal_ref is not None:
        verify_reference(primal_ref);vector=np.load(primal_ref['path'],allow_pickle=False)
        _need(vector.shape==model.c.shape and np.isfinite(vector).all(),'Native primal shape/nonfinite')
        activity=model.A@vector
        _need(np.isfinite(activity).all() and np.max(model.row_lower-activity,initial=0)<=1e-8
            and np.max(activity-model.row_upper,initial=0)<=1e-8
            and np.max(model.lower_bounds-vector,initial=0)<=1e-8 and np.max(vector-model.upper_bounds,initial=0)<=1e-8
            and np.max(abs(vector[model.integrality==1]-np.rint(vector[model.integrality==1])),initial=0)<=1e-8,
            'Native extensive primal violates original coefficients')
        plan=model.plan(vector);first=first_stage_check.check_first_stage(case,tasks,plan)
        _need(first['passed'],'Original first-stage physics failed')
        plan_dict=dict(x=plan.x.tolist(),r=plan.r.tolist(),task_sha256=plan.task_sha256,config_hash=plan.config_hash)
        original=float(model.first_stage.c@vector[:len(model.first_stage.c)])
        _need(len(result['original_witness_checks'])==len(recourses),'Original witness coverage differs')
        for i,row in enumerate(model.manifest['scenarios']):
            check_time();template=recourses[i];scene=scene_loader(i)
            _need(scene.scenario_id==row['scenario_id'] and scene.scenario_sha256==row['scenario_sha256']
                and scene.physical_case_sha256==case.case_sha256 and scene.development is (not formal),'Original witness scene identity differs')
            y=model.original_recourse_vector(i,vector);bub,beq=template.fixed_rhs(plan)
            _need(np.isfinite(y).all() and np.max(template.A_ub@y-bub,initial=0)<=1e-8
                and np.max(abs(template.A_eq@y-beq),initial=0)<=1e-8
                and np.max(template.lower_bounds-y,initial=0)<=1e-8 and np.max(y-template.upper_bounds,initial=0)<=1e-8,
                'Original recourse coefficient witness fails')
            check=recourse_check.check_recourse(case,tasks,scene,plan,template.unpack(y),check_objective_bound=False)
            _need(check['passed'],'Original recourse physics failed')
            cost=float(template.c@y+model.manifest['scenario_cost_constants_usd'][i]);original+=row['empirical_probability']*cost
            saved=_read(result['original_witness_checks'][i])
            _need(saved['scenario_id']==row['scenario_id'] and saved['template_sha256']==template.template_sha256
                and saved['independently_minimized'] is False,'Saved extensive witness identity/scope differs')
            _close(saved['cost_usd'],cost,'Saved extensive cost differs');physical.append(check)
        objective=float(model.c@vector+constant)
        _close(original,objective,'Original weighted objective differs')
        _close(result['objective_usd'],objective,'Result objective differs')
        _close(native['raw_objective_without_constant_usd'],objective-constant,'Native objective differs')
        _need(lower is None or lower<=objective+1e-4+1e-7*max(1.,abs(objective)),'Native LB exceeds extensive incumbent')
    else:
        _need(result['objective_usd'] is None and native['raw_objective_without_constant_usd'] is None and not result['original_witness_checks'],'Absent native primal cannot become incumbent')
    gap=None if objective is None or lower is None else max(0.,objective-lower)/max(1.,abs(objective))
    _need(result['incumbent_physical_checks_passed'] is (objective is not None),'Incumbent flag differs')
    if gap is None:_need(result['gap'] is None,'Missing bound/primal cannot have gap')
    else:_close(result['gap'],gap,'Native extensive gap differs')
    _need(result['normal_convergence'] is (status==0 and gap is not None and gap<=.005),'Normal convergence flag differs')
    check_time()
    for ref in source_references:verify_reference(ref)
    verify_reference(result_reference)
    answer=dict(status='PASS_INDEPENDENT_SAA_MASTER_AUDIT',result_reference=result_reference,model_manifest=manifest_ref,
        source_references=source_references,assembly_audit=assembly,plan=plan_dict,
        plan_sha256=None if plan_dict is None else canonical_hash(plan_dict),lower_bound_usd=lower,
        extensive_incumbent_objective_usd=objective,extensive_gap=gap,native_status=result['status'],native_solver_status=status,raw_native_status=native_status,diagnostic_only=status not in (0,1) or plan_dict is None,
        incumbent_physical_audited=objective is not None,original_recourse_witness_count=len(physical),
        native_bound_source_only=True,branch_and_bound_proof_replayed=False,full_support_Q_audited=False,
        fixed_plan_reoptimization_required=True,scientific_spec_pass=False,formal_OOS=False)
    if output_path is not None:
        with Path(output_path).open('x') as stream:json.dump(answer,stream,indent=2,allow_nan=False);stream.write('\n')
    return answer
