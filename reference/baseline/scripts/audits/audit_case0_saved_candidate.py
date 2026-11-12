"""Reaudit a saved Case0 EA candidate on its original planning support only.

Standalone operation script, run with the unchanged producer release installed.
No MILP, evaluation/OOS or campaign-state write is performed. Parent job receipts
are authoritative for full elapsed expense, including interrupted preparation.
"""
import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from evcs_experiments.campaign_batch import file_reference,verify_reference
from evcs_experiments.config import get_paths
from evcs_experiments.first_stage import FirstStagePlan
from evcs_experiments.first_stage_check import check_first_stage
from evcs_experiments.recourse import assemble_recourse,solve_fixed_plan
from evcs_experiments.recourse_check import check_recourse
from evcs_experiments.case0_planning import audit_case0_recourse,_json
from evcs_experiments.case0_planning_audit import _kkt
from evcs_experiments.probability import solve_worst_distribution
from evcs_experiments.probability_check import check_probability_solution
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.runtime.jobctl import atomic_json


def need(ok,message):
    if not bool(ok):raise ValueError(message)


def read(ref):
    verify_reference(ref)
    return json.loads(Path(ref['path']).read_text())


def remaining(deadline):
    value=deadline-time.monotonic()
    if value<=0:raise TimeoutError('Saved candidate audit deadline exhausted')
    return value


def close(left,right,label):
    need(math.isfinite(left) and math.isfinite(right) and
         abs(left-right)<=1e-4+1e-7*max(1.,abs(left),abs(right)),label+' differs')


def _saved_item(entry,template,plan,case,tasks,scene):
    need(entry['scenario_id']==scene.scenario_id and entry['template_sha256']==template.template_sha256,
         'Checkpoint scenario/template changed')
    for ref in entry['files'].values():verify_reference(ref)
    saved=read(entry['files']['fixed_solver'])
    with np.load(entry['files']['primal']['path'],allow_pickle=False) as data:
        need(data.files==['fixed'],'Checkpoint primal array inventory differs');primal=data['fixed'].copy()
    kkt=_kkt(template,plan,primal,saved)
    physical=check_recourse(case,tasks,scene,plan,template.unpack(primal),check_objective_bound=True)
    need(physical['passed'],'Reused fixed-Q physical check failed')
    close(float(physical['costs']['Qop']),float(kkt['qop_usd']),'Reused physical/KKT cost')
    close(float(entry['q_usd']),float(kkt['qop_usd']),'Checkpoint Q cost')
    return physical,kkt


def audit_candidate(*,master,header,context,output_dir,deadline,binding,resume=None,proposal=None,method_id='EA_DRO'):
    """Numeric kernel; context can be a small synthetic physical fixture in tests."""
    remaining(deadline);out=Path(output_dir)
    need(method_id in ('EA_DRO','OR_DRO'),'Saved candidate supports only original DRO methods')
    if method_id=='OR_DRO':
        need(getattr(context,'lead_hours',None)==0 and binding.get('method_id')=='OR_DRO' and proposal is None,
             'OR saved candidate requires its original lead0 context and explicit method binding')
    else:
        need(getattr(context,'lead_hours',1)==1,'EA saved candidate requires original lead1 context')
    need(out.is_dir() and not any(out.iterdir()),'Candidate numeric output must be newly reserved and empty')
    ids=tuple(context.planning_ids)
    if proposal is None:
        need(master['status']=='TIME_LIMIT' and master['lower_bound_usd'] is None and master['mip_gap'] is None and
             master['solution'] is not None and header['risk_model']=='DRO' and
             master['solution']['model_sha256']==header['model_sha256'],'Wrong saved no-bound EA master')
        solution=master['solution'];reported_objective=master['objective_usd']
    else:
        need(master is None and resume is None and header['risk_model']=='DRO' and
             proposal['model_sha256']==header['model_sha256'] and proposal['changed_plan'] is True and
             proposal['first_stage_numeric_passed'] is True and proposal['planning_acceptance'] is False,
             'Wrong compact proposed candidate or mixed legacy master')
        solution=proposal['plan'];reported_objective=None
        need(canonical_hash(solution)==proposal['plan_sha256'],'Proposed candidate identity differs')
    plan=FirstStagePlan(solution['x'],solution['r'],solution['task_sha256'],solution['config_hash'])
    need(plan.task_sha256==context.tasks.task_sha256==header['task_sha256'] and
         plan.config_hash==header['config_hash']==context.first.config_hash and
         header['first_stage_sha256']==context.first.matrix_sha256 and
         header['scenario_ids']==list(ids) and header['task_ids']==list(context.first.task_ids) and
         header['periods']==list(context.first.periods),'Saved candidate physical/task/support identity differs')
    need(len(ids)>0 and len(set(ids))==len(ids) and
         len(header['recourse_template_sha256s'])==len(header['scenario_sha256s'])==len(ids) and
         header['scenario_cost_constants_usd']==[0.]*len(ids),'Saved candidate support/constants differ')
    plan_json=dict(x=plan.x.tolist(),r=plan.r.tolist(),task_sha256=plan.task_sha256,config_hash=plan.config_hash)
    plan_hash=canonical_hash(plan_json)
    first=check_first_stage(context.case,context.tasks,plan)
    atomic_json(out/'candidate.json',dict(kind='CASE0_SAVED_CANDIDATE_V1',plan=plan_json,plan_sha256=plan_hash,
        source_binding=binding,reported_master_objective_usd=reported_objective,independent_full_master_vector_available=False,
        candidate_origin='saved_native_master' if proposal is None else 'compact_DRO_epigraph_proposal'))
    atomic_json(out/'first_stage_check.json',_json(first));need(first['passed'],'Saved candidate first-stage physical check failed')
    prior=[]
    if resume is not None:
        need(resume['kind']=='CASE0_SAVED_CANDIDATE_CHECKPOINT_V1' and resume['source_binding']==binding and
             resume['plan_sha256']==plan_hash and resume['scenario_ids']==list(ids),'Checkpoint candidate/source binding differs')
        prior=resume['completed'];need(len(prior)<=len(ids) and [e['index'] for e in prior]==list(range(len(prior))),
                                     'Checkpoint ordering/completeness differs')
    probability_set=context.probability_set;remaining(deadline)
    need(tuple(probability_set.support_ids)==ids and probability_set.ambiguity_sha256==header['ambiguity_sha256'],
         'Original planning ambiguity/support identity differs')
    completed=[];rows=[];q=[]
    def checkpoint():
        atomic_json(out/'checkpoint.json',dict(kind='CASE0_SAVED_CANDIDATE_CHECKPOINT_V1',source_binding=binding,
            plan_sha256=plan_hash,scenario_ids=list(ids),completed=completed,lower_bound_usd=None,gap=None,
            scientific_planning_acceptance=False,formal_OOS=False))
    checkpoint()
    for i,sid in enumerate(ids):
        began=time.monotonic();remaining(deadline);scene=context.scenario(i)
        need(scene.scenario_id==sid and scene.development is True and scene.split=='planning' and scene.regime=='R1',
             'Only original development planning R1 scenes are allowed')
        template=assemble_recourse(context.case,context.tasks,scene);remaining(deadline)
        need(template.template_sha256==header['recourse_template_sha256s'][i] and
             template.scenario_sha256==header['scenario_sha256s'][i], 'Original saved recourse support identity differs')
        if i<len(prior):
            entry=prior[i];physical,kkt=_saved_item(entry,template,plan,context.case,context.tasks,scene)
            entry=dict(entry,reused_from_checkpoint=True,revalidation_seconds=time.monotonic()-began)
        else:
            folder=out/f'scenario_{i:03d}';folder.mkdir()
            fixed=solve_fixed_plan(template,plan,time_limit_seconds=remaining(deadline))
            # Persist raw duals and primal before acceptance; failed checks remain reviewable.
            atomic_json(folder/'fixed_solver.json',_json(fixed.as_dict()))
            if fixed.solution is not None:np.savez_compressed(folder/'primal.npz',fixed=template.pack(fixed.solution))
            remaining(deadline);kkt=audit_case0_recourse(template,plan,fixed)
            physical=check_recourse(context.case,context.tasks,scene,plan,fixed.solution,check_objective_bound=True)
            need(physical['passed'],'Independent saved candidate fixed-Q physical check failed')
            close(float(physical['costs']['Qop']),float(kkt['qop_usd']),'Fixed-Q physical/KKT cost')
            atomic_json(folder/'checks.json',_json(dict(kkt=kkt,physical=physical)))
            entry=dict(index=i,scenario_id=sid,template_sha256=template.template_sha256,q_usd=float(kkt['qop_usd']),
                elapsed_seconds=time.monotonic()-began,reused_from_checkpoint=False,
                files={k:file_reference(folder/n) for k,n in (('fixed_solver','fixed_solver.json'),('primal','primal.npz'),('checks','checks.json'))})
            del fixed
        q.append(float(kkt['qop_usd']));rows.append(dict(scenario_id=sid,**physical['costs']))
        completed.append(entry);checkpoint();del template,scene,physical,kkt
        remaining(deadline)
    probability=solve_worst_distribution(probability_set,np.asarray(q),q_support_ids=ids,time_limit_seconds=remaining(deadline))
    check=check_probability_solution(probability_set,np.asarray(q),probability,q_support_ids=ids)
    atomic_json(out/'probability.json',_json(probability.as_dict()));atomic_json(out/'probability_check.json',_json(check.as_dict()))
    need(probability.certified_optimal and check.passed,'Original probability primal/dual optimality check failed')
    ub=float(first['costs']['Fpre_usd']+probability.dual_objective_usd)
    remaining(deadline)
    result=dict(kind='CASE0_SAVED_CANDIDATE_UB_AUDIT_V1' if proposal is None else 'CASE0_PROPOSED_CANDIDATE_UB_AUDIT_V1',passed=True,method_id=method_id,source_binding=binding,
        model_sha256=header['model_sha256'],plan=plan_json,plan_sha256=plan_hash,scenario_ids=list(ids),
        first_stage_physical_passed=True,full_support_fixed_Q_and_KKT_passed=True,probability_primal_dual_passed=True,
        q_usd=q,cost_components=rows,Fpre_usd=first['costs']['Fpre_usd'],risk_value_usd=float(probability.dual_objective_usd),
        upper_bound_usd=ub,native_lower_bound_usd=None,lower_bound_usd=None,gap=None,gap_passed=False,
        original_master_reported_objective_usd=reported_objective,original_master_vector_independently_replayed=False,
        native_full_vector_missing_from_saved_master=(proposal is None),
        candidate_origin='saved_native_master' if proposal is None else 'compact_DRO_epigraph_proposal',planning_acceptance=False,full_numerical_acceptance=False,
        formal_OOS=False,Case1_acceptance=False,completed_scenario_references=completed)
    atomic_json(out/'result.json',_json(result));return result


def run_request(request_path,expected_sha256):
    began=time.monotonic();ref=file_reference(request_path);need(ref['sha256']==expected_sha256,'Candidate request changed')
    request=read(ref)
    need(set(request)=={'kind','references','output_directory','budget_seconds','resume_checkpoint_reference'} and
         request['kind']=='CASE0_SAVED_CANDIDATE_AUDIT_REQUEST_V1' and
         set(request['references'])=={'master','model','input_identity','config','source_manifest','state','settlement','review'},'Wrong saved candidate request schema')
    refs=request['references'];review=read(refs['review']);without=dict(request,references={k:v for k,v in refs.items() if k!='review'})
    need(review['kind']=='CASE0_SAVED_CANDIDATE_AUDIT_REVIEW_V1' and review['passed'] is True and
         review['request_without_review']==without and review['auditor_sha256']==file_reference(__file__)['sha256'],
         'Exact saved candidate source/request review required')
    snapshot=read(refs['state']);settlement=read(refs['settlement']);key='size0/EA_DRO'
    need(settlement['kind']=='CASE0_R5_NATIVE_LIMIT_PARENT_SETTLEMENT_V1' and
         settlement['scientific_pass'] is False and settlement['gap_available'] is False and settlement['OOS_opened'] is False and
         settlement['references']['master']==refs['master'] and settlement['references']['source_manifest']==refs['source_manifest'] and
         snapshot['native_limit_settlement_reference']==snapshot['planning_parent_settlement_reference']==refs['settlement'] and
         snapshot['active_config_reference']==refs['config'] and snapshot['source_release_reference']==refs['source_manifest'] and
         snapshot['compute_seconds']==settlement['closing_compute_seconds'] and
         snapshot['configuration_spent_seconds']=={key:settlement['closing_configuration_seconds']} and
         snapshot['configuration_master_spent_seconds']=={key:settlement['closing_master_seconds']} and
         snapshot['completed']=={} and snapshot['evaluation_opened'] is False and snapshot['active_job_id'] is None and
         snapshot['active_configuration'] is None and snapshot['planning_parent_pending_settlement'] is False,
         'Candidate immutable authority snapshot/settlement differs')
    seconds=request['budget_seconds'];need(type(seconds) in (int,float) and math.isfinite(seconds) and
         0<seconds<=1800-snapshot['configuration_spent_seconds'][key] and
         seconds<=21600-snapshot['compute_seconds'], 'Candidate budget exceeds original settled allowance')
    close(settlement['remaining_configuration_seconds'],1800-settlement['closing_configuration_seconds'],'Remaining configuration')
    close(settlement['remaining_master_seconds'],1200-settlement['closing_master_seconds'],'Remaining master')
    deadline=began+seconds;paths=get_paths();root=paths.campaign_root
    need(root.name=='ieee33-v54-q1-case0-expanded','Saved candidate is expanded Case0 only')
    need(Path(refs['state']['path']).is_relative_to(root/'operations') and
         Path(refs['state']['path'])!=root/'state.json','Bind an immutable operation state snapshot before launching')
    jobid=os.environ.get('EVCS_JOB_ID');jobdir=Path(os.environ.get('EVCS_JOB_DIRECTORY','')).resolve()
    need(jobid and jobdir==root/'jobs'/jobid and (jobdir/'manifest.json').is_file(),
         'Candidate audit must run inside its exact Case0 jobctl parent')
    parent=json.loads((jobdir/'manifest.json').read_text())
    need(parent['job_id']==jobid and parent['outroot']==str(root/'jobs') and parent['threads']==1 and
         parent['timeout_seconds']+parent['grace_seconds']<=1800-snapshot['configuration_spent_seconds'][key] and
         seconds<parent['timeout_seconds'], 'Candidate parent budget/resource envelope differs')
    from evcs_experiments.case0_isolation import verify_current_cgroup
    verify_current_cgroup()
    source=read(refs['source_manifest']);need(Path(refs['source_manifest']['path'])==paths.project_root/'RELEASE_MANIFEST.json',
        'Use the unchanged original installed producer release')
    config=read(refs['config'])
    from evcs_experiments.case0_campaign import verify_sources
    from evcs_experiments.case0_profiles import validate_config,context_options,EXPANDED
    verify_sources(config);need(validate_config(config,root)==EXPANDED,'Wrong original expanded configuration')
    inventory={r['path']:r['sha256'] for r in source['files']}
    need(len(inventory)==len(source['files']) and all(inventory.get(k)==v for k,v in config['source_sha256'].items()),
         'Producer release/source inventory differs')
    inputs=root/'planning/size0/EA_DRO/inputs';attempt=inputs.parent/'compact-metadata-1'
    need(Path(refs['master']['path'])==attempt/'solve/master.json' and Path(refs['model']['path'])==attempt/'solve/model.json' and
         Path(refs['input_identity']['path'])==inputs/'identity.json','Foreign original candidate evidence paths')
    need((inputs/'calibration_manifest.json').is_file(),
         'Original calibration manifest must exist; saved-candidate audit cannot generate input files')
    identity=read(refs['input_identity']);master=read(refs['master']);header=read(refs['model'])
    from evcs_experiments.case0_inputs import Case0Context
    context=Case0Context(inputs,case_root=paths.data_root,lead_hours=1,**context_options(config),**config['sizes'][0])
    need(context.identity==identity and len(context.planning_ids)==8,'Original eight planning scenes/inputs differ')
    # Operational state may advance between reviewed batches; immutable scientific
    # identity remains common to a checkpoint and its explicitly reviewed resume.
    binding={k:v for k,v in refs.items() if k not in ('review','state','settlement')}
    need(request['resume_checkpoint_reference'] is None,
         'This first-run production admission does not authorize a later resumed parent; checkpoints remain available for separately reviewed recovery')
    resume=None
    out=Path(request['output_directory']).resolve()
    need(out.is_relative_to(root/'operations') and not out.is_relative_to(attempt) and out!=root/'operations','Candidate output must be new operations directory')
    out.mkdir(parents=True,exist_ok=False)
    try:
        result=audit_candidate(master=master,header=header,context=context,output_dir=out,deadline=deadline,binding=binding,resume=resume)
        return result
    except BaseException as error:
        atomic_json(out/'failure.json',dict(type=type(error).__name__,error=str(error),request_reference=ref,
            planning_acceptance=False,formal_OOS=False));raise
    finally:
        atomic_json(out/'operation_accounting.json',dict(actual_seconds=time.monotonic()-began,request_reference=ref,
            master_seconds=0.,scope='Entire standalone audit entry; parent receipt adds process/import/closing overhead. Original cumulative global/configuration ledger maintained by owner.'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',required=True);p.add_argument('--expected-sha256',required=True)
    a=p.parse_args();r=run_request(a.request,a.expected_sha256)
    print(json.dumps({k:r[k] for k in ('kind','upper_bound_usd','lower_bound_usd','gap','planning_acceptance')}))


if __name__=='__main__':main()
