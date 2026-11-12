"""Audit a changed compact-master proposal on the original eight Case0 scenes.

A new plan gets new Q/KKT/physical and probability certificates. This operation
never solves another planning master, opens OOS, or changes campaign state.
"""
import argparse,importlib.util,json,math,os,time
from pathlib import Path
from evcs_experiments.campaign_batch import file_reference,verify_reference
from evcs_experiments.config import get_paths
from evcs_experiments.case0_campaign import verify_sources
from evcs_experiments.case0_profiles import validate_config,context_options,EXPANDED
from evcs_experiments.case0_isolation import verify_current_cgroup
from evcs_experiments.case0_inputs import Case0Context
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.runtime.jobctl import atomic_json


def need(ok,msg):
    if not bool(ok):raise ValueError(msg)

def read(ref):
    verify_reference(ref);return json.loads(Path(ref['path']).read_text())

def run_request(path,expected):
    started=time.monotonic();rr=file_reference(path);need(rr['sha256']==expected,'Proposal audit request changed')
    req=read(rr);refs=req['references'];need(set(req)=={'kind','references','output_directory','budget_seconds'} and
        req['kind']=='CASE0_PROPOSED_CANDIDATE_AUDIT_REQUEST_V1' and set(refs)=={'candidate','model','input_identity',
        'config','source_manifest','state','settlement','kernel','old_ub','review'},'Proposal audit schema differs')
    review=read(refs['review']);base=dict(req,references={k:v for k,v in refs.items() if k!='review'})
    need(review['kind']=='CASE0_PROPOSED_CANDIDATE_AUDIT_REVIEW_V1' and review['passed'] is True and
         review['request_without_review']==base and review['auditor_sha256']==file_reference(__file__)['sha256'] and
         review['kernel_reference']==refs['kernel'],'Exact proposal audit source/request review required')
    state=read(refs['state']);settlement=read(refs['settlement']);key='size0/EA_DRO'
    need(state['compact_candidate_settlement_reference']==refs['settlement'] and
         state['compact_candidate_result_reference']==refs['candidate'] and state['saved_candidate_ub_reference']==refs['old_ub'] and
         state['active_config_reference']==refs['config'] and state['source_release_reference']==refs['source_manifest'] and
         state['compute_seconds']==settlement['closing_compute_seconds'] and
         state['configuration_spent_seconds']=={key:settlement['closing_configuration_seconds']} and
         state['configuration_master_spent_seconds']=={key:settlement['closing_master_seconds']} and
         not state['active_job_id'] and not state['active_configuration'] and not state['planning_parent_pending_settlement'] and
         not state.get('pending_compact_candidate') and not state.get('pending_proposed_candidate_audit') and
         not state['evaluation_opened'] and state['completed']=={},'Proposal audit settled authority differs')
    need(settlement['kind']=='CASE0_COMPACT_CANDIDATE_SETTLEMENT_V1' and settlement['passed'] is True and
         settlement['result_reference']==refs['candidate'] and settlement['ub_result_reference']==refs['old_ub'] and
         settlement['scientific_planning_acceptance'] is False,'Compact settlement result/origin differs')
    terminal=read(settlement['terminal_review_reference'])
    need(terminal['kind']=='CASE0_COMPACT_CANDIDATE_TERMINAL_REVIEW_V1' and terminal['passed'] is True and
         terminal['numerical_recheck_passed'] is True and terminal['references']['result']==refs['candidate'],
         'Original compact candidate independent terminal review required')
    seconds=req['budget_seconds'];need(type(seconds) in (int,float) and math.isfinite(seconds) and
        0<seconds<=1800-settlement['closing_configuration_seconds'] and seconds<=21600-state['compute_seconds'],
        'Proposal audit original budget exceeded')
    paths=get_paths();root=paths.campaign_root
    need(root.name=='ieee33-v54-q1-case0-expanded' and Path(refs['state']['path']).is_relative_to(root/'operations') and
         Path(refs['state']['path'])!=root/'state.json','Immutable expanded campaign snapshot required')
    jobid=os.environ.get('EVCS_JOB_ID');jobdir=Path(os.environ.get('EVCS_JOB_DIRECTORY','')).resolve()
    need(jobid and jobdir==root/'jobs'/jobid,'Exact Case0 parent required')
    parent=json.loads((jobdir/'manifest.json').read_text());guard=review['guard_reference'];verify_reference(guard)
    need(Path(guard['path']).parent==Path(rr['path']).parent==Path(__file__).resolve().parent and
         Path(refs['kernel']['path']).parent==Path(rr['path']).parent,'Audited operation source paths differ')
    need(parent['job_id']==jobid and parent['outroot']==str(root/'jobs') and parent['threads']==1 and
         parent['argv']==[str(paths.project_root/'.venv/bin/python'),guard['path']] and
         seconds<parent['timeout_seconds'] and parent['timeout_seconds']+parent['grace_seconds']<=1800-settlement['closing_configuration_seconds'],
         'Proposal audit parent/guard envelope differs')
    verify_current_cgroup();source=read(refs['source_manifest']);config=read(refs['config']);verify_sources(config)
    need(Path(refs['source_manifest']['path'])==paths.project_root/'RELEASE_MANIFEST.json' and validate_config(config,root)==EXPANDED,
         'Original producer/config required')
    inventory={r['path']:r['sha256'] for r in source['files']}
    need(len(inventory)==len(source['files']) and all(inventory.get(k)==v for k,v in config['source_sha256'].items()),'Original source inventory differs')
    candidate=read(refs['candidate']);old=read(refs['old_ub']);header=read(refs['model']);identity=read(refs['input_identity'])
    need(candidate['kind']=='CASE0_COMPACT_DRO_CANDIDATE_RESULT_V1' and
         candidate['source_binding']['ub_result']==refs['old_ub'] and candidate['prior_plan_sha256']==old['plan_sha256'] and
         candidate['plan_sha256']==terminal['plan_sha256'],'Proposed historical anchor or terminal identity differs')
    need(candidate['changed_plan'] is True and candidate['first_stage_numeric_passed'] is True and
         candidate['planning_acceptance'] is False and candidate['plan_sha256']==canonical_hash(candidate['plan']) and
         candidate['plan_sha256']!=old['plan_sha256'] and candidate['model_sha256']==old['model_sha256']==header['model_sha256'],
         'A distinct valid proposed plan and original model are required')
    need(all(candidate['source_binding'][k]==refs[k] for k in ('model','input_identity','config','source_manifest')),
         'Proposed source binding differs')
    inputs=root/'planning/size0/EA_DRO/inputs'
    need(Path(refs['model']['path'])==inputs.parent/'compact-metadata-1/solve/model.json' and
         Path(refs['input_identity']['path'])==inputs/'identity.json' and (inputs/'calibration_manifest.json').is_file(),
         'Only original saved inputs allowed')
    context=Case0Context(inputs,case_root=paths.data_root,lead_hours=1,**context_options(config),**config['sizes'][0])
    need(context.identity==identity and len(context.planning_ids)==8,'Original eight support context differs')
    verify_reference(refs['kernel']);sp=importlib.util.spec_from_file_location('case0_proposed_numeric_kernel',refs['kernel']['path'])
    kernel=importlib.util.module_from_spec(sp);sp.loader.exec_module(kernel)
    out=Path(req['output_directory']).resolve();need(out.is_relative_to(root/'operations') and out!=root/'operations','New operation output required')
    out.mkdir(parents=True,exist_ok=False)
    try:
        return kernel.audit_candidate(master=None,proposal=candidate,header=header,context=context,output_dir=out,
            deadline=started+seconds,binding={k:v for k,v in refs.items() if k not in ('review','state','settlement')})
    except BaseException as e:
        atomic_json(out/'failure.json',dict(type=type(e).__name__,error=str(e),request_reference=rr,planning_acceptance=False));raise
    finally:
        atomic_json(out/'operation_accounting.json',dict(actual_seconds=time.monotonic()-started,master_seconds=0.,
            request_reference=rr,scope='Original full parent once config/global; all source/context/Q/audit work nested; no planning master or OOS.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--expected-sha256',required=True)
    a=p.parse_args();r=run_request(a.request,a.expected_sha256)
    print(json.dumps({k:r[k] for k in ('kind','upper_bound_usd','planning_acceptance')}))
