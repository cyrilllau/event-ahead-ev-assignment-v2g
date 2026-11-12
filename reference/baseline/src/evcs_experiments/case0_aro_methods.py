"""Bounded remaining Case0 methods after an explicitly accepted exact EA plan.

Uses the unchanged installed configuration worker; never calls its EA branch.
The outer owned job must be terminally settled after this inner accounting.
"""
import argparse,importlib.util,json,math,os,sys,time,traceback
from pathlib import Path
from evcs_experiments import case0_campaign as campaign
from evcs_experiments.campaign_batch import file_reference,verify_reference
from evcs_experiments.config import get_paths
from evcs_experiments.runtime.jobctl import atomic_json

METHODS=('ARO',)

def need(value,message):
    if not bool(value):raise ValueError(message)
def read(ref):
    verify_reference(ref);return json.loads(Path(ref['path']).read_text())
def load(ref,name):
    verify_reference(ref);spec=importlib.util.spec_from_file_location(name,ref['path']);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def load_accepted_or(root,state):
    """Bind this exact user-accepted OR candidate and already audited bound."""
    ref=state.get('accepted_approximate_plans',{}).get('size0/OR_DRO');need(ref is not None,'Explicit accepted OR required')
    a=read(ref);need(a['kind']=='CASE0_EXACT_PLAN_APPROXIMATE_ACCEPTANCE_V1' and a['method_id']=='OR_DRO' and
        a['accepted_for_comparison'] is True and a['gap_passed'] is False and a['normal_gap_passed'] is False and
        a['user_gap_threshold']==.01 and a['plan_sha256']=='69c6f86b944fcc87338fc2b3fb449e3037a64cfc4798226d47e5d9aaa5458520','Exact OR acceptance scope differs')
    refs=a['references'];u=read(refs['ub_reference']);b=read(refs['bound_reference']);t=read(refs['settlement_reference'])
    need(u['method_id']=='OR_DRO' and u['plan_sha256']==a['plan_sha256']==b['plan_sha256'] and u['model_sha256']==a['model_sha256']==b['model_sha256'],'Accepted OR identity changed')
    need(u['passed'] and u['first_stage_physical_passed'] and u['full_support_fixed_Q_and_KKT_passed'] and u['probability_primal_dual_passed'] and t['passed'] and t['independent_native_check']['passed'] and t['independent_outward_relaxation_check']['passed'],'Accepted OR numerical evidence missing')
    need(t['result_reference']==refs['bound_reference'] and u['upper_bound_usd']==b['upper_bound_usd']==t['upper_bound_usd']==a['upper_bound_usd'] and b['lower_bound_usd']==t['lower_bound_usd']==a['lower_bound_usd'],'Accepted OR bounds changed')
    gap=(a['upper_bound_usd']-a['lower_bound_usd'])/abs(a['upper_bound_usd'])
    need(0<=gap<=.01 and abs(gap-a['gap'])<1e-12 and state['quality_deferral_resolutions']['size0/OR_DRO']['acceptance_reference']==ref,'Accepted OR gap/resolution changed')
    return dict(acceptance_reference=ref,plan_sha256=a['plan_sha256'],gap=gap,gap_passed=False,accepted_for_comparison=True,source_references=refs)

def check_saa_deferral(state):
    d=state.get('deferred_quality_plans',{}).get('size0/SAA',{});need(d.get('accepted_for_comparison') is False,'Explicit SAA quality deferral required')
    r=read(d['bound_reference']);t=read(d['settlement_reference']);u=state['completed']['size0/SAA'];need(file_reference(u['path'])['sha256']==u['sha256'],'SAA original result changed')
    need(t['passed'] and t['result_reference']==d['bound_reference'] and r['gap']==d['gap'] and r['gap']>.005 and not r['planning_acceptance'],'SAA quality deferral proof differs')
    return d

def run_remaining(root,operation,*,seconds,accepted_api,api=campaign):
    started=time.monotonic();deadline=started+seconds;root=Path(root);operation=Path(operation)
    state=api.read(root/'state.json');config=api.active_config(root);api.verify_sources(config)
    need(root.name=='ieee33-v54-q1-case0-expanded' and state['size_index']==0,'Only locked expanded group')
    need(not state['evaluation_opened'] and state['active_configuration'] is None and not state.get('remaining_methods_parent_pending_settlement'),'Prior configuration/evaluation/unsettled parent blocks continuation')
    need(config['configuration_seconds']==1800 and config['master_seconds']==1200 and config['total_compute_seconds']==21600 and config['gap_target']==.005,'Original budgets/normal gap differ')
    need(tuple(config['planning_methods'])==('EA_DRO','OR_DRO','SAA','ARO'),'Original method group differs')
    accepted_or=load_accepted_or(root,state);saa_deferral=check_saa_deferral(state)
    atomic_json(operation/'accepted_or_comparison.json',accepted_or)
    opening_compute=state['compute_seconds'];opening_ea=(state['configuration_spent_seconds']['size0/EA_DRO'],state['configuration_master_spent_seconds']['size0/EA_DRO'])
    accepted=accepted_api.load_accepted_plan(root,state)
    need(accepted is not None and accepted['comparison_result']['accepted_for_comparison'] is True and accepted['comparison_result']['gap_passed'] is False,'Exact accepted EA bridge required')
    record_path=operation/'accepted_ea_comparison.json';need(not record_path.exists(),'Fresh comparison record required')
    atomic_json(record_path,accepted)
    state['accepted_ea_record_reference']=file_reference(record_path)
    charged=started;method_rows=[]
    def charge():
        nonlocal charged
        now=time.monotonic();state['compute_seconds']+=now-charged;charged=now
    try:
        for method in METHODS:
            key='size0/'+method
            if key in state['completed']:
                saved=state['completed'][key];reference=file_reference(saved['path']);need(reference['sha256']==saved['sha256'],'Completed result hash changed')
                result=read(reference)
                need(result['method_id']==method and result['backend']==config['backend'] and result['numerical_valid'] is True and result['gap_passed'] is True and 0<=result['gap']<=.005,'Completed remaining-method record is not normally accepted')
                folder,spent=api.configuration_attempt(root,state,key)
                accounting=api.read(folder/'accounting.json')
                need(accounting['method']==method and accounting['result_reference']==reference and accounting['failed'] is False and accounting['configuration_wall_seconds']==spent==state['configuration_spent_seconds'][key] and accounting['master_seconds']==state['configuration_master_spent_seconds'][key] and 0<=spent<=1800 and 0<=accounting['master_seconds']<=1200,'Completed method accounting/source differs or budget exceeded')
                continue
            charge()
            remaining=min(deadline-time.monotonic(),config['total_compute_seconds']-state['compute_seconds'])
            if remaining<1810:
                state['status']='REMAINING_METHODS_PARENT_BOUNDARY';state['next_action']='Current bounded parent cannot fit another full1800s configuration. Settle parent, then admit remaining locked methods; do not rerun accepted EA.';break
            folder,spent=api.configuration_attempt(root,state,key)
            need(spent==0 and state.get('configuration_master_spent_seconds',{}).get(key,0)==0 and not folder.exists(),'New remaining configuration must have no prior attempt; interrupted work needs explicit settlement/recovery')
            method_started=time.monotonic();folder.mkdir(parents=True,exist_ok=False)
            state.update(status='REMAINING_METHODS_RUNNING',active_configuration=key,next_action='Healthy remaining-method planning; accepted EA is retained without another solve. Inspect once, do not restart.')
            api.save_state(root,state)
            atomic_json(folder/'start.json',dict(method=method,config_sha256=api.digest(Path(state['active_config_reference']['path'])),configuration_started_monotonic=method_started,started_at_utc=api.utcnow(),prior_configuration_seconds=0.,prior_master_seconds=0.,accepted_ea_record_reference=state['accepted_ea_record_reference'],source_release_reference=state['source_release_reference'],active_config_reference=state['active_config_reference']))
            result=None;master_seconds=None;failed=False
            try:
                result=api.bounded_configuration(method,method_started,folder)
                need(result['method_id']==method and result['backend']==config['backend'] and result['numerical_valid'] is True and result['formal_OOS'] is False,'Remaining worker identity/numerical result differs')
                need(result['gap_passed'] is (result['gap']<=.005) and math.isfinite(result['gap']) and result['gap']>=0,'Unchanged normal gap flag required')
                master_seconds=float(result['master_wall_seconds']);need(math.isfinite(master_seconds) and 0<=master_seconds<=1200,'Actual master budget invalid/exceeded')
                state['completed'][key]=dict(path=str(folder/'solve/result.json'),sha256=api.digest(folder/'solve/result.json'),gap=result['gap'],numerical_valid=True,gap_passed=result['gap_passed'])
                state['status']='REMAINING_METHODS_RUNNING' if result['gap_passed'] else 'DIAGNOSIS_REQUIRED'
                if not result['gap_passed']:state['next_action']='This remaining method has valid time-limited evidence but missed its unchanged0.5% target. Independently audit and diagnose; EA/OR user acceptances cannot waive ARO.'
            except BaseException as error:
                failed=True;atomic_json(folder/'failure.json',dict(type=type(error).__name__,error=str(error),traceback=traceback.format_exc()))
                state.update(status='FAILURE_REQUIRES_REVIEW',next_action='Preserve failed remaining-method evidence and fees; independently inspect before another launch. EA must never be rerun.')
                if isinstance(error,(KeyboardInterrupt,SystemExit)):raise
            finally:
                elapsed=time.monotonic()-method_started
                if master_seconds is None:
                    path=folder/'solve/master.json'
                    if path.exists():
                        raw=api.read(path);value=raw.get('wall_seconds')
                        if type(value) in (int,float) and math.isfinite(value) and value>=0:master_seconds=float(value)
                actual_master_known=master_seconds is not None
                if master_seconds is None:master_seconds=elapsed
                charge();state['active_configuration']=None
                state.setdefault('configuration_spent_seconds',{})[key]=elapsed
                state.setdefault('configuration_master_spent_seconds',{})[key]=master_seconds
                receipt=dict(method=method,configuration_wall_seconds=elapsed,master_seconds=master_seconds,master_seconds_measured=actual_master_known,master_charge_scope='Saved actual extensive master wall' if actual_master_known else 'No master receipt: whole failed configuration is conservative master upper bound, never clamped',global_charge_nested=True,failed=failed,result_reference=None if result is None else file_reference(folder/'solve/result.json'))
                atomic_json(folder/'accounting.json',receipt);method_rows.append(file_reference(folder/'accounting.json'));api.save_state(root,state)
            if state['status']!='REMAINING_METHODS_RUNNING':break
        else:
            state.update(status='ARO_RESULT_PENDING_SAA_QUALITY_AND_ALL_PLAN_AUDIT',next_action='ARO result present; SAA quality explicitly deferred. EA and OR accepted without further refinement. Settle parent and audit ARO; resolve SAA and lock all plans/evaluator before OOS.')
    finally:
        charge()
        need((state['configuration_spent_seconds']['size0/EA_DRO'],state['configuration_master_spent_seconds']['size0/EA_DRO'])==opening_ea,'Accepted EA costs changed during other methods')
        state['remaining_methods_parent_pending_settlement']=True
        atomic_json(operation/'inner_accounting.json',dict(kind='CASE0_REMAINING_METHODS_INNER_ACCOUNTING_V1',opening_compute_seconds=opening_compute,closing_inner_compute_seconds=state['compute_seconds'],inner_charged_seconds=state['compute_seconds']-opening_compute,method_accounting_references=method_rows,accepted_ea_record_reference=state['accepted_ea_record_reference'],complete_inner=True,parent_settlement_required=True,scope='Inner work charged global once; method/master amounts are nested. Terminal settlement adds only nonnegative parent-minus-inner residual, with all external metadata separately once.'))
        api.save_state(root,state)
    return state

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',required=True);parser.add_argument('--expected-sha256',required=True);args=parser.parse_args()
    rr=file_reference(args.request);need(rr['sha256']==args.expected_sha256,'Changed remaining request');request=read(rr)
    need(request['kind']=='CASE0_REMAINING_METHODS_REQUEST_V1' and request['methods']==list(METHODS),'Exact remaining-only request required')
    review=read(request['review_reference']);need(review['kind']=='CASE0_REMAINING_METHODS_PRODUCTION_REVIEW_V1' and review['passed'] is True and review['request_without_review']=={k:v for k,v in request.items() if k!='review_reference'} and review['driver_sha256']==file_reference(__file__)['sha256'],'Actual source/request independent review required')
    refs=request['references'];paths=get_paths();root=paths.campaign_root
    need(root.name=='ieee33-v54-q1-case0-expanded' and Path(refs['source_manifest']['path'])==paths.project_root/'RELEASE_MANIFEST.json','Original isolated R5 runtime required')
    from evcs_experiments.case0_isolation import verify_current_cgroup
    verify_current_cgroup()
    for ref in refs.values():verify_reference(ref)
    need(file_reference(campaign.__file__)==refs['campaign_source'],'Unchanged installed worker/controller source required')
    state=campaign.read(root/'state.json');opening=read(refs['state']);jobid=os.environ['EVCS_JOB_ID'];job=Path(os.environ['EVCS_JOB_DIRECTORY']).resolve()
    need(job==root/'jobs'/jobid and state['active_job_id']==jobid and state['pending_remaining_methods']==file_reference(Path(request['operation_directory'])/'launch_admission.json'),'Exact preregistered owned parent required')
    need(state['compute_seconds']==opening['compute_seconds'] and state['accepted_approximate_plans']==opening['accepted_approximate_plans'] and state['active_config_reference']==refs['config']==opening['active_config_reference'] and state['source_release_reference']==refs['source_manifest'],'Source/acceptance/opening budget differs')
    need(all(state.get(k,{})==opening.get(k,{}) for k in ('configuration_spent_seconds','configuration_master_spent_seconds','completed','configuration_attempt_roots','deferred_quality_plans')),'Configuration accounting/attempt history changed after admission')
    manifest=campaign.read(job/'manifest.json');verify_reference(review['guard_reference'])
    cap=manifest['timeout_seconds']+manifest['grace_seconds'];need(manifest['argv']==[str(paths.project_root/'.venv/bin/python'),review['guard_reference']['path']] and manifest['threads']==1 and 0<request['seconds']<manifest['timeout_seconds'] and cap<=min(7200,21600-state['compute_seconds']),'Owned total parent envelope differs')
    operation=Path(request['operation_directory']).resolve();need(operation.is_relative_to(root/'operations') and operation!=root/'operations','Fresh operation path required')
    helper=load(refs['accepted_plan_source'],'evcs_experiments.case0_accepted_plan')
    return run_remaining(root,operation,seconds=request['seconds'],accepted_api=helper)

if __name__=='__main__':main()
