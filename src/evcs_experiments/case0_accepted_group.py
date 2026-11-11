"""Read-only comparison views for the four explicitly accepted Case0.1 plans.

This joins existing numerical evidence and exact user decisions. It does not
replace physical audits, open OOS, invoke solvers, or rewrite native results.
"""
import json,math
from pathlib import Path
from .campaign_batch import file_reference,verify_reference
from .random_streams import canonical_hash
from .case0_accepted_plan import load_accepted_plan
from .case0_aro_methods import load_accepted_or

ACCEPTANCE_HASHES={
 'SAA':'3b18932a0c9328ae64a7791e4408cfb12963e21b5e23c7a8ef3e0bbed500de2f',
 'ARO':'ff3ed6058a96cca510c43beb4d697f941aa7d370e96094a262b6906b72d049ee',
 'OR_DRO':'54db3e59f29611a6816ce8a27ee832258fbc858068c1b5cce9d88f6acdd18898'}

def need(ok,msg):
 if not bool(ok):raise ValueError(msg)
def load_other(root,state,method):
 need(method in ACCEPTANCE_HASHES,'Unsupported accepted method');root=Path(root).resolve();need(root.name=='ieee33-v54-q1-case0-expanded' and state['size_index']==0,'Wrong campaign')
 key='size0/'+method;r=state['accepted_approximate_plans'][key];need(r['sha256']==ACCEPTANCE_HASHES[method],'Exact user decision required')
 checked={}
 def read(ref):
  verify_reference(ref);checked[ref['path']]=ref;return json.loads(Path(ref['path']).read_text())
 a=read(r);need(a['accepted_for_comparison'] and a['normal_gap_passed'] is False and a['user_gap_threshold']==.01,'Acceptance scope differs')
 refs=a['references'];is_or=method=='OR_DRO'
 if is_or:
  load_accepted_or(root,state);u_ref=refs['ub_reference']
 else:u_ref=refs['original_result']
 u=read(u_ref);b=read(refs['bound_reference']);t=read(refs['settlement_reference'])
 need(u['method_id']==method and canonical_hash(u['plan'])==u['plan_sha256']==a['plan_sha256']==b['plan_sha256'] and u['model_sha256']==a['model_sha256']==b['model_sha256'],'Accepted plan/model identity differs')
 need(t['passed'] and t['result_reference']==refs['bound_reference'] and t['independent_native_check']['passed'],'Native bound terminal proof missing')
 need(u['upper_bound_usd']==a['upper_bound_usd']==b['upper_bound_usd']==t['upper_bound_usd'] and a['lower_bound_usd']==b['lower_bound_usd']==t['lower_bound_usd'],'Accepted bounds differ')
 gap=(a['upper_bound_usd']-a['lower_bound_usd'])/max(1.,abs(a['upper_bound_usd']))
 need(math.isfinite(gap) and 0<=gap<=.01 and abs(gap-a['gap'])<1e-12,'Accepted gap differs')
 need(state['quality_deferral_resolutions'][key]['acceptance_reference']==r,'Acceptance does not resolve historical deferral')
 base=root/'planning'/key;identity_ref=file_reference(base/'inputs/identity.json');identity=read(identity_ref);header_ref=file_reference(base/'solve/model.json');h=read(header_ref)
 need(h['model_sha256']==a['model_sha256'] and identity['lead_h']==(0 if is_or else 1) and identity['profile']=='expanded48-v1' and identity['target']=='zeta_2020' and len(identity['vehicle_ids'])==48 and identity['variant_offset']==30000,'Original information/profile differs')
 need(u['scenario_ids']==h['scenario_ids'] and len(u['scenario_ids'])==len(set(u['scenario_ids']))==len(u['q_usd'])==8,'Original complete support differs')
 if not is_or:
  need(u['numerical_valid'] and u['full_support_independently_evaluated'] and u['gap_passed'] is False,'Original full-Q numerical record required')
  need(state['completed'][key]['sha256']==u_ref['sha256'],'Original completed result changed')
  for i in range(8):
   q=base/'solve'/f'scenario_{i:03d}';check=read(file_reference(q/'checks.json'));solver=read(file_reference(q/'fixed_solver.json'))
   need(check['kkt']['passed'] and check['physical']['passed'] and solver['status']=='OPTIMAL' and solver['certificate']['template_sha256']==h['recourse_template_sha256s'][i],'Original physical/Q receipt differs')
   need(abs(solver['objective_usd']-u['q_usd'][i])<=1e-4+1e-7*abs(u['q_usd'][i]),'Q objective differs')
   pr=file_reference(q/'primal.npz');verify_reference(pr);checked[pr['path']]=pr
 config=read(state['active_config_reference']);read(state['source_release_reference'])
 need(state['configuration_spent_seconds'][key]<=1800 and state['configuration_master_spent_seconds'][key]<=1200,'Original budget exceeded')
 result=dict(kind='CASE0_USER_ACCEPTED_COMPARISON_RESULT_V1',method_id=method,plan=u['plan'],plan_sha256=u['plan_sha256'],model_sha256=u['model_sha256'],scenario_ids=u['scenario_ids'],q_usd=u['q_usd'],upper_bound_usd=a['upper_bound_usd'],lower_bound_usd=a['lower_bound_usd'],gap=gap,gap_passed=False,accepted_for_comparison=True,normal_gap_target=.005,user_gap_threshold=.01,numerical_valid=True,backend=config['backend'],configuration_wall_seconds=state['configuration_spent_seconds'][key],original_native_status=u.get('native_status','TIME_LIMIT'),original_native_gap=u.get('gap'),acceptance_reference=r,original_result_reference=u_ref,auxiliary_bound_reference=refs['bound_reference'],formal_OOS=False)
 return dict(kind='CASE0_ACCEPTED_COMPARISON_PLAN_EVIDENCE_V1',acceptance_reference=r,result_reference=u_ref,input_identity_reference=identity_ref,producer_config_reference=state['active_config_reference'],producer_source_manifest_reference=state['source_release_reference'],producer_source_sha256=config['source_sha256'],comparison_result=result,artifact_references=list(checked.values()),scope='Source-bound existing proof join; no fresh numerical/physical solve')

def load_accepted_group(root,state=None):
 root=Path(root);state=json.loads((root/'state.json').read_text()) if state is None else state
 need(set(state['accepted_approximate_plans'])=={'size0/'+m for m in ('EA_DRO','OR_DRO','SAA','ARO')},'Four exact acceptances required')
 rows={'EA_DRO':load_accepted_plan(root,state)};need(rows['EA_DRO'] is not None,'EA acceptance missing')
 rows.update({m:load_other(root,state,m) for m in ('OR_DRO','SAA','ARO')})
 return dict(kind='CASE0_FOUR_USER_ACCEPTED_PLAN_GROUP_V1',methods=rows,all_user_accepted=True,planning_solver_calls=0,evaluation_opened=False,comparison_entry_integration_complete=False)
