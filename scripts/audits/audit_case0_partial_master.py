"""Bounded stage2: solve the saved one-row augmented auxiliary master only."""
import argparse,importlib.util,json,math,os,time
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
from evcs_experiments.campaign_batch import file_reference,verify_reference
from evcs_experiments.probability import ProbabilitySet
from evcs_experiments.relaxation_bound import coefficient_sha256
from evcs_experiments.native_gurobi import solve_milp
from evcs_experiments.case0_planning import _json
from evcs_experiments.runtime.jobctl import atomic_json


def need(v,m):
 if not bool(v):raise ValueError(m)
def read(r):verify_reference(r);return json.loads(Path(r['path']).read_text())
def load(r,name):
 verify_reference(r);s=importlib.util.spec_from_file_location(name,r['path']);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def model(matrix,arrays):
 verify_reference(matrix);verify_reference(arrays)
 with np.load(arrays['path'],allow_pickle=False) as f:d={k:f[k].copy() for k in f.files}
 need(set(d)=={'c','integrality','lower_bounds','upper_bounds','row_lower','row_upper','objective_constant_usd'},'Saved model array schema differs')
 d['matrix']=load_npz(matrix['path']).tocsr();d['objective_constant_usd']=float(d['objective_constant_usd']);return d

def run_numeric(refs,out,deadline,native_seconds,helper,native_auditor):
 out=Path(out)
 need(type(native_seconds) in (int,float) and math.isfinite(native_seconds) and 0<native_seconds<=8,'Native grant exceeds8s')
 q=read(refs['q_result']);ex=read(refs['exclusion']);sel=read(refs['selection']);cut=read(refs['cut']);old=read(refs['old_ub'])
 need(q['kind']=='CASE0_PARTIAL_SCENARIO_CUT_RESULT_V1' and q['passed'] is True and q['new_Q_solve_count']==1 and q['master_solve_count']==0 and
      q['planning_acceptance'] is False and q['full_candidate_verified'] is False,'One-scenario evidence required')
 need(q['incumbent_exclusion_reference']==refs['exclusion'] and q['selection_reference']==refs['selection'] and q['cut_reference']==refs['cut'] and
      q['partial_model_audit_reference']==refs['partial_audit'],'Q result child references differ')
 need(ex['master_call_required'] is True and ex['incumbent_excluded'] is True,'No required master: do not repeat saturated model')
 for k in ('base_matrix','base_arrays','base_audit','base_vector','old_ub'):
  need(q['source_binding'][k]==refs[k],'Original Q base evidence differs: '+k)
 need(q['upper_bound_usd']==old['upper_bound_usd'] and q['old_plan_sha256']==old['plan_sha256'],'Original accepted UB differs')
 root=Path(refs['q_result']['path']).parent
 for k,n in [('matrix','augmented_matrix.npz'),('arrays','augmented_arrays.npz'),('probability_set','probability_set.json'),('gradient','new_gradient.npy')]:
  need(Path(refs[k]['path'])==root/n,'Foreign saved partial artifact: '+k)
 p=read(refs['probability_set']);P=ProbabilitySet(tuple(p['support_ids']),tuple(p['feature_ids']),p['features'],p['lower'],p['upper'])
 need(P.ambiguity_sha256==p['ambiguity_sha256'],'Original P changed')
 base=model(refs['base_matrix'],refs['base_arrays']);aug=model(refs['matrix'],refs['arrays']);ba=read(refs['base_audit'])
 verify_reference(refs['gradient']);g=np.load(refs['gradient']['path'],allow_pickle=False)
 verify_reference(refs['base_vector']);v=np.load(refs['base_vector']['path'],allow_pickle=False)
 previous=read(refs['partial_audit'])
 check=helper.audit_partial_model(aug,base,ba,P,sel,cut,g,v,source_references=previous['source_references'],deadline=deadline)
 need(check==previous and check['master_call_required'] is True,'Saved augmented model independent audit differs')
 atomic_json(out/'model_reaudit.json',_json(check));sha=coefficient_sha256(aug)
 remaining=deadline-time.monotonic();need(remaining>4.,'Insufficient native and final-audit reserve')
 grant=min(float(native_seconds),remaining-4.)
 atomic_json(out/'native_started.json',dict(grant_seconds=grant,coefficient_sha256=sha,no_new_Q=True))
 solved=solve_milp(aug['c'],aug['integrality'],aug['lower_bounds'],aug['upper_bounds'],aug['matrix'],aug['row_lower'],aug['row_upper'],
   options=dict(time_limit=grant,mip_rel_gap=.005,threads=1,solver_feasibility_tolerance=1e-9),log_path=out/'native.log')
 atomic_json(out/'native_record.json',_json(solved.native_record))
 if solved.x is not None:np.save(out/'auxiliary_vector.npy',solved.x,allow_pickle=False)
 native_check=native_auditor.audit_partial_native_bound(aug,check,solved.native_record,native_status=int(solved.status),native_bound=solved.mip_dual_bound,
   incumbent=solved.x,expected_coefficient_sha256=sha,fixed_upper_bound=q['upper_bound_usd'],deadline=deadline)
 atomic_json(out/'native_audit.json',_json(native_check))
 lower=max(q['lower_bound_usd'],native_check['lower_bound_usd']);upper=q['upper_bound_usd'];gap=max(0.,upper-lower)/max(1.,abs(upper))
 need(time.monotonic()<deadline,'Final bound deadline expired')
 result=dict(kind='CASE0_PARTIAL_MASTER_BOUND_RESULT_V1',status='AUDITED_PARTIAL_SCENARIO_AUXILIARY_BOUND',source_binding=refs,
   model_sha256=q['model_sha256'],old_plan_sha256=q['old_plan_sha256'],selected_index=q['selected_index'],scenario_id=q['scenario_id'],
   lower_bound_usd=lower,prior_lower_bound_usd=q['lower_bound_usd'],new_native_lower_bound_usd=native_check['lower_bound_usd'],
   upper_bound_usd=upper,gap=gap,gap_passed=False,diagnostic_gap_within_threshold=gap<=.005,native_status=native_check['native_status'],
   native_audit_reference=file_reference(out/'native_audit.json'),new_Q_solve_count=0,master_solve_count=1,
   new_plan_accepted=False,full_candidate_verified=False,planning_acceptance=False,formal_OOS=False,requires_terminal_parent_settlement=True)
 atomic_json(out/'result.json',_json(result));return result


def run_request(path,sha):
 started=time.monotonic();rr=file_reference(path);need(rr['sha256']==sha,'Changed request');r=read(rr)
 need(set(r)=={'kind','references','output_directory','budget_seconds','native_seconds','review_reference'} and r['kind']=='CASE0_PARTIAL_MASTER_REQUEST_V1','Partial master request schema differs')
 review=read(r['review_reference']);need(review['kind']=='CASE0_PARTIAL_MASTER_PRODUCTION_REVIEW_V1' and review['passed'] is True and
  review['request_without_review']=={k:v for k,v in r.items() if k!='review_reference'} and review['auditor_sha256']==file_reference(__file__)['sha256'],'Exact production review required')
 refs=r['references'];s=read(refs['state']);sett=read(refs['settlement']);key='size0/EA_DRO'
 need(sett['kind']=='CASE0_PARTIAL_SCENARIO_CUT_SETTLEMENT_V1' and sett['passed'] is True and sett['result_reference']==refs['q_result'],'Accepted Q settlement/result differs')
 terminal=read(sett['terminal_review_reference'])
 need(terminal['kind']=='CASE0_PARTIAL_SCENARIO_CUT_TERMINAL_REVIEW_V1' and terminal['passed'] is True and terminal['numerical_recheck_passed'] is True and terminal['references']['result']==refs['q_result'],'Independent original Q terminal audit required')
 need(s['partial_scenario_cut_settlement_reference']==refs['settlement'] and s['partial_scenario_cut_result_reference']==refs['q_result'] and
  s['source_release_reference']==refs['source_manifest'] and s['active_config_reference']==refs['config'] and not s['active_job_id'] and
  not s['active_configuration'] and not s['evaluation_opened'] and s['completed']=={},'Original settled partial state differs')
 for name,value in [('compute',s['compute_seconds']),('configuration',s['configuration_spent_seconds'][key]),('master',s['configuration_master_spent_seconds'][key])]:
  need(review['opening_'+name+'_seconds']==value==sett['closing_'+name+'_seconds'],'Opening cumulative budget differs from actual settled charge')
 seconds,native=r['budget_seconds'],r['native_seconds'];need(all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in (seconds,native)) and
  native<=8 and native<seconds<=14,'Inner native/total limits differ')
 from evcs_experiments.config import get_paths
 from evcs_experiments.case0_isolation import verify_current_cgroup
 from evcs_experiments.case0_campaign import verify_sources
 paths=get_paths();verify_current_cgroup();verify_sources(read(refs['config']));verify_reference(refs['source_manifest'])
 need(Path(refs['source_manifest']['path'])==paths.project_root/'RELEASE_MANIFEST.json' and paths.campaign_root.name=='ieee33-v54-q1-case0-expanded','Original expanded runtime required')
 jobid=os.environ.get('EVCS_JOB_ID');jd=Path(os.environ.get('EVCS_JOB_DIRECTORY','')).resolve();need(jobid and jd==paths.campaign_root/'jobs'/jobid,'Owned job required')
 manifest=json.loads((jd/'manifest.json').read_text());verify_reference(review['guard_reference'])
 reserve=manifest['timeout_seconds']+manifest['grace_seconds']
 need(manifest['argv']==[str(paths.project_root/'.venv/bin/python'),review['guard_reference']['path']] and manifest['threads']==1 and
  reserve<=min(18,1800-s['configuration_spent_seconds'][key],1200-s['configuration_master_spent_seconds'][key],21600-s['compute_seconds']) and seconds<manifest['timeout_seconds'],'Entire stage2 parent budget differs')
 helper=load(refs['partial_helper'],'evcs_experiments.case0_partial_cut');checker=load(refs['native_auditor'],'case0_partial_native_auditor')
 out=Path(r['output_directory']).resolve();need(out.is_relative_to(paths.campaign_root/'operations') and out!=paths.campaign_root/'operations','Fresh operation output required');out.mkdir(parents=True,exist_ok=False)
 try:return run_numeric(refs,out,started+seconds,native,helper,checker)
 except BaseException as e:atomic_json(out/'failure.json',dict(type=type(e).__name__,error=str(e),new_Q_solve_count=0,planning_acceptance=False));raise
 finally:atomic_json(out/'accounting.json',dict(actual_inner_seconds=time.monotonic()-started,scope='Entire actual jobctl parent charged once master/config/global; inner time is nested, not added twice.'))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--expected-sha256',required=True);a=p.parse_args();print(json.dumps(run_request(a.request,a.expected_sha256)))
