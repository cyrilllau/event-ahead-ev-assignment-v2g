"""One new original-scenario dual cut at a saved auxiliary incumbent.

Explicit partial evaluation: no new upper bound, full-candidate or OOS claim.
The unchanged original extensive arrays define the selected LP and its coupling.
"""
import argparse,gc,hashlib,importlib.util,json,math,os,time,warnings
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz,save_npz
from scipy.optimize import linprog,OptimizeWarning
from evcs_experiments.campaign_batch import file_reference,verify_reference
from evcs_experiments.dual_cut_bound import build_dual_cut
from evcs_experiments.probability import ProbabilitySet
from evcs_experiments.probability_check import check_probability_solution
from evcs_experiments.scenario_features import FEATURE_NAMES
from evcs_experiments.relaxation_bound import coefficient_sha256
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.case0_planning import _json
from evcs_experiments.runtime.jobctl import atomic_json

HASH_SHA='7333d02ff1668ad04d75bc73f8b139570401093679a10a6160fd411a0c2c3582'

def need(ok,msg):
 if not bool(ok):raise ValueError(msg)
def read(ref):
 verify_reference(ref);return json.loads(Path(ref['path']).read_text())
def load_module(ref,name,expected=None):
 verify_reference(ref);need(expected is None or ref['sha256']==expected,'Bound helper source differs')
 spec=importlib.util.spec_from_file_location(name,ref['path']);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def left(deadline):
 x=deadline-time.monotonic()
 if x<=0:raise TimeoutError('Partial scenario deadline exhausted')
 return x
def maximum(x):return float(np.max(x,initial=0.))
def close(a,b,label):
 need(math.isfinite(a) and math.isfinite(b) and abs(a-b)<=1e-4+1e-7*max(1.,abs(a),abs(b)),label+' differs')
def model_files(matrix_ref,arrays_ref):
 verify_reference(matrix_ref);verify_reference(arrays_ref);A=load_npz(matrix_ref['path']).tocsr()
 with np.load(arrays_ref['path'],allow_pickle=False) as f:d={k:f[k].copy() for k in f.files}
 d['matrix']=A;d['objective_constant_usd']=float(d['objective_constant_usd']);return d

def original_probability(A,arrays,header):
 ids=tuple(header['scenario_ids']);sl=header['variable_slices'];theta=sl['theta0'][0];features=[]
 for sid in ids:
  b=[b for b in header['block_map'] if b['scope']==sid and b['kind']=='risk_epigraph'];need(len(b)==1 and b[0]['row_stop']==b[0]['row_start']+1,'Original risk block differs')
  tail=A[b[0]['row_start'],theta:].toarray().ravel();k=len(FEATURE_NAMES)
  need(tail.shape==(1+2*k,) and tail[0]==-1 and np.array_equal(tail[1:1+k],-tail[1+k:]),'Original ambiguity tail differs');features.append(tail[1:1+k])
 P=ProbabilitySet(ids,FEATURE_NAMES,np.asarray(features),-arrays['c'][slice(*sl['alpha'])],arrays['c'][slice(*sl['beta'])])
 need(P.ambiguity_sha256==header['ambiguity_sha256'] and arrays['c'][theta]==1.,'Original P identity differs');return P

def extract_scenario(A,arrays,header,i,P):
 """Same signed original A/B extraction as frozen saved-dual kernel; one index."""
 ids=header['scenario_ids'];need(type(i) is int and 0<=i<len(ids),'Selected support index differs')
 sid=ids[i];sl=header['variable_slices'];H=sl['x'][1];nf=sl['first_stage'][1];a,b=sl['recourses'][sid];theta=sl['theta0'][0];th=header['recourse_template_sha256s'][i]
 need(sl['x']==[0,H] and sl['r']==[H,nf] and sl['first_stage']==[0,nf] and nf<=a<b<=theta,'Original scenario axes differ')
 index={};cursor=0
 for r in header['block_map']:
  need(r['row_start']==cursor and r['row_stop']>=cursor and (r['scope'],r['kind']) not in index,'Original row blocks differ');cursor=r['row_stop'];index[(r['scope'],r['kind'])]=r
 need(cursor==A.shape[0] and len(index)==1+3*len(ids),'Original full block inventory differs')
 t=dict(scenario_id=sid,template_sha256=th,lower_bounds=arrays['lower_bounds'][a:b].copy(),upper_bounds=arrays['upper_bounds'][a:b].copy())
 for suffix in ('ub','eq'):
  row=index[(sid,'recourse_'+suffix)];need(row['source_sha256']==th,'Original scenario block provenance differs')
  ra,rb=row['row_start'],row['row_stop'];block=A[ra:rb];own=block[:,a:b].tocsr();coupling=block[:,:nf].tocsr();rl,ru=arrays['row_lower'][ra:rb],arrays['row_upper'][ra:rb]
  need(own.nnz+coupling.nnz==block.nnz,'Foreign recourse column');need(np.all(np.isneginf(rl)) if suffix=='ub' else np.array_equal(rl,ru),'Original row sense differs')
  t.update({f'A_{suffix}':own,f'Bx_{suffix}':-coupling[:,:H],f'Br_{suffix}':-coupling[:,H:],f'b_{suffix}':ru.copy()})
 risk=index[(sid,'risk_epigraph')];rr=risk['row_start'];row=A[rr]
 need(risk['row_stop']==rr+1 and risk['source_sha256']==th and row[:,:a].nnz==0 and row[:,b:theta].nnz==0 and np.array_equal(row[:,theta:].toarray().ravel(),np.r_[-1.,P.features[i],-P.features[i]]) and np.isneginf(arrays['row_lower'][rr]) and arrays['row_upper'][rr]==0. and header['scenario_cost_constants_usd'][i]==0.,'Original selected risk objective differs')
 t['c']=row[:,a:b].toarray().ravel();return t

def audit_selected_lp(template,z,primal,duals,raw):
 """Check the actual original fixed RHS and derive a globally valid minorant."""
 t=template;H=t['Bx_ub'].shape[1]
 need(raw['solver_status']==0 and raw['success'] is True,'Selected original LP lacks optimal primal/dual')
 bub=t['b_ub']+t['Bx_ub']@z[:H]+t['Br_ub']@z[H:];beq=t['b_eq']+t['Bx_eq']@z[:H]+t['Br_eq']@z[H:]
 v=np.asarray(primal,float);need(v.shape==t['c'].shape and np.isfinite(v).all(),'Invalid selected primal')
 error=max(maximum(t['A_ub']@v-bub),maximum(abs(t['A_eq']@v-beq)),maximum(t['lower_bounds']-v),maximum(v-t['upper_bounds']))
 need(error<=1e-7,'Original selected primal violates rows/bounds');q=float(t['c']@v);close(q,raw['objective_without_constant_usd'],'Selected original Q')
 cut,g=build_dual_cut(t,duals,scenario_constant_usd=0.,anchor_first_stage=z,anchor_q_usd=q);need(cut['passed'],'Selected original global dual cut failed')
 y,e,lo,hi=(np.asarray(duals[k],float) for k in ('inequality','equality','lower','upper'));fl,fu=np.isfinite(t['lower_bounds']),np.isfinite(t['upper_bounds'])
 intercept=float(t['b_ub']@y+t['b_eq']@e+t['lower_bounds'][fl]@lo[fl]+t['upper_bounds'][fu]@hi[fu])
 gradient=np.r_[t['Bx_ub'].T@y+t['Bx_eq'].T@e,t['Br_ub'].T@y+t['Br_eq'].T@e]
 need(np.array_equal(g,gradient),'Independent coupling contraction differs');close(intercept,cut['intercept_usd'],'Independent base RHS intercept')
 return cut,g,dict(kind='CASE0_ONE_SCENARIO_ORIGINAL_LP_AUDIT_V1',passed=True,primal_max_error=error,selected_q_usd=q,global_minorant_verified=True,full_candidate_verified=False,new_upper_bound_verified=False,formal_OOS=False)

def run_numeric(refs,out,deadline,lp_seconds,helper):
 header=read(refs['model']);original=model_files(refs['matrix'],refs['model_arrays']);checker=load_module(refs['model_hash_checker'],'case0_partial_original_hash',HASH_SHA)
 checker.check_model_hash(original['matrix'],{k:v for k,v in original.items() if k!='matrix'},header);left(deadline)
 base=model_files(refs['base_matrix'],refs['base_arrays']);audit=read(refs['base_audit']);prior=read(refs['base_result']);construction=read(refs['base_construction']);old=read(refs['old_ub']);prob=read(refs['old_probability'])
 need(prior['status']=='AUDITED_TWO_ANCHOR_AUXILIARY_BOUND' and prior['model_sha256']==old['model_sha256']==header['model_sha256'] and prior['plan_sha256']==old['plan_sha256'] and prior['source_binding']['ub_result']==refs['old_ub'] and prior['auxiliary_model_sha256']==coefficient_sha256(base)==audit['coefficient_sha256']==construction['master_coefficient_sha256'],'Accepted original/base identity differs')
 need(prior['auxiliary_vector_reference']==refs['base_vector'] and old['source_binding']['model']==refs['model'],'Old bound/vector source differs');verify_reference(refs['base_vector']);v=np.load(refs['base_vector']['path'],allow_pickle=False)
 need(prior['native_status']==0 and prior['auxiliary_mip_gap']==0.,'Skip logic requires an optimal original auxiliary incumbent')
 native_audit=read(prior['native_bound_audit_reference']);need(native_audit['passed'] is True and native_audit['native_status']=='OPTIMAL' and native_audit['coefficient_sha256']==coefficient_sha256(base),'Optimal base native bound audit differs')
 close(prior['new_native_lower_bound_usd'],prior['auxiliary_objective_usd'],'Saturated prior auxiliary bound')
 close(float(base['c']@v+base['objective_constant_usd']),prior['auxiliary_objective_usd'],'Saved optimal auxiliary objective')
 P=original_probability(original['matrix'],original,header);need(check_probability_solution(P,np.asarray(old['q_usd']),prob,q_support_ids=tuple(header['scenario_ids'])).passed,'Old fixed source selection weights are not original feasible P')
 cut_manifest=read(refs['base_cuts']);sets=[]
 for group in cut_manifest['groups']:
  cuts={}
  for i,pair in enumerate(group):
   record=read(pair['cut']);verify_reference(pair['gradient']);cuts[i]=(record['cut'],np.load(pair['gradient']['path'],allow_pickle=False))
  sets.append(cuts)
 need(len(sets)==2 and [c[0]['cut_sha256'] for group in sets for c in group.values()]==audit['cut_sha256s'],'Accepted dual-cut axis differs')
 atomic_json(out/'probability_set.json',P.as_dict())
 selection=helper.select_scenario(base,audit,P,sets,prob,v,source_references=list(refs.values()),deadline=deadline)
 atomic_json(out/'selection.json',_json(selection));np.save(out/'anchor_auxiliary_vector.npy',v,allow_pickle=False)
 # Selection is persisted BEFORE the only new recourse solve.
 i=selection['selected_index'];z=v[:header['variable_slices']['first_stage'][1]].copy();t=extract_scenario(original['matrix'],original,header,i,P)
 matrices=('A_ub','A_eq','Bx_ub','Bx_eq','Br_ub','Br_eq')
 for k in matrices:save_npz(out/(k+'.npz'),t[k])
 values={k:t[k] for k in ('b_ub','b_eq','c','lower_bounds','upper_bounds')};H=t['Bx_ub'].shape[1]
 values.update(fixed_b_ub=t['b_ub']+t['Bx_ub']@z[:H]+t['Br_ub']@z[H:],fixed_b_eq=t['b_eq']+t['Bx_eq']@z[:H]+t['Br_eq']@z[H:]);np.savez_compressed(out/'lp_arrays.npz',**values)
 atomic_json(out/'template_identity.json',dict(scenario_id=t['scenario_id'],template_sha256=t['template_sha256'],index=i,model_sha256=header['model_sha256'],source_binding=refs))
 del original;gc.collect();grant=min(lp_seconds,left(deadline)-8.);need(grant>0,'No reserved LP/audit budget')
 opts=dict(time_limit=float(grant),threads=1,primal_feasibility_tolerance=1e-8,dual_feasibility_tolerance=1e-8,ipm_optimality_tolerance=1e-8)
 atomic_json(out/'lp_started.json',dict(selected_index=i,selection_reference=file_reference(out/'selection.json'),options=opts,call_limit=1))
 began=time.monotonic()
 with warnings.catch_warnings():
  warnings.filterwarnings('ignore',category=OptimizeWarning,message='Unrecognized options detected.*')
  solved=linprog(t['c'],A_ub=t['A_ub'],b_ub=values['fixed_b_ub'],A_eq=t['A_eq'],b_eq=values['fixed_b_eq'],bounds=np.column_stack((t['lower_bounds'],t['upper_bounds'])),method='highs',options=opts)
 raw=dict(solver='SciPy linprog/HiGHS original selected arrays',solver_status=int(solved.status),success=bool(solved.success),message=str(solved.message),objective_without_constant_usd=None if solved.fun is None else float(solved.fun),iterations=int(solved.nit or 0),options=opts,wall_seconds=time.monotonic()-began)
 atomic_json(out/'lp_raw.json',raw);need(solved.x is not None and solved.status==0,'Selected LP incomplete; no retry')
 d={k:np.asarray(getattr(solved,n).marginals,float) for k,n in zip(('inequality','equality','lower','upper'),('ineqlin','eqlin','lower','upper'))};np.savez_compressed(out/'lp_witness.npz',primal=solved.x,**d)
 cut,g,check=audit_selected_lp(t,z,solved.x,d,raw);atomic_json(out/'selected_lp_audit.json',_json(check));atomic_json(out/'new_cut.json',_json(cut));np.save(out/'new_gradient.npy',g,allow_pickle=False)
 atomic_json(out/'auxiliary_preparation_started.json',dict(started_at_utc=datetime.now(timezone.utc).isoformat(),scope='Append/audit/save began; missing final accounting must never be treated as zero master expense'))
 prep_started=time.monotonic()
 try:
  augmented,partial_audit,exclusion=helper.append_partial_cut(base,audit,P,selection,cut,g,full_incumbent=v,source_references=[file_reference(out/'selected_lp_audit.json'),file_reference(out/'new_cut.json'),file_reference(out/'new_gradient.npy')],deadline=deadline)
  atomic_json(out/'partial_model_audit.json',_json(partial_audit));atomic_json(out/'incumbent_exclusion.json',_json(exclusion));save_npz(out/'augmented_matrix.npz',augmented['matrix']);np.savez_compressed(out/'augmented_arrays.npz',**{k:augmented[k] for k in ('c','integrality','lower_bounds','upper_bounds','row_lower','row_upper','objective_constant_usd')})
 finally:
  atomic_json(out/'auxiliary_preparation_accounting.json',dict(actual_seconds=time.monotonic()-prep_started,scope='Measured new partial-model append/audit/save; once master, nestedparentconfig/global'))
 need(time.monotonic()-prep_started<=2.,'Partial auxiliary preparation exceeded reserved master portion')
 result=dict(kind='CASE0_PARTIAL_SCENARIO_CUT_RESULT_V1',passed=True,source_binding=refs,model_sha256=header['model_sha256'],old_plan_sha256=old['plan_sha256'],selected_index=i,scenario_id=t['scenario_id'],selected_q_usd=check['selected_q_usd'],new_Q_solve_count=1,master_solve_count=0,upper_bound_usd=old['upper_bound_usd'],lower_bound_usd=prior['lower_bound_usd'],gap=prior['gap'],new_upper_bound_verified=False,full_candidate_verified=False,planning_acceptance=False,formal_OOS=False,selection_reference=file_reference(out/'selection.json'),cut_reference=file_reference(out/'new_cut.json'),partial_model_audit_reference=file_reference(out/'partial_model_audit.json'),incumbent_exclusion_reference=file_reference(out/'incumbent_exclusion.json'))
 left(deadline);atomic_json(out/'result.json',_json(result));return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--expected-sha256',required=True);a=p.parse_args();started=time.monotonic();rr=file_reference(a.request);need(rr['sha256']==a.expected_sha256,'Request changed');r=read(rr)
 need(set(r)=={'kind','references','output_directory','budget_seconds','lp_seconds','review_reference'} and r['kind']=='CASE0_PARTIAL_SCENARIO_CUT_REQUEST_V1','Partial request schema differs')
 review=read(r['review_reference']);need(review['kind']=='CASE0_PARTIAL_SCENARIO_CUT_PRODUCTION_REVIEW_V1' and review['passed'] is True and review['request_without_review']=={k:v for k,v in r.items() if k!='review_reference'} and review['auditor_sha256']==file_reference(__file__)['sha256'],'Exact production review required')
 refs=r['references'];need(set(refs)=={'model','matrix','model_arrays','model_hash_checker','base_result','base_matrix','base_arrays','base_audit','base_construction','base_vector','base_cuts','old_ub','old_probability','source_manifest','config','state','settlement','partial_helper'},'Partial source inventory differs')
 s=read(refs['state']);sett=read(refs['settlement']);key='size0/EA_DRO'
 need(s['original_extensive_closure_settlement_reference']==refs['settlement'] and s['source_release_reference']==refs['source_manifest'] and s['active_config_reference']==refs['config'] and s['two_anchor_bound_result_reference']==refs['base_result'] and s['saved_candidate_ub_reference']==refs['old_ub'] and s['compute_seconds']==sett['closing_compute_seconds'] and s['configuration_spent_seconds'][key]==sett['closing_configuration_seconds'] and s['configuration_master_spent_seconds'][key]==sett['closing_master_seconds'] and not s['active_job_id'] and not s['active_configuration'] and not s['evaluation_opened'] and s['completed']=={},'Settled partial source/budget snapshot differs')
 seconds,lp=r['budget_seconds'],r['lp_seconds'];need(all(type(x) in (float,int) and math.isfinite(x) and x>0 for x in (seconds,lp)) and lp<=35 and lp<seconds<=50 and seconds<1800-s['configuration_spent_seconds'][key],'Partial budget exceeded')
 from evcs_experiments.config import get_paths
 from evcs_experiments.case0_isolation import verify_current_cgroup
 from evcs_experiments.case0_campaign import verify_sources
 paths=get_paths();verify_current_cgroup();verify_sources(read(refs['config']));verify_reference(refs['source_manifest']);need(Path(refs['source_manifest']['path'])==paths.project_root/'RELEASE_MANIFEST.json','Producer release differs')
 helper=load_module(refs['partial_helper'],'evcs_experiments.case0_partial_cut');out=Path(r['output_directory']).resolve();need(paths.campaign_root.name=='ieee33-v54-q1-case0-expanded' and out!=paths.campaign_root/'operations' and out.is_relative_to(paths.campaign_root/'operations'),'Fresh expanded output only')
 jobid=os.environ.get('EVCS_JOB_ID');jd=Path(os.environ.get('EVCS_JOB_DIRECTORY','')).resolve();need(jobid and jd==paths.campaign_root/'jobs'/jobid,'Exact ownedjob required');manifest=json.loads((jd/'manifest.json').read_text());verify_reference(review['guard_reference']);need(manifest['argv']==[str(paths.project_root/'.venv/bin/python'),review['guard_reference']['path']] and manifest['threads']==1 and manifest['timeout_seconds']+manifest['grace_seconds']<=min(55,1800-s['configuration_spent_seconds'][key],21600-s['compute_seconds']) and seconds<manifest['timeout_seconds'],'Owned partial parent envelope differs')
 out.mkdir(parents=True,exist_ok=False)
 try:run_numeric(refs,out,started+seconds,lp,helper)
 except BaseException as e:atomic_json(out/'failure.json',dict(type=type(e).__name__,error=str(e),new_upper_bound_verified=False));raise
 finally:atomic_json(out/'operation_accounting.json',dict(actual_seconds=time.monotonic()-started,master_seconds=(read(file_reference(out/'auxiliary_preparation_accounting.json'))['actual_seconds'] if (out/'auxiliary_preparation_accounting.json').exists() else 0.),scope='Entire selectedQ parent onceconfig/global; measured auxiliary append/audit/save portion oncemaster, otherQwork notmaster. Nested times notaddedagain.'))
if __name__=='__main__':main()
