"""Source-bound, integer-domain SAA affine bound only; no recourse solver calls."""
import argparse,hashlib,importlib.util,json,math,os,sys,time
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz,save_npz
from evcs_experiments import campaign_batch as batch
from evcs_experiments.candidate_dual_adapter import first_digest
from evcs_experiments.relaxation_bound import coefficient_sha256
from evcs_experiments.native_gurobi import solve_milp
from evcs_experiments.native_gurobi import _check_vector
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.runtime.jobctl import atomic_json


def need(v,m):
 if not bool(v):raise ValueError(m)
def read(ref):batch.verify_reference(ref);return json.loads(Path(ref['path']).read_text())
def load(ref,name):
 batch.verify_reference(ref)
 if name in sys.modules:
  need(batch.file_reference(sys.modules[name].__file__)==ref,'Existing module source differs');return sys.modules[name]
 s=importlib.util.spec_from_file_location(name,ref['path']);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def tick(deadline):
 if not math.isfinite(deadline) or time.monotonic()>=deadline:raise TimeoutError('SAA bound-only parent exhausted')
def jsonable(v):
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,dict):return {k:jsonable(x) for k,x in v.items()}
 if isinstance(v,(tuple,list)):return [jsonable(x) for x in v]
 return v

def load_first(reference,expected_sha,plan,fpre):
 r=read(reference);need(r['kind']=='first_stage','Original first-stage record required')
 refs={Path(x['path']).name:x for x in r['artifacts']}
 need(len(refs)==len(r['artifacts'])==3 and set(refs)=={'identity.json','matrix.npz','arrays.npz'} and
      all(Path(x['path']).parent==Path(reference['path']).parent for x in refs.values()),'Original first-stage artifacts differ')
 for ref in refs.values():batch.verify_reference(ref)
 meta=read(refs['identity.json']);A=load_npz(refs['matrix.npz']['path']).tocsr()
 with np.load(refs['arrays.npz']['path'],allow_pickle=False) as z:v={k:z[k].copy() for k in z.files}
 need(set(v)=={'c','row_lower','row_upper','lower_bounds','upper_bounds','integrality'},'Original first-stage array schema differs')
 need(first_digest(meta,A,v)==meta['matrix_sha256']==expected_sha and meta['task_sha256']==plan['task_sha256'] and meta['config_hash']==plan['config_hash'],'Original first-stage semantic identity differs')
 model=dict(matrix=A,**v,objective_constant_usd=0.);anchor=np.r_[plan['x'],np.asarray(plan['r']).ravel()]
 cert=_check_vector(v['c'],v['integrality'],v['lower_bounds'],v['upper_bounds'],A,v['row_lower'],v['row_upper'],anchor)
 need(cert['passed'],'Accepted anchor violates original first-stage domain')
 need(abs(float(v['c']@anchor)-fpre)<=1e-4+1e-7*max(1.,abs(fpre)),'Original first-stage cost differs')
 return model


def audit_affine_model(model,first,ids,cuts,constants):
 need(len(ids)>0 and set(cuts)==set(range(len(ids))),'Full SAA cut axis required')
 n=len(first['c']);c=np.asarray(first['c']).copy();offset=first['objective_constant_usd'];hashes=[]
 for i,sid in enumerate(ids):
  cut,g=cuts[i];body=dict(cut);sha=body.pop('cut_sha256')
  need(canonical_hash(body)==sha and cut['passed'] is True and cut['scenario_id']==sid and cut['scenario_constant_usd']==constants[i], 'Audited cut identity differs')
  need(cut['gradient_sha256']==hashlib.sha256(np.asarray(g,dtype='<f8').tobytes()).hexdigest() and np.asarray(g).shape==(n,),'Audited gradient differs')
  c+=np.asarray(g)/len(ids);offset+=cut['intercept_usd']/len(ids);hashes.append(sha)
 A,B=model['matrix'].tocsr(),first['matrix'].tocsr()
 need(A.shape==B.shape and np.array_equal(A.data,B.data) and np.array_equal(A.indices,B.indices) and np.array_equal(A.indptr,B.indptr),'Original first-stage constraints changed')
 for k in ('integrality','lower_bounds','upper_bounds','row_lower','row_upper'):need(np.array_equal(model[k],first[k]),'Original first-stage domain changed')
 need(np.array_equal(model['c'],c) and model['objective_constant_usd']==offset and np.isfinite(c).all() and math.isfinite(offset),'Uniform affine objective/constants differ')
 r=dict(kind='INDEPENDENT_SAA_AFFINE_MODEL_AUDIT_V1',passed=True,coefficient_sha256=coefficient_sha256(model),
  original_coefficient_sha256=coefficient_sha256(first),cut_sha256s=hashes,support_ids=list(ids),risk_certificate_kind='fixed_empirical',integer_domain_retained=True,solver_calls=0,scientific_acceptance=False)
 return dict(r,audit_sha256=canonical_hash(r))


def run_numeric(request,out,deadline,adapter,combiner,native_auditor):
 refs=request['references'];a=read(refs['candidate_audit']);q=read(refs['fixed_q_request']);env=read(refs['template_envelope'])
 ctx=adapter.SAASavedDualContext(candidate_audit_reference=refs['candidate_audit'],request_reference=refs['fixed_q_request'],
  checkpoint_reference=refs['fixed_q_checkpoint'],template_envelope_reference=refs['template_envelope'],expected_first_stage_sha256=env['first_stage_sha256'],deadline=deadline)
 first=load_first(refs['first_stage'],env['first_stage_sha256'],q['plan'],a['fixed_q_audit']['first_stage_cost_usd'])
 cuts={};evidence=[]
 for i in range(len(ctx.ids)):
  tick(deadline);cut,g,e=ctx.derive(i);cuts[i]=(cut,g);evidence.append(e)
  atomic_json(out/f'cut_{i:03d}.json',jsonable(cut));np.save(out/f'gradient_{i:03d}.npy',g,allow_pickle=False)
 atomic_json(out/'cut_audits.json',jsonable(evidence))
 model,construction=combiner.build_saa_dual_bound(first,ctx.ids,cuts,constants_usd=q['constants_usd'])
 audit=audit_affine_model(model,first,ctx.ids,cuts,q['constants_usd'])
 atomic_json(out/'construction.json',jsonable(construction));atomic_json(out/'model_audit.json',jsonable(audit));save_npz(out/'matrix.npz',model['matrix'])
 np.savez_compressed(out/'arrays.npz',**{k:v for k,v in model.items() if k!='matrix'})
 grant=min(request['solver_seconds'],deadline-time.monotonic()-request['post_master_reserve_seconds']);need(grant>0,'Preparation exhausted bounded native grant')
 atomic_json(out/'native_started.json',dict(grant_seconds=grant,coefficient_sha256=audit['coefficient_sha256']))
 z=solve_milp(model['c'],model['integrality'],model['lower_bounds'],model['upper_bounds'],model['matrix'],model['row_lower'],model['row_upper'],
  options=dict(time_limit=grant,mip_rel_gap=.005,threads=1,solver_feasibility_tolerance=1e-9),log_path=out/'native.log')
 atomic_json(out/'native_record.json',jsonable(z.native_record))
 if z.x is not None:np.save(out/'auxiliary_vector.npy',z.x,allow_pickle=False)
 check=native_auditor.audit_saa_affine_native_bound(model,audit,z.native_record,native_status=int(z.status),native_bound=z.mip_dual_bound,incumbent=z.x,
  expected_coefficient_sha256=audit['coefficient_sha256'],fixed_upper_bound=a['upper_bound_usd'],deadline=deadline)
 atomic_json(out/'native_audit.json',jsonable(check));lower=max(a['lower_bound_usd'],check['lower_bound_usd']);upper=a['upper_bound_usd']
 result=dict(kind='S05_SAA_SAVED_DUAL_BOUND_RESULT_V1',status='AUDITED_AUXILIARY_BOUND_PENDING_PARENT_SETTLEMENT',source_binding=refs,
  lower_bound_usd=lower,previous_lower_bound_usd=a['lower_bound_usd'],new_native_lower_bound_usd=check['lower_bound_usd'],upper_bound_usd=upper,
  gap=max(0.,upper-lower)/max(1.,abs(upper)),original_plan_sha256=q['plan_sha256'],original_native_status=a['native_status'],auxiliary_native_status=check['native_status'],
  original_saa_status_not_relabelled=True,full_support_count=len(ctx.ids),new_Q_solve_count=0,auxiliary_master_count=1,planning_acceptance=False,
  new_plan_accepted=False,scientific_spec_pass=False,formal_OOS=False,requires_external_parent_settlement=True)
 tick(deadline);atomic_json(out/'result.json',jsonable(result));return result


def accepted_original(refs):
 join=read(refs['final_budget_join']);review=read(refs['final_acceptance_review']);a=read(refs['candidate_audit']);final=read(refs['final_record'])
 need(review['passed'] is True and review['kind']=='SAA_FINAL_SEAL_TERMINAL_AND_BUDGET_INDEPENDENT_REVIEW_V1' and
  review['settlement_reference']==refs['settlement'] and review['final_solver_record_reference']==refs['final_record'] and review['candidate_audit_reference']==refs['candidate_audit'],'Original independently accepted final record required')
 need(join['passed'] is True and join['candidate_audit_reference']==refs['candidate_audit'] and join['final_solver_record_reference']==refs['final_record'] and
  join['settlement_reference']==refs['settlement'] and join['full_support_verified'] is True,'Original numerical/final budget join differs')
 need(all(review['scientific_result'][k]==a[k] for k in ('lower_bound_usd','upper_bound_usd','native_status')),'Independent original numerical values differ')
 need(join['lower_bound_usd']==a['lower_bound_usd'] and join['upper_bound_usd']==a['upper_bound_usd'] and join['native_status']==a['native_status'],'Accepted original SAA numerical values differ')
 return join


def reviewed_admission(adm,request_reference):
 review=read(adm['independent_review_reference'])
 need(adm['kind']=='SAA_SAVED_DUAL_BOUND_ADMISSION_V1' and review['kind']=='SAA_SAVED_DUAL_BOUND_PRODUCTION_REVIEW_V1' and review['passed'] is True and
  review['admission_payload_sha256']==canonical_hash({k:v for k,v in adm.items() if k!='independent_review_reference'}) and
  review['driver_reference']==adm['driver_reference'] and review['request_reference']==adm['request_reference']==request_reference,'Bound admission independent source review differs')
 return review


def continuation_budget(request,join,last):
 """One independently reviewed pre-native failure retains all original cost."""
 if 'failed_parent_continuation' not in request:
  need(last==request['references']['settlement'] and join['effective_spent_seconds']==request['opening_effective_seconds'],'Original acceptance budget differs')
  return
 card=read(request['failed_parent_continuation'])
 need(card['kind']=='SAA_BOUND_PRE_NATIVE_FAILURE_CONTINUATION_REVIEW_V1' and card['passed'] is True and card['pre_native_failure'] is True,'Failed parent continuation review required')
 settlement=read(card['settlement_reference']);reservation=read(settlement['reservation_reference'])
 terminal=read(settlement['terminal_job_result_reference']);old=read(card['previous_request_reference'])
 adm=read(reservation['admission_reference']);failure=read(card['failure_reference'])
 need(card['settlement_reference']==last and reservation['previous_settlement_reference']==request['references']['settlement'] and
      reservation['anchor_reference']==request['references']['ledger_anchor'] and adm['request_reference']==card['previous_request_reference'],'Failed parent continuation history differs')
 need(terminal['complete'] is True and terminal['state']=='FAILED' and settlement['new_queue_charge_seconds']==0 and
      settlement['effective_spent_seconds']==request['opening_effective_seconds'] and
      reservation['effective_spent_seconds']==join['effective_spent_seconds'] and
      settlement['effective_spent_seconds']==join['effective_spent_seconds']+settlement['actual_parent_wall_seconds'], 'Failed parent cost reset or incomplete terminal')
 for key in ('candidate_audit','fixed_q_request','fixed_q_checkpoint','template_envelope','first_stage','final_budget_join','final_acceptance_review','final_record','settlement','ledger_anchor'):
  need(old['references'][key]==request['references'][key],'Failed parent original evidence changed: '+key)
 need(old['kind']==request['kind'] and old['domain']==request['domain'] and
      failure['type']=='ValueError' and failure['error']=='Original full Q checkpoint differs' and
      Path(card['failure_reference']['path'])==Path(old['output_directory'])/'failure.json','Different failure scope')
 need(not any((Path(old['output_directory'])/name).exists() for name in ('native_started.json','native_record.json','result.json','cut_000.json')), 'Pre-native failure scope changed')


def main():
 began=time.monotonic();p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--expected-sha256',required=True);p.add_argument('--admission',required=True);p.add_argument('--admission-sha256',required=True);args=p.parse_args()
 rr=batch.file_reference(args.request);ar=batch.file_reference(args.admission);need(rr['sha256']==args.expected_sha256 and ar['sha256']==args.admission_sha256,'Changed bound request/admission')
 request=read(rr);adm=read(ar);refs=request['references'];need(request['kind']=='S05_SAA_SAVED_DUAL_BOUND_ONLY_V1' and request['domain']=='formal' and adm['request_reference']==rr,'Exact formal bound-only request required')
 need(batch.file_reference(__file__)==adm['driver_reference'],'Actual driver source differs')
 reviewed_admission(adm,rr)
 for ref in request['source_references']:batch.verify_reference(ref)
 need(adm['driver_reference'] in request['source_references'],'Missing actual driver source')
 for name in ('campaign_batch','candidate_dual_adapter','relaxation_bound','native_gurobi','random_streams','runtime.jobctl'):
  need(batch.file_reference(sys.modules['evcs_experiments.'+name].__file__) in request['source_references'],'Missing actual bound kernel source: '+name)
 for key in ('ledger_source','adapter_source','combiner_source','native_auditor_source'):
  need(refs[key] in request['source_references'],'Missing reviewed overlay source: '+key)
 ledger=load(refs['ledger_source'],'evcs_experiments.saa_recovery_ledger');ledger.SOURCE_UPGRADE_REFERENCE=refs['ledger_source_upgrade']
 anchor,extra,last,pending,_=ledger._chain(refs['ledger_anchor']);need(pending is None and last==adm['expected_last_settlement_reference'],'Unsettled/stale original ledger head')
 opening=read(last)['closing_queue_snapshot_reference'];qsec=read(last)['closing_queue_seconds'];need(qsec+extra==request['opening_effective_seconds']==adm['opening_effective_seconds'],'Original effective budget differs')
 join=accepted_original(refs);continuation_budget(request,join,last)
 need(0<request['solver_seconds']<=600 and request['post_master_reserve_seconds']>=100 and 0<request['wall_seconds']<1500,'Bounded scientific resources differ')
 jobid=os.environ.get('EVCS_JOB_ID');job=Path(os.environ.get('EVCS_JOB_DIRECTORY','')).resolve();need(jobid==adm['parent_job_id'] and str(job)==adm['parent_job_directory'],'Owned bound parent required')
 authority=batch.read_json(Path(anchor['campaign_root'])/'state.json');claim=authority['saa_bound_only_pending']
 need(authority['active_jobs']==[jobid] and claim['job_id']==jobid and claim['admission_reference']==ar and claim['request_reference']==rr and claim['expected_last_settlement_reference']==last,'Authoritative bound reservation must be registered before dispatch')
 manifest=batch.file_reference(job/'manifest.json');reservation=ledger.reserve_recovery_parent(refs['ledger_anchor'],parent_job_request_reference=claim['job_request_reference'],parent_manifest_reference=manifest,admission_reference=ar,opening_queue_snapshot_reference=opening)
 out=Path(request['output_directory']);need(out.is_absolute() and out.is_relative_to(Path(anchor['campaign_root'])/'operations'),'Fresh operation output required');out.mkdir(parents=True,exist_ok=False)
 atomic_json(out/'reservation_reference.json',reservation)
 try:
  adapter=load(refs['adapter_source'],'evcs_experiments.saa_saved_dual_adapter');combiner=load(refs['combiner_source'],'evcs_experiments.saa_dual_bound');checker=load(refs['native_auditor_source'],'saa_affine_native_auditor')
  for name,module in tuple(sys.modules.items()):
   if name.startswith('evcs_experiments.') and getattr(module,'__file__',None) and name!='evcs_experiments.saa_recovery_ledger':
    need(batch.file_reference(module.__file__) in request['source_references'],'Missing actual loaded package source: '+name)
  run_numeric(request,out,began+request['wall_seconds'],adapter,combiner,checker)
 except BaseException as e:atomic_json(out/'failure.json',dict(type=type(e).__name__,error=str(e),old_bounds_retained=True));raise
 finally:atomic_json(out/'accounting.json',dict(actual_inner_seconds=time.monotonic()-began,reservation_reference=reservation,new_queue_charge_seconds=0,
  scope='Entire actual terminal jobctl parent added once as external E; inner/source/derive/native/audit times nested. External settlement required, no queue receipt fabricated.'))

if __name__=='__main__':main()
