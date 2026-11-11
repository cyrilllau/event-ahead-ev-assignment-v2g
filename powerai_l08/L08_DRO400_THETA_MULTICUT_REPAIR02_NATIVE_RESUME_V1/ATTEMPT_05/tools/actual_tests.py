from aggregate_core import *
from fractions import Fraction
import subprocess,inspect

def control_tests():
 from control_policy import (bound_state,route_after_bound_update,candidate_route,pending_candidate,request_handover,commit_phase,drive,json_safe,check_science,schema_adapter)
 folder=OUT/'tests/control';folder.mkdir(parents=True);checks={}
 checks['strict_upper_rechecks_gap']=route_after_bound_update(100,100.5,150,certification_done_for_audited_point=True)=='RESUME_SCIENCE_STRICT_GAP_NOT_REACHED'
 checks['post_native_bound_immediate']=bound_state(199,200)['audited_gap']['reached']
 checks['duplicate_cut_does_not_hide_new_point']=candidate_route(complete_cached=False,certified_cut_violates=False,oracle_tight=True,master_proof_complete=True)=='VALIDATE_AND_EVALUATE_FULL400'
 known={'start':dict(activity=100.,tightness=0.)}
 checks['start_only']=not pending_candidate('start',100.,known)
 checks['known_lower_rho']=pending_candidate('start',90.,known)
 checks['handover']=request_handover(root_completed=True,seconds_after_root=120,pending_evaluation_or_separation=True) and not request_handover(root_completed=False,seconds_after_root=1200,pending_evaluation_or_separation=True)
 checks['correct_science_source']=check_science(read(ARO/'CONFIG.json'),read(R36/'02_SOURCE_AND_CONFIG/INPUT_IDENTITY.json'),SCIENCE)
 checks['wrong_identity_rejected']=expect_failure(lambda:check_science({'identity':'wrong'}, {'identity':SCIENCE}, SCIENCE))
 adapter=read(OUT/'SID_SCHEMA_ADAPTER_CERTIFICATE.json')['records'][0]
 checks['legacy_schema_actual_source']=schema_adapter(read(ARO/'oracles/0000/CACHE_COMPLETE.json'),adapter['source_proof'],adapter['matrix_proof'])['passed']
 checks['missing_proof_rejected']=expect_failure(lambda:schema_adapter({'nx':60780},{},{}))
 checks['nonfinite_json']=json.dumps(json_safe({'inf':float('inf'),'nan':float('nan')}),allow_nan=False)=='{"inf": null, "nan": null}'
 for phase in ['CUT_PENDING_INSTALL','MASTER_SOLVED_PENDING_EVALUATION','FINAL_CERTIFY']:
  state=dict(phase=phase,cuts=[],x_eval=None,x_master=None,x_best=None)
  checks[phase]=commit_phase(folder/(phase+'.json'),state,write,rt.verify_ref)['phase']==phase
 events=[];drive(lambda:events.append('prepare'),lambda:events.append('test'),lambda:events.append('execute'));checks['same_entry_drive']=events==['prepare','test','execute']
 with (folder/'EXIT.stdout').open('xb') as so,(folder/'EXIT.stderr').open('xb') as se:
  child=subprocess.Popen([sys.executable,'-c','raise SystemExit(7)'],stdout=so,stderr=se);rc=child.wait()
 write(folder/'EXIT_WAIT.json',dict(actual_parent_wait=True,returncode=rc));checks['real_nonzero_wait']=rc==7
 checks['output_isolation']=expect_failure(lambda:write(CTX.source_root/'MUST_NOT_EXIST.json',{}))
 assert all(checks.values());write(folder/'RESULT.json',dict(passed=True,checks=checks))

def native_callback_test():
 import formal
 folder=OUT/'tests/native_callback';folder.mkdir(parents=True)
 m=gp.Model('TEST_CALLBACK_START');m.Params.OutputFlag=0;x=m.addMVar(3,vtype='B');r=m.addVar(lb=0);m.addConstr(x.sum()>=1);m.setObjective(x.sum()+r);x.Start=np.array([1.,0.,0.]);r.Start=0;m.update()
 key=hashlib.sha256(np.array([1.,0.,0.]).tobytes()).hexdigest();bounds=[]
 queue,rec=formal.native_master(m,x,r,folder/'CALL', {key:dict(activity=0.,tightness=0.)},0,lambda v,evidence:bounds.append(v))
 assert rec['status']==2 and rec['reason'] is None and rec['bound'] is not None
 unavailable=formal.attr(m,'Pi');assert unavailable is None
 m.dispose();write(folder/'RESULT.json',dict(passed=True,native=rec,root_completion='OBSERVED' if rec['root_completed'] else 'NOT_OBSERVED',unavailable_attribute_safe=True,bounds=bounds))

def worker_reuse_test():
 sids=[g[0] for g in read(OUT/'SCENARIO_EQUIVALENCE.json')['groups'][:2]];folder=OUT/'tests/workers';folder.mkdir(parents=True)
 for sid in sids:
  src=OUT/'tests/real_small/ITER_00'/f'{sid:04d}'/'RESULT.json';rec=read(src);target=OUT/'EVALUATION_CACHE'/e.task_id(SCIENCE,sid,np.zeros(60780),'RECOURSE')/'RESULT.json';write(target,rec,True)
 policy=read(ARO/'PATCH_POLICY.json');policy.update(frozen_pool=sids,run_id=CTX.run['run_id'],max_local_retries=2,legacy_task_ids=[])
 for batch in range(2):
  workers=e.Workers(OUT,dict(CTX.run,source=str(R41),execution_root=str(OUT),source_root=str(CTX.source_root),config=CTX.config),CTX.deadline,folder/f'SESSION_{batch}',oracle_cls=Oracle,policy=policy)
  try:
   records=workers.evaluate(np.zeros(60780),sids,SCIENCE,lambda n:None);assert {r['sid'] for r in records}==set(sids)
  finally:workers.close();workers.close()
 write(folder/'RESULT.json',dict(passed=True,batches=2,separate_sessions=True,idempotent_close=True,actual_worker_waits=True))

def expect_failure(fn):
 try:fn()
 except (ValueError,AssertionError):return True
 raise AssertionError('NEGATIVE_TEST_NOT_REJECTED')

def function_tests():
 folder=OUT/'tests/functions';folder.mkdir(parents=True);results={}
 q=np.array([0.,10.,100.]);p=np.array([.5,.25,.25]);results['DRO_not_max_or_mean']=len({float(p@q),float(max(q)),float(q.mean())})==3
 m,x,r=master();results['integer_domain_preserved']=m.NumBinVars==25770 and m.NumVars==60781;m.dispose()
 results['partial_400_rejected']=expect_failure(lambda:full_vector([dict(sid=0,qL=0)],'qL'))
 results['invalid_p_rejected']=expect_failure(lambda:aggregate(p,[],np.zeros(2),np.ones(2),False))
 W=sp.csr_matrix([[1.]]);T=sp.csr_matrix([[.3,.2]]);a=dict(sense=np.array(['<']),c=np.array([-2.]),finite_lo=np.array([0.]),finite_hi=np.array([3.]),xlo=np.zeros(2),xhi=np.ones(2),h=np.array([2.]))
 al,bb,pi=b.replay_cut(W,T,a,np.array([0.]));results['finite_bound_RC_term']=al<=-6 and al>-6.00001
 records=[];aa=[.1,-.3,.7];betas=[[1/3,.1],[-.4,1/7],[.2,-.3]]
 for sid in range(3):
  path=folder/f'CUT_{sid}.npz';save(path,alpha=aa[sid],beta=np.array(betas[sid]));records.append(dict(sid=sid,cut=str(path),global_cut_valid=True))
 alpha,beta,proof=aggregate(p,records,np.zeros(2),np.ones(2),True)
 for v in [(0,0),(0,1),(1,0),(1,1)]:
  exact=sum(Fraction(float(p[s]))*(Fraction(aa[s])+sum(Fraction(betas[s][j])*v[j] for j in range(2))) for s in range(3));stored=Fraction(alpha)+sum(Fraction(float(beta[j]))*v[j] for j in range(2));assert stored<=exact
 results['actual_aggregate_exact_fraction_box_check']=True
 results['equivalent_Q_keeps_probability_columns']=float(np.array([.25,.75])@np.array([7.,7.]))==7 and not np.array_equal(np.array([1.,2.]),np.array([2.,1.]))
 results['cached_cut_reseparates_current_rho']=(alpha+beta@np.ones(2)-0)!=(alpha+beta@np.ones(2)-10)
 results['nonfinite_native_bound_keeps_history']=merge_bound(12536.5,float('nan'))==12536.5 and merge_bound(12536.5,float('-inf'))==12536.5
 path=folder/'IMMUTABLE.json';write(path,dict(a=1),True);write(path,dict(a=1),True);results['duplicate_immutable_publication']=read(path)==dict(a=1)
 assert all(results.values());write(folder/'RESULT.json',dict(passed=True,checks=results))

def reference_regression():
 folder=OUT/'tests/original_P_reference';folder.mkdir(parents=True)
 root=R36/'evaluations/7cc27e720e13d3cbe46a1a25a9e704f9495bbcf045cf2df212cb702bd24fdf27';complete=read(root/'COMPLETE.json');z=np.load(root/'probability/CERTIFICATE.npz');assert sha(root/'probability/CERTIFICATE.npz')==complete['risk']['certificate_sha']
 q=np.zeros(400)
 for sid in range(400):
  src=root/f'{sid:04d}';v=read(src/'COMPLETE.json');ce=read(src/'CERTIFICATE.json');assert sha(src/'PRIMAL.npz')==ce['primal_sha'] and sha(src/'DUAL_AND_BOUNDS.npz')==ce['dual_sha'];assert v['physical_sha']==sha(orig.scene_path('TRAIN',sid)/'physical.npz') and v['recovery_sha']==sha(orig.scene_path('TRAIN',sid)/'recovery.npz');assert v['replay_sha']==sha(orig.verify_replay('TRAIN',sid));assert v['plan_sha']==complete['plan_hash'];q[sid]=ce['qU']
 assert np.array_equal(q,z['q'])
 ru=b.risk_upper(q,float(z['beta0']),z['plus'],z['minus']);F=float(first()[1]['cost']@plan_point(complete['plan']));UB=F+ru['risk_upper'];assert abs(UB-39088.41746327296)<1e-4
 write(folder/'RESULT.json',dict(passed=True,F=F,replayed_risk_upper=ru['risk_upper'],UB=UB,reference=39088.41746327296,source=ref(root/'COMPLETE.json'),new_LP_calls=0,all400_source_certificates_SHA=True,upper_level='numerically_audited_recourse_plus_directed_probability_dual'))

def real_small_test():
 folder=OUT/'tests/real_small';folder.mkdir(parents=True);eq=read(OUT/'SCENARIO_EQUIVALENCE.json');sids=[eq['groups'][0][0],eq['groups'][1][0]];plan=read(OUT/'INITIAL_PLAN.json')['plan'][:2];qs=orig.routes();permitted={q['route_id'] for q in plan};a=first()[1];P=[np.array([.25,.75]),np.array([.75,.25])]
 def restrict(m,x):
  by={h['e']:h for h in plan};names=first()[2]
  for i,n in enumerate(names):
   if n.startswith('route[') and qs[int(n[6:-1])]['route_id'] not in permitted:x[i].UB=0
   if n.startswith('start['):
    ev,k=map(int,n[6:-1].split(','))
    if ev not in by or k!=by[ev]['k']:x[i].UB=0
   if n.startswith('end['):
    ev,j=map(int,n[4:-1].split(','))
    if ev not in by or j!=by[ev]['r']+1:x[i].UB=0
  m.update()
 # Build each real recourse with the unchanged original constructor once,
 # prove the archived full-domain W/T and objective split against that build.
 for sid in sids:
  m=gp.Model('REAL_CONSTRUCTOR_SPLIT');m.Params.OutputFlag=0;fs=orig.add_first(m,qs,30,8,orig.full_bounds());orig.add_strengthening(m,fs);m.update();rc=orig.add_scene(m,fs,None,sid);m.setObjective(rc['objective']);m.update();o=Oracle(sid,folder/'VERIFY')
  assert m.ObjCon==0 and not np.any(np.asarray(m.getAttr('Obj'))[:60780]);assert all(v=='C' for v in m.getAttr('VType')[60780:]);AA=m.getA().tocsr();assert (AA[90768:,:60780]-o.T).nnz==0 and (AA[90768:,60780:]-o.W).nnz==0;assert np.array_equal(m.getAttr('Obj')[60780:],o.a['c']);m.dispose();o.close(discard=True)
 # The same common constructor has no sid-dependent branch for d, kappa or
 # VType: objective is base.parts@y, auxiliary addVars all default continuous.
 source=inspect.getsource(orig.add_recourse);assert 'objective=sum(parts)' in source and 'vtype=' not in source
 write(folder/'SPLIT_ASSUMPTIONS.json',dict(passed=True,actual_sids=sids,common_source=ref(Path(inspect.getsourcefile(orig.add_recourse))),d_zero=True,kappa_zero=True,all_recourse_continuous=True))
 m,x,rho=master('TEST_ONLY_DIRECT');restrict(m,x);objectives=[]
 for sid in sids:
  o=Oracle(sid,folder/'DIRECT_INPUT');y=m.addMVar(len(o.a['c']),lb=o.a['lo'],ub=o.a['hi']);m.addMConstr(sp.hstack([o.T,o.W]).tocsr(),gp.MVar.fromlist(x.tolist()+y.tolist()),o.a['sense'],o.a['h']);objectives.append(o.a['c']@y);o.close(discard=True)
 for p in P:m.addConstr(rho>=sum(float(p[i])*objectives[i] for i in range(2)))
 b.configure_oracle(m,folder/'DIRECT.log',min(180,CTX.prep-time.monotonic()-30));m.Params.IntFeasTol=1e-9;m.Params.MIPGap=0;m.Params.MIPGapAbs=1e-6;e.optimize(m,'TEST_REAL_DIRECT_SMALL',dict(sids=sids,test_probability_vertices=[p.tolist() for p in P]));assert m.Status==2;direct=float(m.ObjVal);m.dispose()
 env=gp.Env(empty=True);env.setParam('OutputFlag',0);env.start();oracles=[Oracle(s,folder/'DECOMPOSED',env=env) for s in sids];m,x,rho=master('TEST_ONLY_AGGREGATE');restrict(m,x);point=np.zeros(60780);history=[]
 try:
  for iteration in range(12):
   CTX.check(True);records=[]
   for i,o in enumerate(oracles):
    rec=o.solve_task(point,f'TEST_{iteration}_{o.sid}',folder/f'ITER_{iteration:02d}'/f'{o.sid:04d}',min(CTX.prep-20,CTX.deadline));records.append(dict(rec,sid=i))
   q=np.array([r['qU_audited'] for r in records]);p=max(P,key=lambda p:float(p@q));al,be,proof=aggregate(p,records,a['lo'],a['hi'],True);add_cut(m,x,rho,al,be,'test_aggregate_'+str(iteration));m.update();b.configure_oracle(m,folder/f'AGG_{iteration:02d}.log',min(90,CTX.prep-time.monotonic()-30));m.Params.MIPGap=0;m.Params.MIPGapAbs=1e-6;m.Params.IntFeasTol=1e-9;e.optimize(m,'TEST_REAL_AGGREGATE_SMALL',dict(iteration=iteration));assert m.Status==2;lb=float(m.ObjBound);ub=float(a['cost']@point+p@q);history.append(dict(iteration=iteration,LB=lb,UB=ub,direct=direct))
   assert lb<=direct+1e-4 and ub>=direct-1e-4
   if ub-lb<=1e-4:break
   point=canonical(x.X,folder/f'POINT_{iteration:02d}')
  else:raise ValueError('REAL_SMALL_EQUIVALENCE_NOT_CLOSED')
 finally:
  m.dispose()
  for o in oracles:o.close(discard=True)
  env.dispose()
 write(folder/'RESULT.json',dict(passed=True,sids=sids,direct=direct,history=history,TEST_only=True,formal_domain_restrictions=False))

def tests():
 folder=OUT/'tests';folder.mkdir(exist_ok=True)
 with (folder/'SYNTHETIC.stdout.log').open('xb') as so,(folder/'SYNTHETIC.stderr.log').open('xb') as se:
  p=subprocess.Popen([sys.executable,'-B',str(Path(CTX.config['execution_root'])/'tools/math_preflight.py'),'--output',str(folder/'SYNTHETIC_OUTPUT.json')],cwd=folder,stdout=so,stderr=se);t=time.monotonic();rc=p.wait(timeout=max(1,CTX.prep-time.monotonic()-30))
 write(folder/'SYNTHETIC_WAIT.json',dict(actual_parent_wait=True,returncode=rc,elapsed=time.monotonic()-t));assert rc==0
 control_tests();function_tests();reference_regression();native_callback_test();real_small_test();worker_reuse_test();write(folder/'TEST_RESULT.json',dict(passed=True,source_sha={str(p):sha(p) for p in (OUT/'tools').glob('*.py')},epoch=time.time()))
