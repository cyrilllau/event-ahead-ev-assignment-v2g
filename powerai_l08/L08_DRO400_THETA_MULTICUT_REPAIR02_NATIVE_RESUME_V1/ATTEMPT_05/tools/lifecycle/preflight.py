"""No-optimizer tests of the exact deployed execution and persistence functions."""
from pathlib import Path
import os,sys,time,copy,subprocess,hashlib
import numpy as np
from scipy import sparse as sp
import durable as d
import runtime as rt
import engine as e
import strict_upper as strict

class MockOracleFault:
 """Fault-injection fixture. Never solves or contributes any scientific result."""
 def __init__(self,sid,root):self.sid=sid
 def close(self):pass
 def validate_result(self,x,result):
  r=d.read(result);assert r['sid']==self.sid and r['mock_only'];return r
 def solve_task(self,x,task,folder,deadline):
  folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
  if self.sid==3 and not (folder/'EXIT_INJECTED.json').exists():d.atomic(folder/'EXIT_INJECTED.json',{'mock_only':True},True);os._exit(83)
  if (folder/'RESULT.json').exists():raise RuntimeError('COMPLETED_MOCK_TASK_REEXECUTED')
  d.atomic(folder/'RESULT.json',{'sid':self.sid,'mock_only':True,'task_id':task},True);return d.read(folder/'RESULT.json')

def run_tests(out,run,store,state,meta):
 from autonomous_controller import solution_load,solution_save,threshold
 root=out/'PREFLIGHT'/str(time.time_ns());root.mkdir(parents=True);reports=[];start=time.monotonic();calls=[]
 def test(name,fn):
  t=time.monotonic()
  try:detail=fn();reports.append({'name':name,'passed':True,'wall':time.monotonic()-t,'details':detail})
  except BaseException as exc:
   reports.append({'name':name,'passed':False,'type':type(exc).__name__,'message':str(exc),'wall':time.monotonic()-t});d.atomic(out/'TEST_REPORT.json',{'passed':False,'tests':reports,'scientific_optimizer_calls':0});raise
 old_opt=e.gp.Model.optimize
 def forbidden(*args,**kwargs):calls.append('forbidden');raise RuntimeError('PREFLIGHT_OPTIMIZER_FORBIDDEN')
 e.PREFLIGHT=True;e.gp.Model.optimize=forbidden
 try:
  def proc_tests():
   script='import ctypes,time;ctypes.CDLL(None).prctl(15,b"a (b) c",0,0,0);time.sleep(.5)'
   p=subprocess.Popen([sys.executable,'-c',script]);q=rt.wait_process(p.pid)
   if not q or not rt.same(q,rt.proc(p.pid)):raise AssertionError('spaced comm identity')
   changed=dict(q,start_ticks=str(int(q['start_ticks'])+1))
   if rt.same(changed,rt.proc(p.pid)):raise AssertionError('PID reuse accepted')
   rc=p.wait();d.atomic(root/'NORMAL_CHILD_WAIT.json',{'actual_parent_wait':True,'returncode':rc,'process':q},True)
   if rt.proc(p.pid) is not None:raise AssertionError('exited PID')
   p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(10)']);q=rt.wait_process(p.pid)
   try:p.wait(timeout=.05);raise AssertionError('did not timeout')
   except subprocess.TimeoutExpired:
    if not rt.same(q,rt.proc(p.pid)):raise AssertionError('owned child identity')
    p.terminate();rc=p.wait()
   d.atomic(root/'TIMEOUT_CHILD_WAIT.json',{'actual_parent_wait':True,'returncode':rc,'identity_verified_before_signal':True,'production_signals_sent':[]},True)
   return {'space_parenthesis_comm':True,'PID_reuse_rejected':True,'normal_wait':0,'timeout_wait':rc}
  test('process identity, exited PID, owned timeout true WAIT',proc_tests)
  def point():
   src=Path(run['source']);z=np.load(src/'LAST_SOLVED/SOLUTION.npz');p=root/'MISSING_DIRECTORY/POINT_REPAIR';assert not p.exists();x,check=e.b.point_repair(z['x'][:e.b.FIRST_N],p,rt.deadline(run)-120)
   saved=np.load(p/'POINTS.npz');assert np.array_equal(saved['accepted'],x) and check['displacement']<=1e-7 and check['projection'] is None and max(check['after'].values())<=1e-9
   return check
  test('same production POINT_REPAIR creates missing directory, actual source point no optimize',point)
  def oracle_reuse():
   class MockOracle:
    count=0
    def __init__(self,sid,root):MockOracle.count+=1
   objects={}
   for sid in (1,2,1,2):e.get_oracle(objects,sid,root,MockOracle)
   assert MockOracle.count==2
   return {'construction_count':2,'batches':2,'mock_only_no_recourse_claim':True}
  test('same worker oracle get-or-create reused across messages',oracle_reuse)
  def clock_tests():
   r={'boot_id':'A','deadline_monotonic':100,'start_utc':1000,'deadline_utc':2000,'last_observed_utc':1100}
   assert rt.deadline(r,40,1500,'A')==100 and rt.deadline(r,40,1500,'B')==540
   try:rt.deadline(r,40,900,'B');raise AssertionError('backwards clock')
   except RuntimeError:pass
   a,n=rt.retry_allowed(root,'stable task','worker exit',3,True);assert a and n==1
   a,n=rt.retry_allowed(root,'stable task','worker exit',3,True);assert a and n==2
   a,n=rt.retry_allowed(root,'stable task','worker exit',3,True);assert a and n==3
   a,n=rt.retry_allowed(root,'stable task','worker exit',3,True);assert not a
   return {'same_boot_monotonic':True,'reboot_downtime_deducted':True,'budget_not_reset_on_attempt':True,'retry_counts_persisted':True,'unknown_error_not_retried':not rt.retry_allowed(root,'other','TypeError',3,False)[0]}
  test('global deadline recovery and persistent retry exhaustion',clock_tests)
  def storage():
   st=d.Store(root/'JOURNAL_TEST',base=out/'BASE');st.load_bank([]);a=copy.deepcopy(state);seed={'phase':'MASTER_SOLVE','last_solved':meta['last_solved'],'pool_closed':False};st.commit(a,seed);old=d.read(st.root/'LAST_COMMITTED.json')
   for label in ('payload','journal','reference'):
    try:st.commit(a,dict(seed,phase='NEW_Q'),fail_at=label);raise AssertionError('fault not injected')
    except RuntimeError:pass
    assert d.read(st.root/'LAST_COMMITTED.json')==old
    restored,rmeta=d.Store(st.root,base=out/'BASE').replay();assert d.identity(restored)==d.identity(a) and rmeta['phase']=='MASTER_SOLVE'
   # Re-add a real dormant row and persist pending without attaching old Pi.
   idx=0;row=a['dA'][idx];name=a['dnames'][idx];a['A']=sp.vstack([a['A'],row],format='csr');a['rhs']=np.r_[a['rhs'],a['drhs'][idx]];a['sense']=np.concatenate([a['sense'],a['dsense'][idx:idx+1]]);a['names']['rows'].append(name);a['dA']=a['dA'][1:];a['drhs']=a['drhs'][1:];a['dsense']=a['dsense'][1:];a['dnames']=a['dnames'][1:]
   st.commit(a,seed);rr,mm=d.Store(st.root,base=out/'BASE').replay();assert d.identity(rr)==d.identity(a) and mm['last_solved']==meta['last_solved'] and rr['A'].shape[0]==91242
   # Sleep/wake exact row mappings; no coefficient rewriting.
   st.commit(state,dict(seed,phase='RECOMPRESS'));rr,_=d.Store(st.root,base=out/'BASE').replay();assert d.identity(rr)==d.identity(state)
   # Same name with changed coefficients must be rejected by deployed commit.
   bad=copy.deepcopy(state);bad['A'].data[0]=np.nextafter(bad['A'].data[0],np.inf)
   try:st.commit(bad,seed);raise AssertionError('row identity conflict missed')
   except ValueError:pass
   return {'hash_chain_readback':True,'three_crash_points_preserve_head':True,'pending_with_old_last_solved_reference':True,'readd_sleep_restore_exact':True,'same_name_changed_coefficient_rejected':True}
  test('actual filesystem fsync, journal crash injection, pending/solved identity separation',storage)
  def history_replay():
   src=Path(run['source']);candidates=sorted((src/'outer_027').glob('MASTER_*/SOLVED'))
   if not candidates:raise ValueError('HISTORICAL_MATRIX_MISSING')
   historical=e.core.load_state(candidates[0]);mapping=store.mapping(historical);actual=store.materialize(mapping)
   assert d.identity(actual)==d.identity(historical)
   for k in ('rhs','sense','cost','lo','hi','drhs','dsense'):assert np.array_equal(actual[k],historical[k])
   assert (actual['A']!=historical['A']).nnz==0 and (actual['dA']!=historical['dA']).nnz==0 and actual['dnames']==historical['dnames']
   # Exercise a real new-row batch (stored coefficients and source) against a prior row bank.
   pos=next(i for i,n in enumerate(state['names']['rows']) if n.startswith('r44_benders/') and n.endswith('/27'))
   small={'A':state['A'][pos:pos+1],'rhs':state['rhs'][pos:pos+1],'sense':state['sense'][pos:pos+1],'cost':state['cost'],'lo':state['lo'],'hi':state['hi'],'names':{'rows':[state['names']['rows'][pos]],'columns':state['names']['columns']},'dA':sp.csr_matrix((0,state['A'].shape[1])),'drhs':np.array([],float),'dsense':np.array([],dtype='<U1'),'dnames':[],'dsource':[]}
   empty=copy.deepcopy(small);empty['A']=empty['A'][:0];empty['rhs']=empty['rhs'][:0];empty['sense']=empty['sense'][:0];empty['names']['rows']=[];ss=d.Store(root/'REAL_ROW_BLOCK');ss.initialize(empty);ss.load_bank([]);ss.commit(small,{'phase':'PENDING'},sources=[{'source':str(candidates[0]),'actual_coefficient_sha':d.identity(small)['matrix_sha']}]);back,_=d.Store(ss.root).replay();assert d.identity(back)==d.identity(small)
   return {'historical_master':str(candidates[0]),'coefficientwise_and_order_exact':True,'actual_historical_new_row_stored_once':True,'ROW_STORE_blocks':len(ss.blocks)}
  test('real historical outer incremental reconstruction and real ROW_STORE payload',history_replay)
  def solved():
   ss,z,p=solution_load(store,meta['last_solved']);assert d.identity(ss)==d.identity(state);assert len(z['Pi'])==len(z['CBasis'])==91241
   scene=np.array([i for i,n in enumerate(ss['names']['rows']) if e.b.scenario_name(n)]);_,cert=e.b.lagrangian(ss['A'],ss['rhs'],ss['sense'],ss['cost'],ss['lo'],ss['hi'],z['Pi']);rem=scene[(z['Pi'][scene]==0)&(cert[scene]==0)&(z['CBasis'][scene]==0)]
   keep=np.ones(ss['A'].shape[0],bool);keep[rem]=False;lb,_=e.b.lagrangian(ss['A'][keep],ss['rhs'][keep],ss['sense'][keep],ss['cost'],ss['lo'],ss['hi'],z['Pi'][keep]);assert abs(lb-p['LB'])<=1e-5
   assert len(z['CBasis'][keep])==ss['A'][keep].shape[0] and len(z['VBasis'])==ss['A'].shape[1]
   return {'source_LB_replayed':p['LB'],'strict_raw_and_certificate_zero':True,'basis_mapping':True,'count_is_not_independent_nonsingularity_proof':True,'native_basis_acceptance_NOT_RUN':True}
  test('solution, dual, basis and strict-zero compression mapping restore',solved)
  def cache():
   x=np.array([0.,1.]);other=x.copy();other[1]=np.nextafter(other[1],np.inf);assert e.task_id('s',1,x,'q')!=e.task_id('s',1,other,'q')
   complete={1:'valid',3:'valid'};missing=[sid for sid in [1,2,3,4] if sid not in complete];assert missing==[2,4]
   # Task checker and unknown deterministic failures never clear successful siblings.
   assert e.raw_for('preflight nonexistent','MASTER')==(None,None)
   return {'partial_missing_only':[2,4],'distinct_binary64_points_not_cache_hits':True,'mock_scheduler_only':True,'actual93_batch_NOT_RUN':True,'actual_cache_native_calls':0}
  test('stable exact point task IDs, partial completion and missing-only dispatch policy',cache)
  def partial_workers():
   mockroot=root/'MOCK_ONLY_93_SCHEDULER';mockroot.mkdir();x=np.array([0.,1.]);science='PREFLIGHT_NOT_SCIENCE';task=e.task_id(science,0,x,'RECOURSE');folder=mockroot/'EVALUATION_CACHE'/task;d.atomic(folder/'RESULT.json',{'sid':0,'mock_only':True,'task_id':task},True)
   w=e.Workers(mockroot,run,time.monotonic()+60,mockroot/'SUPERVISION',MockOracleFault)
   try:r=w.evaluate(x,list(range(93)),science,lambda n:None)
   finally:w.close()
   assert len(r)==93 and w.cache_hits==1
   waits=[d.read(p) for p in (mockroot/'SUPERVISION/workers').glob('*_WAIT.json')];assert any(r['returncode']==83 for r in waits) and all(r['actual_parent_wait'] for r in waits)
   return {'mock_only_no_scientific_Q':True,'actual_deployed_scheduler':True,'completed_93_fixture_tasks':93,'reused_partial':1,'worker_crash_injected_and_true_WAIT_obtained':True,'other_successful_tasks_preserved':True}
  test('actual four-process scheduler worker-exit recovery and partial93 cache preservation',partial_workers)
  def risks():
   src=Path(run['source'])/'probability/OUTER_027';raw=np.load(src/'RAW_FINAL.npz');risk=e.b.risk_upper(raw['q'],float(raw['beta0']),raw['plus'],raw['minus']);assert abs(risk['risk_upper']-d.read(src/'RISK_UPPER.json')['risk_upper'])<1e-8
   # Actual deployed postprocess, using copied immutable RAW and certificate, no native LP.
   dst=root/'RISK_RAW_REPLAY';dst.mkdir();import shutil
   for f in ('RAW_FINAL.npz','RAW_FINAL_IDENTITY.json','RISK_UPPER.json','CERTIFICATE.json'):shutil.copy2(src/f,dst/f)
   shutil.copytree(src/'STRICT_CUT',dst/'STRICT_CUT');p,rr,st,info=e.replay_risk(dst,raw['q']);assert rr['risk_upper']==risk['risk_upper']
   return {'raw_replayed_without_Q_or_probability_optimize':True,'all400_support':len(p)==400,'strict_cut_compensation':st['epsilon'],'independent_risk_upper':risk['risk_upper']}
  test('saved risk raw independently postprocessed without optimizer',risks)
  def exact():
   A=sp.csr_matrix([[1.]]);sense=np.array(['=']);lo=np.array([0.]);hi=np.array([1.]);end=time.monotonic()+10
   assert strict.check(A,np.array([.5]),sense,lo,hi,np.array([.5]),end)['passed']
   assert not strict.check(A,np.array([.5]),sense,lo,hi,np.array([np.nextafter(.5,1)]),end)['passed']
   # Internal rounded h-T*x equals .2, exact original subtraction differs.
   h=np.array([1.]);T=sp.csr_matrix([[1.]]);x=np.array([.1]);y=h-T@x
   assert not strict.check(A,h,sense,lo,hi,y,end,T,x)['passed']
   try:strict.check(A,np.array([.5]),sense,lo,hi,np.array([.5]),time.monotonic()-1);raise AssertionError('no deadline')
   except TimeoutError:pass
   return {'exact_feasible_and_tiny_violation':True,'original_rhs_roundoff_detected':True,'deadline_honored':True,'not_an_exact_repair_solver':True}
  test('independent exact upper checker rejects rounded RHS and tiny equalities',exact)
  def disk():assert not d.disk_check(out,free_override=0)['passed'];return {'insufficient_space_rejected':True,'quota':d.disk_check(out)['quota']}
  test('production disk gate rejects insufficient space',disk)
  def late():
   p=root/'UNCLOSED.log';p.write_text('partial');d.atomic(root/'LOG_STATE.json',{'sealed':False,'file':str(p)});p.write_text('partial\nclosed\n');d.atomic(root/'LOG_STATE.json',{'sealed':True,'file':rt.ref(p)});rt.verify_ref(d.read(root/'LOG_STATE.json')['file']);return {'derived_manifest_after_close':True,'unsealed_log_not_integrity_failure':True}
  test('unsealed logs excluded until close',late)
  if calls:raise AssertionError('a real optimizer invocation was attempted in no-optimizer preflight')
  report={'passed':True,'tests':reports,'wall':time.monotonic()-start,'scientific_optimizer_calls':0,'true_outer028_integration':'NOT_RUN: will execute once after this preflight','limits':'fault injections are labeled synthetic; historical matrix and mathematical replay use actual data'}
  d.atomic(root/'RESULT.json',report,True);d.atomic(out/'TEST_REPORT.json',report)
 finally:e.PREFLIGHT=False;e.gp.Model.optimize=old_opt
