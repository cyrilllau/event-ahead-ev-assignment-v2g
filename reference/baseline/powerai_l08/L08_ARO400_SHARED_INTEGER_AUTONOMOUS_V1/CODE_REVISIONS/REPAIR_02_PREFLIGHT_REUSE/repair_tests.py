from control import *
import control,aro_core
def control_tests():
 out=REV/'tests'/('UNIT_'+str(time.time_ns()));out.mkdir(parents=True);checks={}
 assert check_external_gap({'best_LB':9.75,'best_UB_audited':10},has_incumbent=True,cached_evaluation=True)=='FINAL_CERTIFY';checks['cached_gap_quarter_enters_certification']=True
 assert check_external_gap({'best_LB':5,'best_UB_audited':10},has_incumbent=True,cached_evaluation=True)=='REPLAY_CACHED_EVALUATION'
 assert check_external_gap({'best_LB':5,'best_UB_audited':10},has_incumbent=True,cached_evaluation=True,separated=True)=='CONTINUE_MASTER'
 for value in (np.nan,np.inf):
  try:check_external_gap({'best_LB':value,'best_UB_audited':10});raise AssertionError('NONFINITE_BOUND_ACCEPTED')
  except ValueError:pass
 try:canonical(np.full(60780,np.nan),out/'NAN',0);raise AssertionError('NAN_POINT_ACCEPTED')
 except ValueError:pass
 checks['NaN_Inf_and_noninteger_guard']=True
 old=control.ROOT;control.ROOT=out/'STATE';control.ROOT.mkdir()
 try:
  s={'phase':'SOLVE','outer':2};commit_state(s);new=candidate_transition(s,point_id='POINT',point_ref={'path':'unit','sha256':'unit'},eta=4);new['master_source']={'call':'test'};new['master_model_id']='MODEL'
  try:commit_state(new,'before')
  except RuntimeError:pass
  assert read_state()==s
  try:commit_state(new,'after')
  except RuntimeError:pass
  assert read_state()==new and s['outer']==2 and new['outer']==3;checks['atomic_complete_candidate_transition_crashes']=True
 finally:control.ROOT=old
 from memory_worker import Workers
 for session in ('BATCH_A','BATCH_B'):
  obj=Workers.__new__(Workers);obj.closed=False;obj.items={};obj.last_accounting={};obj.attempt=out/'SESSIONS'/session;obj.deadline=time.monotonic()+30;obj.close();before=sha(obj.attempt/'WORKER_CLEANUP.json');obj.close();assert sha(obj.attempt/'WORKER_CLEANUP.json')==before
 checks['two_sessions_and_idempotent_close']=True
 # Run cache separation on the formal function using a non-native recording model.
 p=out/'CUT.npz';d.publish_npz(p,alpha=10.,beta=np.array([-10.]));record={'sid':0,'cut':str(p)}
 class Recording:
  def __init__(self):self.rows=[]
  def addConstr(self,row,name):self.rows.append((row,name));return row
  def update(self):pass
 class Expr:
  def __ge__(self,other):return ('>=',str(other))
 m=Recording();original=control.add_support;control.add_support=lambda model,x,eta,alpha,beta,name:model.addConstr((alpha,beta.tolist()),name)
 try:new=separate(m,[0],Expr(),[record],np.array([0.]),0,[]);assert len(new)==1 and len(m.rows)==1;assert not separate(m,[0],Expr(),[record],np.array([0.]),0,new)
 finally:control.add_support=original
 checks['cached_separation_zero_native_calls_stable_dedup']=True
 assert bound_update(4,None)==4;assert check_external_gap({'best_LB':0,'best_UB_audited':None},has_incumbent=False)=='CONTINUE_MASTER';checks['missing_bound_and_incumbent']=True
 import strict_upper
 exact=strict_upper.check(sp.csr_matrix([[1.]]),np.array([.1]),np.array(['=']),np.array([0.]),np.array([1.]),np.array([.10000000000000002]),time.monotonic()+5);assert not exact['passed'];checks['exact_primal_failure_does_not_certify']=True
 write(REV/'tests/CONTROL_RESULT.json',{'passed':True,'checks':checks,'evidence_directory':str(out),'formal_functions_called':True,'native_calls':0,'mock_scope':'only pure control, recording model, and empty worker sessions; native and real-scenario checks are separate'},True)
