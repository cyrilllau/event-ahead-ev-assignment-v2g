from r36_model import *
from exact_transform import original_expressions
def incident_tests():
 out=R/'tests/incident';out.mkdir(parents=True,exist_ok=True);sid=read(R/'tests/REAL_SCENE_SELECTION.json')['choices']['no_fault'];pp=plans();a=load_scene('TRAIN',sid);net=layout()['net'];assert not incident(net,a)[0].any()
 qs=routes();st=route_states('L08',qs,'TRAIN',sid);m=gp.Model('R36_ZETA_CONTRACT_ONLY');parameters(m,out/'NATIVE.log',1);fs=add_first(m,qs,30,8,full_bounds());fix_first(fs,pp);rc=add_recourse(m,fs,net,a,st,sid);yy=original_expressions(rc['y'],rc['pfmap']);chosen=None
 for h in pp:
  state=st[next(i for i,q in enumerate(qs) if q['route_id']==h['route_id'])];win=actual(h['B']+100,state['arrival'],h['k'],h['r'],state['success'])
  if win['available_ac']>=1 and win['mu'].any():chosen=(h,int(np.flatnonzero(win['mu'])[0]));break
 assert chosen;h,t=chosen;m.addConstr(yy[rc['original'].index['p'][h['e'],h['c'],t]]>=1,name='TEST_ONLY_POSITIVE_P_WHEN_ZETA_ZERO');m.setObjective(rc['objective']);m.update()
 for v in m.getVars():
  if v.VType==GRB.BINARY:v.VType=GRB.CONTINUOUS
 m.update();dump(out/'BEGIN.json',dict(epoch=time.time(),scenario=sid,selected_vehicle=h['e'],station=h['c'],slot=t,zeta=0,forced_kw=1));m.optimize();assert m.Status==GRB.OPTIMAL
 y=lift(rc['y'].X,rc['pfmap']);tasks=[task(next(q for q in qs if q['route_id']==p['route_id']),p['k'],p['r']) for p in pp];contracts=np.zeros(30)
 for p in pp:contracts[p['e']]=p['contract']
 mat=assemble(net,tasks,a,route_states('L08',tasks,'TRAIN',sid),30,full_bounds());res=mat.residual(y,np.r_[np.ones(len(tasks)),contracts]);assert res<=1e-6 and y[mat.index['short']].max()<=1e-7
 rec=dict(passed=True,native_status=m.Status,runtime=m.Runtime,scenario=sid,positive_power_when_zeta_zero=float(y[mat.index['p'][h['e'],h['c'],t]]),shortfall_when_no_incident=float(y[mat.index['short']].sum()),original_residual=res,power_zeta_gate_absent=True,first_stage_incident_admission_absent=True);npz(out/'ORIGINAL_POINT.npz',y=y);dump(out/'END.json',rec);m.dispose();return rec
