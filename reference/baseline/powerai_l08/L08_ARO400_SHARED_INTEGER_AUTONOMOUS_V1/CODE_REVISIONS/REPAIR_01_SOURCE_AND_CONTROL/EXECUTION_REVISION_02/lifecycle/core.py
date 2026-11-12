from __future__ import annotations
import os, sys, time, json, hashlib, traceback, copy, shutil
from pathlib import Path
import numpy as np
from scipy import sparse as sp
import gurobipy as gp
import base_r44 as b
from probability_patch_adapter import certify_candidate_compat

OUT=b.OUT
SRC=b.ROOT/'RESUME_04'
LIMIT=1800.0
original_identity=b.state_identity
def identity(state):
    v=original_identity(state)
    v['rhs_sense_sha']=hashlib.sha256(state['rhs'].tobytes()+state['sense'].tobytes()).hexdigest()
    v['objective_constant']=0.0
    return v
b.state_identity=identity
def write(p,x):b.atomic_json(p,x)
def save_npz(path,**arrays):
    p=Path(path); root=OUT.resolve()
    if root not in p.resolve().parents:raise ValueError('OUTPUT_OUTSIDE_OUT')
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError(str(p))
    temp=p.with_name(p.name+'.tmp')
    with temp.open('xb') as f:
        np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(temp,p)

def load_state(d):
    z=dict(np.load(d/'ARRAYS.npz'))
    z.update(A=sp.load_npz(d/'A.npz').tocsr(),names=b.read(d/'NAMES.json'),dA=sp.load_npz(d/'DORMANT_A.npz').tocsr())
    dz=np.load(d/'DORMANT_ARRAYS.npz');dn=b.read(d/'DORMANT_NAMES.json')
    z.update(drhs=dz['rhs'],dsense=dz['sense'],dnames=dn['rows'],dsource=dn['source'])
    return z

def legacy_recover_not_used(directory):
    directory.mkdir(parents=True,exist_ok=True)
    last=load_state(SRC/'LAST_SOLVED'); state=load_state(SRC/'WORKING_PENDING')
    sol=dict(np.load(SRC/'LAST_SOLVED/SOLUTION.npz'));sol.update(dict(np.load(SRC/'LAST_SOLVED/BASIS.npz')))
    assert last['A'].shape==(91298,60874) and last['A'].nnz==1651240
    assert state['A'].shape==(91386,60874) and state['A'].nnz==2000749
    assert (last['A']!=state['A'][:91298]).nnz==0
    for k in ('rhs','sense'):assert np.array_equal(last[k],state[k][:91298])
    for k in ('cost','lo','hi'):assert np.array_equal(last[k],state[k])
    assert last['names']['columns']==state['names']['columns'] and last['names']['rows']==state['names']['rows'][:91298]
    assert (last['dA']!=state['dA']).nnz==0
    pool=list(map(int,b.read(b.R41/'LOCK.json')['pool']))
    qbars=np.array([b.read(b.R41/'oracles'/f'{sid:04d}'/'QBAR_PROOF.json')['Qbar'] for sid in pool])
    b.GLOBAL_POOL=pool;b.GLOBAL_QBARS=qbars.tolist()
    assert len(pool)==93 and len(set(pool))==93
    theta_lo=state['lo'][b.FIRST_N:b.FIRST_N+93];theta_hi=state['hi'][b.FIRST_N:b.FIRST_N+93]
    assert np.all(theta_lo>=0) and np.all(theta_hi<=qbars) and np.isfinite(qbars).all()
    proofs=b.read(b.R42/'GLOBAL_BOUND_PROOFS.json')
    byid={int(x['sid']):x for x in proofs}
    assert all(byid[s]['Qbar']==q for s,q in zip(pool,qbars))
    write(directory/'THETA_BOUND_AUDIT.json',{'passed':True,'lo':theta_lo,'hi':theta_hi,'Qbar':qbars,'ids':pool,'proof_source':b.record(b.R42/'GLOBAL_BOUND_PROOFS.json'),'per_sid_proofs':[b.record(b.R41/'oracles'/f'{s:04d}'/'QBAR_PROOF.json') for s in pool]})
    audit=b.endpoint_residual_audit(last['A'],last['rhs'],last['sense'],last['lo'],last['hi'],last['names']['rows'],last['dA'],last['drhs'],last['dsense'],sol['x'])
    assert audit['passed'];write(directory/'ENDPOINT_RESIDUAL_AUDIT.json',audit)
    lb,_=b.lagrangian(last['A'],last['rhs'],last['sense'],last['cost'],last['lo'],last['hi'],sol['Pi'])
    assert abs(lb-10116.746983790255)<1e-9
    transferred,_=b.lagrangian(state['A'],state['rhs'],state['sense'],state['cost'],state['lo'],state['hi'],np.r_[sol['Pi'],np.zeros(88)])
    write(directory/'LB_REPLAY.json',{'LB':lb,'pending_transferred_LB':transferred,'transferred_certificate':True,'native_new_solve':False,'source':b.record(SRC/'LAST_SOLVED/LB_CERTIFICATE.json'),'identity':identity(last)})
    point=np.load(SRC/'outer_000/POINT_REPAIR/POINTS.npz')['accepted']
    point_sha=hashlib.sha256(point.tobytes()).hexdigest()
    assert point_sha=='90d2b71c4fd4ecec09ac7f6570012acf56e50fdd46b529de20b97cd6430eaa05'
    assert np.array_equal(point,sol['x'][:b.FIRST_N])
    results=[];q=np.zeros(400);index=[]
    for sid in pool:
        d=SRC/'oracles'/f'{sid:04d}'/'OUTER_000'
        rec=b.read(d/'RESULT.json');z=np.load(d/'POINT.npz');cut=np.load(d/'CUT.npz')
        assert np.array_equal(z['x'],point) and rec['point_sha']==point_sha
        oracle=b.Oracle(sid,OUT/'recovery_validation')
        check,alpha,beta,pi,ql,qu=oracle.validate_point(point,z['y'],z['Pi']);oracle.close()
        assert np.array_equal(beta,cut['beta']) and alpha==float(cut['alpha']) and abs(qu-rec['qU_audited'])<1e-8
        assert rec['sid']==sid
        q[sid]=qu;results.append(rec)
        index.append({'sid':sid,'point_sha':point_sha,'qU_audited':qu,'sources':[b.record(d/n) for n in ('RESULT.json','POINT.npz','CUT.npz','NATIVE/BEGIN.json','NATIVE/END.json')],'replayed_global_cut':True,'residual':check})
    delta=[]
    for i,name in enumerate(state['names']['rows'][91298:],91298):
        sid=int(name.split('/')[1]);cut=np.load(SRC/'oracles'/f'{sid:04d}'/'OUTER_000/CUT.npz');beta=cut['beta'];nz=np.flatnonzero(beta)
        row=sp.csr_matrix((np.r_[-beta[nz],1.],(np.zeros(len(nz)+1,int),np.r_[nz,b.FIRST_N+pool.index(sid)])),shape=(1,60874))
        assert (row!=state['A'].getrow(i)).nnz==0 and state['rhs'][i]==float(cut['alpha']) and state['sense'][i]=='>'
        delta.append({'row':name,'sid':sid,'source':b.record(SRC/'oracles'/f'{sid:04d}'/'OUTER_000/CUT.npz')})
    fingerprints={b.row_fingerprint(state['A'].getrow(i),state['rhs'][i],state['sense'][i]) for i,n in enumerate(state['names']['rows']) if b.scenario_name(n)}
    dormant={b.row_fingerprint(state['dA'].getrow(i),state['drhs'][i],state['dsense'][i]) for i in range(state['dA'].shape[0])}
    assert not fingerprints&dormant and len(state['dnames'])==2577
    write(directory/'PENDING_88_AUDIT.json',{'passed':True,'rows':delta,'basis_seed_only':True,'no_pending_native_Pi':True,'identity':identity(state)})
    write(directory/'RECOVERED_BATCH_INDEX.json',{'cache_reuse':93,'new_recourse_calls':0,'accepted_x_sha':point_sha,'pool':pool,'scenarios':index,'remaining_307_Q_zero':True})
    save_npz(directory/'RECOVERED_Q.npz',accepted_x=point,q=q,pool=pool)
    state.update(VBasis=sol['VBasis'],CBasis=np.r_[sol['CBasis'],np.zeros(88,int)],LB=lb)
    return state,last,sol,point,q,results

def certify(p,qbars,ids,out):
    p0=np.load(b.R42/'PROBABILITY_INTERIOR_STRICT/P_STAR.npz')['p']
    rec=certify_candidate_compat(p,qbars,ids,out,p0=p0,dd=b.moment_data(),check=b.probability_check,enclosure=b.enclosure,owned_root=OUT)
    write(out/'SOURCES.json',{'anchor':b.record(b.R42/'PROBABILITY_INTERIOR_STRICT/P_STAR.npz'),'moments':b.record(b.R29/'05_DRO_MOMENTS/EFFECTIVE_INTERVALS.npz'),'bounds':b.record(OUT/'RECOVERY/THETA_BOUND_AUDIT.json')})
    return rec

class Probability(b.Probability):
    def __init__(self,out):
        super().__init__(out)
        previous=np.load(SRC/'probability/OUTER_009/RAW_FINAL.npz')
        self.add([(str(kind),int(i)) for kind,i in zip(previous['active_kind'],previous['active_id'])])
        write(Path(out)/'RESTORED_ACTIVE_SIDES.json',{'source':b.record(SRC/'probability/OUTER_009/RAW_FINAL.npz'),'active_sides':len(self.active),'probability_basis_restored':False})
    def solve(self,q,label,deadline):
        d=self.out/label;d.mkdir()
        q=np.asarray(q,float);normal=max(1.,float(abs(q).max()));self.model.setObjective((q/normal)@self.p,gp.GRB.MAXIMIZE)
        history=[];closed=False
        for k in range(200):
            remaining=deadline-time.monotonic()-20
            if remaining<=0:raise TimeoutError('PROBABILITY_DEADLINE')
            call=d/f'CALL_{k:03d}';call.mkdir()
            for key,val in dict(Threads=1,Method=1,FeasibilityTol=1e-9,OptimalityTol=1e-9,SoftMemLimit=6,TimeLimit=remaining,OutputFlag=1,LogFile=str(call/'NATIVE.log')).items():setattr(self.model.Params,key,val)
            write(call/'BEGIN.json',{'epoch':time.time(),'remaining':remaining})
            cid=b.tracked_optimize(self.model,'PROBABILITY_ROWGEN',{'label':label,'iteration':k})
            write(call/'END.json',{'status':self.model.Status,'runtime':self.model.Runtime,'call_id':cid})
            if self.model.Status!=gp.GRB.OPTIMAL:raise TimeoutError(('PROBABILITY_PARTIAL',self.model.Status))
            p=np.asarray(self.p.X);bp=np.zeros(45139);bm=np.zeros(45139)
            for (kind,i),row in self.active.items():
                if kind=='U':bp[i]=max(0,row.Pi)*normal
                else:bm[i]=max(0,-row.Pi)*normal
            raw=dict(p=p,q=q,beta0=float(self.simplex.Pi)*normal,plus=bp,minus=bm,scale=self.a['scale'],active_kind=np.array([x[0] for x in self.active]),active_id=np.array([x[1] for x in self.active]))
            save_npz(call/'RAW_NATIVE.npz',**raw)
            write(call/'RAW_IDENTITY.json',{'status':'NATIVE_SAVED','call_id':cid,'epoch':time.time(),'monotonic':time.monotonic(),'raw':b.record(call/'RAW_NATIVE.npz')})
            ex=self.a['phi']@p;lv=self.a['lower']-ex;uv=ex-self.a['upper']
            violations=sorted([(float(v),'L',int(i)) for i,v in enumerate(lv) if v>1e-9]+[(float(v),'U',int(i)) for i,v in enumerate(uv) if v>1e-9],reverse=True)
            history.append({'iteration':k,'violations':len(violations),'max':violations[0][0] if violations else 0,'call_id':cid})
            write(d/'HISTORY.json',history)
            if not violations:closed=True;break
            new=[(kind,i) for _,kind,i in violations if (kind,i) not in self.active][:500]
            if not new:break
            self.add(new)
        save_npz(d/'RAW_FINAL.npz',**raw)
        write(d/'RAW_FINAL_IDENTITY.json',{'status':'ROWGEN_CLOSED' if closed else 'ROWGEN_NUMERICAL_PARTIAL','raw':b.record(d/'RAW_FINAL.npz'),'all400':True,'all_effective_rows':45139,'all_original_rows':322311})
        risk=b.risk_upper(q,float(raw['beta0']),bp,bm)
        save_npz(d/'RISK_UPPER.npz',beta0=risk['beta0'],plus=bp,minus=bm,scale=self.a['scale'],q=q)
        write(d/'RISK_UPPER.json',{k:v for k,v in risk.items() if k not in ('plus','minus')})
        strict=certify(p,np.array(b.GLOBAL_QBARS),b.GLOBAL_POOL,d/'STRICT_CUT')
        info={'risk_upper':risk['risk_upper'],'risk_primal':float(p@q),'rowgen_closed':closed,'raw_final':b.record(d/'RAW_FINAL.npz'),'strict':strict}
        write(d/'CERTIFICATE.json',info)
        return p,risk,strict,info

def solve_master(state,model,v,d,deadline,best,first):
    d.mkdir();remaining=deadline-time.monotonic()-15
    if remaining<=0:raise TimeoutError('MASTER_DEADLINE')
    b.master_config(model,d/'NATIVE.log',remaining)
    write(d/'BEGIN.json',{'epoch':time.time(),'identity':identity(state),'remaining':remaining})
    cid=b.tracked_optimize(model,'MASTER',{'directory':str(d)})
    accepted=b.basis_accepted(d/'NATIVE.log')
    write(d/'END.json',{'status':model.Status,'basis_accepted':accepted,'call_id':cid,'runtime':model.Runtime})
    if first and not accepted:raise RuntimeError('BASIS_NOT_ACCEPTED')
    pi=np.asarray(model.getAttr('Pi'));rc=np.asarray(model.getAttr('RC'))
    lb,cert=b.lagrangian(state['A'],state['rhs'],state['sense'],state['cost'],state['lo'],state['hi'],pi)
    best=max(best,lb)
    save_npz(d/'DUAL.npz',Pi=cert,RC=rc)
    write(d/'LOWER_CERTIFICATE.json',{'current_LB':lb,'best_LB':best,'identity':identity(state)})
    if model.Status!=gp.GRB.OPTIMAL:raise TimeoutError(('MASTER_PARTIAL',model.Status))
    result={'x':np.asarray(v.X),'Pi':pi,'RC':rc,'VBasis':np.asarray(model.getAttr('VBasis'),int),'CBasis':np.asarray(model.getAttr('CBasis'),int)}
    audit=b.row_audit(state['A'],state['rhs'],state['sense'],state['lo'],state['hi'],result['x'])
    rv=b.directed_row_violations(state['A'],state['rhs'],state['sense'],result['x']);other=np.array([not b.scenario_name(n) for n in state['names']['rows']])
    write(d/'PRIMAL_AUDIT.json',{'all_rows':audit,'physical_probability_max':float(max(0,np.max(rv[other]))),'primal_objective_is_not_strict_UB':True})
    if not np.isfinite(result['x']).all() or max(0,np.max(rv[other]))>1e-9 or audit['bound']>1e-9:raise RuntimeError('FIRST_STAGE_OR_NUMERIC_FAILED')
    b.export_state(d/'SOLVED',state,b.freeze_solved(state,result,lb))
    return result,best,lb

def verify_export(d,expected):
    loaded=load_state(d)
    assert identity(loaded)==identity(expected)
    for k in ('rhs','sense','cost','lo','hi'):assert np.array_equal(loaded[k],expected[k])
    z=np.load(d/'SOLUTION.npz');bz=np.load(d/'BASIS.npz')
    assert len(z['Pi'])==len(bz['CBasis'])==expected['A'].shape[0] and len(z['x'])==len(z['RC'])==len(bz['VBasis'])==expected['A'].shape[1]
    return True

TIMES={}
def timed(name,fn):
    def call(*args,**kwargs):
        t=time.monotonic()
        try:return fn(*args,**kwargs)
        finally:TIMES[name]=TIMES.get(name,0.)+time.monotonic()-t
    return call
for _name,_fn in [('certificate', 'lagrangian'),('scan','scan_dormant'),('readd','append_rows'),('compression','maybe_recompress'),('state_copy','freeze_solved')]:
    setattr(b,_fn,timed(_name,getattr(b,_fn)))
np.savez_compressed=timed('npz_compress_write',np.savez_compressed)
verify_export=timed('checkpoint_readback',verify_export)

def recover(directory):
    directory.mkdir(parents=True,exist_ok=True)
    t=time.monotonic()
    last=load_state(SRC/'LAST_SOLVED');state=load_state(SRC/'WORKING_PENDING')
    assert identity(last)==b.read(SRC/'LAST_SOLVED/IDENTITY.json')|{} or all(identity(last)[k]==b.read(SRC/'LAST_SOLVED/IDENTITY.json')[k] for k in identity(last))
    assert all(identity(state)[k]==b.read(SRC/'WORKING_PENDING/IDENTITY.json')[k] for k in identity(state))
    assert last['A'].shape==(91330,60874) and last['A'].nnz==1810195
    assert state['A'].shape==(91332,60874) and state['A'].nnz==1831280
    assert b.matrix_sha(last['A'])=='83e4a1c8e8451febbe0e9d718c0da75f89e7b2c75a2f5c89f7341884b4a1d490'
    assert b.matrix_sha(state['A'])=='eedbf7413a76f805b8226748278387e446a7d1cd673802e656a3d3127b319d98'
    assert (last['A']!=state['A'][:91330]).nnz==0
    for k in ('rhs','sense'):assert np.array_equal(last[k],state[k][:91330])
    for k in ('cost','lo','hi'):assert np.array_equal(last[k],state[k])
    assert last['names']['columns']==state['names']['columns'] and last['names']['rows']==state['names']['rows'][:91330]
    tails=['benders/49/11','r44_benders/244/1'];ids=[1412,2976]
    assert state['names']['rows'][91330:]==tails and [last['dnames'][i] for i in ids]==tails
    assert (state['A'][91330:]!=last['dA'][ids]).nnz==0
    for k,dkey in [('rhs','drhs'),('sense','dsense')]:assert np.array_equal(state[k][91330:],last[dkey][ids])
    keep=np.ones(len(last['dnames']),bool);keep[ids]=False
    assert (state['dA']!=last['dA'][keep]).nnz==0
    assert state['dnames']==[n for i,n in enumerate(last['dnames']) if keep[i]]
    for k in ('drhs','dsense'):assert np.array_equal(state[k],last[k][keep])
    active={b.row_fingerprint(state['A'].getrow(i),state['rhs'][i],state['sense'][i]) for i,n in enumerate(state['names']['rows']) if b.scenario_name(n)}
    dormant={b.row_fingerprint(state['dA'].getrow(i),state['drhs'][i],state['dsense'][i]) for i in range(state['dA'].shape[0])}
    assert len(active)==323 and len(dormant)==3438 and not active&dormant and len(active|dormant)==3761
    assert sum(b.probability_name(n) for n in state['names']['rows'])==241
    sol=dict(np.load(SRC/'LAST_SOLVED/SOLUTION.npz'));sol.update(dict(np.load(SRC/'LAST_SOLVED/BASIS.npz')))
    lb,_=b.lagrangian(last['A'],last['rhs'],last['sense'],last['cost'],last['lo'],last['hi'],sol['Pi']);assert abs(lb-10849.17288074456)<1e-9
    transferred,_=b.lagrangian(state['A'],state['rhs'],state['sense'],state['cost'],state['lo'],state['hi'],np.r_[sol['Pi'],np.zeros(2)])
    rv=b.directed_row_violations(last['A'],last['rhs'],last['sense'],sol['x'])
    physical=np.array([not b.scenario_name(n) and not b.probability_name(n) for n in last['names']['rows']]);probability=np.array([b.probability_name(n) for n in last['names']['rows']]);scenes=~(physical|probability)
    audit={'physical':float(max(0,np.max(rv[physical]))),'probability':float(max(0,np.max(rv[probability]))),'benders':float(max(0,np.max(rv[scenes]))),'bounds':b.row_audit(last['A'],last['rhs'],last['sense'],last['lo'],last['hi'],sol['x'])['bound']}
    assert audit['physical']<=1e-9 and audit['probability']<=1e-9 and audit['bounds']<=1e-9
    write(directory/'INITIAL_AUDIT.json',{'last':identity(last),'pending':identity(state),'pending_tail':tails,'residuals':audit,'LB':lb,'transferred_LB':transferred,'certificate_kind':'TRANSFERRED_CERTIFICATE','seed_only':True,'requires_reoptimization':True})
    pool=list(map(int,b.read(b.R41/'LOCK.json')['pool']));b.GLOBAL_POOL=pool;b.GLOBAL_QBARS=[b.read(b.R41/'oracles'/f'{s:04d}'/'QBAR_PROOF.json')['Qbar'] for s in pool]
    assert np.all(state['lo'][b.FIRST_N:b.FIRST_N+93]>=0) and np.all(state['hi'][b.FIRST_N:b.FIRST_N+93]<=b.GLOBAL_QBARS)
    write(directory/'THETA_BOUND_AUDIT.json',{'passed':True,'pool':pool,'Qbar':b.GLOBAL_QBARS,'source':b.record(SRC/'RECOVERY/THETA_BOUND_AUDIT.json')})
    state.update(VBasis=sol['VBasis'],CBasis=np.r_[sol['CBasis'],np.zeros(2,int)],LB=10849.17288074456)
    TIMES['model_pool_loading']=TIMES.get('model_pool_loading',0)+time.monotonic()-t
    return state,last,sol,None,None,None

def best_evaluation_audit(directory):
    directory.mkdir(parents=True,exist_ok=True)
    ev=np.load(SRC/'outer_006/EVALUATION.npz');rec=b.read(SRC/'outer_006/EVALUATION.json');x=ev['accepted_x'];q=ev['q']
    h=hashlib.sha256(x.tobytes()).hexdigest();assert h==rec['accepted_x_sha']=='7cc1fa0c9ec2cde38d236fa3fe6fda34f486df7d6f00b4222568098240ee135a'
    entries=[]
    for sid in b.GLOBAL_POOL:
        d=SRC/'oracles'/f'{sid:04d}'/'OUTER_006';z=np.load(d/'POINT.npz');r=b.read(d/'RESULT.json')
        assert np.array_equal(z['x'],x) and r['point_sha']==h and r['sid']==sid and q[sid]==r['qU_audited']
        oracle=b.Oracle(sid,OUT/'inheritance_validation');check,*_,qu=oracle.validate_point(x,z['y'],z['Pi']);oracle.close();assert abs(qu-q[sid])<1e-8
        entries.append({'sid':sid,'point':b.record(d/'POINT.npz'),'result':b.record(d/'RESULT.json'),'residual':check})
    assert all(q[i]==0 for i in range(400) if i not in b.GLOBAL_POOL)
    raw=np.load(SRC/'probability/OUTER_006/RAW_FINAL.npz');assert np.array_equal(raw['q'],q) and np.array_equal(raw['p'],ev['p'])
    risk=b.risk_upper(q,float(raw['beta0']),raw['plus'],raw['minus']);saved=b.read(SRC/'probability/OUTER_006/RISK_UPPER.json')
    assert abs(risk['risk_upper']-saved['risk_upper'])<1e-8 and saved['risk_upper']==rec['risk_upper']
    first=np.load(b.R41/'identity/FIRST_ARRAYS.npz');_,f=b.enclosure(sp.csr_matrix(first['cost'][None,:]),x)
    assert float(f[0])==rec['F_upper'] and rec['audited_UB']==13444.76574120264
    cert=np.load(SRC/'probability/OUTER_006/STRICT_CUT/PROBABILITY.npz');assert b.probability_check(cert['pstar'])['passed']
    write(directory/'BEST_EVALUATION_INHERITANCE.json',{'passed':True,'status':'NUMERICALLY_AUDITED','value':rec['audited_UB'],'source':b.record(SRC/'outer_006/EVALUATION.json'),'point_sha':h,'F_upper':rec['F_upper'],'risk_upper':rec['risk_upper'],'strict_UB':None,'same_point93':entries})
    return rec['audited_UB']

def worker():
    global scenario_fps,prob_fps
    start=time.monotonic();deadline=start+LIMIT
    write(OUT/'BEGIN.json',{'epoch':time.time(),'monotonic':start,'budget':LIMIT,'historical_scientific_calls':{'RESUME_03':103,'RESUME_04':915},'prior_formal_wall':898.1556149451062,'starting_phase':'READD_CONTINUATION'})
    state,last,sol,accepted,q,recovered=recover(OUT/'RECOVERY')
    scenario_fps={b.row_fingerprint(state['A'].getrow(i),state['rhs'][i],state['sense'][i]) for i,n in enumerate(state['names']['rows']) if b.scenario_name(n)}
    scenario_fps|={b.row_fingerprint(state['dA'].getrow(i),state['drhs'][i],state['dsense'][i]) for i in range(state['dA'].shape[0])}
    prob_fps={b.row_fingerprint(state['A'].getrow(i),state['rhs'][i],state['sense'][i]) for i,n in enumerate(state['names']['rows']) if b.probability_name(n)}
    best=state['LB'];initial=best;last_snapshot=b.freeze_solved(last,sol,best)
    model,v=b.build_master(state,OUT/'INITIAL.log');workers=None;prob=None
    counters={'new_Q_dispatched':0,'new_Q_complete':0,'complete_outer':0,'recovered_tail_closed':False,'attempted_outer':0,'attempted_probability_episodes':0,'certified_probability_episodes':0,'master_calls':0,'readded':0,'recompressed':0}
    phase='READD_CONTINUATION';status='RUNNING';audited=best_evaluation_audit(OUT/'INHERITED_BEST');prob_wall=0.;history=[];raw=sol['x'];first=True
    try:
        for outer in range(9,30):
            if deadline-time.monotonic()<60:status='BUDGET_NO_NEW_BATCH';break
            od=OUT/f'outer_{outer:03d}';od.mkdir();counters['attempted_outer']+=1
            if outer>=10:
                if workers is None:workers=b.Workers(b.GLOBAL_POOL,OUT)
                if prob is None:prob=Probability(OUT/'probability')
                phase='POINT_REPAIR';accepted,pa=b.point_repair(raw[:b.FIRST_N],od/'POINT_REPAIR',deadline)
                phase='NEW_Q';counters['new_Q_dispatched']+=1
                results=workers.run(accepted,f'OUTER_{outer:03d}',deadline)
                assert len(results)==93 and {r['sid'] for r in results}==set(b.GLOBAL_POOL)
                counters['new_Q_complete']+=1;q=np.zeros(400)
                for rr in results:
                    q[rr['sid']]=rr['qU_audited'];z=np.load(rr['cut']);alpha=float(z['alpha']);beta=z['beta']
                    violation=float(np.longdouble(alpha)+np.asarray(beta,np.longdouble)@np.asarray(raw[:b.FIRST_N],np.longdouble)-raw[b.FIRST_N+b.GLOBAL_POOL.index(rr['sid'])])
                    if violation>1e-6:b.add_scenario_cut(state,model,v,rr['sid'],alpha,beta,f'r44_benders/{rr["sid"]}/{outer}',scenario_fps)
            if outer>=10:
                phase='PROBABILITY';counters['attempted_probability_episodes']+=1;t=time.monotonic()
                try:p,risk,strict,rec=prob.solve(q,f'OUTER_{outer:03d}',deadline)
                finally:prob_wall+=time.monotonic()-t
                counters['certified_probability_episodes']+=1
                b.add_probability_cut(state,model,v,p,strict['epsilon'],f'r44_probability/{outer}',prob_fps)
                fz=np.load(b.R41/'identity/FIRST_ARRAYS.npz');_,fu=b.enclosure(sp.csr_matrix(fz['cost'][None,:]),accepted)
                current_ub=float(np.nextafter(float(fu[0])+risk['risk_upper'],np.inf));audited=current_ub if audited is None else min(audited,current_ub)
                save_npz(od/'EVALUATION.npz',accepted_x=accepted,q=q,p=p)
                write(od/'EVALUATION.json',{'accepted_x_sha':hashlib.sha256(accepted.tobytes()).hexdigest(),'F_upper':float(fu[0]),'risk_upper':risk['risk_upper'],'audited_UB':current_ub,'strict_UB':None,'q_source':'new_complete_Q','remaining_307_Q_zero':True})
            phase='MASTER';closed=False
            for j in range(20):
                result,best,current_lb=solve_master(state,model,v,od/f'MASTER_{j:03d}',deadline,best,first);first=False;counters['master_calls']+=1
                last_snapshot=b.freeze_solved(state,result,current_lb)
                local,viol=b.scan_dormant(state['dA'],state['drhs'],state['dsense'],result['x'])
                save_npz(od/f'SCAN_{j:03d}.npz',readd_local=local,upper_violation=viol)
                if not len(local):closed=True;break
                phase='READD'
                b.append_rows(state,model,v,state['dA'][local],state['drhs'][local],state['dsense'][local],[state['dnames'][i] for i in local])
                keep=np.ones(len(state['dnames']),bool);keep[local]=False
                state['dA']=state['dA'][keep].tocsr();state['drhs']=state['drhs'][keep];state['dsense']=state['dsense'][keep];state['dnames']=[n for i,n in enumerate(state['dnames']) if keep[i]];state['dsource']=[n for i,n in enumerate(state['dsource']) if keep[i]]
                counters['readded']+=len(local)
            if not closed:raise TimeoutError('POOL_PARTIAL')
            phase='CHECKPOINT'
            model,v,result,recomp=b.maybe_recompress(state,model,v,result,outer)
            if recomp['executed']:counters['recompressed']+=recomp['candidate_rows']
            current_lb,_=b.lagrangian(state['A'],state['rhs'],state['sense'],state['cost'],state['lo'],state['hi'],result['Pi'])
            last_snapshot=b.freeze_solved(state,result,current_lb)
            checkpoint=od/'CHECKPOINT';b.export_state(checkpoint,state,last_snapshot);verified=verify_export(checkpoint,state)
            acceptance={'pool_closed':True,'checkpoint_full_readback':verified,'best_LB':best,'current_LB':current_lb,'recovered_historical_tail':outer==9}
            write(od/'OUTER_ACCEPTANCE.json',acceptance)
            if outer==9:
                counters['recovered_tail_closed']=True;write(OUT/'RECOVERED_TAIL_ACCEPTANCE.json',acceptance)
            else:counters['complete_outer']+=1
            raw=result['x']
            history.append({'outer':outer,'wall':time.monotonic()-start,'best_LB':best,'audited_UB':audited,'nnz':state['A'].nnz,'active_scenario':sum(b.scenario_name(n) for n in state['names']['rows']),'dormant':len(state['dnames'])})
            write(OUT/'LIVE.json',dict(history[-1],counters=counters))
            if audited is not None and 0<=audited-best<=1:
                status='NUMERICAL_TARGET_REACHED_CERTIFICATION_PENDING';break
        else:status='BATCH_LIMIT'
        if status=='RUNNING':status='BUDGET'
    except TimeoutError:
        status='PARTIAL_BUDGET';write(OUT/'PARTIAL.json',{'phase':phase,'traceback':traceback.format_exc()})
    except OSError:
        status='INFRASTRUCTURE_FAILED';write(OUT/'FAILURE.json',{'phase':phase,'traceback':traceback.format_exc()})
    except Exception:
        status='CERTIFICATE_OR_EXECUTION_FAILED';write(OUT/'FAILURE.json',{'phase':phase,'traceback':traceback.format_exc()})
    finally:
        stopped=time.monotonic()
        if workers is not None:workers.close()
        if prob is not None:prob.close()
        for p in OUT.glob('outer_*/MASTER_*/LOWER_CERTIFICATE.json'):
            best=max(best,b.read(p)['best_LB'])
        b.export_state(OUT/'LAST_SOLVED',last_snapshot['state'],last_snapshot)
        b.export_state(OUT/'WORKING_PENDING',state)
        model.dispose();calls=b.optimizer_ledger_summary()
        counters['native_probability_calls']=calls['by_type'].get('PROBABILITY_ROWGEN',0)
        ledger=[json.loads(x) for x in (OUT/'OPTIMIZER_CALL_LEDGER.jsonl').read_text().splitlines()]
        native={kind:sum(x.get('runtime',0) for x in ledger if x['phase']=='END' and x['call_type']==kind) for kind in calls['by_type']}
        write(OUT/'STAGE_TIMING.json',{'main_process_measured_inclusive_stages':TIMES,'native_runtime_by_type':native,'parallel_native_not_wall':True,'stages_overlap_do_not_sum_as_wall':True,'serialization_is_measured_not_residual_wall':True})
        complete={'status':status,'phase':phase,'counters':counters,'calls':calls,'historical_calls':{'RESUME_03':103,'RESUME_04':915},'cumulative_calls':1018+calls['explicit_begin_count'],'LB':best,'LB_gain':best-initial,'audited_UB':audited,'strict_UB':None,'history':history,'probability_wall':prob_wall,'scientific_flow_wall':stopped-start,'cleanup_wall':time.monotonic()-stopped,'wall':time.monotonic()-start,'budget':LIMIT,'active_nnz':state['A'].nnz,'production_signals_sent':[]}
        write(OUT/'COMPLETE.json',complete);write(OUT/'END.json',{'epoch':time.time(),'status':status})
    if status.endswith('FAILED'):raise SystemExit(1)

scenario_fps=set();prob_fps=set()

def legacy_preflight_not_used():
    global OUT
    root=OUT;test=root/f'PREFLIGHT_{len(list(root.glob("PREFLIGHT_*")))+1:02d}';test.mkdir(exist_ok=False)
    b.OUT=test
    # Reuse formal execution-chain tests with optimize intercepted.
    t=time.monotonic()
    result=b.preflight()
    b.OUT=root
    if not result['passed']:raise RuntimeError('BASE_PREFLIGHT_FAILED')
    OUT=test/'REAL_RECOVERY';b.OUT=OUT;OUT.mkdir()
    original=gp.Model.optimize
    def forbidden(*args,**kwargs):raise RuntimeError('PREFLIGHT_OPTIMIZER_FORBIDDEN')
    gp.Model.optimize=forbidden;b.PREFLIGHT_MODE=True
    try:
        state,last,sol,point,q,results=recover(OUT/'RECOVERY')
        p0=np.load(b.R42/'PROBABILITY_INTERIOR_STRICT/P_STAR.npz')['p']
        check=b.probability_check(p0);assert check['passed']
        cert=certify(p0,np.array(b.GLOBAL_QBARS),b.GLOBAL_POOL,OUT/'REFERENCE_CERTIFICATE')
        # Persist the same raw saver used by Probability.solve, then inject an
        # exception and confirm the completed payload remains readable.
        raw_path=OUT/'FAULT_INJECTION/RAW_FINAL.npz'
        save_npz(raw_path,p=p0,q=q,plus=np.zeros(45139),minus=np.zeros(45139))
        try:raise RuntimeError('INJECTED_AFTER_RAW_SAVE')
        except RuntimeError:assert np.array_equal(np.load(raw_path)['p'],p0)
        overwrite=False;outside=False
        try:save_npz(raw_path,p=p0)
        except FileExistsError:overwrite=True
        try:save_npz(root.parent/'outside.npz',p=p0)
        except ValueError:outside=True
        assert overwrite and outside
        write(OUT/'RESULT.json',{'passed':True,'real_anchor':check,'certificate':cert,'cache_reuse':93,'pending88_verified':True,'raw_survives_later_failure':True,'overwrite_rejected':True,'outside_rejected':True,'scientific_optimizer_calls':0})
    finally:
        gp.Model.optimize=original;b.PREFLIGHT_MODE=False;OUT=root;b.OUT=root
    write(root/'PREFLIGHT_RESULT.json',{'passed':True,'wall':time.monotonic()-t,'test_directory':str(test),'files':[b.record(p) for p in sorted((root/'tools').glob('*.py'))],'real_recovery':b.record(test/'REAL_RECOVERY/RESULT.json'),'simulation_not_scientific_progress':True})

def inventory():
    paths,missing=b.required_inputs();before=b.protection_snapshot(paths)
    checks={'inputs_complete':not missing,'prior_lease_released':b.read(SRC/'execution/main/LEASE.json')['owner'] is None}
    old=b.read(SRC/'execution/main/PROCESS.json')
    for kind in ('parent','child'):checks[kind+'_exited']=not b.same_identity(b.proc(old[kind]['pid']),old[kind])
    prior=b.read(SRC/'PROTECTION_AFTER.json')['DET'];now=before['DET']
    checks['DET_identity']=b.same_identity(prior['actual'],now['actual']) and prior['boot_id']==now['boot_id']
    checks['DET_model']=prior['model']['sha256']==now['model']['sha256']
    checks['static_inputs']=before['input_hashes']==b.read(SRC/'PROTECTION_AFTER.json')['input_hashes']
    checks['source_sealed']=all(b.sha(Path(r['path']))==r['sha256'] for r in b.read(SRC/'SEALED_MANIFEST.json')['files'])
    mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
    checks['resources']=mem['MemAvailable']>=32*1024**3
    write(OUT/'SAFETY_GATE.json',{'passed':all(checks.values()),'checks':checks,'MemAvailable':mem['MemAvailable'],'reserved_bytes':24*1024**3,'production_signals_sent':[]})
    write(OUT/'PROTECTION_BEFORE.json',before)
    if not all(checks.values()):raise RuntimeError('RESUME04_SAFETY_GATE_FAILED')
    return before,paths

def preflight():
    global OUT
    root=OUT;test=root/'PREFLIGHT';test.mkdir(exist_ok=True)
    start=time.monotonic()
    helper=root/'tools/prepare_continuation_seed.py'
    rc=b.owned_run([str(b.PYTHON),'-B',str(helper),'--source',str(SRC),'--output',str(test/'SEED')],test/'SEED_EXECUTION')
    if rc:raise RuntimeError('CONTINUATION_SEED_FAILED')
    original=gp.Model.optimize
    def forbidden(*args,**kwargs):raise RuntimeError('PREFLIGHT_OPTIMIZER_FORBIDDEN')
    gp.Model.optimize=forbidden;b.PREFLIGHT_MODE=True
    OUT=test/'ENTRY';OUT.mkdir();b.OUT=OUT
    try:
        state,last,sol,*_=recover(OUT/'RECOVERY')
        seed=np.load(test/'SEED/PENDING_BASIS_SEED.npz')
        assert np.array_equal(seed['VBasis'],state['VBasis']) and np.array_equal(seed['CBasis'],state['CBasis'])
        audited=best_evaluation_audit(OUT/'BEST')
        p0=np.load(b.R42/'PROBABILITY_INTERIOR_STRICT/P_STAR.npz')['p'];c=b.probability_check(p0);assert c['passed']
        assert list(range(10,30))==list(range(10,10+20))
        write(OUT/'ENTRY_TESTS.json',{'passed':True,'seed_full_match':True,'audited_UB':audited,'p0_check':c,'new_global_outer_range':[10,29],'starting_phase':'READD_CONTINUATION','no_optimizer_calls':True,'no_R43_restore_called':True})
    finally:
        gp.Model.optimize=original;b.PREFLIGHT_MODE=False;OUT=root;b.OUT=root
    write(root/'PREFLIGHT_RESULT.json',{'passed':True,'wall':time.monotonic()-start,'source_files':[b.record(p) for p in sorted((root/'tools').glob('*.py'))],'test_result':b.record(test/'ENTRY/ENTRY_TESTS.json')})

def parent():
    if (OUT/'BEGIN.json').exists() or (OUT/'execution/main').exists():raise RuntimeError('NO_SECOND_SCIENTIFIC_ATTEMPT')
    before,paths=inventory()
    preflight()
    frozen={str(p):b.sha(p) for p in (OUT/'tools').glob('*.py')}
    write(OUT/'SOURCE_FROZEN.json',{'hashes':frozen,'passed':True})
    command=[str(b.PYTHON),'-B',str(Path(__file__)),'worker']
    rc=b.owned_run(command,OUT/'execution/main')
    after=b.protection_snapshot(paths)
    after['DET_identity_unchanged']=b.same_identity(before['DET']['actual'],after['DET']['actual'])
    after['static_inputs_unchanged']=before['input_hashes']==after['input_hashes']
    after['source_frozen_unchanged']=all(b.sha(Path(p))==h for p,h in frozen.items())
    after['SRC_sealed_unchanged']=all(b.sha(Path(r['path']))==r['sha256'] for r in b.read(SRC/'SEALED_MANIFEST.json')['files'])
    write(OUT/'PROTECTION_AFTER.json',after)
    wait=b.read(OUT/'execution/main/WAIT.json');complete=b.read(OUT/'COMPLETE.json') if (OUT/'COMPLETE.json').exists() else {'status':'FAILED_BEFORE_LOOP','calls':b.optimizer_ledger_summary()}
    write(OUT/'FINAL_EXECUTION.json',{'prior_RESUME03_04_formal_wall':898.1556149451062,'new_budget':LIMIT,'new_wall':wait['elapsed'],'cumulative_formal_wall':898.1556149451062+wait['elapsed'],'preflight_wall':b.read(OUT/'PREFLIGHT_RESULT.json')['wall'],'WAIT':wait,'scientific':complete,'automatic_CONTINUE02':False})
    (OUT/'FINAL_REPORT_ZH.md').write_text('R44 CONTINUE_01\n\n'+json.dumps(b.read(OUT/'FINAL_EXECUTION.json'),indent=2)+'\n生产保护：'+json.dumps({k:v for k,v in after.items() if k.endswith('unchanged')})+'\n无整数DRO、无full400新recourse；无新公平A/B，不作一般加速倍数声明。')
    b.sealed_manifest(OUT)
    if rc:raise SystemExit(rc)

if __name__=='__main__':
    try:
        if len(sys.argv)>1 and sys.argv[1]=='worker':worker()
        else:parent()
    except Exception:
        write(OUT/('ENTRY_FAILURE_'+str(os.getpid())+'.json'),{'traceback':traceback.format_exc(),'automatic_resume':False})
        raise
