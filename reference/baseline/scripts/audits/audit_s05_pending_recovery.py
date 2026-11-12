"""Audit saved formal pending RMP/LP witnesses without calling any optimizer.

Full byte hashes include the large descriptive identity maps. Only their small
scientific header is decoded; template digests are rebuilt from saved matrices.
An audit success is neither a probability certificate nor permission to resume.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
from scipy.sparse import load_npz
from evcs_experiments.config import get_paths
from evcs_experiments.first_stage_check import check_first_stage
from evcs_experiments.recourse_check import check_recourse
from evcs_experiments.formal_physical_inputs import FormalPhysicalBank
from evcs_experiments.physical_inputs import load_physical_case, load_frozen_tasks
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.probability import ProbabilitySet

ORDER=('rav','p','q','short','shed','line_p','line_q','u','soc_dep')


def need(ok,message):
    if not bool(ok): raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(),parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))


def npz(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k].item() if z[k].ndim==0 else z[k].copy() for k in z.files}


def maximum(a): return float(np.max(a,initial=0.))


def close(a,b,label,tol=1e-8):
    a,b=np.asarray(a),np.asarray(b)
    need(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
         and maximum(abs(a-b))<=tol,label)


def same(a,b,label): need(np.array_equal(a,b),label)


def metadata_header(path,limit=4*1024*1024):
    """Decode the reviewed exporter's fields before descriptive variable_map.

    The caller must independently hash the complete file, including all maps.
    No numerical coefficient is stored only in the omitted descriptive maps.
    """
    with Path(path).open() as f:
        text=''
        while '\n  "variable_map":' not in text:
            part=f.read(8192)
            need(part and len(text)+len(part)<=limit,'Unknown/oversized identity header')
            text+=part
    return json.loads(text.split('\n  "variable_map":',1)[0].rstrip().removesuffix(',')+'\n}')


def template_digest(meta,matrices,arrays,first_sha):
    identity=dict(task_sha256=meta['task_sha256'],config_hash=meta['config_hash'],
        scenario_sha256=meta['scenario_sha256'],first_stage_sha256=first_sha,
        shapes=meta['variable_shapes'],task_ids=meta['task_ids'],bus_ids=meta['bus_ids'],
        line_ids=meta['line_ids'],station_ids=meta['station_ids'],periods=meta['periods'],
        references=meta['reference_bus_ids_by_period'],cmax_usd=meta['cmax_usd'])
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode())
    assembled={**{k.lower().replace('bx','x').replace('br','r'):v for k,v in matrices.items()},
               'b_ub':arrays['b_ub'],'b_eq':arrays['b_eq']}
    for key in sorted(assembled):
        value=assembled[key];values=(value.data,value.indices,value.indptr) if hasattr(value,'indptr') else (value,)
        digest.update(key.encode())
        for array in values:
            a=np.asarray(array,dtype='<i8' if array.dtype.kind in 'iu' else '<f8')
            digest.update(str(a.shape).encode());digest.update(a.tobytes())
    for key in ('lower_bounds','upper_bounds','component_objectives'):
        digest.update(np.asarray(arrays[key],dtype='<f8').tobytes())
    return digest.hexdigest()


def model_digest(model,A,d):
    meta=dict(risk_model=model['risk_model'],scenario_ids=model['scenario_ids'],scenario_sha256s=model['scenario_sha256s'],
        template_sha256s=model['recourse_template_sha256s'],first_stage_sha256=model['first_stage_sha256'],
        ambiguity_sha256=model['ambiguity_sha256'],empirical_probabilities=model['empirical_probabilities'],
        slices=model['variable_slices'],objective_constant_usd=model['objective_constant_usd'])
    h=hashlib.sha256(json.dumps(meta,sort_keys=True,separators=(',',':')).encode())
    for a in (A.data,A.indices,A.indptr,d['row_lower'],d['row_upper'],d['lower_bounds'],d['upper_bounds'],d['c'],d['integrality'],np.array(model['scenario_cost_constants_usd'])):
        v=np.asarray(a,dtype='<i8' if np.asarray(a).dtype.kind in 'iu' else '<f8')
        h.update(str(v.shape).encode());h.update(v.tobytes())
    return h.hexdigest()


def audit_kkt(m,v,plan,z,dual,solver,fixed):
    """Original-coefficient primal/dual feasibility and gap, no solver call."""
    cert=solver['certificate']
    need(solver['status']=='OPTIMAL' and solver['optimal'] is True and solver['solver_status']==0,'LP is not OPTIMAL')
    y,lam,lo,hi=[np.asarray(dual[k]) for k in ('inequality','equality','lower','upper')]
    need(all(np.isfinite(a).all() for a in (z,y,lam,lo,hi)),'Nonfinite LP primal/dual')
    for k,a in zip(('inequality','equality','lower','upper'),(y,lam,lo,hi)):
        same(a,np.asarray(cert['dual_'+k+'_marginals']),'Saved dual differs from original solver')
    bu=v['b_ub']+m['Bx_ub']@plan.x+m['Br_ub']@plan.r.ravel()
    be=v['b_eq']+m['Bx_eq']@plan.x+m['Br_eq']@plan.r.ravel()
    same(bu,fixed['b_ub'],'Fixed RHS ub changed');same(be,fixed['b_eq'],'Fixed RHS eq changed')
    lb,ub=v['lower_bounds'],v['upper_bounds'];fl,fu=np.isfinite(lb),np.isfinite(ub)
    slack=bu-m['A_ub']@z;eq=m['A_eq']@z-be
    stationarity=v['c']-m['A_ub'].T@y-m['A_eq'].T@lam-lo-hi
    primal=max(maximum(-slack),maximum(abs(eq)),maximum(lb-z),maximum(z-ub))
    station=maximum(abs(stationarity));sign=max(maximum(y),maximum(-lo),maximum(hi))
    infinite=max(maximum(abs(lo[~fl])),maximum(abs(hi[~fu])))
    pobj=float(v['c']@z);dobj=float(bu@y+be@lam+lb[fl]@lo[fl]+ub[fu]@hi[fu])
    gap=abs(pobj-dobj);tol=1e-4+1e-7*max(1.,abs(pobj),abs(dobj))
    need(primal<=1e-7 and max(station,sign,infinite)<=1e-8 and gap<=tol,'Original LP KKT failed')
    complementarity=dict(inequality=y*slack,lower=lo[fl]*(z-lb)[fl],upper=hi[fu]*(ub-z)[fu])
    for key,a in dict(stationarity=stationarity,inequality_slack=slack,equality_residual=eq,
        **{k+'_complementarity':a for k,a in complementarity.items()}).items():
        close(dual[key],a,'Saved KKT arithmetic differs: '+key,1e-8)
    for k,x in (('dual_objective_usd',dobj),('dual_bound_usd',dobj),('primal_bound_usd',pobj)):
        close(cert[k],x,'Claimed LP bound differs: '+k,tol)
    close(solver['objective_usd'],pobj,'Claimed Q differs',tol)
    need(cert['options']['threads']==1,'LP thread policy differs')
    return dict(primal_residual=primal,stationarity=station,dual_sign_violation=sign,
        infinite_bound_dual_violation=infinite,duality_gap_usd=gap,primal_objective_usd=pobj,dual_objective_usd=dobj,
        maximum_complementarity=max(maximum(abs(a)) for a in complementarity.values()))


def rmp_residuals(A,v,z):
    rows=A@z;integers=np.asarray(v['integrality'])!=0
    row=max(maximum(v['row_lower']-rows),maximum(rows-v['row_upper']))
    bound=max(maximum(v['lower_bounds']-z),maximum(z-v['upper_bounds']))
    integer=maximum(abs(z[integers]-np.rint(z[integers])))
    need(max(row,bound,integer)<=1e-8,'Original RMP primal feasibility failed')
    return dict(max_row_violation=row,max_bound_violation=bound,max_integrality_violation=integer)


class Audit:
    def __init__(self,wall_seconds):
        self.started=time.monotonic();self.wall=wall_seconds;self.refs={};self.bytes=0
    def remaining(self):
        if time.monotonic()-self.started>=self.wall: raise TimeoutError('Pending evidence audit wall exhausted')
    def ref(self,path):
        self.remaining();p=Path(path).resolve(strict=True);h=hashlib.sha256()
        with p.open('rb') as f:
            while chunk:=f.read(4*1024*1024):self.remaining();h.update(chunk)
        self.bytes+=p.stat().st_size
        return dict(path=str(p),sha256=h.hexdigest(),size_bytes=p.stat().st_size)
    def verify(self,row):
        expected=dict(path=row['path'],sha256=row['sha256'],size_bytes=row.get('size_bytes',row.get('bytes')))
        need(Path(row['path']).is_absolute(),'Relative evidence reference')
        actual=self.refs.get(row['path'])
        if actual is None:actual=self.ref(row['path']);self.refs[row['path']]=actual
        need(actual==expected,'Original bytes changed: '+row['path']);return actual
    def record(self,row):
        self.verify(row);p=Path(row['path']);value=read(p)
        expected={str(x.resolve()) for x in p.parent.iterdir() if x.is_file() and x.name!='record.json'}
        need({x['path'] for x in value['artifacts']}==expected,'Raw artifact inventory differs')
        for ref in value['artifacts']:self.verify(ref)
        return value
    def write(self,path,value):
        with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def audit(request,expected_request_sha256,output,wall_seconds=590,diagnostic_first_atoms=None):
    need(math.isfinite(wall_seconds) and 0<wall_seconds<=7190,'Invalid audit wall budget')
    a=Audit(wall_seconds);rr=a.ref(request);need(rr['sha256']==expected_request_sha256,'Audit request changed')
    a.verify(rr);req=read(request);refs=req['references'];paths=get_paths()
    source=Path(refs['source_manifest']['path']).resolve().parent;planroot=Path(req['plan_root']).resolve();raw=planroot/'raw'
    a.verify(refs['raw_request']);raw_preview=read(refs['raw_request']['path'])
    summary_ref=raw_preview['input_binding']['summary'];a.verify(summary_ref)
    out=Path(output).resolve();bankroot=Path(summary_ref['path']).resolve().parent
    cardroot=Path(refs['bank_acceptance']['path']).resolve().parent
    need(source==paths.project_root.resolve(),'Installed source differs from pinned original release')
    need(not any(out.is_relative_to(p) for p in (planroot,source,paths.data_root.resolve(),paths.frozen_source_root.resolve(),bankroot,cardroot)),'Audit output overlaps protected evidence')
    out.mkdir(parents=True,exist_ok=False);completed=[];current=None;rmp_report=None
    try:
        for ref in refs.values():a.verify(ref)
        need(a.ref(__file__)==refs['audit_script'],'Audit executable is not request-bound')
        manifest=read(refs['source_manifest']['path']);sourcepaths={}
        for member in manifest['files']:
            p=(source/member['path']).resolve();need(p.is_relative_to(source),'Source path escape')
            ref=dict(path=str(p),sha256=member['sha256'],size_bytes=member['size_bytes']);a.verify(ref);sourcepaths[str(p)]=ref
        original_ref=refs.get('original_source_manifest',refs['source_manifest'])
        original_root=Path(original_ref['path']).resolve().parent
        need(original_root==source,'Audit must use the original installed package')
        original_sources={}
        for member in read(original_ref['path'])['files']:
            p=(original_root/member['path']).resolve();need(p.is_relative_to(original_root),'Original source path escape')
            a.verify(dict(path=str(p),sha256=member['sha256'],size_bytes=member['size_bytes']))
            if member['path'].startswith('src/evcs_experiments/') and member['path'].endswith('.py'):
                original_sources[member['path'][len('src/evcs_experiments/'):]]=member['sha256']
        current_sources={str(Path(p).relative_to(source/'src/evcs_experiments')):r['sha256'] for p,r in sourcepaths.items()
                         if Path(p).is_relative_to(source/'src/evcs_experiments') and p.endswith('.py')}
        changed={n for n in set(original_sources)|set(current_sources) if original_sources.get(n)!=current_sources.get(n)}
        need(set(original_sources)==set(current_sources) and not changed,'Audit runtime differs from original source')
        engine_names=('ccg.py','extensive.py','first_stage.py','recourse.py','probability.py','probability_phase1.py',
                      'random_streams.py','lazy_recourses.py','zero_presolve.py','native_highs.py','native_gurobi.py')
        job=read(refs['job_result']['path']);need(job['complete'] is True and job['state']=='FAILED','Expected actual terminal failure')
        cp=read(refs['checkpoint']['path']);state=cp['state'];pending=state['pending'];history=read(refs['history']['path'])
        need(refs['checkpoint']['path']==str(planroot/'driver/checkpoint.json') and refs['history']['path']==str(raw/'history.json'),'Wrong pending origin')
        need(cp['sha256']==canonical_hash(state) and state['problem_sha256']==canonical_hash(state['identity']),'Checkpoint semantic hash changed')
        need(state['engine_sha256']==canonical_hash({n:original_sources[n] for n in engine_names}),'Original engine source differs')
        need(state['status']=='PROBABILITY_FAILED' and state['trace']==[] and state['best'] is None and state['lower_bound_usd'] is None and state['upper_bound_usd'] is None,'Unexpected algorithm state')
        need(pending['iteration']==1 and len(pending['q_usd'])==len(pending['recourse_certificates'])==160,'Full pending q vector required')
        rawreq=read(refs['raw_request']['path']);need(rawreq['problem_identity']==state['identity'] and rawreq['domain']=='formal' and rawreq['formal_OOS'] is False,'Raw/checkpoint identity differs')
        binding=rawreq['input_binding'];card=read(refs['bank_acceptance']['path'])
        need(rawreq['exporter_sha256']==original_sources['formal_planning.py'] and
             binding['planning_source']=={n:s for n,s in original_sources.items() if '/' not in n},'Original exporter/source binding differs')
        need(card['status']=='PASS_COMPONENT' and card['component_id']=='FORMAL_BASELINE_INPUTS','Original bank component missing')
        need(card['summary_reference']==binding['summary'] and card['audit_reference']==binding['audit'],'Bank acceptance/physical source differs')
        for key,value in card.items():
            if key.endswith('_reference'):a.verify(value)
        for key in ('summary','audit','task_source','formal_protocol_lock','bank_source_manifest'):a.verify(binding[key])
        row=binding['target'];a.verify(row['planning_index']);index=read(row['planning_index']['path'])
        need(row['target_event']==req['target_id']=='michael_2018' and row['H']==1008 and row['planning_count']==160,'Wrong formal target/H/support')
        ids=[x['scenario_id'] for x in index['scenarios']]
        need(len(ids)==len(set(ids))==160 and ids==state['identity']['scenario_ids'],'Complete ordered support differs')
        need(index['development'] is False and index['split']=='planning' and index['lead_h']==1 and index['regime']=='R1','Non-formal or changed input scope')
        case=load_physical_case(paths.data_root);tasks=load_frozen_tasks(binding['task_source']['path'],case,binding['task_source']['sha256'])
        need(len(tasks.task_ids)==1008 and tuple(index['task_ids'])==tasks.task_ids,'Fixed H/order differs')
        bank=FormalPhysicalBank(row['planning_index']['path'],case,tasks,expected_index_sha256=row['planning_index']['sha256'],case_root=paths.data_root)
        firstrow=a.record(a.ref(raw/'first_stage/record.json'));firstmeta=metadata_header(raw/'first_stage/identity.json')
        need(firstmeta['matrix_sha256']==state['identity']['first_stage_sha256'],'First-stage source differs')
        firstA=load_npz(raw/'first_stage/matrix.npz').tocsr();firstv=npz(raw/'first_stage/arrays.npz')
        plan=SimpleNamespace(x=np.asarray(pending['plan']['x']),r=np.asarray(pending['plan']['r']),task_sha256=pending['plan']['task_sha256'],config_hash=pending['plan']['config_hash'])
        planhash=canonical_hash(pending['plan']);firstcheck=check_first_stage(case,tasks,plan)
        need(firstcheck['passed'],'Independent first-stage physical check failed')
        need([r['path'] for r in history]==[str(p) for p in sorted((raw/'events').glob('*/record.json'))],'Raw event inventory/order differs')
        records=[read(r['path']) for r in history]
        for ref in history:a.verify(ref)
        need([r['event'] for r in records]==['phase1','rmp']+['recourse']*160+['probability'],'Unexpected incomplete callback inventory')
        phase=a.record(history[0]);rmp=a.record(history[1]);probability=a.record(history[-1])
        need(phase['iteration']==0 and rmp['iteration']==probability['iteration']==1,'Callback iteration differs')
        rd=Path(history[1]['path']).parent;model=read(rd/'model.json');solver=read(rd/'solver.json');mv=npz(rd/'model_arrays.npz');sol=npz(rd/'solution.npz');A=load_npz(rd/'matrix.npz').tocsr();z=sol['vector']
        need(model_digest(model,A,mv)==model['model_sha256']==pending['rmp_model_sha256'],'RMP coefficient semantic hash differs')
        need(solver==pending['rmp'],'Pending RMP differs from raw result')
        active=pending['active_indices'];need(active==state['active_indices']==[67,87,98,137],'Initial support differs')
        need(model['scenario_ids']==[ids[i] for i in active] and model['risk_model']=='DRO','RMP support/risk differs')
        H,T=plan.r.shape;nf=H+H*T;slices=model['variable_slices'];same(z[:nf],np.r_[plan.x,plan.r.ravel()],'RMP plan differs')
        same(sol['x'],plan.x,'RMP x changed');same(sol['r'],plan.r,'RMP reserve changed')
        rmp_report=rmp_residuals(A,mv,z)
        firstblock=model['block_map'][0];nfr=firstblock['row_stop']
        need((A[:nfr,:nf]-firstA).nnz==0 and A[:nfr,nf:].nnz==0,'RMP first-stage coefficients differ')
        for key in ('row_lower','row_upper'):same(mv[key][:nfr],firstv[key],'First-stage row bound differs')
        for key in ('lower_bounds','upper_bounds','c','integrality'):same(mv[key][:nf],firstv[key],'First-stage variable data differs')
        constants=np.asarray(state['identity']['constants_usd']);need(constants.shape==(160,),'Scenario constants scope')
        probdir=Path(history[-1]['path']).parent;probinput=read(probdir/'probability_input.json')['effective_ambiguity'];phi=np.asarray(probinput['features']);lo=np.asarray(probinput['lower']);hi=np.asarray(probinput['upper'])
        qinput=read(probdir/'q_input.json');need(qinput['support_ids']==ids and qinput['q_usd']==pending['q_usd'] and qinput['plan']==pending['plan'],'Original probability q input differs')
        need(qinput['recourse_evidence']==history[2:-1],'Probability q evidence references differ')
        need(phi.shape==(160,5) and probinput['ambiguity_sha256']==state['identity']['effective_ambiguity_sha256'],'Probability feature identity differs')
        need(probinput['support_ids']==ids,'Probability support axis differs')
        ambiguity=ProbabilitySet(tuple(ids),tuple(probinput['feature_ids']),phi,lo,hi)
        need(ambiguity.ambiguity_sha256==probinput['ambiguity_sha256'],'Probability matrix content digest differs')
        theta=slices['theta0'][0];alpha=slice(*slices['alpha']);beta=slice(*slices['beta'])
        dual=dict(theta0=float(z[theta]),alpha=z[alpha].tolist(),beta=z[beta].tolist(),threshold=None)
        need(dual==pending['raw_rmp_dual'],'Original RMP dual witness differs')
        expectedc=np.zeros_like(z);expectedc[:nf]=firstv['c'];expectedc[theta]=1;expectedc[alpha]=-lo;expectedc[beta]=hi
        same(mv['c'],expectedc,'RMP DRO objective coefficients differ')
        need(np.isneginf(mv['lower_bounds'][theta]) and np.isposinf(mv['upper_bounds'][theta]),'RMP intercept no longer free')
        for sl in (alpha,beta):need(np.all(mv['lower_bounds'][sl]==0) and np.isposinf(mv['upper_bounds'][sl]).all(),'DRO sign/bound changed')
        objective=float(mv['c']@z+mv['objective_constant_usd']);close(objective,solver['objective_usd'],'RMP objective differs',1e-4+1e-7*abs(objective))
        need(solver['status']=='TIME_LIMIT' and solver['optimal'] is False and solver['lower_bound_usd'] is None,'Do not promote the actual limited RMP')
        need(solver['raw_solver']['raw_lower_bound_without_constant_usd'] is None and solver['certificate']['solver_bound_usable'] is False,'Unexpected raw lower-bound scope')
        native=solver['raw_solver']['native_interface'];need(native['status']=='TIME_LIMIT' and native['raw_model_status']==9 and native['valid_incumbent'],'Native RMP status differs')
        rmp_report.update(status='TIME_LIMIT',objective_usd=objective,lower_bound_usd=None,solver_bound_usable=False,master_seconds=solver['wall_seconds'],physical_first_stage_cost_usd=firstcheck['costs']['Fpre_usd'])
        a.write(out/'rmp.json',rmp_report)
        a.write(out/'original_probability_failure.json',dict(status=pending['last_incomplete_probability']['status'],certified_optimal=pending['last_incomplete_probability']['certified_optimal'],solver_reference=a.verify(next(x for x in probability['artifacts'] if Path(x['path']).name=='solver.json')),promoted_to_PASS=False))
        count=160 if diagnostic_first_atoms is None else diagnostic_first_atoms
        need(type(count) is int and 1<=count<=160,'Invalid diagnostic atom count')
        for i in range(count):
            current=i;a.remaining();tick=time.monotonic();event=a.record(history[i+2]);d=Path(history[i+2]['path']).parent
            need(event['scenario_index']==i and event['scenario_id']==ids[i] and event['plan_sha256']==planhash and event['iteration']==1,'Recourse order/fixed plan differs')
            need(read(d/'plan.json')==pending['plan'],'A recourse changed x/r/identity')
            tr=read(d/'template_reference.json');template=a.record(tr);td=Path(tr['path']).parent
            need(template['scenario_index']==i and template['scenario_id']==ids[i],'Template addressing differs')
            meta=metadata_header(td/'identity.json');v=npz(td/'arrays.npz');m={k:load_npz(td/(k+'.npz')).tocsr() for k in ('A_ub','A_eq','Bx_ub','Bx_eq','Br_ub','Br_eq')}
            digest=template_digest(meta,m,v,firstmeta['matrix_sha256'])
            need(digest==template['template_sha256']==meta['template_sha256']==pending['recourse_certificates'][i]['template_sha256'],'Original template coefficient digest differs')
            need(meta['scenario_sha256']==index['scenarios'][i]['scenario_sha256'] and tuple(meta['task_ids'])==tasks.task_ids and meta['task_sha256']==tasks.task_sha256,'Template physical identity differs')
            atom=read(index['scenarios'][i]['manifest']['path']);a.verify(index['scenarios'][i]['manifest'])
            for ref in atom['files'].values():a.verify(ref)
            scene=bank.load(i,expected_scenario_sha256=index['scenarios'][i]['scenario_sha256'])
            solution=npz(d/'solution.npz');witness=SimpleNamespace(**solution);vector=npz(d/'primal_vector.npz')['vector']
            same(vector,np.concatenate([np.asarray(solution[k]).ravel() for k in ORDER]),'LP vector/physical witness differs')
            physical=check_recourse(case,tasks,scene,plan,witness,check_objective_bound=True)
            need(physical['passed'],'Independent original-unit LP physical check failed')
            lp=read(d/'solver.json');need(lp['certificate']['template_sha256']==digest,'LP solver template identity differs')
            cert=pending['recourse_certificates'][i]
            need(cert['index']==i and cert['scenario_id']==ids[i] and cert['solver_result_sha256']==canonical_hash(lp),'Pending LP raw solver identity differs')
            kkt=audit_kkt(m,v,plan,vector,npz(d/'dual_kkt.npz'),lp,npz(d/'fixed_rhs.npz'))
            close(physical['costs']['Qop'],kkt['primal_objective_usd'],'Independent physical Q differs',1e-4+1e-7*abs(kkt['primal_objective_usd']))
            need(event['q_usd']==pending['q_usd'][i]==lp['objective_usd']+constants[i],'Original q scalar changed')
            original_costs=np.zeros_like(v['component_objectives']);vs=meta['variable_slices'];dt=case.dt_h
            original_costs[0,slice(*vs['shed'])]=np.broadcast_to(case.c_shed_usd_per_kwh[None,:]*dt,solution['shed'].shape).ravel()
            original_costs[1,slice(*vs['short'])]=case.c_short_usd_per_kwh*dt
            original_costs[2,slice(*vs['p'])]=np.broadcast_to(tasks.c_v2g_usd_per_kwh[:,None]*dt,solution['p'].shape).ravel()
            original_costs[3,slice(*vs['p'])]=np.broadcast_to(tasks.c_deg_usd_per_kwh[:,None]*dt/tasks.eta_dis[:,None],solution['p'].shape).ravel()
            same(v['component_objectives'],original_costs,'LP costs differ from independent physical prices/units')
            same(v['c'],v['component_objectives'].sum(axis=0),'LP objective aggregation differs')
            if i in active:
                local=active.index(i);need(model['recourse_template_sha256s'][local]==digest and model['scenario_sha256s'][local]==meta['scenario_sha256'],'RMP template binding differs')
                ys=slice(*slices['recourses'][ids[i]])
                for key in ('lower_bounds','upper_bounds'):same(mv[key][ys],v[key],'RMP recourse bounds differ')
                for sense in ('ub','eq'):
                    block=next(b for b in model['block_map'] if b['scope']==ids[i] and b['kind']=='recourse_'+sense);rs=slice(block['row_start'],block['row_stop'])
                    need((A[rs,ys]-m['A_'+sense]).nnz==0 and (A[rs,:H]+m['Bx_'+sense]).nnz==0 and (A[rs,H:nf]+m['Br_'+sense]).nnz==0,'RMP physical blocks differ')
                    same(mv['row_upper'][rs],v['b_'+sense],'RMP recourse RHS differs')
                    same(mv['row_lower'][rs],v['b_eq'] if sense=='eq' else np.full(len(v['b_ub']),-np.inf),'RMP row sense differs')
                rw=SimpleNamespace(**npz(rd/f'witness_{i:03d}.npz'))
                same(z[ys],np.concatenate([np.asarray(getattr(rw,k)).ravel() for k in ORDER]),'RMP recourse vector differs')
                need(check_recourse(case,tasks,scene,plan,rw,check_objective_bound=False)['passed'],'Original RMP recourse physics failed')
                block=next(b for b in model['block_map'] if b['scope']==ids[i] and b['kind']=='risk_epigraph')
                expected=np.zeros_like(z);expected[ys]=v['c'];expected[theta]=-1;expected[alpha]=phi[i];expected[beta]=-phi[i]
                same(A[block['row_start']].toarray().ravel(),expected,'Original risk row differs')
                need(block['row_stop']==block['row_start']+1 and mv['row_upper'][block['row_start']]==-constants[i] and np.isneginf(mv['row_lower'][block['row_start']]),'Risk row constant/sense differs')
            completed.append(dict(index=i,scenario_id=ids[i],q_usd=pending['q_usd'][i],template_sha256=digest,physical_passed=True,**kkt,seconds=time.monotonic()-tick))
            a.write(out/f'lp_{i:03d}.json',completed[-1])
        residual=np.asarray(pending['q_usd'])-(dual['theta0']+phi@(np.asarray(dual['beta'])-np.asarray(dual['alpha'])))
        included=maximum(residual[active]);full=maximum(residual)
        a.write(out/'raw_rmp_dual_residuals.json',dict(support_ids=ids,residuals_usd=residual.tolist(),active_indices=active,
            included_maximum_usd=included,full_support_maximum_usd=full,separation_absolute_usd=state['config']['separation_absolute_usd'],separation_relative=state['config']['separation_relative'],raw_witness_unchanged=True))
        if count==160:
            expectedtemplates={str(raw/'templates'/f'{i:03d}'/'record.json') for i in range(160)}
            need(expectedtemplates=={str(p) for p in (raw/'templates').glob('*/record.json')},'Template coverage has extras/missing files')
            rawfiles={str(p.resolve()) for p in raw.rglob('*') if p.is_file()}
            need(rawfiles<=set(a.refs),'Unhashed original raw files remain')
        for name,module in tuple(sys.modules.items()):
            if name=='evcs_experiments' or name.startswith('evcs_experiments.'):
                filename=getattr(module,'__file__',None)
                if filename is not None:
                    p=str(Path(filename).resolve());need(p in sourcepaths,'Mixed/unbound imported source');a.verify(sourcepaths[p])
        a.remaining();status='PASS_SAVED_RMP_PRIMAL_AND_160_LP_AUDIT' if diagnostic_first_atoms is None else 'DIAGNOSTIC_PREFIX_ONLY_NOT_ACCEPTED'
        result=dict(status=status,completed_lp_count=len(completed),expected_lp_count=160,rmp=rmp_report,
            included_dual_violation_usd=included,full_support_dual_violation_usd=full,
            original_engine_status=state['status'],original_probability_status=pending['last_incomplete_probability']['status'],
            probability_audited=False,probability_PASS=False,recovery_authorized=False,scientific_acceptance=False,
            original_LB=None,original_UB=None,original_best=None,original_trace_count=0,formal_OOS=False,solver_calls=0,
            hashed_files=len(a.refs),hashed_bytes=a.bytes,wall_seconds=time.monotonic()-a.started)
        a.write(out/'input_references.json',list(a.refs.values()));a.write(out/'audit.json',result)
        a.write(out/'artifact_manifest.json',[a.ref(p) for p in sorted(out.iterdir()) if p.is_file()]);a.remaining()
        return result
    except Exception as exc:
        for name in ('audit.json','artifact_manifest.json'):
            if (out/name).exists():(out/name).rename(out/('unaccepted_'+name))
        result=dict(status='TIME_LIMIT_INCOMPLETE_AUDIT' if isinstance(exc,TimeoutError) else 'FAIL_PENDING_EVIDENCE_AUDIT',
            current_index=current,completed_lp_count=len(completed),error_type=type(exc).__name__,error=str(exc),
            probability_PASS=False,recovery_authorized=False,scientific_acceptance=False,solver_calls=0,formal_OOS=False)
        a.write(out/'audit.json',result)
        if not (out/'input_references.json').exists():a.write(out/'input_references.json',list(a.refs.values()))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request',type=Path,required=True);p.add_argument('--expected-request-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--wall-seconds',type=float,default=590)
    p.add_argument('--diagnostic-first-atoms',type=int)
    args=vars(p.parse_args());result=audit(**args);print(json.dumps(result,allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())
