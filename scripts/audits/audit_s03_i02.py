#!/usr/bin/env python3
"""Audit saved S03-I02 inputs, raw witnesses and certificates without solving.

Only independent physical/probability checkers and historical hash utilities
are imported. No generator, model assembler, or optimizer is called. The MIP
bound is a recorded solver bound, not an independently reconstructed MIP tree.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import re
from types import SimpleNamespace
import numpy as np
from scipy.sparse import load_npz
from evcs_experiments.config import get_paths
from evcs_experiments.evidence import sha256, historical_path
from evcs_experiments.first_stage_check import check_first_stage
from evcs_experiments.recourse_check import check_recourse
from evcs_experiments.probability_check import check_probability_solution

_spec = importlib.util.spec_from_file_location('s03_i01_audit_helpers', get_paths().project_root/'scripts/audits/audit_s03_i01.py')
old = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(old)
need, read, close, maximum, canonical = old.need, old.read, old.close, old.maximum, old.canonical
I01_SHA = 'a42357640bd50cd65d2f801b965250e839e8f969789b000c73e10f89faab5112'
# Independently fixed hand optima and fixture definitions; never trust expected in inputs.json.
GOLDENS = {
 'three_ev_moment': {'H':3,'T':2,'loads':[4.,8.,12.],'act':8.,'constants':[1.,2.,3.],
   'values':{'SAA':(30.266666666666666,2),'DRO':(32.12,3),'ARO':(34.2,3)}},
 'two_ev_fixed_probability': {'H':2,'T':3,'loads':[0.,4.,8.,8.],'act':18.,'constants':[0.]*4,
   'values':{'SAA':(39.,1),'DRO':(39.,1),'ARO':(43.2,2)}},
 'four_ev_simplex': {'H':4,'T':2,'loads':[0.,4.,8.,12.,16.,16.,8.,4.],'act':8.,'constants':[0.]*8,
   'values':{'SAA':(35.,2),'DRO':(41.6,4),'ARO':(41.6,4)}}}
ORDER = ('rav','p','q','short','shed','line_p','line_q','u','soc_dep')


def namespace(d):
    return SimpleNamespace(**{k:np.asarray(v) if isinstance(v,list) else v for k,v in d.items()})


def npz(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k].item() if z[k].ndim==0 else z[k].copy() for k in z.files}


def witness(path):
    return namespace(npz(path))


def flatten(y):
    return np.concatenate([np.asarray(getattr(y,k),dtype=float).ravel() for k in ORDER])


def phase1(folder, expected):
    phi,lo,hi,ids=expected; raw=npz(folder/'inputs.npz'); r=read(folder/'solution.json')
    for k,v in (('features',phi),('lower',lo),('upper',hi)):
        close(raw[k],v,'Phase-I accepted input changed: '+k,0.)
    need(list(raw['support_ids'])==list(ids) and tuple(raw['feature_ids'])==old.FEATURES,'Phase-I axes changed')
    ah=canonical({'support_ids':list(ids),'feature_ids':list(old.FEATURES),'features':phi.tolist(),
      'lower':lo.tolist(),'upper':hi.tolist(),'feature_orientation':'scenario_by_feature','dual_convention':'alpha_lower_beta_upper'})
    need(r['ambiguity_sha256']==ah and r['support_ids']==list(ids) and r['feature_ids']==list(old.FEATURES),'Phase-I identity changed')
    p=old.array(r['probabilities'],(len(ids),),'phase-I p'); mean=phi.T@p
    violation=max(maximum(-p),abs(p.sum()-1),maximum(lo-mean),maximum(mean-hi))
    need(violation<=1e-8,'Phase-I original probability constraints failed')
    indices=np.flatnonzero(p!=0); need(r['initial_support_indices']==indices.tolist(),'Phase-I dropped nonzero mass')
    need(r['initial_support_ids']==[ids[i] for i in indices],'Phase-I subset identity mismatch')
    sub=old.array(r['initial_probabilities'],(len(indices),),'initial p'); lift=np.zeros(len(p));lift[indices]=sub
    need(np.array_equal(lift,p),'Phase-I clipped/renormalized/lost tiny mass')
    need(max(abs(sub.sum()-1),maximum(-sub),maximum(lo-phi[indices].T@sub),maximum(phi[indices].T@sub-hi))<=1e-8,'Restricted witness infeasible')
    need(r['witness_sha256']==canonical({'ambiguity_sha256':ah,'support_ids':list(ids),'probabilities':p.tolist()}),'Phase-I witness hash changed')
    rawsolver=r['solver']; close(rawsolver['raw_solution'],p,'Phase-I raw p changed',0.)
    # Independently inspect the numerical active system; zeros are EXACT zeros.
    mag=np.maximum(1.,np.maximum(np.maximum(abs(lo),abs(hi)),abs(phi).T@abs(p)))
    active_tol=64*np.finfo(float).eps*mag
    rows=[np.ones(len(indices))]
    for j in range(phi.shape[1]):
        if abs(mean[j]-lo[j])<=active_tol[j]:rows.append(phi[indices,j])
        if abs(mean[j]-hi[j])<=active_tol[j]:rows.append(phi[indices,j])
    mat=np.asarray(rows);mat=mat/np.maximum(1.,abs(mat).max(axis=1))[:,None]
    singular=np.linalg.svd(mat,compute_uv=False);ranktol=max(mat.shape)*np.finfo(float).eps*max(singular,default=0)
    rank=int(np.count_nonzero(singular>ranktol)); fullrank=len(p)-len(indices)+rank
    need(np.all(p>=0) and fullrank==len(p),'Phase-I numerical basicness not established')
    cert=r['certificate'];need(r['feasible'] and r['basis_verified'] and r['status']=='FEASIBLE_BASIC','Phase-I reported status differs')
    need(cert['basis']['full_active_rank']==fullrank and cert['support_threshold']==0 and not cert['probabilities_clipped_or_renormalized'],'Phase-I certificate claims changed')
    saved={'status':rawsolver['status'],'success':rawsolver['success'],'raw_solution_scaled':rawsolver['raw_solution'],
      'raw_objective_scaled':rawsolver['raw_objective'],'options':rawsolver['options']}
    for group in ('ineqlin','eqlin','lower','upper'):saved[group+'_marginals_scaled']=rawsolver[group+'_marginals']
    kkt=old.raw_lp(saved,np.zeros(len(p)),np.vstack((phi.T,-phi.T)),np.r_[hi,-lo],np.ones((1,len(p))),np.ones(1),np.zeros(len(p)),np.full(len(p),np.inf),p)
    return {'support':len(p),'initial_support':len(indices),'full_active_rank':fullrank,'max_original_violation':float(violation),'raw_LP':kkt}


def model_digest(model,A,d):
    meta={'risk_model':model['risk_model'],'scenario_ids':model['scenario_ids'],'scenario_sha256s':model['scenario_sha256s'],
      'template_sha256s':model['recourse_template_sha256s'],'first_stage_sha256':model['first_stage_sha256'],
      'ambiguity_sha256':model['ambiguity_sha256'],'empirical_probabilities':model['empirical_probabilities'],
      'slices':model['variable_slices'],'objective_constant_usd':model['objective_constant_usd']}
    h=hashlib.sha256(json.dumps(meta,sort_keys=True,separators=(',',':')).encode())
    for a in (A.data,A.indices,A.indptr,d['row_lower'],d['row_upper'],d['lower_bounds'],d['upper_bounds'],d['c'],d['integrality'],np.array(model['scenario_cost_constants_usd'])):
        v=np.asarray(a,dtype='<i8' if np.asarray(a).dtype.kind in 'iu' else '<f8');h.update(str(v.shape).encode());h.update(v.tobytes())
    return h.hexdigest()


def cost_vector(case,tasks,solution):
    cost={k:np.zeros_like(getattr(solution,k),dtype=float) for k in ORDER}
    cost['p'][:]=(tasks.c_v2g_usd_per_kwh+tasks.c_deg_usd_per_kwh/tasks.eta_dis)[:,None]*case.dt_h
    cost['short'][:]=case.c_short_usd_per_kwh*case.dt_h
    cost['shed'][:]=case.c_shed_usd_per_kwh[None,:]*case.dt_h
    return np.concatenate([cost[k].ravel() for k in ORDER])


def fixed_lp(A,b,ae,be,lower,upper,c,z,record):
    cert=record['certificate'];n=len(z)
    need(record['optimal'] and record['solver_status']==0 and cert['optimal_certificate_valid'],'Fixed LP not certified optimal')
    y=old.array(cert['dual_inequality_marginals'],(len(b),),'LP inequality dual')
    lam=old.array(cert['dual_equality_marginals'],(len(be),),'LP equality dual')
    lm=old.array(cert['dual_lower_marginals'],(n,),'LP lower dual');um=old.array(cert['dual_upper_marginals'],(n,),'LP upper dual')
    primal=max(maximum(A@z-b),maximum(abs(ae@z-be)),maximum(lower-z),maximum(z-upper))
    stationarity=maximum(abs(c-A.T@y-ae.T@lam-lm-um));sign=max(maximum(y),maximum(-lm),maximum(um))
    fl,fu=np.isfinite(lower),np.isfinite(upper)
    infinite=max(maximum(abs(lm[~fl])),maximum(abs(um[~fu])))
    primalobj=float(c@z);dual=float(b@y+be@lam+lower[fl]@lm[fl]+upper[fu]@um[fu]);gap=abs(primalobj-dual)
    tol=1e-4+1e-7*max(1.,abs(primalobj),abs(dual))
    need(primal<=1e-7 and max(stationarity,sign,infinite)<=1e-8 and gap<=tol,'Fixed LP raw KKT failed')
    for key,val in (('dual_objective_usd',dual),('dual_bound_usd',dual),('primal_bound_usd',primalobj)):
        close(cert[key],val,'Fixed LP bound/objective claim changed',tol)
    close(record['objective_usd'],primalobj,'Fixed LP objective changed',tol)
    need(cert['options']['threads']==1,'Fixed LP thread policy changed')
    return {'primal_residual':primal,'stationarity':stationarity,'gap_usd':gap}


def audit_toy(folder,risk):
    inp=read(folder/'inputs.json');name=folder.name;g=GOLDENS[name];H,T=g['H'],g['T'];S=len(g['loads'])
    need(inp['name']==name and len(inp['scenarios'])==S,'Golden identity changed')
    case,tasks=namespace(inp['case']),namespace(inp['tasks']);scenes=[namespace(v) for v in inp['scenarios']]
    need(len(tasks.task_ids)==H and list(case.periods)==list(range(1,T+1)) and list(case.bus_ids)==[1] and len(case.line_ids)==0,'Golden geometry changed')
    for field,val in (('p_dis_kw',4.),('p_block_kw',1.),('eta_dis',1.),('c_act_usd',g['act']),('c_v2g_usd_per_kwh',.15),('c_deg_usd_per_kwh',.05)):
        close(getattr(tasks,field),np.full(H,val),'Golden hardware/cost changed: '+field,0.)
    close(tasks.c_cap_usd_per_kwh,np.full((H,T),.1),'Golden capacity cost changed',0.)
    close(case.c_shed_usd_per_kwh,[3.],'Golden shed cost changed',0.)
    for scene,load in zip(scenes,g['loads']):
        close(scene.load_p_kw,np.full((T,1),load),'Golden load changed',0.);close(scene.load_q_kvar,np.zeros((T,1)),'Golden Q changed',0.)
        need(scene.development and scene.split=='development_fixture' and np.all(scene.task_mu_full),'Golden source scope changed')
    constants=np.asarray(inp['scenario_cost_constants_usd']);close(constants,g['constants'],'Golden diagnostic constants changed',0.)
    ids=inp['scenario_ids'];need(ids==[s.scenario_id for s in scenes] and ids==[f'{name}/scenario_{i+1:02d}' for i in range(S)],'Golden scenario axes changed')
    a=namespace(inp['ambiguity']);pi=np.asarray(inp['empirical_probabilities']);close(pi,np.full(S,1/S),'Golden pi changed',0.)
    if name=='three_ev_moment': phi=np.array([[0.],[1.],[2.]]);lo=np.array([.8]);hi=np.array([1.2])
    elif name=='two_ev_fixed_probability':phi=np.vstack((np.eye(3),np.zeros(3)));lo=hi=np.full(3,.25)
    else:phi=np.empty((8,0));lo=hi=np.empty(0)
    close(a.features,phi,'Golden features changed',0.);close(a.lower,lo,'Golden lower changed',0.);close(a.upper,hi,'Golden upper changed',0.)
    d=folder/risk;model=read(d/'model.json');sol=read(d/'solver.json');arr=npz(d/'model_arrays.npz');raw=npz(d/'solution.npz');A=load_npz(d/'matrix.npz').tocsr()
    z=old.array(raw['vector'],(A.shape[1],),'full extensive vector');sl=model['variable_slices'];nfirst=H+H*T
    need(model_digest(model,A,arr)==model['model_sha256']==sol['solution']['model_sha256'],'Extensive model identity changed')
    need(model['scenario_ids']==ids and model['task_ids']==list(tasks.task_ids) and model['risk_model']==risk,'Model axes changed')
    expected_integer=np.zeros(len(z),int);expected_integer[:H]=1
    need(np.array_equal(arr['integrality'],expected_integer),'Only assignment x may be integer')
    row=A@z;viol=max(maximum(arr['row_lower']-row),maximum(row-arr['row_upper']),maximum(arr['lower_bounds']-z),maximum(z-arr['upper_bounds']),maximum(abs(z[:H]-np.rint(z[:H]))))
    need(viol<=1e-8,'Extensive raw row/bound/integer violation')
    plan=SimpleNamespace(x=raw['x'],r=raw['r'],task_sha256=raw['task_sha256'],config_hash=raw['config_hash'])
    close(plan.x,z[:H],'Plan x differs from full vector',0.);close(plan.r,z[H:nfirst].reshape(H,T),'Plan r differs from full vector',0.)
    fc=check_first_stage(case,tasks,plan);need(fc['passed'],'Independent first-stage failed');fpre=fc['costs']['Fpre_usd']
    expected_c=np.zeros(len(z));expected_c[:H]=tasks.c_act_usd
    expected_c[H:nfirst]=(tasks.c_cap_usd_per_kwh*case.dt_h).ravel()
    if risk=='DRO':
        expected_c[sl['theta0'][0]]=1.;expected_c[slice(*sl['alpha'])]=-lo;expected_c[slice(*sl['beta'])]=hi
    elif risk=='ARO':expected_c[sl['risk_threshold'][0]]=1.
    q=[];ycost=[];lp_reports=[];components=[]
    for i,scene in enumerate(scenes):
        sd=d/f'scenario_{i}';yw=witness(sd/'extensive_witness.npz');yf=witness(sd/'fixed_plan_solution.npz');ys=slice(*sl['recourses'][ids[i]])
        close(flatten(yw),z[ys],'Saved extensive recourse differs from raw vector',0.)
        cw=check_recourse(case,tasks,scene,plan,yw,check_objective_bound=False);cf=check_recourse(case,tasks,scene,plan,yf,check_objective_bound=True)
        need(cw['passed'] and cf['passed'],'Independent direct physics failed')
        components.append(cf['costs'])
        cost=cost_vector(case,tasks,yf);fixed=flatten(yf);q.append(cf['costs']['Qop']+constants[i]);ycost.append(cw['costs']['Qop']+constants[i])
        if risk=='SAA':expected_c[ys]=pi[i]*cost
        need(ycost[-1]>=q[-1]-1e-7,'Extensive witness below certified recourse minimum')
        blocks={b['kind']:b for b in model['block_map'] if b['scope']==ids[i]}
        if risk!='SAA':
            riskrow=blocks['risk_epigraph'];need(riskrow['row_stop']==riskrow['row_start']+1,'Risk row size changed')
            expected_row=np.zeros(len(z));expected_row[ys]=cost
            if risk=='DRO':
                expected_row[sl['theta0'][0]]=-1.;expected_row[slice(*sl['alpha'])]=phi[i];expected_row[slice(*sl['beta'])]=-phi[i]
            else:expected_row[sl['risk_threshold'][0]]=-1.
            close(A[riskrow['row_start'],:].toarray().ravel(),expected_row,'Risk epigraph coefficients changed',0.)
            need(arr['row_upper'][riskrow['row_start']]==-constants[i] and np.isneginf(arr['row_lower'][riskrow['row_start']]),'Risk RHS constant/sign changed')
        ub=slice(blocks['recourse_ub']['row_start'],blocks['recourse_ub']['row_stop']);eq=slice(blocks['recourse_eq']['row_start'],blocks['recourse_eq']['row_stop'])
        b=arr['row_upper'][ub]-A[ub,:nfirst]@z[:nfirst];be=arr['row_upper'][eq]-A[eq,:nfirst]@z[:nfirst]
        need(np.array_equal(arr['row_lower'][eq],arr['row_upper'][eq]),'Recourse equality rows changed')
        lp_reports.append(fixed_lp(A[ub,ys],b,A[eq,ys],be,arr['lower_bounds'][ys],arr['upper_bounds'][ys],cost,fixed,read(sd/'fixed_plan_solver.json')))
    close(arr['c'],expected_c,'Full objective coefficients differ from physical/risk costs',0.)
    q=np.array(q);ycost=np.array(ycost);prob=read(d/'fixed_plan_probability.json')
    check=check_probability_solution(a,q,prob,q_support_ids=ids);need(check.passed,'True-q probability witness failed')
    old.audit_raw_probability_lp(phi,lo,hi,q,prob)
    rv={'SAA':float(pi@q),'DRO':float(prob['dual_objective_usd']),'ARO':float(max(q))}[risk]
    upper=fpre+rv;known,selected=g['values'][risk];tol=1e-4+1e-7*max(1.,abs(known),abs(upper))
    obj=float(arr['c']@z+arr['objective_constant_usd']);constant=float(arr['objective_constant_usd'])
    close(constant,float(pi@constants) if risk=='SAA' else 0.,'Objective constant changed',0.)
    rawmip=sol['raw_solver'];lower=float(rawmip['raw_lower_bound_without_constant_usd'])+constant
    need(rawmip['status']==0 and rawmip['success'] and rawmip['options']['threads']==1 and sol['optimal'],'Raw MIP not optimal')
    need(rawmip['raw_mip_gap']<=rawmip['options']['mip_rel_gap'],'Raw MIP gap failed')
    for val,label in ((obj,'matrix objective'),(upper,'true-q upper'),(sol['objective_usd'],'reported objective')):close(val,known,'Hand optimum mismatch: '+label,tol)
    close(obj,float(rawmip['raw_objective_without_constant_usd'])+constant,'Raw MIP objective differs',tol)
    close(sol['lower_bound_usd'],lower,'MIP lower bound restoration changed',tol)
    need(abs(obj-lower)<=tol,'Claimed optimal MIP incumbent and raw bound do not close')
    need(lower<=known+tol and known<=upper+tol and abs(plan.x.sum()-selected)<=1e-8,'Golden selection or LB<=opt<=UB failed')
    eval=read(d/'independent_plan_evaluation.json');close(eval['q_usd'],q,'True q changed',1e-8);close(eval['extensive_witness_costs_usd'],ycost,'Raw y costs changed',1e-8)
    if risk=='DRO':
        theta=z[sl['theta0'][0]];alpha=z[slice(*sl['alpha'])];beta=z[slice(*sl['beta'])]
        need(maximum(ycost-(theta+phi@(beta-alpha)))<=1e-8 and maximum(-alpha)<=1e-8 and maximum(-beta)<=1e-8,'Raw extensive distribution dual failed')
        close(obj,fpre+theta-lo@alpha+hi@beta,'Raw DRO objective signs changed',tol)
    elif risk=='ARO':
        threshold=z[sl['risk_threshold'][0]];need(maximum(ycost-threshold)<=1e-8,'ARO epigraph failed');close(obj,fpre+threshold,'ARO objective changed',tol)
    else:close(obj,fpre+pi@ycost,'SAA objective changed',tol)
    return {'case_id':name,'risk_model':risk,'known_optimum_usd':known,'objective_usd':obj,'lower_bound_usd':lower,'upper_bound_usd':upper,'max_matrix_violation':viol,'fixed_LP_certificates':lp_reports,'nonbinding_witness_costs_not_assumed_optimal':True,'scenario_ids':ids,'q_usd':q.tolist(),'objective_components':components,'Fpre_usd':fpre}


def audit_i01(source):
    ref=source['s03_i01_acceptance'];need(ref['sha256']==I01_SHA,'Unpinned I01 prerequisite')
    card=read(ref['path'],I01_SHA);need(card['status']=='PASS_INCREMENT' and card['increment']=='S03-I01' and not card['full_S03_pass'],'I01 scope changed')
    oldroot=Path(card['source_root']);evidence=historical_path(card['evidence'])
    artifacts=old.artifact_manifest(evidence,card['artifact_manifest_sha256'])
    old.source_manifest(oldroot,read(card['code_manifest']['path'],card['code_manifest']['sha256']))
    old.release_manifest(oldroot,card['current_source_manifest']['sha256']);old.verify_refs(card)
    read(evidence/'increment_validation.json',card['validation_sha256'])
    for path,digest in card['independent_audits'].items():need(read(path,digest)['status']=='PASS_INDEPENDENT_S03_I01_AUDIT','I01 audit did not pass')
    need(bool(card['reviews']),'I01 independent review missing')
    for name,digest in card['reviews'].items():need(sha256(historical_path(Path(ref['path']).parent/name))==digest,'I01 review changed')
    return {'acceptance_sha256':I01_SHA,'artifact_files':artifacts,'historical_source_root':str(oldroot)}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--evidence',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.evidence.resolve();out=args.output.resolve()
    need(not out.exists() and out!=root and root not in out.parents,'Audit output must be new and outside evidence')
    artifacts=old.artifact_manifest(root);paths=get_paths();ex=read(root/'execution.json')
    need(ex['paths']==paths.as_dict(),'Audit config/release differs from execution')
    need(ex['config_path'] and sha256(ex['config_path'])==ex['config_sha256'],'Execution config changed')
    need(sha256(paths.spec_root/'runtime_policy.json')==ex['runtime_policy_sha256'],'Runtime policy changed')
    sources=old.source_manifest(paths.project_root,read(root/'model_code_manifest.json'));release=old.release_manifest(paths.project_root)
    log=(root/'tests.stderr.log').read_text();m=re.search(r'Ran (\d+) tests? in',log)
    need(m and int(m[1])>=264 and log.rstrip().endswith('OK') and not re.search(r'skipp?ed|expected failure|unexpected success',log,re.I),'264+ tests must pass with no skips')
    v=read(root/'increment_validation.json')
    for k,value in {'increment':'S03-I02','status':'PASS_INCREMENT_AUTOMATED','full_S03_pass':False,'phase1_real_support_cases':32,'synthetic_datasets':3,'extensive_solutions':9,'fixed_plan_scenario_solutions':45,'new_production_scenarios':0,'formal_OOS_outcomes_generated':False,'moments_changed':False}.items():need(v[k]==value,'Increment coverage/scope differs: '+k)
    need(v['tests']==int(m[1]),'Test count changed')
    source=read(root/'sources.json');history_i01=audit_i01(source);targets,inputs,history=old.prerequisite_inputs(source)
    rows=read(root/'phase1_records.json');need(len(rows)==32 and {(r['target'],float(r['rho'])) for r in rows}==set(inputs),'Phase-I coverage changed')
    phase=[]
    for r in rows:phase.append(phase1(old.child(root,r['relative_path']),inputs[r['target'],float(r['rho'])]))
    records=read(root/'extensive_records.json');need(len(records)==9 and {(r['case_id'],r['risk_model']) for r in records}=={(n,r) for n in GOLDENS for r in ('SAA','DRO','ARO')},'Extensive coverage changed')
    toys=[audit_toy(root/'toys'/n,r) for n in GOLDENS for r in ('SAA','DRO','ARO')]
    indexed={(r['case_id'],r['risk_model']):r for r in toys}
    csv_summary=old.csv_rows(root/'extensive_reference_comparison.csv')
    need(len(csv_summary)==9 and len({(r['case_id'],r['risk_model']) for r in csv_summary})==9,'Summary CSV coverage changed')
    for row in csv_summary:
        checked=indexed[row['case_id'],row['risk_model']]
        for key,other in (('objective_usd','objective_usd'),('lower_bound_usd','lower_bound_usd'),('evaluated_upper_bound_usd','upper_bound_usd'),('known_optimum_usd','known_optimum_usd')):
            close(float(row[key]),checked[other],'Summary CSV value changed',1e-8)
    csv_cost=old.csv_rows(root/'objective_components.csv');seen=set()
    need(len(csv_cost)==45,'Cost CSV row count changed')
    for row in csv_cost:
        key=(row['case_id'],row['risk_model'],row['scenario_id']);need(key not in seen,'Duplicate cost CSV identity');seen.add(key)
        checked=indexed[key[:2]];i=checked['scenario_ids'].index(row['scenario_id'])
        for cost,value in checked['objective_components'][i].items():close(float(row[cost]),value,'Cost CSV physical component changed',1e-8)
        close(float(row['q_usd']),checked['q_usd'][i],'Cost CSV true q changed',1e-8)
        close(float(row['Fpre_usd']),checked['Fpre_usd'],'Cost CSV first stage changed',1e-8)
        close(float(row['external_diagnostic_constant_usd']),GOLDENS[row['case_id']]['constants'][i],'Cost CSV diagnostic constant changed',0.)
        need(row['Fpre_included_in_q']=='False','Fpre was counted inside recourse q')
    for name in GOLDENS:
        saa,dro,aro=(indexed[name,r]['objective_usd'] for r in ('SAA','DRO','ARO'))
        need(saa<=dro+1e-7 and dro<=aro+1e-7,'Golden empirical-feasible risk ordering failed')
    report={'status':'PASS_INDEPENDENT_S03_I02_AUDIT','increment':'S03-I02','full_S03_pass':False,'tests':int(m[1]),'artifact_files':artifacts,'source_files':sources,'release_files':release,'historical_I01':history_i01,'historical_S02_chain':history,'phase1_cases':32,'extensive_solutions':9,'fixed_plan_LP_certificates':45,'phase1':phase,'toys':toys,'new_scenarios_generated':0,'formal_OOS_outcomes_generated':False,'audit_source_sha256':sha256(__file__),'artifact_manifest_sha256':sha256(root/'artifact_sha256.json'),'limits':['Numerical active-rank check is not exact arithmetic BFS proof.','MIP lower bounds are preserved solver bounds; no independent MIP tree certificate is claimed.','C&CG, resume and full80EV/160support production trial remain outside I02.']}
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':report['status'],'output':str(out),'phase1_cases':32,'extensive_solutions':9,'fixed_plan_LP_certificates':45}));return 0


if __name__=='__main__':raise SystemExit(main())
