#!/usr/bin/env python3
"""Replay stored C&CG matrices, physical LP certificates and history; never solve.

The accepted I02 complete matrices anchor the physical coefficients. This file
does not import the C&CG driver, an assembler, a generator, or an optimizer.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import re

import numpy as np
from scipy.sparse import load_npz

from evcs_experiments.config import get_paths
from evcs_experiments.evidence import sha256
from evcs_experiments.first_stage_check import check_first_stage
from evcs_experiments.recourse_check import check_recourse
from evcs_experiments.probability_check import check_probability_solution

spec = importlib.util.spec_from_file_location('i02_audit_helpers', get_paths().project_root/'scripts/audits/audit_s03_i02.py')
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
need, read, close, maximum, canonical = old.need, old.read, old.close, old.maximum, old.canonical
I02_SHA = '88e7ea6eac9f3f51109f7b478fab07b67f139631d1f1de41f74d3b34cc37596b'
SOURCE_SHA = 'a3bf57c838f64496d5a83cacdf30040484689443d1fc05aed0c19e3c58aad021'
ARTIFACT_SHA = '5e3f05e725fd12af6916443878c61f51ddf9c081580cdb635dbaf605f7b847b9'


def same(a, b, label):
    need(np.array_equal(np.asarray(a), np.asarray(b)), label)


def matrix_same(a, b, label):
    need(a.shape == b.shape and (a-b).nnz == 0, label)


def plan_dict(plan):
    return {'x':plan.x.tolist(), 'r':plan.r.tolist(), 'task_sha256':plan.task_sha256, 'config_hash':plan.config_hash}


def template_anchor(folder, reference, inputs):
    """Match every A,b,Bx,Br and bound to the accepted full physical matrix."""
    rm = read(reference/'DRO/model.json'); ra = old.npz(reference/'DRO/model_arrays.npz')
    A = load_npz(reference/'DRO/matrix.npz').tocsr()
    nf = rm['variable_slices']['first_stage'][1]
    h = len(inputs['tasks']['task_ids']); templates = []
    for i, sid in enumerate(inputs['scenario_ids']):
        d = folder/f'template_{i}'; v = old.npz(d/'arrays.npz'); identity = read(d/'identity.json')
        mats = {k:load_npz(d/(k+'.npz')).tocsr() for k in ('A_ub','A_eq','Bx_ub','Bx_eq','Br_ub','Br_eq')}
        need(identity == {'template_sha256':rm['recourse_template_sha256s'][i],
            'scenario_sha256':rm['scenario_sha256s'][i], 'scenario_id':sid}, 'Template identity differs from I02')
        ys = slice(*rm['variable_slices']['recourses'][sid])
        for sense in ('ub','eq'):
            block = next(b for b in rm['block_map'] if b['scope']==sid and b['kind']=='recourse_'+sense)
            rs = slice(block['row_start'],block['row_stop'])
            matrix_same(mats['A_'+sense],A[rs,ys], 'Physical A changed from I02')
            matrix_same(mats['Bx_'+sense],-A[rs,:h], 'Physical Bx changed from I02')
            matrix_same(mats['Br_'+sense],-A[rs,h:nf], 'Physical Br changed from I02')
            same(v['b_'+sense],ra['row_upper'][rs], 'Physical RHS changed from I02')
        same(v['lower_bounds'],ra['lower_bounds'][ys], 'Physical lower bound changed')
        same(v['upper_bounds'],ra['upper_bounds'][ys], 'Physical upper bound changed')
        templates.append({**v, **mats, **identity})
    return templates, (rm,ra,A)


def fixed_lp(folder, solver_name, solution_name, plan, scene, case, tasks, template):
    record = read(folder/solver_name); y = old.witness(folder/solution_name); z = old.flatten(y)
    checked = check_recourse(case,tasks,scene,plan,y,check_objective_bound=True)
    need(checked['passed'], 'Independent fixed-plan physics failed')
    c = old.cost_vector(case,tasks,y)
    same(c,template['c'],'LP objective differs from independent physical costs')
    b = template['b_ub']+template['Bx_ub']@plan.x+template['Br_ub']@plan.r.ravel()
    be = template['b_eq']+template['Bx_eq']@plan.x+template['Br_eq']@plan.r.ravel()
    certificate = old.fixed_lp(template['A_ub'],b,template['A_eq'],be,
        template['lower_bounds'],template['upper_bounds'],c,z,record)
    close(float(c@z),checked['costs']['Qop'],'Physical and raw LP cost differ',1e-8)
    # Bind the exact exported scalar after independently verifying its value.
    # A different summation order can differ by ulps and therefore by SHA256.
    return float(record['objective_usd']), certificate


def master(folder, indices, inputs, templates, reference, risk, *,
           expected_master_backend='highs', expected_optimizer_threads=1):
    model = read(folder/'model.json'); sol = read(folder/'solver.json')
    d = old.npz(folder/'model_arrays.npz'); zsaved = old.npz(folder/'solution.npz')
    A = load_npz(folder/'matrix.npz').tocsr(); z = np.asarray(zsaved['vector'])
    need(old.model_digest(model,A,d)==model['model_sha256']==sol['solution']['model_sha256'], 'RMP digest changed')
    ids = inputs['scenario_ids']; subset = [ids[i] for i in indices]
    need(model['scenario_ids']==subset and model['risk_model']==risk, 'RMP support/risk changed')
    rm,ra,refA = reference; nf = rm['variable_slices']['first_stage'][1]
    h = len(inputs['tasks']['task_ids']); sl = model['variable_slices']
    same(d['integrality'],np.r_[np.ones(h),np.zeros(len(z)-h)], 'Only x may be integer')
    for key in ('lower_bounds','upper_bounds','c'):
        same(d[key][:nf],ra[key][:nf], 'First-stage '+key+' changed')
    same(model['scenario_sha256s'],[templates[i]['scenario_sha256'] for i in indices],'RMP scene hash changed')
    same(model['recourse_template_sha256s'],[templates[i]['template_sha256'] for i in indices],'RMP template hash changed')
    blocks = model['block_map']; cursor = 0
    for block in blocks:
        need(block['row_start']==cursor,'RMP row coverage gap/overlap'); cursor=block['row_stop']
    need(cursor==A.shape[0] and len(blocks)==1+3*len(indices),'Unexpected RMP rows')
    nfr = rm['block_map'][0]['row_stop']; matrix_same(A[:nfr,:nf],refA[:nfr,:nf],'First-stage matrix changed')
    need(A[:nfr,nf:].nnz==0,'First-stage row depends on recourse')
    for key in ('row_lower','row_upper'): same(d[key][:nfr],ra[key][:nfr],'First-stage RHS changed')
    plan = old.namespace({k:zsaved[k] for k in ('x','r','task_sha256','config_hash')})
    same(np.r_[plan.x,plan.r.ravel()],z[:nf],'Saved plan differs from RMP vector')
    case,tasks = old.namespace(inputs['case']),old.namespace(inputs['tasks'])
    check = check_first_stage(case,tasks,plan); need(check['passed'],'RMP first-stage physics failed')
    fpre = check['costs']['Fpre_usd']; constants=np.asarray(inputs['scenario_cost_constants_usd'])
    phi = np.asarray(inputs['ambiguity']['features']) if risk=='DRO' else np.empty((len(ids),0))
    lo = np.asarray(inputs['ambiguity']['lower']) if risk=='DRO' else np.empty(0)
    hi = np.asarray(inputs['ambiguity']['upper']) if risk=='DRO' else np.empty(0)
    expected_c=np.zeros(len(z));expected_c[:nf]=ra['c'][:nf]
    intercept=sl['theta0'][0] if risk=='DRO' else sl['risk_threshold'][0]
    need(np.isneginf(d['lower_bounds'][intercept]) and np.isposinf(d['upper_bounds'][intercept]),'Risk intercept must be free')
    expected_c[intercept]=1.
    if risk=='DRO':
        expected_c[slice(*sl['alpha'])]=-lo;expected_c[slice(*sl['beta'])]=hi
        for name in ('alpha','beta'):
            same(d['lower_bounds'][slice(*sl[name])],np.zeros(len(lo)),'Risk multiplier sign changed')
            same(d['upper_bounds'][slice(*sl[name])],np.full(len(lo),np.inf),'Risk multiplier cap changed')
    for i in indices:
        sid=ids[i];template=templates[i];ys=slice(*sl['recourses'][sid])
        for key in ('lower_bounds','upper_bounds'):same(d[key][ys],template[key],'Recourse variable bounds changed')
        for sense in ('ub','eq'):
            b=next(b for b in blocks if b['scope']==sid and b['kind']=='recourse_'+sense)
            rs=slice(b['row_start'],b['row_stop'])
            matrix_same(A[rs,:h],-template['Bx_'+sense],'RMP Bx sign/coefficients')
            matrix_same(A[rs,h:nf],-template['Br_'+sense],'RMP Br sign/coefficients')
            matrix_same(A[rs,ys],template['A_'+sense],'RMP physical A changed')
            other=[j for j in range(nf,len(z)) if j not in range(ys.start,ys.stop)]
            need(A[rs,:][:,other].nnz==0,'Recourse row coupled to unrelated variables')
            same(d['row_upper'][rs],template['b_'+sense],'RMP original RHS changed')
            same(d['row_lower'][rs],template['b_eq'] if sense=='eq' else np.full(len(template['b_ub']),-np.inf),'RMP sense changed')
        y=old.witness(folder/f'witness_{i}.npz');same(old.flatten(y),z[ys],'RMP recourse witness changed')
        checky=check_recourse(case,tasks,old.namespace(inputs['scenarios'][i]),plan,y,check_objective_bound=False)
        need(checky['passed'],'RMP direct recourse physics failed')
        c=old.cost_vector(case,tasks,y);same(c,template['c'],'Template cost disagrees with physics')
        b=next(b for b in blocks if b['scope']==sid and b['kind']=='risk_epigraph')
        expected=np.zeros(len(z));expected[ys]=c;expected[intercept]=-1.
        if risk=='DRO':expected[slice(*sl['alpha'])]=phi[i];expected[slice(*sl['beta'])]=-phi[i]
        same(A[b['row_start']].toarray().ravel(),expected,'Risk row signs/coefficients changed')
        need(b['row_stop']==b['row_start']+1 and d['row_upper'][b['row_start']]==-constants[i]
             and np.isneginf(d['row_lower'][b['row_start']]),'Risk constant/sense changed')
    same(d['c'],expected_c,'Risk objective changed');need(d['objective_constant_usd']==0.,'Unexpected objective offset')
    violation=max(maximum(d['row_lower']-A@z),maximum(A@z-d['row_upper']),maximum(d['lower_bounds']-z),
                  maximum(z-d['upper_bounds']),maximum(abs(z[:h]-np.rint(z[:h]))))
    need(violation<=1e-8,'RMP raw feasibility failed')
    raw=sol['raw_solver'];need(raw['status']==0 and raw['success'],'RMP not successfully solved')
    need(expected_master_backend in ('highs','gurobi') and
         isinstance(expected_optimizer_threads,int) and not isinstance(expected_optimizer_threads,bool) and
         1<=expected_optimizer_threads<=4,'Invalid expected master backend/thread identity')
    need(raw['options']['threads']==expected_optimizer_threads,'RMP optimizer thread identity changed')
    if expected_master_backend=='highs':
        need(expected_optimizer_threads==1 and raw['solver'] in
             ('scipy.optimize.milp/HiGHS','SciPy-bundled HiGHS native/setSolution'),
             'Expected the one-thread HiGHS reference backend')
    else:
        native=raw['native_interface'];options=native['effective_options']
        need(raw['solver']=='gurobipy/Gurobi' and native['backend']=='gurobipy' and
             native['gurobi_version']==[12,0,3] and
             options['Threads']==expected_optimizer_threads==raw['options']['threads'],
             'Gurobi backend/version/thread identity changed')
        need(native['raw_model_status']==2 and native['status']=='OPTIMAL' and native['is_mip'] and
             native['valid_incumbent'] and native['objective_matches'] and
             native['solver_bound_consistent'] and native['bound_source']=='MIP_ObjBound',
             'Gurobi successful MIP certificate identity changed')
        for key in ('FeasibilityTol','OptimalityTol','IntFeasTol'):
            need(options[key]==1e-8,'Gurobi original tolerance changed: '+key)
        need(options['MIPGap']==raw['options']['mip_rel_gap'],'Gurobi MIP-gap option changed')
        close(native['incumbent_without_constant_usd'],raw['raw_objective_without_constant_usd'],
              'Gurobi raw objective identity changed')
        close(native['solver_bound_without_constant_usd'],raw['raw_lower_bound_without_constant_usd'],
              'Gurobi native bound identity changed')
    need(raw['raw_mip_gap']<=raw['options']['mip_rel_gap'],'Raw MIP gap failed')
    obj=float(d['c']@z);bound=float(raw['raw_lower_bound_without_constant_usd'])
    close(obj,sol['objective_usd'],'RMP reported objective');close(obj,raw['raw_objective_without_constant_usd'],'RMP raw objective')
    close(bound,sol['lower_bound_usd'],'RMP raw bound changed')
    need(bound<=obj+1e-4+1e-7*max(1.,abs(obj)),'RMP bound contradiction')
    dual={'theta0':None,'alpha':None,'beta':None,'threshold':None}
    if risk=='DRO':dual.update(theta0=float(z[intercept]),alpha=z[slice(*sl['alpha'])].tolist(),beta=z[slice(*sl['beta'])].tolist())
    else:dual['threshold']=float(z[intercept])
    return plan, fpre, dual, bound, sol


def probability(a,q,record,ids):
    check=check_probability_solution(a,q,record,q_support_ids=ids)
    need(check.passed,'Independent probability certificate failed')
    old.old.audit_raw_probability_lp(a.features,a.lower,a.upper,q,record)
    return float(record['dual_objective_usd'])


def checkpoint_check(value, final, expected_engine):
    state=value['state'];need(value['sha256']==canonical(state),'Checkpoint checksum failed')
    need(state['identity']==final['identity'] and state['problem_sha256']==canonical(state['identity']), 'Checkpoint problem identity changed')
    need(state['engine_sha256']==expected_engine and state['config']==final['config'],'Checkpoint source/config changed')
    need(state['pending'] is None,'Completed-iteration snapshot has pending evaluation')
    n=len(state['trace']);need(state['trace']==final['trace'][:n] and n>0,'Checkpoint trace is not accepted prefix')
    last=state['trace'][-1]
    need(state['next_iteration']==last['iteration']+1,'Checkpoint iteration address changed')
    close(state['lower_bound_usd'],last['lower_bound_usd'],'Checkpoint LB differs from history')
    close(state['upper_bound_usd'],last['upper_bound_usd'],'Checkpoint UB differs from history')
    need(state['best']['iteration']==last['best_iteration'],'Checkpoint best plan iteration changed')
    bestrow=next(r for r in state['trace'] if r['iteration']==last['best_iteration'])
    stored=state['best'];raw=bestrow['rmp']['solution']
    expected_plan={key:raw[key] for key in ('x','r','task_sha256','config_hash')}
    need(stored['plan']==expected_plan and stored['plan_sha256']==canonical(expected_plan),'Checkpoint best plan changed')
    need(stored['q_usd']==bestrow['q_usd'] and stored['probability']==bestrow['probability_audit']
         and stored['original_rmp_dual']==bestrow['raw_rmp_dual']
         and stored['original_rmp_residual_usd']==bestrow['residual_usd'],'Checkpoint best evidence rebound')
    close(stored['objective_usd'],bestrow['candidate_upper_bound_usd'],'Checkpoint best objective changed')
    close(state['elapsed_seconds'],sum(x['elapsed_seconds'] for x in state['sessions']),'Checkpoint cumulative time reset',1e-8)
    need(state['elapsed_seconds']>=last['elapsed_seconds'] and all(x['elapsed_seconds']<=x['wall_budget_seconds'] for x in state['sessions']), 'Checkpoint time invalid')
    active=list(last['active_indices'])
    if last['added_scenario_index'] is not None:active.append(last['added_scenario_index'])
    need(state['active_indices']==active,'Checkpoint support changed')
    return state


def audit_run(folder, risk, mode, reference_root, engine_sha, *,
              expected_master_backend='highs', expected_optimizer_threads=1):
    inputs=read(folder/'inputs.json');reference=reference_root/'toys'/folder.name
    need(inputs==read(reference/'inputs.json'),'Fixture changed from accepted I02')
    templates,anchor=template_anchor(folder,reference,inputs)
    d=folder/risk/mode;result=read(d/'result.json');events=read(d/'events.json')
    need(result['config'].get('master_backend','highs')==expected_master_backend and
         result['config'].get('optimizer_threads',1)==expected_optimizer_threads,
         'C&CG checkpoint backend/thread identity changed')
    case,tasks=old.namespace(inputs['case']),old.namespace(inputs['tasks']);ids=inputs['scenario_ids']
    a=old.namespace(inputs['ambiguity'])
    if risk=='ARO':a.features=np.empty((len(ids),0));a.lower=np.empty(0);a.upper=np.empty(0);a.feature_ids=[]
    a.support_ids=ids
    a.ambiguity_sha256=canonical({'support_ids':ids,'feature_ids':list(a.feature_ids),'features':a.features.tolist(),
        'lower':a.lower.tolist(),'upper':a.upper.tolist(),'feature_orientation':'scenario_by_feature','dual_convention':'alpha_lower_beta_upper'})
    need(result['status']=='CONVERGED' and result['pending'] is None,'Incomplete C&CG result')
    need(result['identity']['scenario_ids']==ids and result['identity']['risk_model']==risk,'C&CG identity changed')
    need(result['identity']['effective_ambiguity_sha256']==a.ambiguity_sha256,'Effective ambiguity changed')
    need(events[0]['event']=='phase1' and events[0]['iteration']==0,'Missing Phase-I event')
    phase=read(old.old.child(d,events[0]['path'])/'solver.json');p=np.asarray(phase['probabilities'])
    need(phase==result['phase1'] and phase['support_ids']==ids and phase['ambiguity_sha256']==a.ambiguity_sha256,'Phase-I binding changed')
    need(np.all(p>=0) and max(abs(p.sum()-1),maximum(a.lower-a.features.T@p),maximum(a.features.T@p-a.upper))<=1e-8,'Phase-I infeasible')
    active=np.flatnonzero(p!=0).tolist();same(active,phase['initial_support_indices'],'Phase-I dropped nonzero p')
    same(p[active],phase['initial_probabilities'],'Phase-I altered raw mass')
    lb=None;ub=None;best=None;cursor=1;lp_count=0;matrix_count=0;max_lp_gap=0.
    for k,row in enumerate(result['trace'],1):
        need(row['iteration']==k and row['active_indices']==active,'C&CG iteration/support progression changed')
        e=events[cursor];need(e['event']=='rmp' and e['iteration']==k,'Missing RMP event');cursor+=1
        md=old.old.child(d,e['path'])
        plan,fpre,dual,bound,raw=master(md,active,inputs,templates,anchor,risk,
            expected_master_backend=expected_master_backend,expected_optimizer_threads=expected_optimizer_threads)
        matrix_count+=1
        need(row['rmp']==raw and row['raw_rmp_dual']==dual and row['rmp_model_sha256']==raw['solution']['model_sha256'],'Trace changed original RMP/dual')
        need(row['plan_sha256']==canonical(plan_dict(plan)),'Trace plan identity mismatch')
        q=[]
        for i in range(len(ids)):
            e=events[cursor];cursor+=1
            need(e['event']=='recourse' and e['iteration']==k and e['scenario_index']==i,'Missing/reshuffled recourse event')
            rd=old.old.child(d,e['path']);need(read(rd/'plan.json')==plan_dict(plan),'Recourse plan changed')
            cost,cert=fixed_lp(rd,'solver.json','solution.npz',plan,old.namespace(inputs['scenarios'][i]),case,tasks,templates[i])
            q.append(cost+inputs['scenario_cost_constants_usd'][i]);lp_count+=1;max_lp_gap=max(max_lp_gap,cert['gap_usd'])
        q=np.asarray(q);e=events[cursor];cursor+=1
        need(e['event']=='probability' and e['iteration']==k,'Missing probability event')
        prob=read(old.old.child(d,e['path'])/'solver.json');need(row['probability_audit']==prob,'Trace probability witness changed')
        candidate=fpre+probability(a,q,prob,ids)
        residual=q-dual['threshold'] if risk=='ARO' else q-dual['theta0']+a.features@(np.asarray(dual['alpha'])-dual['beta'])
        close(row['q_usd'],q,'Trace q is not actual full-support recourse',1e-8)
        close(row['residual_usd'],residual,'Trace uses wrong RMP dual',1e-8)
        lb=bound if lb is None else max(lb,bound)
        if ub is None or candidate<ub:
            ub=candidate;best={'iteration':k,'plan':plan_dict(plan),'q':q.copy(),'probability':prob,'dual':dual,'residual':residual.copy(),'Fpre':fpre}
        known=old.GOLDENS[folder.name]['values'][risk][0]
        need(lb<=known+1e-7 and known<=ub+1e-7,'LB<=hand optimum<=UB failed')
        close(row['candidate_upper_bound_usd'],candidate,'Candidate UB changed');close(row['lower_bound_usd'],lb,'Historical LB changed');close(row['upper_bound_usd'],ub,'Historical UB changed')
        need(row['best_iteration']==best['iteration'],'Best-UB iteration changed')
        scale=max(1.,float(max(abs(q))));tol=1e-4+1e-8*scale;gap=max(0.,ub-lb)/max(1.,abs(ub))
        close(row['gap_relative'],gap,'Wrong historical gap');close(row['separation_scale_usd'],scale,'Wrong separation scale')
        close(row['separation_tolerance_usd'],tol,'Wrong separation tolerance')
        close(row['max_full_residual_usd'],maximum(residual),'Wrong full residual')
        close(row['max_included_residual_usd'],maximum(residual[active]),'Included residual omitted')
        need(maximum(residual[active])<=tol,'Included positive residual is a failure')
        omitted=[i for i in range(len(ids)) if i not in active]
        need(row['no_omitted_scenarios']==(not omitted),'Wrong omitted-set convention')
        new=None if not omitted else omitted[int(np.argmax(residual[omitted]))]
        if new is None:need(row['max_new_residual_usd'] is None,'Empty omitted set must be explicit')
        else:close(row['max_new_residual_usd'],residual[new],'Wrong omitted maximum')
        terminal=maximum(residual)<=tol and gap<=result['config']['optimality_relative_gap']
        expected_new=new if not terminal and new is not None and residual[new]>tol else None
        need(row['added_scenario_index']==expected_new,'Scenario addition differs from paper rule')
        if expected_new is not None:active.append(expected_new)
        need(terminal==(k==len(result['trace'])),'Convergence must satisfy BOTH current residual and historical gap')
        e=events[cursor];cursor+=1;need(e['event']=='iteration' and e['iteration']==k,'Missing iteration event')
        need(read(old.old.child(d,e['path'])/'trace.json')==row,'Saved trace copies differ')
    need(cursor==len(events),'Unaccounted solver events')
    b=result['best'];need(b['plan']==best['plan'] and b['iteration']==best['iteration'] and b['plan_sha256']==canonical(best['plan']),'Returned plan is not historical best')
    same(b['q_usd'],best['q'],'Returned best q changed');need(b['probability']==best['probability'] and b['original_rmp_dual']==best['dual'],'Best evidence rebound to another iteration')
    same(b['original_rmp_residual_usd'],best['residual'],'Original best-plan residual changed')
    close(b['objective_usd'],ub,'Best objective changed');close(b['Fpre_usd'],best['Fpre'],'Best Fpre changed')
    close(result['lower_bound_usd'],lb,'Final LB changed');close(result['upper_bound_usd'],ub,'Final UB changed')
    final_cp=checkpoint_check(read(d/'checkpoint.json'),result,engine_sha);need(final_cp==result,'Final checkpoint/result differ')
    previous=0.;snapshots=sorted(d.glob('checkpoint_session_*.json'),key=lambda p:int(p.stem.rsplit('_',1)[1]))
    need(len(snapshots)==len(result['sessions']),'Missing session checkpoints')
    for cp in snapshots:
        s=checkpoint_check(read(cp),result,engine_sha);need(s['elapsed_seconds']>=previous,'Elapsed time decreased');previous=s['elapsed_seconds']
    plan=old.namespace(best['plan']);q=[]
    for i in range(len(ids)):
        cost,cert=fixed_lp(d/'best_plan_recheck'/str(i),'solver.json','solution.npz',plan,
            old.namespace(inputs['scenarios'][i]),case,tasks,templates[i]);q.append(cost+inputs['scenario_cost_constants_usd'][i]);lp_count+=1;max_lp_gap=max(max_lp_gap,cert['gap_usd'])
    close(q,b['q_usd'],'Independent best-plan q differs',1e-8)
    auditprob=read(d/'best_plan_recheck/probability.json');auditub=best['Fpre']+probability(a,np.asarray(q),auditprob,ids)
    close(auditub,ub,'Independent best-plan UB differs',1e-7)
    close(auditub,old.GOLDENS[folder.name]['values'][risk][0],'Hand optimum mismatch',1e-7)
    return {'case_id':folder.name,'risk_model':risk,'mode':mode,'RMP_matrices':matrix_count,'fixed_LP_certificates':lp_count,
            'lower_bound_usd':lb,'algorithm_upper_bound_usd':ub,'audit_upper_bound_usd':auditub,'max_LP_gap_usd':max_lp_gap,'sessions':len(snapshots)}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--evidence',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.evidence.resolve();out=args.output.resolve();paths=get_paths()
    need(not out.exists() and root not in out.parents,'Audit output must be new and outside evidence')
    files=old.old.artifact_manifest(root,ARTIFACT_SHA)
    sources=old.old.source_manifest(paths.project_root,read(root/'model_code_manifest.json'))
    release=old.old.release_manifest(paths.project_root,SOURCE_SHA)
    ex=read(root/'execution.json');need(ex['paths']==paths.as_dict(),'Audit configuration differs from execution')
    need(sha256(ex['config_path'])==ex['config_sha256'] and sha256(paths.spec_root/'runtime_policy.json')==ex['runtime_policy_sha256'],'Execution config/policy changed')
    source=read(root/'sources.json');ref=source['s03_i02_acceptance'];need(ref['sha256']==I02_SHA,'Unpinned I02')
    card=read(ref['path'],I02_SHA);need(card['increment']=='S03-I02' and card['status']=='PASS_INCREMENT' and not card['full_S03_pass'],'Wrong I02 scope')
    reference=Path(card['evidence']);old.old.artifact_manifest(reference,card['artifact_manifest_sha256'])
    old.old.source_manifest(Path(card['source_root']),read(card['code_manifest']['path'],card['code_manifest']['sha256']))
    old.old.release_manifest(Path(card['source_root']),card['current_source_manifest']['sha256']);old.old.verify_refs(card)
    for name,digest in card['reviews'].items():need(sha256(Path(ref['path']).parent/name)==digest,'I02 review changed')
    for path,digest in card['independent_audits'].items():need(read(path,digest)['status']=='PASS_INDEPENDENT_S03_I02_AUDIT','I02 audit not accepted')
    old.audit_i01({'s03_i01_acceptance':source['inherited']['s03_i01_acceptance']})
    old.old.prerequisite_inputs({'inherited':source['inherited']})
    v=read(root/'increment_validation.json');need(v['increment']=='S03-I03' and v['status']=='PASS_INCREMENT_AUTOMATED' and v['tests']==280 and v['runs']==12 and not v['full_S03_pass'] and not v['formal_OOS_outcomes_generated'],'Coverage/scope changed')
    log=(root/'tests.stderr.log').read_text();need(re.search(r'Ran 280 tests in',log) and log.rstrip().endswith('OK') and not re.search(r'skipp?ed|expected failure|unexpected success',log,re.I),'Tests did not pass')
    names=('ccg.py','extensive.py','first_stage.py','recourse.py','probability.py','probability_phase1.py','random_streams.py')
    engine=canonical({n:sha256(paths.project_root/'src/evcs_experiments'/n) for n in names})
    rows=[audit_run(root/'toys'/name,risk,mode,reference,engine) for name in old.GOLDENS for risk in ('DRO','ARO') for mode in ('continuous','resumed')]
    for name in old.GOLDENS:
        for risk in ('DRO','ARO'):
            pair=[r for r in rows if r['case_id']==name and r['risk_model']==risk]
            close(pair[0]['audit_upper_bound_usd'],pair[1]['audit_upper_bound_usd'],'Resume differs from continuous')
    report={'status':'PASS_INDEPENDENT_S03_I03_AUDIT','increment':'S03-I03','full_S03_pass':False,'tests':280,
        'runs':rows,'RMP_matrices':sum(r['RMP_matrices'] for r in rows),'fixed_LP_certificates':sum(r['fixed_LP_certificates'] for r in rows),
        'artifact_files':files,'source_files':sources,'release_files':release,'source_manifest_sha256':SOURCE_SHA,
        'artifact_manifest_sha256':ARTIFACT_SHA,'audit_source_sha256':sha256(__file__),
        'limits':['Numerical solver/KKT bounds within recorded tolerances, not exact arithmetic.',
                  'Synthetic physical fixtures only; full80EV/160support performance trial not accepted.',
                  'Resume is algorithm state plus completed LPs; no solver search-tree warm start.']}
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:report[k] for k in ('status','RMP_matrices','fixed_LP_certificates','artifact_files')}))
    return 0


if __name__=='__main__':raise SystemExit(main())
