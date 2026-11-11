#!/usr/bin/env python3
"""Read saved I03 physical/MILP evidence and numeric duals; never solve or draw."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import csv
import json
import re

import numpy as np
from scipy import sparse

from evcs_experiments.config import get_paths
from evcs_experiments.evidence import historical_path, sha256, frozen_runner_root, frozen_control_root
from evcs_experiments.first_stage import FirstStagePlan
from evcs_experiments.physical_inputs import load_scenario_physical_inputs
from evcs_experiments.recourse import RecourseSolution
from evcs_experiments.recourse_check import check_recourse
from evcs_experiments.validation.validate_s02_recourse import load_case_from_i01, SOLUTION_FIELDS


def need(ok, message):
    if not bool(ok): raise ValueError(message)


def read(path, expected=None):
    path = historical_path(path)
    if expected is not None: need(sha256(path) == expected, f"SHA256 mismatch: {path}")
    return json.loads(path.read_text())


def child(root, name):
    p = Path(name)
    need(not p.is_absolute() and '..' not in p.parts, f"Unsafe relative path: {name}")
    q = (root/p).resolve()
    need(root.resolve() in q.parents, f"Artifact escaped root: {name}")
    return q


def manifest(root, expected=None):
    values = read(root/'artifact_sha256.json', expected)
    for name, digest in values.items(): need(sha256(child(root,name)) == digest, f"Artifact changed: {name}")
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    need(actual == set(values)|{'artifact_sha256.json'}, 'Manifest omits or invents an artifact')
    return len(values)


def release_manifest(root, expected=None):
    rows = read(root/'RELEASE_MANIFEST.json',expected)['files']
    need(len({r['path'] for r in rows}) == len(rows), 'Duplicate release identity')
    for row in rows:
        p = child(root,row['path'])
        need(p.stat().st_size == row['size_bytes'] and sha256(p) == row['sha256'], f"Release file changed: {p}")
    return len(rows)


def sources(root, code):
    values = code['current_source_hashes']
    need(bool(values), 'Empty source manifest')
    for name,digest in values.items(): need(sha256(child(root,name)) == digest, f"Source changed: {name}")
    return len(values)


def acceptance(path, digest, increment):
    card = read(path,digest)
    need(card['status']=='PASS_INCREMENT' and card['increment']==increment and not card['full_S02_pass'],
         f"Wrong accepted prerequisite: {increment}")
    root = Path(card['source_root']); evidence = historical_path(card['evidence'])
    files = manifest(evidence,card['artifact_manifest_sha256'])
    code = read(card['code_manifest']['path'],card['code_manifest']['sha256'])
    source_count = sources(root,code)
    bound = card['current_source_manifest']
    need(Path(bound['path']).resolve() == (root/'RELEASE_MANIFEST.json').resolve(), 'Historical release manifest path changed')
    release_count = release_manifest(root,bound['sha256'])
    if 'validation_sha256' in card: read(evidence/'increment_validation.json',card['validation_sha256'])
    need(bool(card['independent_audits']), 'Missing accepted independent audit')
    for p,d in card['independent_audits'].items():
        audit = read(p,d)
        need(audit['status'].startswith('PASS_INDEPENDENT'), 'Prerequisite independent audit did not pass')
    execution = read(evidence/'execution.json')
    need(execution['config_path'] and sha256(execution['config_path'])==execution['config_sha256'], 'Historical config changed')
    need(sha256(root/'docs/experiments/runtime_policy.json')==execution['runtime_policy_sha256'], 'Historical policy changed')
    return evidence, {'increment':increment,'artifacts':files,'source_files':source_count,'release_files':release_count}


def load_plan(path):
    with np.load(path,allow_pickle=False) as a:
        return FirstStagePlan(a['x'],a['r'],str(a['task_sha256']),str(a['config_hash']))


def load_solution(path):
    with np.load(path,allow_pickle=False) as a:
        return RecourseSolution(**{k:a[k] for k in SOLUTION_FIELDS}, task_sha256=str(a['task_sha256']),
            config_hash=str(a['config_hash']),scenario_sha256=str(a['scenario_sha256']))


def same_plan(a,b):
    need(a.task_sha256==b.task_sha256 and a.config_hash==b.config_hash and
         np.array_equal(a.x,b.x) and np.array_equal(a.r,b.r), 'Saved fixed plan was changed')


def close(a,b,message,tol=1e-7):
    aa,bb=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    need(aa.shape==bb.shape and np.isfinite(aa).all() and np.isfinite(bb).all() and
         np.allclose(aa,bb,rtol=0,atol=tol),message)


def cost_check(actual,saved):
    for key in ('Fshed','Fshort','FV2G','Fdeg','Qop'): close(actual[key],saved[key],f'Cost changed: {key}',1e-4)


def matrices(root):
    ms={k:sparse.load_npz(root/(k+'.npz')) for k in ('A_ub','A_eq','Bx_ub','Bx_eq','Br_ub','Br_eq')}
    with np.load(root/'template_arrays.npz',allow_pickle=False) as a: coeff={k:a[k] for k in a.files}
    for m in ms.values(): need(np.isfinite(m.data).all(), 'Nonfinite sparse matrix')
    for k in ('b_ub','b_eq','c','component_objectives'): need(np.isfinite(coeff[k]).all(), f'Nonfinite {k}')
    need(not np.isnan(coeff['lower_bounds']).any() and not np.isnan(coeff['upper_bounds']).any(), 'NaN variable bounds')
    return ms,coeff


def vector(solution,slices,c):
    z=np.empty_like(c); cursor=0
    need(tuple(slices)==SOLUTION_FIELDS, 'Variable ordering/schema changed')
    for key,(start,end) in slices.items():
        a=getattr(solution,key).ravel()
        need(start==cursor and end-start==a.size, f'Bad variable slice: {key}')
        z[start:end]=a;cursor=end
    need(cursor==z.size, 'Unmapped recourse columns')
    return z


def primal(ms,co,z,plan):
    bub=co['b_ub']+ms['Bx_ub']@plan.x+ms['Br_ub']@plan.r.ravel()
    beq=co['b_eq']+ms['Bx_eq']@plan.x+ms['Br_eq']@plan.r.ravel()
    lo,hi=co['lower_bounds'],co['upper_bounds']
    violation=float(max(np.max(ms['A_ub']@z-bub,initial=0.),np.max(np.abs(ms['A_eq']@z-beq),initial=0.),
                        np.max(lo-z,initial=0.),np.max(z-hi,initial=0.)))
    need(violation<=1e-7,'Saved CSR primal certificate failed')
    return bub,beq,violation


def dual(root,ms,co,z,bub,beq):
    with np.load(root/'dual_marginals.npz',allow_pickle=False) as a:
        y,lam,lo,hi=(a[k] for k in ('dual_inequality_marginals','dual_equality_marginals','dual_lower_marginals','dual_upper_marginals'))
    need(y.shape==bub.shape and lam.shape==beq.shape and lo.shape==hi.shape==z.shape and
         all(np.isfinite(a).all() for a in (y,lam,lo,hi)), 'Invalid dual axes or nonfinite witness')
    stationarity=float(np.max(np.abs(co['c']-ms['A_ub'].T@y-ms['A_eq'].T@lam-lo-hi),initial=0.))
    signs=float(max(np.max(y,initial=0.),np.max(-lo,initial=0.),np.max(hi,initial=0.)))
    lower,upper=co['lower_bounds'],co['upper_bounds'];fl,fu=np.isfinite(lower),np.isfinite(upper)
    infinite=float(max(np.max(np.abs(lo[~fl]),initial=0.),np.max(np.abs(hi[~fu]),initial=0.)))
    value=float(bub@y+beq@lam+lower[fl]@lo[fl]+upper[fu]@hi[fu]);obj=float(co['c']@z)
    gap=abs(obj-value);tol=1e-4+1e-7*max(1.,abs(obj),abs(value))
    need(max(stationarity,signs,infinite)<=1e-8 and gap<=tol,'Saved numeric dual certificate failed')
    saved=read(root/'solver_result.json')
    need(saved['status']=='OPTIMAL' and saved['optimal'] and saved['certificate']['optimal_certificate_valid'], 'Missing optimal LP claim')
    close(saved['objective_usd'],obj,'Solver objective differs',1e-4)
    close(saved['certificate']['dual_objective_usd'],value,'Saved dual objective differs',1e-4)
    return {'primal_usd':obj,'dual_usd':value,'gap_usd':gap,'stationarity':stationarity,'sign_violation':signs,'infinite_bound_marginal':infinite}


def audit_real(root,case,tasks,scene,plan,q0,prior=None):
    meta=read(root/'metadata.json');same_plan(load_plan(root/'plan.npz'),plan)
    need(meta['H']==1008 and meta['task_sha256']==tasks.task_sha256 and meta['scenario_sha256']==scene.scenario_sha256 and
         meta['scenario_id']==scene.scenario_id and meta['q0_diagnostic']==q0,'Real case identity changed')
    need(meta['line_outage_periods']==int(scene.line_operating_outages.sum()),'Outage summary changed')
    need(meta['unqualified_tasks']==int((~scene.task_varsigma).sum()) and
         meta['selected_unqualified_tasks']==int(((plan.x>.5)&~scene.task_varsigma).sum()),'Task qualification summary changed')
    ms,co=matrices(root)
    if q0:
        old,oldco=matrices(prior);oldmeta=read(prior/'template_metadata.json')
        need(meta['original_Q1_template_sha256']==oldmeta['template_sha256'], 'Q0 not paired to accepted Q1 template')
        for k in ms: need(ms[k].shape==old[k].shape and (ms[k]!=old[k]).nnz==0, f'Q0 changed {k}')
        for k in co:
            expected=oldco[k].copy()
            if k in ('lower_bounds','upper_bounds'):
                a,b=meta['variable_slices']['q'];expected[a:b]=0.
            need(np.array_equal(co[k],expected),f'Q0 altered coefficients beyond q=0: {k}')
    reports=[]
    for label in ('fallback','solution'):
        sol=load_solution(root/(label+'.npz'))
        checked=check_recourse(case,tasks,scene,plan,sol,check_objective_bound=True)
        need(checked['passed'],f'Real direct physics failed: {root.name}/{label}')
        saved=read(root/('fallback_check.json' if label=='fallback' else 'physical_check.json'))
        need(saved['passed'],'Saved physical report did not pass');cost_check(checked['costs'],saved['costs'])
        z=vector(sol,meta['variable_slices'],co['c']);bub,beq,res=primal(ms,co,z,plan)
        close(float(co['c']@z),checked['costs']['Qop'],'Matrix cost differs',1e-4)
        close(co['component_objectives']@z,[checked['costs'][k] for k in ('Fshed','Fshort','FV2G','Fdeg')], 'Component coefficients differ',1e-4)
        served=scene.load_p_kw-sol.shed
        if q0: need(np.max(np.abs(sol.q),initial=0.)<=1e-8 and np.max(np.abs(served),initial=0.)<=1e-5,'Q0 physical zero-service diagnostic failed')
        row={'case':root.name,'solution':label,'costs':checked['costs'],'matrix_max_violation':res,
             'max_normalized_violation':checked['max_normalized_violation'],'served_kwh':float(served.sum()*case.dt_h)}
        if label=='solution':
            row['dual']=dual(root,ms,co,z,bub,beq)
            cost_check(checked['costs'],read(root/'solver_result.json')['objective_components'])
            summary=read(root/'summary.json');cost_check(checked['costs'],summary)
            close(summary['served_kwh'],row['served_kwh'],'Served-energy summary differs')
            close(summary['max_hourly_served_kw'],float(np.max(served.sum(axis=1),initial=0.)),'Hourly service summary differs')
        reports.append(row)
    return reports


def toy_oracles():
    # Only explicit input constructors, no call to export_toys/LP/MILP assembly.
    from evcs_experiments.validation.s02_toys import make_fixture as f, golden_single_bus, golden_reverse_flow, golden_islands
    theta=(2*np.arange(16)+1)*np.pi/16;active=np.sin(theta)>0
    q=float(np.min((18.75*np.cos(np.pi/16)-15*np.cos(theta))[active]/np.sin(theta)[active]))
    items={}
    def add(name,fixture,cost=None,fail=(),arrays=None):items[name]=(fixture,cost,tuple(fail),arrays or {})
    add('single_bus_paid_service',golden_single_bus(),1.02,arrays={'p':[[6.]],'q':[[3.]],'rav':[[10.]],'soc_dep':[.725]})
    add('positive_Q_rav15_boundary',f(load_p=[[15.]],tan_phi=q/15,reserve=15.),2.55,arrays={'q':[[q]],'rav':[[15.]]})
    add('negative_Q_rav15_boundary',f(task_station_indices=(0,0),load_p=[[15.]],tan_phi=0.,reserve=[[15.],[1.]]),2.55,arrays={'q':[[-q],[q]]})
    add('Q_only_rav_zero',f(task_station_indices=(0,0),load_p=[[6.]],tan_phi=18.75/6,reserve=[[6.],[1.]]),2.02,arrays={'rav':[[6.],[0.]],'p':[[6.],[0.]],'q':[[0.],[18.75]]})
    add('no_selected_EV',f(load_p=[[6.]],tan_phi=.5,selected=0.),18.,arrays={'p':[[0.]],'q':[[0.]],'shed':[[6.]]})
    add('unavailable_unqualified_EV',f(load_p=[[6.]],tan_phi=.5,available=False,arrival_soc=.1),28.,arrays={'p':[[0.]],'q':[[0.]],'soc_dep':[.1]})
    add('rejected_p0_rav15_q15',f(task_station_indices=(0,0),load_p=[[6.]],tan_phi=2.5,reserve=[[15.],[6.]]),fail=('task_capability',),arrays={'p':[[0.],[6.]],'rav':[[15.],[6.]],'q':[[15.],[0.]]})
    add('reserve_without_activation',f(periods=2,load_p=[[0.],[0.]],tan_phi=0.,reserve=15.,arrival_soc=.3875),0.,arrays={'rav':[[15.,15.]],'p':[[0.,0.]]})
    chain=f(periods=2,load_p=[[10.],[0.]],tan_phi=0.,reserve=15.,arrival_soc=.45)
    add('actual_discharge_reserve_chain',chain,6.7,arrays={'rav':[[15.,10.]],'soc_dep':[.325]})
    add('rejected_actual_discharge_chain',chain,fail=('sustainability',))
    add('departure_SOC_binding',f(load_p=[[15.]],tan_phi=.6,reserve=15.,eta=.95,arrival_soc=.2+15/(.95*80)),1.8+.05*15/.95,arrays={'soc_dep':[.2],'p':[[15.]]})
    add('tau2_efficiency_SOC',f(periods=3,tau=2.,eta=.8,arrival_soc=.7,reserve=[[15.,15.,0.]],load_p=[[8.],[0.],[0.]],tan_phi=0.),4.46,arrays={'rav':[[15.,12.,0.]],'soc_dep':[.575]})
    station=golden_single_bus();station.case.station_rating_kva=np.array([10.])
    add('rejected_station_polygon',station,fail=('station_capability',));add('derated_station_polygon',station)
    add('reverse_signed_flow',golden_reverse_flow(),1.02,arrays={'line_p':[[-6.]],'line_q':[[-3.]],'u':[[.991,1.]]})
    add('two_islands_one_without_EVCS',golden_islands(),15.68,arrays={'shed':[[0.,0.,5.]],'line_p':[[4.,0.]]})
    add('two_islands_zero_selected',golden_islands(selected=0.),27.,arrays={'shed':[[0.,4.,5.]],'p':[[0.]]})
    for name,load in [('joint_MILP_useful',15.),('joint_MILP_not_useful',1.)]:
        fixture=f(load_p=[[load]],tan_phi=.6,reserve=15.);fixture.tasks.p_block_kw[:]=15.
        fixture.tasks.qualification_soc=np.maximum(.3,fixture.tasks.soc_departure_min+15/(fixture.tasks.eta_dis*fixture.tasks.e_cap_kwh))
        add(name,fixture)
    return items,q


def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,(tuple,list)):return [serial(v) for v in value]
    if isinstance(value,dict):return {str(k):serial(v) for k,v in value.items()}
    if hasattr(value,'__dict__'):return serial(vars(value))
    return value


def audit_toys(root):
    manifest(root);report=read(root/'physical_check_report.json');oracles,q=toy_oracles()
    need(report['status']=='PASS_TOY_EXPECTATIONS_AUTOMATED' and not report['full_S02_pass'] and report['fixture_count']==19,'Wrong toy scope/count')
    need({r['case_id'] for r in report['cases']}==set(oracles) and len(report['cases'])==19,'Toy case set changed')
    records={r['case_id']:r for r in report['cases']}
    with (root/'objective_components.csv').open(newline='') as stream:cost_rows=list(csv.DictReader(stream))
    need(len(cost_rows)==19 and {r['case_id'] for r in cost_rows}==set(oracles),'Toy cost-table coverage changed')
    cost_rows={r['case_id']:r for r in cost_rows}
    results=[]
    for name,(oracle,cost,fail,arrays) in oracles.items():
        path=root/name;inputs=read(path/'fixture_inputs.json');expected=read(path/'expected.json')
        for key in ('case','tasks','scenario'):
            need(inputs[key]==serial(getattr(oracle,key)),f'Golden input differs: {name}/{key}')
        f=SimpleNamespace(**{key:SimpleNamespace(**inputs[key]) for key in ('case','tasks','scenario')})
        plan=load_plan(path/'plan.npz')
        if not name.startswith('joint_MILP'):same_plan(plan,oracle.plan)
        need(bool(expected.get('physical_pass',True))==(not bool(fail)),'Golden physical verdict changed')
        sol=load_solution(path/('rejected_candidate.npz' if fail else 'solution.npz'))
        checked=check_recourse(f.case,f.tasks,f.scenario,plan,sol)
        need(checked['passed']==(not bool(fail)),f'Toy direct physical verdict differs: {name}')
        row_record=records[name]
        need(row_record['expected_physical_pass']==(not bool(fail)) and row_record['observed_physical_pass']==checked['passed'] and row_record['expectations_passed'], 'Toy summary verdict differs')
        for residual in fail:need(not checked['residuals'][residual]['passed'],f'Missing expected violation: {name}/{residual}')
        if fail:need(set(expected['failed_residuals'])==set(fail),'Golden rejected constraints changed')
        if cost is not None:
            close(checked['costs']['Qop'],cost,f'Hand-calculated cost differs: {name}')
            close(expected['Qop'],cost,'Saved golden cost differs')
        for field,value in arrays.items():close(getattr(sol,field),value,f'Golden array differs: {name}/{field}')
        for field,value in expected.get('arrays',{}).items():close(getattr(sol,field),value,f'Saved expected array differs: {name}/{field}')
        saved=read(path/'physical_check.json');need(saved['passed']==checked['passed'],'Saved toy verdict differs');cost_check(checked['costs'],saved['costs'])
        cost_check(checked['costs'],cost_rows[name])
        close(float(cost_rows[name]['Fpre']),checked['first_stage']['costs']['Fpre_usd'],'Toy Fpre table differs')
        close(float(cost_rows[name]['total_usd']),checked['first_stage']['costs']['Fpre_usd']+checked['costs']['Qop'],'Toy total table differs')
        if (path/'fallback.npz').exists():
            fb=check_recourse(f.case,f.tasks,f.scenario,plan,load_solution(path/'fallback.npz'),check_objective_bound=True)
            need(fb['passed'],f'Toy fallback failed: {name}')
        row={'case':name,'expected_feasible':not bool(fail),'physical_replayed':True,'costs':checked['costs'],'failed_residuals':list(fail)}
        if name.startswith('joint_MILP'):
            with np.load(path/'joint_arrays.npz',allow_pickle=False) as a:co={k:a[k] for k in a.files}
            with np.load(path/'joint_primal.npz',allow_pickle=False) as a:z=a['vector']
            A=sparse.load_npz(path/'joint_matrix.npz');need(np.isfinite(z).all(),'Nonfinite joint primal')
            close(z,np.concatenate((plan.x,plan.r.ravel(),*(getattr(sol,k).ravel() for k in SOLUTION_FIELDS))),'Joint vector differs from saved physical solution',0.)
            activity=A@z
            violation=max(np.max(co['lower']-activity,initial=0.),np.max(activity-co['upper'],initial=0.),np.max(co['variable_lower']-z,initial=0.),np.max(z-co['variable_upper'],initial=0.))
            ints=co['integrality']!=0
            need(ints.sum()==1 and ints[0] and co['variable_lower'][0]==0 and co['variable_upper'][0]==1,'Joint x was fixed or integrality changed')
            need(violation<=1e-7 and np.max(np.abs(z[ints]-np.rint(z[ints])),initial=0.)<=1e-8,'Joint primal/integrality failed')
            total=checked['first_stage']['costs']['Fpre_usd']+checked['costs']['Qop'];close(float(co['c']@z),total,'Joint cost reconstruction failed')
            # Exact two-branch proof: x=0 sheds L at 3/kWh; x=1 forces r=15,
            # costs 25+.02*15 and optimally serves all L at (.12+.05)/kWh.
            load=float(oracle.scenario.load_p_kw[0,0]);branches=[3*load,25.3+.17*load]
            optimum=min(branches);choice=int(branches[1]<branches[0])
            close(plan.x,[choice],'Joint activation disagrees with independent two-branch proof')
            close(total,optimum,'Joint optimum disagrees with independent two-branch proof')
            solver=read(path/'solver_result.json')
            need(solver['status']=='OPTIMAL' and not solver['first_stage_fixed_in_optimization'] and solver['joint_integer_variables']==1,'Not an actual joint MILP')
            close(solver['objective_usd'],optimum,'Joint reported optimum differs');close(solver['dual_bound_usd'],optimum,'Joint MIP bound differs')
            need(solver['mip_gap']<=1e-9,'Joint MIP gap failed')
            close(expected['selected'],choice,'Saved joint choice differs');close(expected['total_usd'],optimum,'Saved joint cost differs')
            row['joint_certificate']={'branch_objectives_usd':branches,'optimum_usd':optimum,'selected':choice,'max_primal_violation':float(violation),'proof':'independent two-binary-branch hand calculation plus saved MILP primal'}
        results.append(row)
    facet=read(root/'facet_reference_check.json');close(facet['q_limit_at_rav15_kvar'],q,'Facet boundary changed',1e-12)
    need(facet['passed'] and facet['active_coordinate']=='rav' and max(facet['max_normal_error'],facet['max_rhs_error'])<=1e-14,'Independent facet reference failed')
    return results


def audit_legacy(e,paths):
    replay=e/'legacy_replay';copy_root=replay/'IEEE33_paper_aligned_case_data_v1'
    m=read(replay/'legacy_replay_manifest.json')
    need(m['copy_sha256_verified'] and m['source_files_unchanged'] and not m['changed_source_files'], 'Legacy copy/source preservation did not pass')
    need(Path(m['case_root']).resolve()==copy_root.resolve() and Path(m['project_root']).resolve()==paths.project_root.resolve(),'Legacy copy identity differs')
    execution=m['execution'];need(execution['state']=='SUCCEEDED' and execution['returncode']==0,'Legacy copied execution failed')
    script=copy_root/'12_reactive_support_correction/verification/verify_reactive_support.py'
    need(Path(execution['argv'][1]).resolve()==script.resolve(),'Wrong legacy replay script')
    roots=[(paths.data_root,Path('.')),(paths.project_root/'archive/11_readiness_audit',Path('11_readiness_audit')),
           (paths.project_root/'archive/12_reactive_support_correction',Path('12_reactive_support_correction')),
           (frozen_runner_root(),Path('15_experiment_runner')),(frozen_control_root(),Path('14_experiment_specs'))]
    expected={}
    for root,prefix in roots:
        for file in root.rglob('*'):
            if file.is_file() and '__pycache__' not in file.parts and file.suffix!='.pyc':
                name=str(prefix/file.relative_to(root));need(name not in expected,'Overlapping legacy source roots');expected[name]=file.resolve()
    need(len(m['files'])==len(expected) and {r['case_relative_path'] for r in m['files']}==set(expected),'Legacy manifest source coverage differs')
    rewritten={f'12_reactive_support_correction/verification/{n}' for n in ('checks.csv','case_summaries.csv','task_dispatch_and_SOC.csv','network_hourly.csv','station_dispatch.csv','actual_colocated_roster.csv','verification_summary.json')}
    for row in m['files']:
        name=row['case_relative_path'];source=expected[name]
        need(Path(row['source']).resolve()==source and source.stat().st_size==row['size_bytes'] and sha256(source)==row['sha256'],f'Legacy original changed: {name}')
        copied=child(copy_root,name)
        if name not in rewritten:need(sha256(copied)==row['sha256'],f'Legacy copied input changed: {name}')
    folder=copy_root/'12_reactive_support_correction/verification';summary=read(folder/'verification_summary.json')
    with (folder/'checks.csv').open(newline='') as stream:checks=list(csv.DictReader(stream))
    with (paths.project_root/'archive/12_reactive_support_correction/verification/checks.csv').open(newline='') as stream:original=list(csv.DictReader(stream))
    need(summary['status']=='PASS' and summary['checks']==13 and len(checks)==13 and [r['check'] for r in checks]==[r['check'] for r in original] and all(r['passed']=='True' for r in checks),'Legacy thirteen checks did not pass')
    for name,digest in summary['source_sha256'].items():need(sha256(child(copy_root,name))==digest,'Legacy replay input binding differs')
    consolidated=read(e/'legacy_13_summary.json')
    need(consolidated['status']=='PASS_LEGACY_13_REPLAY' and consolidated['checks']==checks and consolidated['original_source_files_unchanged'],'Legacy consolidation differs')
    with (e/'q1_13_coverage.csv').open(newline='') as stream:coverage=list(csv.DictReader(stream))
    need(len(coverage)==13,'Q1 coverage table must have exactly thirteen rows')
    expected_refs=['facet_reference_check.json','positive_Q_rav15_boundary','negative_Q_rav15_boundary','Q_only_rav_zero',
        'no_selected_EV','unavailable_unqualified_EV','rejected_p0_rav15_q15','joint_MILP_useful;joint_MILP_not_useful',
        'actual_discharge_reserve_chain;reserve_without_activation;tau2_efficiency_SOC;departure_SOC_binding',
        'derated_station_polygon;rejected_station_polygon',None,None,None]
    for i,row in enumerate(coverage):
        target=';'.join('toy_solutions/'+part for part in expected_refs[i].split(';')) if expected_refs[i] else 'paired_Q0_Q1.json'
        need(row['criterion_id']==f'Q1-{i+1:02d}' and row['legacy_check']==checks[i]['check'] and row['legacy_replay_passed']=='True' and row['production_evidence']==target,'Q1 coverage link does not match replayed evidence')
    return {'checks':13,'source_files_unchanged':len(m['files']),'copied_case_root':str(copy_root),'scope':'Historical diagnostic replay only; distinct from production-model evidence'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--evidence',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();e=args.evidence.resolve();out=args.output.resolve()
    need(not out.exists() and out!=e and e not in out.parents,'Audit must be a new file outside run evidence')
    count=manifest(e);paths=get_paths();execution=read(e/'execution.json')
    need(Path(execution['paths']['project_root']).resolve()==paths.project_root.resolve(),'Audit release differs from execution')
    need(execution['paths']==paths.as_dict(),'Resolved audit configuration paths differ from execution')
    current_sources=sources(paths.project_root,read(e/'model_code_manifest.json'));release_files=release_manifest(paths.project_root)
    need(execution['config_path'] and sha256(execution['config_path'])==execution['config_sha256'],'Execution config changed')
    need(sha256(paths.spec_root/'runtime_policy.json')==execution['runtime_policy_sha256'],'Runtime policy changed')
    log=(e/'tests.stderr.log').read_text();match=re.search(r'Ran (\d+) tests? in',log)
    need(match and int(match[1])>=183 and log.rstrip().endswith('OK') and not re.search(r'skipp?ed|expected failure|unexpected success',log,re.I),'Full 183+ tests must pass without skips')
    for name in ('test_recourse.','test_recourse_check.','test_recourse_physics.','test_s02_closure.'):need(name in log,f'Missing test module: {name}')
    validation=read(e/'increment_validation.json')
    need(validation['increment']=='S02-I03' and validation['status']=='PASS_INCREMENT_AUTOMATED' and not validation['full_S02_pass'] and validation['real_case_count']==3 and validation['H']==1008 and validation['new_scenarios_generated']==0 and not validation['formal_OOS_outcomes_generated'],'Incorrect I03 scope')
    reference=read(e/'input_reference.json');need(not reference['task_subset_used'] and not reference['formal_OOS'] and reference['new_scenarios_generated']==0,'Scenario/task scope changed')
    prior,chain2=acceptance(reference['i02_acceptance'],reference['i02_acceptance_sha256'],'S02-I02')
    ref=read(prior/'input_reference.json');need(ref==reference['i01_reference'],'I01 reference was relabeled')
    _,chain1=acceptance(ref['i01_acceptance'],ref['i01_acceptance_sha256'],'S02-I01')
    case,tasks,scene,plans,record=load_case_from_i01(ref['i01_acceptance'],ref['i01_acceptance_sha256'])
    need(len(tasks.task_ids)==1008 and len(case.bus_ids)==33 and len(case.periods)==24,'Real dimensions changed')
    selection=read(e/'stress_selection.json')
    need(selection['selected_index']==141 and not selection['selection_uses_optimization_outcomes'],'Stress was not prespecified index 141')
    need(selection['npz_sha256']==record['scenario_source']['sha256'] and selection['ledger_sha256']==record['scenario_ledger']['sha256'],'Stress source not accepted I01 bank')
    stress=load_scenario_physical_inputs(historical_path(selection['npz']),historical_path(selection['ledger']),141,case,tasks,
        expected_npz_sha256=selection['npz_sha256'],expected_ledger_sha256=selection['ledger_sha256'],expected_scenario_sha256=selection['scenario_sha256'])
    need(stress.scenario_id=='michael_2018/development/planning/michael_2018/v018/R1' and int(stress.line_operating_outages.sum())==622 and np.all(stress.task_varsigma),'Stress identity/input count changed')
    real=e/'real_cases';names={'intact_same_paid_plan_Q0','disrupted_zero','disrupted_minimum_fixture'}
    need({p.name for p in real.iterdir()}==names,'Wrong real case set')
    reports=audit_real(real/'intact_same_paid_plan_Q0',case,tasks,scene,plans['minimum_fixture'],True,prior)
    for name in ('zero','minimum_fixture'):reports+=audit_real(real/('disrupted_'+name),case,tasks,stress,plans[name],False)
    toys=audit_toys(e/'toy_solutions');need(read(e/'toy_report.json')==read(e/'toy_solutions/physical_check_report.json'),'Toy summary differs')
    legacy=audit_legacy(e,paths)
    with np.load(prior/'minimum_fixture/solution.npz',allow_pickle=False) as a:
        q1_served=float(np.sum(scene.load_p_kw-a['shed'])*case.dt_h)
        need(q1_served>0,'Accepted paired Q1 has no positive service')
    with np.load(prior/'zero/solution.npz',allow_pickle=False) as a:
        zeroq=float(np.max(np.abs(a['q']),initial=0.));zeroservice=float(np.max(np.abs(scene.load_p_kw-a['shed']),initial=0.))
        need(zeroq<=1e-8 and zeroservice<=1e-5,'Accepted zero-selection Q/service gate failed')
    paired=read(e/'paired_Q0_Q1.json')
    need(paired['same_paid_plan'] and paired['same_scenario_sha256']==scene.scenario_sha256 and paired['same_task_sha256']==tasks.task_sha256 and Path(paired['Q1_accepted_evidence']).resolve()==prior.resolve(),'Q0/Q1 pair identity changed')
    close(paired['Q1_served_kwh'],q1_served,'Paired Q1 service changed');close(paired['Q0_served_kwh'],reports[1]['served_kwh'],'Paired Q0 service changed')
    close(paired['zero_plan_max_task_abs_q_kvar'],zeroq,'Paired zero Q summary changed');close(paired['zero_plan_max_bus_served_kw'],zeroservice,'Paired zero service changed')
    with (e/'objective_components.csv').open(newline='') as stream:cost_rows=list(csv.DictReader(stream))
    need(len(cost_rows)==3 and {r['case'] for r in cost_rows}==names,'Real objective table case coverage differs')
    solved_rows={r['case']:r for r in reports if r['solution']=='solution'}
    for row in cost_rows:cost_check(solved_rows[row['case']]['costs'],row)
    gates=[('settings_facets_exact_PF',['case74manifest','facet_reference_check']),('positive_Q_rav15',['positive_Q_rav15_boundary']),
        ('negative_Q_rav15',['negative_Q_rav15_boundary']),('Q_only',['Q_only_rav_zero']),('unselected_Q_zero',['no_selected_EV']),
        ('unavailable_Q_zero',['unavailable_unqualified_EV']),('rav_headroom_rejection',['rejected_p0_rav15_q15']),
        ('joint_paid_activation',['joint_MILP_useful','joint_MILP_not_useful']),('SOC_and_sustainability',['departure_SOC_binding','actual_discharge_reserve_chain','reserve_without_activation','tau2_efficiency_SOC']),
        ('station_polygon',['rejected_station_polygon','derated_station_polygon']),('real_Q0_zero_service',['intact_same_paid_plan_Q0']),
        ('paired_real_Q1_service',['accepted_I02/minimum_fixture']),('all_station_Q_zero_when_unselected',['accepted_I02/zero'])]
    result={'status':'PASS_INDEPENDENT_S02_I03_AUDIT','increment':'S02-I03','full_S02_pass':False,'tests':int(match[1]),
        'scope':'Saved real primal/dual, 19 explicit toys, independent joint binary-branch proof, and 13 Q1 evidence links; no solve or draw',
        'artifact_files':count,'current_source_files':current_sources,'current_release_files':release_files,'accepted_chain':[chain2,chain1],
        'real_reports':reports,'toy_reports':toys,'historical_legacy_replay':legacy,'Q1_gates':[{'id':i+1,'name':name,'status':'PASS_EVIDENCE_REPLAY','evidence':evidence} for i,(name,evidence) in enumerate(gates)],
        'stress_index':141,'stress_line_hour_outages':622,'stress_unqualified_tasks':0,
        'artifact_manifest_sha256':sha256(e/'artifact_sha256.json'),'audit_source_sha256':sha256(__file__),
        'remaining_gate':'Root independent stage review and explicit scientific acceptance; this audit never marks complete S02 PASS'}
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':result['status'],'output':str(out),'real_solutions':len(reports),'toys':len(toys),'Q1_gates':13}))
    return 0


if __name__=='__main__':raise SystemExit(main())
