"""S03-I01: probability LP certificates on accepted development feature supports.

The four q vectors are explicit algebraic diagnostics, NOT solved recourse
costs or policy performance. No new scenarios, calibration, or OOS are used.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import numpy as np
from evcs_experiments.config import get_paths
from evcs_experiments.evidence import historical_path,sha256,source_provenance,verify_accepted_sources
from evcs_experiments.physical_inputs import load_physical_case
from evcs_experiments.probability import ProbabilitySet,solve_worst_distribution
from evcs_experiments.probability_check import check_probability_solution
from evcs_experiments.validation.validate_s02_first_stage import checked_json,write_json,write_mapping

FEATURES=('Phi_W','Phi_L','Phi_R','Phi_D','Phi_load')
RHOS=(.5,1.,1.5,2.)


def _check_files(root,entries):
    for name,value in entries.items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Invalid relative evidence path')
        expected=value['sha256'] if isinstance(value,dict) else value
        if sha256(Path(root)/name)!=expected:raise ValueError('Changed accepted evidence: '+name)


def _physical_acceptance(path,digest):
    card=checked_json(path,digest)
    if card['status'] not in ('PASS','PASS_INCREMENT'):raise ValueError('Physical prerequisite unaccepted')
    root=Path(card['source_root']);e=Path(card['evidence'])
    _check_files(e,checked_json(e/'artifact_sha256.json',card['artifact_manifest_sha256']))
    code=checked_json(card['code_manifest']['path'],card['code_manifest']['sha256'])
    _check_files(root,code['current_source_hashes'])
    release=checked_json(card['current_source_manifest']['path'],card['current_source_manifest']['sha256'])
    _check_files(root,{row['path']:row['sha256'] for row in release['files']})
    for name,expected in card['independent_audits'].items():
        audit=checked_json(name,expected)
        if not audit['status'].startswith('PASS_INDEPENDENT'):raise ValueError('Independent prerequisite audit failed')
    for item in card.get('prerequisite_acceptances',[]):_physical_acceptance(item['path'],item['sha256'])
    return card


def load_prerequisites(s01_path,s01_sha,s02_path,s02_sha):
    s02=_physical_acceptance(s02_path,s02_sha)
    if s02['spec_id']!='S02' or s02['status']!='PASS' or not s02['full_S02_pass']:
        raise ValueError('Complete S02 must be independently accepted')
    s01=checked_json(s01_path,s01_sha)
    if s01['spec_id']!='S01' or s01['status']!='PASS' or s01['formal_OOS_outcomes_generated']:
        raise ValueError('Requires accepted development S01')
    manifest=checked_json(s01['deliverables_manifest'],s01['deliverables_manifest_sha256'])
    _check_files(s01['deliverables'],manifest)
    verify_accepted_sources(s01['source_implementation_snapshot'])
    contract=checked_json(manifest['generator_contract.json']['path'],manifest['generator_contract.json']['sha256'])
    if tuple(contract['feature_order'])!=FEATURES or not contract['development_only'] or not contract['formal_OOS_locked']:
        raise ValueError('Wrong feature schema or unlocked OOS')
    def rows(name):
        with historical_path(manifest[name]['path']).open(newline='') as stream:return list(csv.DictReader(stream))
    feature_rows=rows('features.csv');moment_rows=rows('moment_bounds.csv')
    if len(feature_rows)!=1920 or len(moment_rows)!=160:raise ValueError('Accepted S01 table coverage changed')
    moment_refs={row['source_moments_path']:row['source_moments_sha256'] for row in moment_rows}
    moments={p:checked_json(p,d) for p,d in moment_refs.items()}
    for row in moment_rows:
        m=moments[row['source_moments_path']];i=int(row['feature_index']);bounds=m['bounds_by_rho'][row['rho']]
        if (m['status']!='PASS_MOMENT_CALIBRATION' or row['feature']!=FEATURES[i]
                or float(row['lower'])!=bounds['lower'][i] or float(row['upper'])!=bounds['upper'][i]):
            raise ValueError('Accepted moment-table/source mismatch')
    return {'feature_rows':feature_rows,'moment_rows':moment_rows,'target_order':contract['target_order'],
        'feature_order':list(FEATURES),'source_refs':{'s01_acceptance':{'path':str(s01_path),'sha256':s01_sha},
        's02_acceptance':{'path':str(s02_path),'sha256':s02_sha},'delivery_manifest':{'path':s01['deliverables_manifest'],'sha256':s01['deliverables_manifest_sha256']},
        'features':manifest['features.csv'],'moments':manifest['moment_bounds.csv'],'moment_sources':moment_refs,
        'feature_schema':list(FEATURES),'development_only':True,'formal_OOS_locked':True}}


def diagnostic_q(features):
    return {'zero_q':np.zeros(len(features)),'constant_q':np.full(len(features),260000.),
        'affine_q':100000.+features@np.array([10000.,20000.,30000.,40000.,50000.]),
        'nonlinear_q':100000.+20000.*features[:,0]*features[:,2]+70000.*features[:,3]**2+10000.*features[:,4]**2}


def target_inputs(inputs,target,rho):
    rows=[row for row in inputs['feature_rows'] if row['target_event']==target and row['split']=='planning']
    ids=tuple(row['scenario_id'] for row in rows)
    if len(rows)!=160 or len(set(ids))!=160 or any('/development/planning/' not in i for i in ids):
        raise ValueError('Expected 160 distinct unchanged development planning atoms')
    phi=np.array([[float(row[k]) for k in FEATURES] for row in rows])
    bounds=sorted([r for r in inputs['moment_rows'] if r['target_event']==target and float(r['rho'])==rho],key=lambda r:int(r['feature_index']))
    if len(bounds)!=5 or [r['feature'] for r in bounds]!=list(FEATURES):raise ValueError('Incomplete moment order')
    return ids,phi,np.array([float(r['lower']) for r in bounds]),np.array([float(r['upper']) for r in bounds])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case-root',type=Path,default=get_paths().data_root)
    for name in ('s01','s02'):
        p.add_argument('--'+name+'-acceptance',type=Path,required=True);p.add_argument('--'+name+'-acceptance-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--lp-time-limit',type=float,default=30.)
    args=p.parse_args();paths=get_paths();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);began=time.monotonic()
    config=Path(os.environ['EVCS_CONFIG']).resolve() if 'EVCS_CONFIG' in os.environ else None
    write_json(out/'execution.json',{'argv':sys.argv,'python':sys.version,'paths':paths.as_dict(),'config_path':str(config) if config else None,
        'config_sha256':sha256(config) if config else None,'runtime_policy_sha256':sha256(paths.spec_root/'runtime_policy.json'),
        'scope':'S03-I01 probability primitive only; algebraic q diagnostics, no physical recourse runs or policy optimization',
        'probability_objective_transform':'q_centered_min_v1; original q and scientific gates unchanged'})
    write_json(out/'progress.json',{'phase':'tests','status':'RUNNING'})
    with (out/'tests.stdout.log').open('w') as stdout,(out/'tests.stderr.log').open('w') as stderr:
        test=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(paths.project_root/'tests'),'-p','test_*.py','-v'],cwd=paths.project_root,stdout=stdout,stderr=stderr)
    log=(out/'tests.stderr.log').read_text();match=re.search(r'Ran (\d+) tests? in',log)
    if test.returncode or not match or int(match[1])<230 or not log.rstrip().endswith('OK') or 'skipped=' in log:
        raise ValueError('Full probability and existing tests must pass without skips')
    case=load_physical_case(args.case_root)
    inputs=load_prerequisites(args.s01_acceptance,args.s01_acceptance_sha256,args.s02_acceptance,args.s02_acceptance_sha256)
    write_json(out/'sources.json',inputs['source_refs'])
    records=[];summary=[];probabilities=[];sweep=[]
    for target in inputs['target_order']:
        previous={}
        for rho in RHOS:
            ids,phi,lower,upper=target_inputs(inputs,target,rho)
            ambiguity=ProbabilitySet(ids,FEATURES,phi,lower,upper)
            empirical=phi.mean(axis=0)
            empirical_feasible=bool(np.all(empirical>=lower-1e-8) and np.all(empirical<=upper+1e-8))
            for name,q in diagnostic_q(phi).items():
                relative=f'cases/{target}/rho_{rho}/{name}';folder=out/relative;folder.mkdir(parents=True)
                write_json(out/'progress.json',{'phase':'probability_primal_dual','target':target,'rho':rho,'vector':name,'status':'RUNNING'})
                np.savez_compressed(folder/'inputs.npz',features=phi,lower=lower,upper=upper,q=q,support_ids=np.array(ids),feature_ids=np.array(FEATURES))
                result=solve_worst_distribution(ambiguity,q,q_support_ids=ids,time_limit_seconds=args.lp_time_limit)
                write_json(folder/'solution.json',result.as_dict())
                checked=check_probability_solution(ambiguity,q,result,q_support_ids=ids)
                write_json(folder/'physical_probability_check.json',checked.as_dict())
                if not result.certified_optimal or not checked.passed:raise ValueError('Probability primal/dual independent check failed: '+relative)
                value=result.primal_objective_usd;tol=1e-4+1e-7*max(1.,abs(value),float(np.max(np.abs(q))))
                if value>float(q.max())+tol or (empirical_feasible and value<float(q.mean())-tol):
                    raise ValueError('Probability expectation bounds failed')
                if name in ('zero_q','constant_q') and abs(value-float(q[0]))>tol:raise ValueError('Constant-q analytic optimum failed')
                if name in previous and value<previous[name]-tol:raise ValueError('Nested rho worst expectation decreased')
                previous[name]=value
                record={'relative_path':relative,'target':target,'rho':rho,'vector':name,'ambiguity_sha256':ambiguity.ambiguity_sha256,'q_sha256':result.q_sha256}
                records.append(record)
                row={**record,'status':result.status,'certified_optimal':result.certified_optimal,'independent_passed':checked.passed,
                    'primal_objective_usd':value,'dual_objective_usd':result.dual_objective_usd,'gap_usd':abs(value-result.dual_objective_usd),
                    'q_offset_usd':result.certificate['q_offset_usd'],'q_scale_usd':result.certificate['q_scale_usd'],
                    'certification_scale_usd':result.certificate['certification_scale_usd'],
                    'uniform_expectation_usd':float(q.mean()),'empirical_law_feasible':empirical_feasible,'support_maximum_usd':float(q.max()),'scope':'algebraic diagnostic q; not recourse'}
                summary.append(row)
                for sid,weight in zip(ids,result.probabilities):probabilities.append({**record,'scenario_id':sid,'probability':float(weight)})
        sweep.append({'target':target,'rho_order':list(RHOS),'nested_expectations_checked':True})
    if len(records)!=128 or len(probabilities)!=20480:raise ValueError('Incomplete real-support certificate coverage')
    write_json(out/'records.json',records);write_mapping(out/'probability_duality_audit.csv',summary);write_mapping(out/'worst_distribution.csv',probabilities)
    write_json(out/'rho_sweep_checks.json',sweep);write_json(out/'model_code_manifest.json',source_provenance())
    write_json(out/'increment_validation.json',{'increment':'S03-I01','status':'PASS_INCREMENT_AUTOMATED','full_S03_pass':False,
        'scientific_acceptance_pending':True,'tests':int(match[1]),'targets':8,'rho_values':list(RHOS),'diagnostic_vectors':4,'probability_problems':128,
        'primal_dual_LP_pairs':128,'support_per_problem':160,'feature_count':5,'probability_rows':20480,'case_sha256':case.case_sha256,
        'new_scenarios_generated':0,'moments_changed':False,'physical_recourse_evaluated':False,'formal_OOS_outcomes_generated':False,
        'probability_objective_transform':'q_centered_min_v1; original q and scientific gates unchanged',
        'deferred':'Independent saved probability audit; feasible initialization, full extensive reference, C&CG and global certificates remain future S03 increments',
        'total_wall_seconds':time.monotonic()-began})
    write_json(out/'progress.json',{'phase':'complete','status':'AWAITING_INDEPENDENT_REVIEW'})
    write_json(out/'artifact_sha256.json',{str(f.relative_to(out)):sha256(f) for f in sorted(out.rglob('*')) if f.is_file()})
    print(json.dumps({'status':'PASS_INCREMENT_AUTOMATED','increment':'S03-I01','probability_problems':128,'output':str(out)}));return 0


if __name__=='__main__':raise SystemExit(main())
