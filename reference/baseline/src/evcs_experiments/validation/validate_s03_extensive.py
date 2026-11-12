"""S03-I02 phase-I and small full extensive reference; no production optimization."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import numpy as np
from scipy.sparse import save_npz
from evcs_experiments.config import get_paths
from evcs_experiments.evidence import sha256,source_provenance
from evcs_experiments.first_stage import assemble_first_stage
from evcs_experiments.first_stage_check import check_first_stage
from evcs_experiments.physical_inputs import load_physical_case
from evcs_experiments.probability import ProbabilitySet,solve_worst_distribution
from evcs_experiments.probability_check import check_probability_solution
from evcs_experiments.probability_phase1 import solve_phase1
from evcs_experiments.extensive import assemble_extensive,solve_extensive
from evcs_experiments.recourse import assemble_recourse,solve_fixed_plan
from evcs_experiments.recourse_check import check_recourse
from evcs_experiments.validation.s02_toys import _json_file,_save_solution,_csv_file
from evcs_experiments.validation.s03_toys import build_golden_cases
from evcs_experiments.validation.validate_s02_first_stage import checked_json
from evcs_experiments.validation.validate_s03_probability import load_prerequisites,target_inputs,FEATURES,RHOS,_physical_acceptance

I01_SHA='a42357640bd50cd65d2f801b965250e839e8f969789b000c73e10f89faab5112'


def prerequisites(path,digest):
    if digest!=I01_SHA:raise ValueError('Requires pinned independently accepted S03-I01')
    card=_physical_acceptance(path,digest)
    if card['increment']!='S03-I01' or card['status']!='PASS_INCREMENT' or card['full_S03_pass']:
        raise ValueError('Wrong probability prerequisite scope')
    refs=card['source_refs'];s1=refs['s01_acceptance'];s2=refs['s02_acceptance']
    return card,load_prerequisites(s1['path'],s1['sha256'],s2['path'],s2['sha256'])


def export_phase1(inputs,out,time_limit):
    records=[]
    for target in inputs['target_order']:
        for rho in RHOS:
            ids,phi,lo,hi=target_inputs(inputs,target,rho)
            a=ProbabilitySet(ids,FEATURES,phi,lo,hi);r=solve_phase1(a,time_limit_seconds=time_limit)
            folder=out/'phase1'/target/('rho_'+str(rho));folder.mkdir(parents=True)
            np.savez_compressed(folder/'inputs.npz',features=phi,lower=lo,upper=hi,support_ids=np.array(ids),feature_ids=np.array(FEATURES))
            _json_file(folder/'solution.json',r.as_dict())
            if not r.feasible or not r.basis_verified:raise ValueError('Phase-I feasible basic witness failed: '+target+'/'+str(rho))
            p=np.asarray(r.probabilities);indices=np.asarray(r.initial_support_indices,dtype=int)
            lifted=np.zeros(len(ids));lifted[indices]=r.initial_probabilities
            raw=max(float(np.max(-p,initial=0.)),abs(float(p.sum())-1.),float(np.max(lo-phi.T@p,initial=0.)),float(np.max(phi.T@p-hi,initial=0.)))
            if not np.array_equal(lifted,p) or raw>1e-8:raise ValueError('Phase-I raw witness or support lifting changed')
            _json_file(folder/'direct_check.json',{'passed':True,'raw_max_violation':raw,'lift_exact':True,'support_threshold':0.,'renormalized':False})
            records.append({'target':target,'rho':rho,'relative_path':str(folder.relative_to(out)),'ambiguity_sha256':a.ambiguity_sha256,'complete_support':len(ids),'initial_support':len(indices),'basis_verified':r.basis_verified,'max_raw_violation':raw})
    _json_file(out/'phase1_records.json',records)
    return records


def export_toys(out,time_limit):
    records=[];cost_rows=[]
    for f in build_golden_cases():
        folder=out/'toys'/f.name;folder.mkdir(parents=True)
        # Store every explicit synthetic physical input, including all full-H tasks.
        _json_file(folder/'inputs.json',{'name':f.name,'case':f.case,'tasks':f.tasks,'scenarios':f.scenarios,
            'scenario_ids':f.scenario_ids,'ambiguity':f.probability_set.as_dict(),
            'empirical_probabilities':f.empirical_probabilities,'scenario_cost_constants_usd':f.scenario_cost_constants_usd,
            'expected':f.expected,'hand_branches':f.hand_branches,'hand_formula':f.hand_formula})
        first=assemble_first_stage(f.case,f.tasks)
        recourses=tuple(assemble_recourse(f.case,f.tasks,s) for s in f.scenarios)
        totals={}
        for risk in ('SAA','DRO','ARO'):
            d=folder/risk;d.mkdir()
            model=assemble_extensive(first,recourses,risk_model=risk,scenario_ids=f.scenario_ids,
                probability_set=f.probability_set if risk=='DRO' else None,
                empirical_probabilities=f.empirical_probabilities if risk=='SAA' else None,
                scenario_cost_constants_usd=f.scenario_cost_constants_usd)
            save_npz(d/'matrix.npz',model.A)
            np.savez_compressed(d/'model_arrays.npz',row_lower=model.row_lower,row_upper=model.row_upper,
                lower_bounds=model.lower_bounds,upper_bounds=model.upper_bounds,c=model.c,integrality=model.integrality,
                objective_constant_usd=model.objective_constant_usd)
            _json_file(d/'model.json',model.as_dict())
            solved=solve_extensive(model,time_limit_seconds=time_limit,mip_relative_gap=1e-9)
            _json_file(d/'solver.json',solved.as_dict())
            if not solved.optimal or solved.solution is None:raise ValueError('Small extensive reference not certified optimal: '+f.name+'/'+risk)
            s=solved.solution;plan=s.plan
            np.savez_compressed(d/'solution.npz',vector=s.vector,x=plan.x,r=plan.r,
                task_sha256=plan.task_sha256,config_hash=plan.config_hash)
            fc=check_first_stage(f.case,f.tasks,plan);_json_file(d/'first_stage_check.json',fc)
            if not fc['passed']:raise ValueError('Independent first-stage check failed')
            q=[];witness_costs=[]
            for i,(scene,template,witness) in enumerate(zip(f.scenarios,recourses,s.recourse_solutions)):
                sd=d/('scenario_'+str(i));sd.mkdir()
                _save_solution(sd/'extensive_witness.npz',witness)
                physical=check_recourse(f.case,f.tasks,scene,plan,witness,check_objective_bound=False)
                _json_file(sd/'extensive_physical_check.json',physical)
                if not physical['passed']:raise ValueError('Extensive raw witness failed physical check')
                witness_costs.append(physical['costs']['Qop']+f.scenario_cost_constants_usd[i])
                # Nonbinding extensive recourse copies may be nonoptimal. Resolve
                # the original physical LP at this exact unchanged plan for true q.
                fixed=solve_fixed_plan(template,plan,time_limit_seconds=time_limit)
                _json_file(sd/'fixed_plan_solver.json',fixed.as_dict())
                if not fixed.optimal or not fixed.certificate['optimal_certificate_valid']:raise ValueError('Fixed-plan recourse lacks optimal certificate')
                _save_solution(sd/'fixed_plan_solution.npz',fixed.solution)
                checked=check_recourse(f.case,f.tasks,scene,plan,fixed.solution,check_objective_bound=True)
                _json_file(sd/'fixed_plan_physical_check.json',checked)
                if not checked['passed']:raise ValueError('Fixed-plan independent physical check failed')
                constant=float(f.scenario_cost_constants_usd[i]);q.append(checked['costs']['Qop']+constant)
                cost_rows.append({'case_id':f.name,'risk_model':risk,'scenario_id':f.scenario_ids[i],
                    **checked['costs'],'external_diagnostic_constant_usd':constant,'q_usd':q[-1],
                    'Fpre_usd':fc['costs']['Fpre_usd'],'Fpre_included_in_q':False})
            q=np.asarray(q);prob=solve_worst_distribution(f.probability_set,q,q_support_ids=f.scenario_ids,time_limit_seconds=time_limit)
            probcheck=check_probability_solution(f.probability_set,q,prob,q_support_ids=f.scenario_ids)
            _json_file(d/'fixed_plan_probability.json',prob.as_dict());_json_file(d/'fixed_plan_probability_check.json',probcheck.as_dict())
            if not prob.certified_optimal or not probcheck.passed:raise ValueError('True q probability certificate failed')
            risk_value={'SAA':float(np.asarray(f.empirical_probabilities)@q),'DRO':prob.dual_objective_usd,'ARO':float(q.max())}[risk]
            upper=fc['costs']['Fpre_usd']+risk_value
            expected=f.expected[risk];known=float(expected['objective_usd']);tol=1e-4+1e-7*max(1.,abs(known),abs(upper))
            lb=solved.lower_bound_usd
            if lb is None or not lb<=known+tol or not known<=upper+tol or abs(upper-known)>tol or abs(solved.objective_usd-known)>tol:
                raise ValueError('Golden optimum or certified bound mismatch: '+f.name+'/'+risk)
            if abs(float(plan.x.sum())-expected['selected_count'])>1e-8:raise ValueError('Golden selected count mismatch')
            totals[risk]=upper
            _json_file(d/'independent_plan_evaluation.json',{'status':'PASS_FIXED_PLAN_REFERENCE_CHECK','scenario_ids':f.scenario_ids,
                'q_usd':q,'extensive_witness_costs_usd':witness_costs,'witness_costs_are_not_assumed_optimal':True,
                'Fpre_usd':fc['costs']['Fpre_usd'],'risk_value_usd':risk_value,'full_support_upper_bound_usd':upper,
                'solver_certified_lower_bound_usd':lb,'known_golden_optimum_usd':known,'tolerance_usd':tol,
                'plan_unchanged':True,'C_and_CG_in_scope':False})
            records.append({'case_id':f.name,'risk_model':risk,'relative_path':str(d.relative_to(out)),
                'vehicles':len(set(f.tasks.vehicle_ids)),'tasks':len(f.tasks.task_ids),'periods':len(f.case.periods),'scenarios':len(f.scenarios),
                'selected':float(plan.x.sum()),'objective_usd':solved.objective_usd,'lower_bound_usd':lb,'evaluated_upper_bound_usd':upper,
                'known_optimum_usd':known,'model_sha256':model.model_sha256,'status':solved.status})
        mean=f.probability_set.features.T@np.asarray(f.empirical_probabilities)
        feasible=bool(np.all(mean>=f.probability_set.lower-1e-8) and np.all(mean<=f.probability_set.upper+1e-8))
        if totals['DRO']>totals['ARO']+1e-6 or (feasible and totals['SAA']>totals['DRO']+1e-6):raise ValueError('Risk-model ordering failed')
    _json_file(out/'extensive_records.json',records);_csv_file(out/'extensive_reference_comparison.csv',records)
    _csv_file(out/'objective_components.csv',cost_rows)
    return records,cost_rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case-root',type=Path,default=get_paths().data_root)
    p.add_argument('--s03-i01-acceptance',type=Path,required=True);p.add_argument('--s03-i01-acceptance-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--time-limit',type=float,default=30.)
    args=p.parse_args();paths=get_paths();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);began=time.monotonic()
    config=Path(os.environ['EVCS_CONFIG']).resolve() if 'EVCS_CONFIG' in os.environ else None
    _json_file(out/'execution.json',{'argv':sys.argv,'python':sys.version,'paths':paths.as_dict(),'config_path':config,
        'config_sha256':sha256(config) if config else None,'runtime_policy_sha256':sha256(paths.spec_root/'runtime_policy.json'),
        'scope':'S03-I02 phase-I on accepted development supports and complete synthetic extensive MILP reference only'})
    _json_file(out/'progress.json',{'phase':'tests','status':'RUNNING'})
    with (out/'tests.stdout.log').open('w') as stdout,(out/'tests.stderr.log').open('w') as stderr:
        test=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(paths.project_root/'tests'),'-p','test_*.py','-v'],cwd=paths.project_root,stdout=stdout,stderr=stderr)
    log=(out/'tests.stderr.log').read_text();match=re.search(r'Ran (\d+) tests? in',log)
    if test.returncode or not match or int(match[1])<264 or not log.rstrip().endswith('OK') or 'skipped=' in log:raise ValueError('Full tests must pass without skips')
    case=load_physical_case(args.case_root);card,inputs=prerequisites(args.s03_i01_acceptance,args.s03_i01_acceptance_sha256)
    _json_file(out/'sources.json',{'s03_i01_acceptance':{'path':args.s03_i01_acceptance,'sha256':args.s03_i01_acceptance_sha256},'inherited':inputs['source_refs']})
    _json_file(out/'progress.json',{'phase':'phase_I','status':'RUNNING'});phase1=export_phase1(inputs,out,args.time_limit)
    _json_file(out/'progress.json',{'phase':'synthetic_full_extensive','status':'RUNNING'});records,costs=export_toys(out,args.time_limit)
    _json_file(out/'model_code_manifest.json',source_provenance())
    _json_file(out/'increment_validation.json',{'increment':'S03-I02','status':'PASS_INCREMENT_AUTOMATED','full_S03_pass':False,
        'scientific_acceptance_pending':True,'tests':int(match[1]),'phase1_real_support_cases':len(phase1),'synthetic_datasets':3,
        'extensive_risk_models':3,'extensive_solutions':len(records),'fixed_plan_scenario_solutions':len(costs),
        'case_sha256':case.case_sha256,'new_production_scenarios':0,'formal_OOS_outcomes_generated':False,'moments_changed':False,
        'deferred':'Independent saved source/matrix/physical/phaseI and golden audit; C&CG, resume and 80EV/160support trial still pending',
        'total_wall_seconds':time.monotonic()-began})
    _json_file(out/'progress.json',{'phase':'complete','status':'AWAITING_INDEPENDENT_REVIEW'})
    _json_file(out/'artifact_sha256.json',{str(f.relative_to(out)):sha256(f) for f in sorted(out.rglob('*')) if f.is_file()})
    print(json.dumps({'status':'PASS_INCREMENT_AUTOMATED','increment':'S03-I02','output':str(out),'extensive_solutions':len(records)}));return 0


if __name__=='__main__':raise SystemExit(main())
