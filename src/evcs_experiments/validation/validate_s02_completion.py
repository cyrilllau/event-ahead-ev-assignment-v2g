"""S02-I03 evidence closure. Produces evidence; never signs scientific PASS."""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy import sparse

from evcs_experiments.config import get_paths
from evcs_experiments.evidence import historical_path, sha256, source_provenance
from evcs_experiments.physical_inputs import load_scenario_physical_inputs
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.recourse import assemble_recourse, full_shedding_fallback, solve_fixed_plan
from evcs_experiments.recourse_check import check_recourse
from evcs_experiments.validation.validate_s02_first_stage import checked_json, write_json, write_mapping
from evcs_experiments.validation.validate_s02_recourse import load_case_from_i01, save_solution, DUAL_FIELDS


def accepted_i02(path, expected):
    card = checked_json(path, expected)
    if card['increment'] != 'S02-I02' or card['status'] != 'PASS_INCREMENT' or card['full_S02_pass']:
        raise ValueError('Requires independent I02 acceptance')
    evidence = Path(card['evidence'])
    manifest = checked_json(evidence/'artifact_sha256.json', card['artifact_manifest_sha256'])
    for name, digest in manifest.items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha256(evidence/name) != digest:
            raise ValueError('Accepted I02 evidence changed')
    code = checked_json(card['code_manifest']['path'], card['code_manifest']['sha256'])
    for name, digest in code['current_source_hashes'].items():
        if sha256(Path(card['source_root'])/name) != digest:
            raise ValueError('Historical I02 source changed')
    for name, digest in card['independent_audits'].items():
        checked_json(name, digest)
    return card, evidence


def export_real_case(out, case, tasks, scene, plan, *, q0=False, limit=120.):
    out.mkdir(parents=True, exist_ok=False)
    template = assemble_recourse(case, tasks, scene)
    original_hash = template.template_sha256
    if q0:
        lo, hi = template.lower_bounds.copy(), template.upper_bounds.copy()
        start, end = template.variable_slices['q']
        lo[start:end] = hi[start:end] = 0.
        template = replace(template, lower_bounds=lo, upper_bounds=hi,
            template_sha256=canonical_hash({'Q1_template': original_hash, 'diagnostic_constraint': 'q == 0'}))
    # Save the actual matrices/bounds used, including the explicit Q0 diagnostic.
    for name in ('A_ub','A_eq','Bx_ub','Bx_eq','Br_ub','Br_eq'):
        sparse.save_npz(out/(name+'.npz'), getattr(template,name))
    np.savez_compressed(out/'template_arrays.npz', **{name:getattr(template,name) for name in
        ('b_ub','b_eq','c','lower_bounds','upper_bounds','component_objectives')})
    for filename, rows in (('variable_map.csv',template.variable_map),
            ('constraint_map_ub.csv',template.constraint_map_ub),('constraint_map_eq.csv',template.constraint_map_eq)):
        write_mapping(out/filename,rows)
    write_json(out/'metadata.json', {'H':len(tasks.task_ids),'task_sha256':tasks.task_sha256,
        'scenario_id':scene.scenario_id,'scenario_sha256':scene.scenario_sha256,
        'variable_slices':dict(template.variable_slices),'template_sha256':template.template_sha256,
        'original_Q1_template_sha256':original_hash,'q0_diagnostic':q0,
        'line_outage_periods':int(scene.line_operating_outages.sum()),
        'unqualified_tasks':int((~scene.task_varsigma).sum()),
        'selected_unqualified_tasks':int(((plan.x>0.5)&~scene.task_varsigma).sum())})
    np.savez_compressed(out/'plan.npz',x=plan.x,r=plan.r,task_sha256=plan.task_sha256,config_hash=plan.config_hash)
    fallback = full_shedding_fallback(template,plan)
    save_solution(out/'fallback.npz',fallback)
    fallback_check = check_recourse(case,tasks,scene,plan,fallback,check_objective_bound=True)
    write_json(out/'fallback_check.json',fallback_check)
    if not fallback_check['passed']:
        raise ValueError('Full-shedding fallback failed')
    result = solve_fixed_plan(template,plan,time_limit_seconds=limit)
    certificate = dict(result.certificate)
    duals = {name:certificate.pop(name) for name in DUAL_FIELDS if name in certificate}
    if duals:
        np.savez_compressed(out/'dual_marginals.npz',**duals)
    write_json(out/'solver_result.json',{'status':result.status,'optimal':result.optimal,
        'objective_usd':result.objective_usd,'objective_components':dict(result.objective_components),
        'certificate':certificate,'wall_seconds':result.wall_seconds})
    if result.solution is None:
        raise ValueError('No recourse solution; solver diagnostic preserved')
    save_solution(out/'solution.npz',result.solution)
    checked = check_recourse(case,tasks,scene,plan,result.solution,check_objective_bound=True)
    write_json(out/'physical_check.json',checked)
    served = np.sum(scene.load_p_kw-result.solution.shed,axis=1)
    row = {'case':out.name,'scenario_id':scene.scenario_id,'scenario_sha256':scene.scenario_sha256,
        'q0_diagnostic':q0,'served_kwh':float(served.sum()*case.dt_h),
        'max_hourly_served_kw':float(np.max(served,initial=0.)),**checked['costs'],
        'cmax_usd':checked['cmax_usd'],'physical_passed':checked['passed'],
        'max_normalized_violation':checked['max_normalized_violation']}
    write_json(out/'summary.json',row)
    if not result.optimal or not certificate.get('optimal_certificate_valid') or not checked['passed']:
        raise ValueError('Real-case solve requires primal/dual and independent physical checks')
    if q0 and (np.max(np.abs(result.solution.q),initial=0.)>1e-8 or np.max(np.abs(served),initial=0.)>1e-5):
        raise ValueError('Q0 diagnostic should serve zero with positive fixed power factor and no source')
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-root',type=Path,default=get_paths().data_root)
    parser.add_argument('--i02-acceptance',type=Path,required=True)
    parser.add_argument('--i02-acceptance-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--lp-time-limit',type=float,default=120.)
    parser.add_argument('--stress-index',type=int,default=141)
    args=parser.parse_args()
    paths=get_paths();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    config=Path(os.environ['EVCS_CONFIG']).resolve() if 'EVCS_CONFIG' in os.environ else None
    write_json(out/'execution.json',{'argv':sys.argv,'python':sys.version,'paths':paths.as_dict(),
        'config_path':str(config) if config else None,'config_sha256':sha256(config) if config else None,
        'runtime_policy_sha256':sha256(paths.spec_root/'runtime_policy.json')})
    write_json(out/'progress.json',{'phase':'tests','status':'RUNNING'})
    with (out/'tests.stdout.log').open('w') as stdout,(out/'tests.stderr.log').open('w') as stderr:
        run=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(paths.project_root/'tests'),'-p','test_*.py','-v'],
            cwd=paths.project_root,stdout=stdout,stderr=stderr)
    if run.returncode:
        raise ValueError('Full tests failed; evidence preserved')
    card,prior=accepted_i02(args.i02_acceptance,args.i02_acceptance_sha256)
    ref=json.loads((prior/'input_reference.json').read_text())
    case,tasks,scene,plans,record=load_case_from_i01(ref['i01_acceptance'],ref['i01_acceptance_sha256'],args.case_root)
    write_json(out/'input_reference.json',{'i02_acceptance':str(args.i02_acceptance),
        'i02_acceptance_sha256':args.i02_acceptance_sha256,'i01_reference':ref,
        'task_subset_used':False,'new_scenarios_generated':0,'formal_OOS':False})
    write_json(out/'progress.json',{'phase':'toy_pack','status':'RUNNING'})
    from evcs_experiments.validation.s02_toys import export_toys
    toy_report=export_toys(out/'toy_solutions',lp_time_limit=min(args.lp_time_limit,30.))
    write_json(out/'toy_report.json',toy_report)
    if toy_report['status']!='PASS_TOY_EXPECTATIONS_AUTOMATED':
        raise ValueError('Toy expectations did not pass')
    write_json(out/'progress.json',{'phase':'legacy_13_regression_in_independent_copies','status':'RUNNING'})
    replay=out/'legacy_replay'
    replay_args=[sys.executable,str(paths.project_root/'scripts/replay_legacy.py'),
        '--destination',str(replay),'--timeout-seconds','180',
        '--run','12_reactive_support_correction/verification/verify_reactive_support.py']
    with (out/'legacy_replay.stdout.log').open('w') as stdout,(out/'legacy_replay.stderr.log').open('w') as stderr:
        run=subprocess.run(replay_args,cwd=paths.project_root,stdout=stdout,stderr=stderr,timeout=240)
    if run.returncode:
        raise ValueError('Legacy diagnostic replay failed; copied evidence preserved, originals unchanged')
    copied=replay/'IEEE33_paper_aligned_case_data_v1/12_reactive_support_correction/verification'
    legacy=json.loads((copied/'verification_summary.json').read_text())
    replay_manifest=json.loads((replay/'legacy_replay_manifest.json').read_text())
    with (copied/'checks.csv').open(newline='') as stream:
        legacy_checks=list(csv.DictReader(stream))
    if (legacy['status']!='PASS' or legacy['checks']!=13 or len(legacy_checks)!=13
            or not all(row['passed']=='True' for row in legacy_checks)
            or not replay_manifest['source_files_unchanged']):
        raise ValueError('Original thirteen diagnostic checks or frozen-source preservation failed')
    write_json(out/'legacy_13_summary.json',{'status':'PASS_LEGACY_13_REPLAY','checks':legacy_checks,
        'scope':'Historical constructive regression only; production-model evidence is separately validated',
        'argv':replay_args,'elapsed_seconds':legacy['elapsed_seconds'],
        'original_source_files_unchanged':True})
    write_json(out/'progress.json',{'phase':'full_H_Q0_diagnostic','status':'RUNNING'})
    real=out/'real_cases';real.mkdir()
    rows=[export_real_case(real/'intact_same_paid_plan_Q0',case,tasks,scene,plans['minimum_fixture'],q0=True,limit=args.lp_time_limit)]
    # Prespecified existing development stress case selected from input-only
    # inspection in the independent I02 review; no new draw or outcome selection.
    npz=historical_path(record['scenario_source']['path']);ledger=historical_path(record['scenario_ledger']['path'])
    index=args.stress_index
    stress=load_scenario_physical_inputs(npz,ledger,index,case,tasks,
        expected_npz_sha256=record['scenario_source']['sha256'],expected_ledger_sha256=record['scenario_ledger']['sha256'])
    if int(stress.line_operating_outages.sum())<=0:
        raise ValueError('Prespecified stress input has no line disruption')
    write_json(out/'stress_selection.json',{'rule':'prespecified accepted development index; input-only inspection',
        'selected_index':index,'scenario_id':stress.scenario_id,'scenario_sha256':stress.scenario_sha256,
        'line_outage_periods':int(stress.line_operating_outages.sum()),
        'unqualified_tasks':int((~stress.task_varsigma).sum()),
        'npz':str(npz),'npz_sha256':record['scenario_source']['sha256'],'ledger':str(ledger),
        'ledger_sha256':record['scenario_ledger']['sha256'],'selection_uses_optimization_outcomes':False})
    write_json(out/'progress.json',{'phase':'full_H_disrupted_input','status':'RUNNING','index':index})
    for name,plan in plans.items():
        rows.append(export_real_case(real/('disrupted_'+name),case,tasks,stress,plan,limit=args.lp_time_limit))
    with np.load(prior/'minimum_fixture/plan.npz',allow_pickle=False) as old:
        if not np.array_equal(old['x'],plans['minimum_fixture'].x) or not np.array_equal(old['r'],plans['minimum_fixture'].r):
            raise ValueError('Q0/Q1 comparison did not preserve the identical paid plan')
    with np.load(prior/'minimum_fixture/solution.npz',allow_pickle=False) as old:
        q1_served=float(np.sum(scene.load_p_kw-old['shed'])*case.dt_h)
    with np.load(prior/'zero/solution.npz',allow_pickle=False) as old:
        zero_q=float(np.max(np.abs(old['q']),initial=0.))
        zero_service=float(np.max(np.abs(scene.load_p_kw-old['shed']),initial=0.))
    if q1_served<=1. or zero_q>1e-8 or zero_service>1e-5:
        raise ValueError('Accepted actual Q1 positive/zero-plan regression differs')
    write_json(out/'paired_Q0_Q1.json',{'same_paid_plan':True,'same_scenario_sha256':scene.scenario_sha256,
        'same_task_sha256':tasks.task_sha256,'Q1_accepted_evidence':str(prior),
        'Q1_served_kwh':q1_served,'Q0_served_kwh':rows[0]['served_kwh'],
        'zero_plan_max_task_abs_q_kvar':zero_q,'zero_plan_max_bus_served_kw':zero_service,
        'interpretation':'Constructive same-plan Q1/Q0 diagnostic; not policy-effectiveness comparison'})
    toy_refs=['facet_reference_check.json','positive_Q_rav15_boundary','negative_Q_rav15_boundary',
        'Q_only_rav_zero','no_selected_EV','unavailable_unqualified_EV','rejected_p0_rav15_q15',
        'joint_MILP_useful;joint_MILP_not_useful',
        'actual_discharge_reserve_chain;reserve_without_activation;tau2_efficiency_SOC;departure_SOC_binding',
        'derated_station_polygon;rejected_station_polygon',None,None,None]
    write_mapping(out/'q1_13_coverage.csv',[{'criterion_id':f'Q1-{i+1:02d}',
        'legacy_check':old['check'],'legacy_replay_passed':old['passed'],
        'production_evidence':';'.join('toy_solutions/'+part for part in toy_refs[i].split(';')) if toy_refs[i] else 'paired_Q0_Q1.json',
        'status':'AUTOMATED_EXPECTATIONS_PASS_AWAITING_INDEPENDENT_REVIEW'} for i,old in enumerate(legacy_checks)])
    write_mapping(out/'objective_components.csv',rows)
    write_mapping(out/'variable_constraint_map.csv',[{'scope':'toys_and_joint_first_stage',
        'variables':'toy_solutions/variable_constraint_map.csv','inequalities':'toy_solutions/variable_constraint_map.csv',
        'equalities':'toy_solutions/variable_constraint_map.csv'}]+[{'scope':row['case'],'variables':f"real_cases/{row['case']}/variable_map.csv",
        'inequalities':f"real_cases/{row['case']}/constraint_map_ub.csv",'equalities':f"real_cases/{row['case']}/constraint_map_eq.csv"} for row in rows])
    write_json(out/'physical_check_report.json',{'real_cases':rows,'toy_report':'toy_report.json',
        'accepted_I01_first_stage':ref['i01_acceptance'],'accepted_I02_Q1_positive_service':str(args.i02_acceptance)})
    write_json(out/'model_code_manifest.json',source_provenance())
    write_json(out/'increment_validation.json',{'increment':'S02-I03','status':'PASS_INCREMENT_AUTOMATED',
        'scientific_acceptance_pending':True,'full_S02_pass':False,'real_case_count':len(rows),
        'H':len(tasks.task_ids),'new_scenarios_generated':0,'formal_OOS_outcomes_generated':False,
        'scope':'Toy/13-gate closure plus full-H development stress and same-paid-plan Q0 diagnostic; not policy performance',
        'total_wall_seconds':time.monotonic()-started,
        'deferred':'Independent saved-evidence audit and all-five-standard full S02 scientific acceptance'})
    write_json(out/'progress.json',{'phase':'complete','status':'AWAITING_INDEPENDENT_REVIEW'})
    write_json(out/'artifact_sha256.json',{str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print(json.dumps({'status':'PASS_INCREMENT_AUTOMATED','increment':'S02-I03','output':str(out)}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
