"""Save S03-I03 small-instance C&CG and resume evidence; no production/OOS runs."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np
from scipy.sparse import save_npz

from evcs_experiments.ccg import CCGProblem, run_ccg
from evcs_experiments.config import get_paths
from evcs_experiments.evidence import sha256, source_provenance
from evcs_experiments.first_stage import assemble_first_stage
from evcs_experiments.first_stage_check import check_first_stage
from evcs_experiments.probability import solve_worst_distribution
from evcs_experiments.probability_check import check_probability_solution
from evcs_experiments.recourse import assemble_recourse, solve_fixed_plan
from evcs_experiments.recourse_check import check_recourse
from evcs_experiments.validation.s02_toys import _json_file, _save_solution, _csv_file, _serial
from evcs_experiments.validation.s03_toys import build_golden_cases
from evcs_experiments.validation.validate_s03_probability import _physical_acceptance
from evcs_experiments.validation.validate_s03_extensive import prerequisites

I02_SHA = '88e7ea6eac9f3f51109f7b478fab07b67f139631d1f1de41f74d3b34cc37596b'


def load_i02(path, digest):
    if digest != I02_SHA:
        raise ValueError('S03-I03 requires the pinned independently accepted I02')
    card = _physical_acceptance(path, digest)
    if card['increment'] != 'S03-I02' or card['status'] != 'PASS_INCREMENT' or card['full_S03_pass']:
        raise ValueError('I02 prerequisite scope mismatch')
    for name, expected in card['reviews'].items():
        if sha256(Path(path).parent/name) != expected:
            raise ValueError('I02 independent review changed')
    for key in ('config', 'implementation_review'):
        ref = card[key]
        if sha256(ref['path']) != ref['sha256']:
            raise ValueError('I02 prerequisite identity changed: '+key)
    if sha256(Path(card['evidence'])/'increment_validation.json') != card['validation_sha256']:
        raise ValueError('I02 validation changed')
    ref = card['source_refs']['s03_i01_acceptance']
    prerequisites(ref['path'], ref['sha256'])  # Original I01/S01/S02 hashes; no repeated solve.
    return card


def _save_model(folder, model, result):
    save_npz(folder/'matrix.npz', model.A)
    np.savez_compressed(folder/'model_arrays.npz', row_lower=model.row_lower, row_upper=model.row_upper,
        lower_bounds=model.lower_bounds, upper_bounds=model.upper_bounds, c=model.c,
        integrality=model.integrality, objective_constant_usd=model.objective_constant_usd)
    _json_file(folder/'model.json', model.as_dict())
    _json_file(folder/'solver.json', result.as_dict())
    if 'native_interface' in result.raw_solver and result.raw_solver.get('mip_start_supplied', True):
        from evcs_experiments.extensive import full_shedding_start
        np.savez_compressed(folder/'mip_start.npz', vector=full_shedding_start(model))
    if result.solution is not None:
        s = result.solution
        np.savez_compressed(folder/'solution.npz', vector=s.vector, x=s.plan.x, r=s.plan.r,
            task_sha256=s.plan.task_sha256, config_hash=s.plan.config_hash)


def export_ccg(out, time_limit, reference_root=None, *, lazy=False, zero_presolve=False, native_start=False,
               master_backend='highs', optimizer_threads=1):
    rows, traces = [], []
    for f in build_golden_cases():
        folder = out/'toys'/f.name
        folder.mkdir(parents=True)
        inputs = {'name': f.name, 'case': f.case, 'tasks': f.tasks, 'scenarios': f.scenarios,
            'scenario_ids': f.scenario_ids, 'ambiguity': f.probability_set.as_dict(),
            'empirical_probabilities': f.empirical_probabilities, 'scenario_cost_constants_usd': f.scenario_cost_constants_usd,
            'expected': f.expected, 'hand_branches': f.hand_branches, 'hand_formula': f.hand_formula}
        if reference_root is not None:
            original = reference_root/'toys'/f.name/'inputs.json'
            if json.loads(original.read_text()) != _serial(inputs):
                raise ValueError('Synthetic reference inputs changed from accepted I02')
        _json_file(folder/'inputs.json', inputs)
        first = assemble_first_stage(f.case, f.tasks)
        templates = tuple(assemble_recourse(f.case, f.tasks, s) for s in f.scenarios)
        for i, template in enumerate(templates):
            d = folder/('template_'+str(i)); d.mkdir()
            for name in ('A_ub', 'A_eq', 'Bx_ub', 'Bx_eq', 'Br_ub', 'Br_eq'):
                save_npz(d/(name+'.npz'), getattr(template, name))
            np.savez_compressed(d/'arrays.npz', b_ub=template.b_ub, b_eq=template.b_eq,
                lower_bounds=template.lower_bounds, upper_bounds=template.upper_bounds, c=template.c)
            _json_file(d/'identity.json', {'template_sha256': template.template_sha256,
                'scenario_sha256': template.scenario_sha256, 'scenario_id': f.scenario_ids[i]})
        for risk in ('DRO', 'ARO'):
            recourses = templates
            if lazy:
                from evcs_experiments.lazy_recourses import LazyRecourses
                recourses = LazyRecourses(f.scenario_ids, [s.scenario_sha256 for s in f.scenarios],
                    first.matrix_sha256, {'fixture': f.name, 'template_sha256s': [t.template_sha256 for t in templates]},
                    templates.__getitem__, capacity=1)
            problem = CCGProblem(first, recourses, f.probability_set, f.scenario_cost_constants_usd, risk)
            for mode in ('continuous', 'resumed'):
                d = folder/risk/mode; d.mkdir(parents=True)
                events = []

                def sink(event, iteration, **data):
                    ed = d/'events'/f'{len(events):06d}_{event}'
                    ed.mkdir(parents=True)
                    meta = {'event': event, 'iteration': iteration, 'path': str(ed.relative_to(d))}
                    if event == 'rmp':
                        model, result = data['model'], data['result']
                        _save_model(ed, model, result)
                        if result.solution is not None:
                            plan = result.solution.plan
                            fc = check_first_stage(f.case, f.tasks, plan)
                            _json_file(ed/'first_stage_check.json', fc)
                            if not fc['passed']: raise ValueError('RMP first-stage physical failure')
                            for sid, y in zip(model.scenario_ids, result.solution.recourse_solutions):
                                j = f.scenario_ids.index(sid)
                                _save_solution(ed/f'witness_{j}.npz', y)
                                checked = check_recourse(f.case, f.tasks, f.scenarios[j], plan, y,
                                                         check_objective_bound=False)
                                _json_file(ed/f'physical_{j}.json', checked)
                                if not checked['passed']: raise ValueError('RMP recourse physical failure')
                    elif event == 'recourse':
                        j, plan, result = data['index'], data['plan'], data['result']
                        meta['scenario_index'] = j
                        _json_file(ed/'solver.json', result.as_dict())
                        _json_file(ed/'plan.json', {'x':plan.x, 'r':plan.r,
                            'task_sha256':plan.task_sha256, 'config_hash':plan.config_hash})
                        if result.solution is not None:
                            _save_solution(ed/'solution.npz', result.solution)
                            checked = check_recourse(f.case, f.tasks, f.scenarios[j], plan,
                                                     result.solution, check_objective_bound=result.optimal)
                            _json_file(ed/'physical_check.json', checked)
                            if not checked['passed']: raise ValueError('Fixed-plan physical failure')
                    elif event in ('phase1', 'probability'):
                        _json_file(ed/'solver.json', data['result'].as_dict())
                    elif event == 'iteration':
                        _json_file(ed/'trace.json', data['record'])
                    events.append(meta)
                    _json_file(d/'events.json', events)

                checkpoint = d/'checkpoint.json'
                kwargs = dict(checkpoint_path=checkpoint, wall_seconds=time_limit,
                    rmp_time_limit_seconds=time_limit, oracle_time_limit_seconds=time_limit,
                    mip_relative_gap=1e-9, optimality_relative_gap=1e-8, evidence_sink=sink)
                if zero_presolve:
                    kwargs['zero_presolve'] = True
                if native_start:
                    kwargs.update(native_start=True,solver_log_directory=d/'solver_logs')
                if master_backend != 'highs':
                    kwargs.update(master_backend=master_backend,optimizer_threads=optimizer_threads,
                                  solver_log_directory=d/'solver_logs')
                result = run_ccg(problem, max_iterations=1 if mode == 'resumed' else 100, **kwargs)
                session = 1
                shutil.copyfile(checkpoint, d/f'checkpoint_session_{session}.json')
                while mode == 'resumed' and result.status == 'ITERATION_LIMIT' and session < 20:
                    result = run_ccg(problem, resume=True, max_iterations=1, **kwargs)
                    session += 1
                    shutil.copyfile(checkpoint, d/f'checkpoint_session_{session}.json')
                _json_file(d/'result.json', result.as_dict())
                if result.status != 'CONVERGED':
                    raise ValueError(f'Small C&CG did not converge: {f.name}/{risk}/{mode}: {result.status}')
                # Independently re-solve every scene at the returned historical best plan.
                audit = d/'best_plan_recheck'; audit.mkdir()
                plan = result.best_plan; q = []
                for j, template in enumerate(templates):
                    rd = audit/str(j); rd.mkdir()
                    solved = solve_fixed_plan(template, plan, time_limit_seconds=time_limit)
                    _json_file(rd/'solver.json', solved.as_dict())
                    if not solved.optimal or not solved.certificate['optimal_certificate_valid']:
                        raise ValueError('Returned best-plan LP lacks certificate')
                    _save_solution(rd/'solution.npz', solved.solution)
                    checked = check_recourse(f.case, f.tasks, f.scenarios[j], plan, solved.solution,
                                             check_objective_bound=True)
                    _json_file(rd/'physical_check.json', checked)
                    if not checked['passed']: raise ValueError('Best-plan physics failed')
                    q.append(checked['costs']['Qop']+f.scenario_cost_constants_usd[j])
                prob = solve_worst_distribution(problem.ambiguity, q, q_support_ids=f.scenario_ids,
                                               time_limit_seconds=time_limit)
                checked = check_probability_solution(problem.ambiguity, q, prob, q_support_ids=f.scenario_ids)
                _json_file(audit/'probability.json', prob.as_dict())
                _json_file(audit/'probability_check.json', checked.as_dict())
                if not prob.certified_optimal or not checked.passed: raise ValueError('Best-plan probability check failed')
                upper = first.evaluate(plan).objective_usd+prob.upper_bound_usd
                known = f.expected[risk]['objective_usd']; lb = result.state['lower_bound_usd']
                if abs(upper-known)>1e-6 or lb>known+1e-6 or not np.allclose(q,result.state['best']['q_usd'],atol=1e-8,rtol=0):
                    raise ValueError('Historical best-plan/reference mismatch')
                reference = None if reference_root is None else reference_root/'toys'/f.name/risk/'solver.json'
                if reference is not None and abs(json.loads(reference.read_text())['objective_usd']-upper)>1e-6:
                    raise ValueError('I02 accepted full-reference comparison failed')
                record = {'case_id':f.name, 'risk_model':risk, 'mode':mode, 'status':result.status,
                    'iterations':len(result.state['trace']), 'sessions':session, 'known_optimum_usd':known,
                    'lower_bound_usd':lb, 'algorithm_upper_bound_usd':result.state['upper_bound_usd'],
                    'audit_upper_bound_usd':upper, 'elapsed_seconds':result.state['elapsed_seconds'],
                    'reference_path':None if reference is None else str(reference),
                    'reference_sha256':None if reference is None else sha256(reference)}
                rows.append(record)
                for row in result.state['trace']:
                    traces.append({'case_id':f.name, 'risk_model':risk, 'mode':mode, **row})
    _csv_file(out/'extensive_ccg_comparison.csv', rows)
    _csv_file(out/'ccg_trace.csv', traces)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-root', type=Path, default=get_paths().data_root)
    parser.add_argument('--s03-i02-acceptance', type=Path, required=True)
    parser.add_argument('--s03-i02-acceptance-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--time-limit', type=float, default=30.)
    args = parser.parse_args(); paths = get_paths(); out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False); began = time.monotonic()
    _json_file(out/'progress.json', {'phase':'prerequisites', 'status':'RUNNING'})
    card = load_i02(args.s03_i02_acceptance, args.s03_i02_acceptance_sha256)
    _json_file(out/'sources.json', {'s03_i02_acceptance':{'path':args.s03_i02_acceptance, 'sha256':I02_SHA},
                                 'inherited':card['source_refs']})
    config = Path(os.environ['EVCS_CONFIG']).resolve() if 'EVCS_CONFIG' in os.environ else None
    _json_file(out/'execution.json', {'argv':sys.argv, 'python':sys.version, 'paths':paths.as_dict(),
        'config_path':config, 'config_sha256':sha256(config) if config else None,
        'runtime_policy_sha256':sha256(paths.spec_root/'runtime_policy.json'), 'production_or_OOS':False})
    with (out/'tests.stdout.log').open('w') as stdout, (out/'tests.stderr.log').open('w') as stderr:
        test = subprocess.run([sys.executable,'-m','unittest','discover','-s',str(paths.project_root/'tests'),
                               '-p','test_*.py','-v'], cwd=paths.project_root, stdout=stdout, stderr=stderr)
    log = (out/'tests.stderr.log').read_text(); match = re.search(r'Ran (\d+) tests? in', log)
    if test.returncode or not match or int(match[1])<280 or not log.rstrip().endswith('OK') or re.search(r'skipp?ed|expected failure|unexpected success',log,re.I):
        raise ValueError('Full tests must pass with no skips')
    _json_file(out/'progress.json', {'phase':'ccg_and_resume_goldens', 'status':'RUNNING'})
    records = export_ccg(out, args.time_limit, Path(card['evidence']))
    _json_file(out/'model_code_manifest.json', source_provenance())
    _json_file(out/'increment_validation.json', {'increment':'S03-I03', 'status':'PASS_INCREMENT_AUTOMATED',
        'full_S03_pass':False, 'scientific_acceptance_pending':True, 'tests':int(match[1]),
        'synthetic_datasets':3, 'risk_models':2, 'runs':len(records), 'records':records,
        'new_production_scenarios':0, 'formal_OOS_outcomes_generated':False,
        'deferred':'Independent saved-matrix/trace/checkpoint scientific audit and full80EV/160support performance trial',
        'total_wall_seconds':time.monotonic()-began})
    _json_file(out/'progress.json', {'phase':'complete', 'status':'AWAITING_INDEPENDENT_REVIEW'})
    _json_file(out/'artifact_sha256.json', {str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print(json.dumps({'status':'PASS_INCREMENT_AUTOMATED', 'increment':'S03-I03', 'runs':len(records), 'output':str(out)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
