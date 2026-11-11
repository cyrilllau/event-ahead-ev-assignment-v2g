#!/usr/bin/env python3
"""Replay an unchanged saved Gurobi RMP diagnostic; never invoke a solver.

Independently multiply the original matrix, decode the original physical
column layout, and run direct-physics checkers. The native MIP bound remains
a solver certificate, not an independently reconstructed branch-and-bound
proof. Historical full-support evidence is verified by its pinned hashes.
"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.sparse import load_npz

from evcs_experiments.config import get_paths
from evcs_experiments.first_stage_check import check_first_stage
from evcs_experiments.physical_inputs import (
    load_frozen_tasks, load_physical_case, load_scenario_physical_inputs,
)
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.recourse_check import check_recourse


SOURCE = '732bb3b93057c783d4d5d0c4a21255cc01c1e28c8e353b96503156b55c57ccd2'
CHECKPOINT = '001b47b5add3e0d2aff1c5ee3c09b256cddc2486cf1a6a9d187b42439242c6b0'
FULL_Q_AUDIT = 'd7242c926f0b15cb185a581e4408ade46a78b58b2de771f2ae1cc29e3fd977c8'
REQUEST = '240cecaa524ff42d4c978e487677cec5d5a235ad61314c508074c04b0d53fc8e'
MODEL = '35bf987f2735d150aa25f9302c953346669c77bfefc60a858d08385e228d1256'
ORDER = ('rav', 'p', 'q', 'short', 'shed', 'line_p', 'line_q', 'u', 'soc_dep')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key].copy() for key in values.files}


def model_digest(model, matrix, data):
    meta = dict(risk_model=model['risk_model'], scenario_ids=model['scenario_ids'],
        scenario_sha256s=model['scenario_sha256s'], template_sha256s=model['recourse_template_sha256s'],
        first_stage_sha256=model['first_stage_sha256'], ambiguity_sha256=model['ambiguity_sha256'],
        empirical_probabilities=model['empirical_probabilities'], slices=model['variable_slices'],
        objective_constant_usd=model['objective_constant_usd'])
    digest = hashlib.sha256(json.dumps(meta, sort_keys=True, separators=(',', ':')).encode())
    for value in (matrix.data, matrix.indices, matrix.indptr, data['row_lower'], data['row_upper'],
                  data['lower_bounds'], data['upper_bounds'], data['c'], data['integrality'],
                  np.array(model['scenario_cost_constants_usd'])):
        value = np.asarray(value, dtype='<i8' if np.asarray(value).dtype.kind in 'iu' else '<f8')
        digest.update(str(value.shape).encode()); digest.update(value.tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    began = time.monotonic(); checks = 0
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'audit.json').exists():
        raise ValueError('Refusing to overwrite an audit')

    def need(condition, message):
        nonlocal checks
        checks += 1
        if not bool(condition):
            raise ValueError(message)

    def ref(value):
        need(sha(value['path']) == value['sha256'], 'Changed reference: '+value['path'])

    def save(name, value):
        with (out/name).open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')

    launch = read(args.launch/'launch_context.json')
    for value in launch.values():
        if isinstance(value, dict) and 'path' in value and 'sha256' in value:
            ref(value)
    request_path = args.launch/'gurobi_request.json'
    need(sha(request_path) == REQUEST, 'Unpinned diagnostic request')
    request = read(request_path); evidence = Path(request['output'])
    paths = get_paths(); project = paths.project_root
    need(str(project) == request['project'], 'Wrong installed package project')
    need(sha(project/'RELEASE_MANIFEST.json') == SOURCE, 'Wrong source manifest')
    source = read(project/'RELEASE_MANIFEST.json')
    for value in source['files']:
        ref(dict(path=str(project/value['path']), sha256=value['sha256']))
    for value in request['references'].values():
        ref(value)
    refs = request['references']
    manifest = read(evidence/'artifact_sha256.json')
    for name, digest in manifest.items():
        need(sha(evidence/name) == digest, 'Changed diagnostic artifact: '+name)
    execution = read(evidence/'execution.json'); summary = read(evidence/'summary.json')
    need(execution['request'] == request and execution['request_sha256'] == REQUEST,
         'Execution/request binding mismatch')
    need(execution['script_sha256'] == refs['script']['sha256'] and
         execution['original_matrix_sha256'] == refs['matrix']['sha256'], 'Execution source binding')
    need(execution['OOS'] is False and execution['full_S03_pass'] is False and
         request['no_ccg_checkpoint_mutation'] and request['original_integer_model'], 'Scope changed')
    job_path = paths.campaign_root/'jobs'/launch['job_id']/'result.json'; job = read(job_path)
    need(job['complete'] and job['returncode'] == 0 and job['state'] == 'SUCCEEDED', 'Job did not complete normally')
    need(0 < summary['solver_seconds'] <= summary['wall_seconds'] <= job['elapsed_seconds'] < 600,
         'Invalid timing hierarchy')
    need(summary['status'] == 'TIME_LIMIT' and summary['native_status'] == 9,
         'Expected true time-limited diagnostic')
    need(summary['gurobi_version'] == [12, 0, 3] and summary['effective_parameters']['Threads'] == 4 and
         summary['effective_parameters']['SoftMemLimit'] == 24 and
         summary['effective_parameters']['TimeLimit'] == 480, 'Solver version/resource limits changed')
    need(summary['peak_rss_bytes'] > 0 and 0 < summary['max_solver_memory_gb'] < 24,
         'Invalid measured memory')

    matrix = load_npz(refs['matrix']['path']).tocsr()
    data = arrays(refs['model_arrays']['path']); model = read(refs['model_metadata']['path'])
    values = arrays(evidence/'solution.npz'); vector = values['vector']
    need(model_digest(model, matrix, data) == model['model_sha256'] == MODEL, 'Original model digest changed')
    need(matrix.shape == (1996166, 332795) and matrix.nnz == 9587112 and
         len(model['task_ids']) == 1008 and len(model['scenario_ids']) == 4, 'Original complete model scale changed')
    need(model['first_stage_fixed'] is False and model['all_original_tasks_retained'], 'First-stage domain changed')
    need(vector.shape == data['c'].shape and np.isfinite(vector).all(), 'Malformed raw vector')
    need(np.array_equal(np.flatnonzero(data['integrality']), np.arange(1008)) and
         np.all(data['integrality'][:1008] == 1), 'Original integrality changed')
    lo, hi = data['row_lower'], data['row_upper']
    equality = np.isfinite(lo) & np.isfinite(hi) & (lo == hi)
    mapping = dict(eq=np.flatnonzero(equality), le=np.flatnonzero(np.isfinite(hi) & ~equality),
                   ge=np.flatnonzero(np.isfinite(lo) & ~equality))
    for name, indices in mapping.items():
        need(np.array_equal(indices, values[name+'_source_rows']), 'Original row mapping mismatch: '+name)
    need(sum(len(rows) for rows in mapping.values()) == summary['model_rows'] == matrix.shape[0],
         'Native model row coverage changed')
    activity = matrix @ vector
    residual_arrays = dict(row_lower=lo-activity, row_upper=activity-hi,
        column_lower=data['lower_bounds']-vector, column_upper=vector-data['upper_bounds'],
        integrality=np.abs(vector[data['integrality'] == 1]-np.rint(vector[data['integrality'] == 1])))
    residuals = {name: dict(max_violation=max(0., float(np.max(value, initial=0.))),
        violation_count=int(np.count_nonzero(value > 1e-8))) for name, value in residual_arrays.items()}
    need(all(value['violation_count'] == 0 for value in residuals.values()), 'Original rows, bounds or integer violation')
    objective = float(data['c'] @ vector); constant = float(data['objective_constant_usd'])
    need(constant == model['objective_constant_usd'] == summary['objective_constant_usd'] == 0., 'Objective offset changed')
    need(abs(objective-summary['incumbent_without_constant_usd']) <= 1e-6, 'Original objective differs from ObjVal')
    bound = summary['solver_bound_without_constant_usd']
    need(np.isfinite(bound) and summary['is_mip'] and summary['bound_source'] == 'MIP_ObjBound' and
         bound <= objective and summary['solution_count'] == 2, 'Invalid native MIP bound')
    gap = abs(objective-bound)/abs(objective)
    need(abs(gap-summary['mip_gap']) < 1e-12, 'Native gap inconsistent')
    log = (evidence/'gurobi.log').read_text()
    logged = re.search(r'Best objective ([\deE+.-]+), best bound ([\deE+.-]+), gap ([\d.]+)%', log)
    need(logged is not None and 'Time limit reached' in log, 'Missing native termination/bound log')
    need(abs(float(logged[1])-objective) < 1e-6 and abs(float(logged[2])-bound) < 1e-6 and
         abs(float(logged[3])/100-gap) < 1e-6, 'Native log/result mismatch')
    need('1996166 rows, 332795 columns and 9587112 nonzeros' in log and
         '331787 continuous, 1008 integer' in log and 'Loaded user MIP start' in log,
         'Native log original matrix/hint mismatch')
    iterations = re.search(r'Explored (\d+) nodes \((\d+) simplex iterations\)', log)
    need(iterations is not None and int(iterations[1]) == summary['node_count'] and
         int(iterations[2]) == summary['root_or_tree_lp_iterations'] and
         'Barrier solved model in 34 iterations' in log and summary['barrier_iterations'] == 34,
         'Native iteration accounting mismatch')

    ccg = Path(launch['old_ccg_checkpoint']['path']).parent
    need(sha(ccg/'checkpoint.json') == CHECKPOINT, 'Historical CCG checkpoint changed')
    checkpoint = read(ccg/'checkpoint.json'); state = checkpoint['state']
    need(checkpoint['sha256'] == canonical_hash(state), 'Historical checkpoint internal hash mismatch')
    need(sha(refs['full_q_audit']['path']) == FULL_Q_AUDIT, 'Historical full-q audit changed')
    full_q = read(refs['full_q_audit']['path'])
    need(full_q['total_verified_q'] == len(state['best']['q_usd']) == 160 and
         full_q['checkpoint_sha256'] == CHECKPOINT and not full_q['full_S03_pass'], 'Historical q scope mismatch')
    historical_manifest = ccg/'sessions/0004/artifact_sha256.json'
    need(sha(historical_manifest) == full_q['session_artifact_manifest_sha256'], 'Historical artifact manifest changed')
    historical_files = read(historical_manifest)
    for name, digest in historical_files.items():
        need(sha(ccg/name) == digest, 'Historical full-q artifact changed: '+name)
    binding = read(ccg/'sessions/0001/execution.json')['binding']
    for name in ('npz', 'ledger', 'tasks'):
        ref(binding[name])
    case = load_physical_case(paths.data_root)
    tasks = load_frozen_tasks(binding['tasks']['path'], case, binding['tasks']['sha256'])
    need(tuple(model['task_ids']) == tasks.task_ids and tuple(model['periods']) == case.periods and
         model['task_sha256'] == tasks.task_sha256 and model['config_hash'] == tasks.config_hash,
         'Frozen physical axes changed')
    need(len(read(binding['tasks']['path'])['vehicle_summaries']) == 80, 'Original fleet changed')
    slices = model['variable_slices']; h = len(tasks.task_ids); t = len(case.periods)
    plan = SimpleNamespace(x=vector[slice(*slices['x'])], r=vector[slice(*slices['r'])].reshape(h,t),
                           task_sha256=tasks.task_sha256, config_hash=tasks.config_hash)
    first = check_first_stage(case, tasks, plan)
    need(first['passed'], 'Direct first-stage physics failed'); save('first_stage_check.json', first)
    physical = []
    sizes = ((h,t), (h,t), (h,t), (len(case.station_ids),t), (t,len(case.bus_ids)),
             (t,len(case.line_ids)), (t,len(case.line_ids)), (t,len(case.bus_ids)), (h,))
    for position, index in enumerate(request['original_rmp_support_indices']):
        scene = load_scenario_physical_inputs(binding['npz']['path'], binding['ledger']['path'], index,
            case, tasks, expected_npz_sha256=binding['npz']['sha256'],
            expected_ledger_sha256=binding['ledger']['sha256'],
            expected_scenario_sha256=model['scenario_sha256s'][position])
        need(scene.scenario_id == model['scenario_ids'][position] == state['identity']['scenario_ids'][index],
             'Physical scenario identity changed')
        raw = vector[slice(*slices['recourses'][scene.scenario_id])]; decoded = {}; cursor = 0
        for name, shape in zip(ORDER, sizes):
            count = int(np.prod(shape)); decoded[name] = raw[cursor:cursor+count].reshape(shape); cursor += count
        need(cursor == len(raw), 'Physical recourse column layout mismatch')
        witness = SimpleNamespace(**decoded, task_sha256=tasks.task_sha256,
            config_hash=tasks.config_hash, scenario_sha256=scene.scenario_sha256)
        checked = check_recourse(case, tasks, scene, plan, witness, check_objective_bound=False)
        need(checked['passed'], 'Direct recourse physics failed: '+str(index))
        save('physical_'+str(index)+'.json', checked)
        physical.append(dict(index=index, scenario_id=scene.scenario_id, passed=checked['passed'],
                             costs=checked['costs']))
    need(sha(ccg/'checkpoint.json') == CHECKPOINT, 'Historical checkpoint changed during audit')
    report = dict(status='PASS_ORIGINAL_GUROBI_RMP_DIAGNOSTIC_AUDIT_NOT_S03', checks_passed=checks,
        failure_count=0, full_S03_pass=False, OOS=False, new_optimization_solves=0,
        source_manifest=dict(path=str(project/'RELEASE_MANIFEST.json'), sha256=SOURCE),
        source_files=len(source['files']), diagnostic_script=refs['script'],
        audit_source=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
        request=dict(path=str(request_path), sha256=REQUEST),
        artifact_manifest=dict(path=str(evidence/'artifact_sha256.json'), sha256=sha(evidence/'artifact_sha256.json')),
        diagnostic_artifact_files=len(manifest), original_model_sha256=MODEL,
        original_matrix_shape=list(matrix.shape), original_matrix_nnz=matrix.nnz,
        original_integer_columns=1008, original_residuals=residuals,
        row_mapping_counts={name:len(rows) for name,rows in mapping.items()},
        objective_usd=objective+constant, diagnostic_solver_bound_usd=bound+constant,
        diagnostic_solver_relative_gap=gap, bound_is_native_solver_certificate=True,
        bound_independently_reconstructed=False, accepted_campaign_lower_bound=False,
        incumbent_is_full_support_upper_bound=False, selected_tasks=int(np.count_nonzero(plan.x)),
        total_reserve_kw=float(plan.r.sum()), first_stage_passed=first['passed'], physical_witnesses=physical,
        historical_checkpoint=dict(path=str(ccg/'checkpoint.json'), sha256=CHECKPOINT),
        historical_full_q_audit=refs['full_q_audit'], historical_q_count=160,
        historical_artifact_manifest=dict(path=str(historical_manifest),sha256=sha(historical_manifest)),
        historical_artifact_files=len(historical_files), historical_upper_bound_usd=full_q['upper_bound_usd'],
        historical_algorithm_seconds=state['elapsed_seconds'],
        job_result=dict(path=str(job_path), sha256=sha(job_path)), job_returncode=job['returncode'],
        job_elapsed_seconds=job['elapsed_seconds'], solver_seconds=summary['solver_seconds'],
        diagnostic_wall_seconds=summary['wall_seconds'], peak_rss_bytes=summary['peak_rss_bytes'],
        max_solver_memory_gb=summary['max_solver_memory_gb'], audit_wall_seconds=time.monotonic()-began,
        limits=['Independent primal and direct-physics replay; native MIP bound is provenance/log verified.',
                'Restricted 4-scenario incumbent is not a new 160-scenario upper bound.',
                'Historical 160-q certificate is preserved by hashes, not recomputed or relabeled.'])
    save('audit.json', report)
    save('artifact_sha256.json', {p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps({key:report[key] for key in ('status','failure_count','checks_passed','objective_usd',
        'diagnostic_solver_bound_usd','selected_tasks','total_reserve_kw','audit_wall_seconds')}))


if __name__ == '__main__':
    main()
