"""One-scenario, no-solver validation of a legacy numeric template reader.

Run with an installed evcs_experiments package and an explicitly pinned JSON
request. This does not route production work through the cache or accept a
complete planning support. The first measured read does not imply an empty OS
page cache. All legacy bytes remain unchanged.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, replace
import json
from pathlib import Path
import signal
import time
from unittest.mock import patch

import numpy as np

from evcs_experiments.config import get_paths
from evcs_experiments.evaluation_lock import file_reference, _verify_reference
from evcs_experiments.first_stage import FirstStagePlan, assemble_first_stage
from evcs_experiments.formal_physical_inputs import FormalPhysicalBank
from evcs_experiments.physical_inputs import load_physical_case, load_frozen_tasks
from evcs_experiments.recourse import assemble_recourse, full_shedding_fallback
from evcs_experiments.recourse_check import check_recourse
from evcs_experiments.recourse_store import bind_template, load_template


def need(value, message):
    if not bool(value):
        raise ValueError(message)


def read(ref):
    _verify_reference(ref)
    return json.loads(Path(ref['path']).read_text())


def normal(ref):
    value = dict(ref)
    if 'bytes' in value:
        value['size_bytes'] = value.pop('bytes')
    _verify_reference(value)
    return value


def timed(function):
    started = time.perf_counter()
    result = function()
    return result, time.perf_counter() - started


def array_equal(left, right, name):
    need(np.array_equal(left, right), 'Numeric template comparison differs: ' + name)


def compare(left, right, plan):
    comparisons = []
    for name in ('A_ub', 'A_eq', 'Bx_ub', 'Bx_eq', 'Br_ub', 'Br_eq'):
        a, b = getattr(left, name), getattr(right, name)
        need(a.shape == b.shape, 'CSR shape differs: ' + name)
        for suffix in ('data', 'indices', 'indptr'):
            array_equal(getattr(a, suffix), getattr(b, suffix), name + '.' + suffix)
            comparisons.append(name + '.' + suffix)
    for name in ('b_ub', 'b_eq', 'c', 'lower_bounds', 'upper_bounds',
                 'component_objectives', 'load_p_kw', 'arrival_soc'):
        array_equal(getattr(left, name), getattr(right, name), name)
        comparisons.append(name)
    for name in ('template_sha256', 'task_sha256', 'config_hash', 'scenario_sha256',
                 'cmax_usd', 'task_ids', 'station_ids', 'bus_ids', 'line_ids', 'periods',
                 'reference_bus_ids_by_period', 'variable_shapes', 'variable_slices',
                 'station_task_indices', 'u_ref', 'units'):
        need(getattr(left, name) == getattr(right, name), 'Metadata comparison differs: ' + name)
        comparisons.append(name)
    for index, (a, b) in enumerate(zip(left.fixed_rhs(plan), right.fixed_rhs(plan))):
        array_equal(a, b, 'candidate_fixed_rhs_' + str(index))
        comparisons.append('candidate_fixed_rhs_' + str(index))
    a, b = full_shedding_fallback(left, plan), full_shedding_fallback(right, plan)
    for field in fields(a):
        va, vb = getattr(a, field.name), getattr(b, field.name)
        need(np.array_equal(va, vb) if isinstance(va, np.ndarray) else va == vb,
             'Fallback witness differs: ' + field.name)
    array_equal(left.pack(a), right.pack(b), 'fallback_packed_vector')
    need(left.objective_components(a) == right.objective_components(b), 'Fallback objective components differ')
    return comparisons, b


def run(request, output):
    started = time.perf_counter()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    required = {'raw_request', 'template_record', 'candidate_plan', 'compiler', 'runtime_source_manifest'}
    need(set(request['references']) == required, 'Unknown benchmark references')
    refs = request['references']
    docs = {key: read(ref) for key, ref in refs.items() if key != 'compiler'}
    _verify_reference(refs['compiler'])
    raw, record = docs['raw_request'], docs['template_record']
    need(raw['domain'] == 'formal' and raw['formal_OOS'] is False, 'Wrong raw evidence scope')
    binding = raw['input_binding']
    need(binding['planning_source']['recourse.py'] == refs['compiler']['sha256'], 'Original compiler binding differs')
    import evcs_experiments.recourse as original_module
    need(file_reference(original_module.__file__)['sha256'] == refs['compiler']['sha256'],
         'Installed original recourse compiler differs')
    source_manifest = docs['runtime_source_manifest']
    source_root = Path(refs['runtime_source_manifest']['path']).parent
    for relative in ('src/evcs_experiments/recourse.py', 'src/evcs_experiments/recourse_store.py',
                     'scripts/verification/validate_recourse_store.py'):
        entries = [item for item in source_manifest['files'] if item['path'] == relative]
        need(len(entries) == 1, 'Runtime manifest lacks unique benchmark source: ' + relative)
        need(file_reference(source_root / relative)['sha256'] == entries[0]['sha256'], 'Runtime source changed')
    need(file_reference(__file__)['sha256'] == file_reference(source_root / 'scripts/verification/validate_recourse_store.py')['sha256'],
         'Executed benchmark source differs')
    task_ref = normal(binding['task_source'])
    index_ref = normal(binding['target']['planning_index'])
    index = read(index_ref)
    i = request['scenario_index']
    need(type(i) is int and 0 <= i < len(index['scenarios']), 'Invalid bounded benchmark index')
    need(record['scenario_index'] == i and record['scenario_id'] == index['scenarios'][i]['scenario_id'],
         'Template support ordering differs')

    def setup():
        case = load_physical_case(get_paths().data_root)
        tasks = load_frozen_tasks(task_ref['path'], case, task_ref['sha256'])
        bank = FormalPhysicalBank(index_ref['path'], case, tasks,
            expected_index_sha256=index_ref['sha256'], case_root=get_paths().data_root)
        scene = bank.load(i, expected_scenario_sha256=index['scenarios'][i]['scenario_sha256'])
        first = assemble_first_stage(case, tasks)
        need(first.matrix_sha256 == raw['problem_identity']['first_stage_sha256'], 'Original first-stage identity differs')
        return case, tasks, scene, first

    (case, tasks, scene, first), setup_seconds = timed(setup)
    plan = FirstStagePlan(**docs['candidate_plan'])
    need(first.evaluate(plan).feasible, 'Benchmark candidate is not first-stage feasible')
    source_refs = dict(raw_request=refs['raw_request'], tasks=task_ref, planning_index=index_ref,
                       scenario_manifest=normal(index['scenarios'][i]['manifest']))
    bind_ref, bind_seconds = timed(lambda: bind_template(refs['template_record'], first, case, tasks, scene,
        source_references=source_refs, compiler_reference=refs['compiler'], output=output / 'binding.json'))
    options = dict(expected_template_sha256=record['template_sha256'],
                   expected_scenario_index=i, expected_scenario_id=record['scenario_id'])
    guard = AssertionError('This bounded validation cannot start a solver')
    with patch('evcs_experiments.recourse.linprog', side_effect=guard), patch('scipy.optimize.linprog', side_effect=guard):
        loaded, first_load_seconds = timed(lambda: load_template(bind_ref, first, case, tasks, scene, **options))
        need(loaded.descriptive_maps_loaded is False, 'Verified load decoded descriptive maps')
        warm_seconds = []
        for unused in range(2):
            warm, elapsed = timed(lambda: load_template(bind_ref, first, case, tasks, scene, **options))
            need(warm.template_sha256 == loaded.template_sha256, 'Repeated template identity changed')
            need(warm.descriptive_maps_loaded is False, 'Repeated load decoded descriptive maps')
            warm_seconds.append(elapsed)
        original, assembly_seconds = timed(lambda: assemble_recourse(case, tasks, scene))
        (comparisons, witness), comparison_seconds = timed(lambda: compare(original, loaded, plan))
        physical = check_recourse(case, tasks, scene, plan, witness, check_objective_bound=False)
        need(physical['passed'], 'Loaded-template fallback physical check failed')
        need(loaded.descriptive_maps_loaded is False, 'Numeric comparison decoded descriptive maps')
        changed = np.array(scene.task_arrival_soc, copy=True)
        changed[0] = min(1., changed[0] + 0.001) if changed[0] < 1. else 0.999
        wrong_scene = replace(scene, task_arrival_soc=changed)
        negative = {}
        for label, scene_value, option_value in (
            ('stale_scene_sha_with_changed_actual_SOC', wrong_scene, options),
            ('wrong_expected_template_sha', scene, {**options, 'expected_template_sha256': '0' * 64}),
            ('wrong_scenario_index', scene, {**options, 'expected_scenario_index': i + 1}),
            ('wrong_scenario_id', scene, {**options, 'expected_scenario_id': record['scenario_id'] + '/wrong'})):
            try:
                load_template(bind_ref, first, case, tasks, scene_value, **option_value)
            except ValueError as exc:
                negative[label] = {'rejected': True, 'reason': str(exc)}
            else:
                raise ValueError('Independent negative control was accepted: ' + label)
    result = dict(schema_version=1, status='PASS_ONE_SCENARIO_NUMERIC_TEMPLATE_PARITY',
        scientific_scope='one original physical template; no optimization result or full-support acceptance',
        scenario_index=i, scenario_id=record['scenario_id'], template_sha256=record['template_sha256'],
        source_references=refs, binding_reference=bind_ref, candidate_plan_sha256=refs['candidate_plan']['sha256'],
        references_rehashed_on_every_load=True, formal_OOS=False, solver_calls=0, full_support_verified=False,
        production_routing_changed=False, matrix_comparisons=comparisons, exact_numeric_equality=True,
        descriptive_maps_decoded=False,
        fallback_physical_check=physical, independent_negative_controls=negative,
        timing_seconds=dict(physical_inputs_and_first_stage=setup_seconds, bind_import=bind_seconds,
            first_verified_load=first_load_seconds, repeated_verified_loads=warm_seconds,
            original_reassembly=assembly_seconds, exact_comparison=comparison_seconds,
            total=time.perf_counter()-started),
        timing_scope='First read in this process follows binding import; OS page cache is uncontrolled. '
                     'Repeated reads also rehash references and all artifacts. No 160-scenario speedup is certified.',
        identity_json_bytes=next(r['size_bytes'] for r in record['artifacts'] if Path(r['path']).name == 'identity.json'),
        numeric_npz_bytes=sum(r['size_bytes'] for r in record['artifacts'] if Path(r['path']).suffix == '.npz'))
    with (output / 'result.json').open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    def expired(unused_signal, unused_frame):
        raise TimeoutError('Bounded one-template validation exceeded 55 seconds')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(55)
    result = run(json.loads(Path(args.request).read_text()), args.output)
    signal.alarm(0)
    print(json.dumps({key: result[key] for key in ('status', 'scenario_id', 'timing_seconds')}, indent=2))


if __name__ == '__main__':
    main()
