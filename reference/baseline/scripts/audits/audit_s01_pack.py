#!/usr/bin/env python3
"""Independent row/schema/formula audit of the deterministic S01 delivery pack.

Only path and hash infrastructure is reused; no scientific generator, exporter
formulas, or experiment kernels are imported and no sample is drawn.
The supplied I08 acceptance anchors all raw source paths. Results are written
only to a new audit directory outside the exported pack.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import zipfile

import numpy as np
from evcs_experiments.config import get_paths
from evcs_experiments.evidence import (frozen_runner_root, frozen_source_file, historical_path, legacy_transfer_manifest, LEGACY_CASE_NAME, source_provenance, verify_scientific_kernel_identity)

FEATURES = ('Phi_W', 'Phi_L', 'Phi_R', 'Phi_D', 'Phi_load')


def check(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open(newline='', encoding='utf-8') as source:
        return list(csv.DictReader(source))


def number(actual, expected, label, tolerance=0.):
    check(np.isfinite(float(actual)) and abs(float(actual) - float(expected)) <= tolerance, label)


def parse_cell(value, expected):
    if isinstance(expected, (list, tuple, dict)):
        return json.loads(value)
    if isinstance(expected, bool):
        check(value in ('True', 'False'), 'Bad Boolean CSV cell')
        return value == 'True'
    if isinstance(expected, int):
        return int(value)
    if isinstance(expected, float):
        return float(value)
    return value


def schema(path):
    result = {}
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            check(item.filename.endswith('.npy'), 'Unexpected NPZ member')
            with archive.open(item) as f:
                version = np.lib.format.read_magic(f)
                if version == (1, 0):
                    shape, order, dtype = np.lib.format.read_array_header_1_0(f)
                else:
                    check(version == (2, 0), 'Unsupported NPY version')
                    shape, order, dtype = np.lib.format.read_array_header_2_0(f)
            check(not dtype.hasobject, 'Object scientific payload')
            result[item.filename[:-4]] = {'shape': list(shape), 'dtype': dtype.str, 'fortran_order': bool(order)}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pack', required=True, type=Path)
    p.add_argument('--i08-acceptance', required=True, type=Path)
    p.add_argument('--audit-output', required=True, type=Path)
    a = p.parse_args()
    pack, audit = a.pack.resolve(), a.audit_output.resolve()
    check(pack != audit and pack not in audit.parents, 'Audit directory must be outside source pack')
    audit.mkdir(parents=True, exist_ok=False)
    hashes = {}

    def sha(path):
        path = Path(path).resolve()
        if path not in hashes:
            digest = hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
                    digest.update(chunk)
            hashes[path] = digest.hexdigest()
        return hashes[path]

    def reference(value, expected_path=None):
        path = historical_path(value['path']).resolve()
        if expected_path is not None:
            check(path == historical_path(expected_path).resolve(), 'Reference points to wrong source')
        check(path.is_file() and sha(path) == value['sha256'] and path.stat().st_size == value['bytes'],
              'Reference byte/hash mismatch: ' + str(path))
        return path

    manifest = read(pack / 'artifact_manifest.json')
    check(set(manifest) == {x.name for x in pack.iterdir() if x.is_file()} - {'artifact_manifest.json'}, 'Pack manifest is incomplete')
    for name, ref in manifest.items():
        reference(ref, pack / name)
    contract, generation = read(pack / 'generator_contract.json'), read(pack / 'generation_audit.json')
    check(contract['development_only'] and contract['formal_OOS_locked'] and not contract['formal_outcomes_generated'], 'Wrong development/formal scope')
    check(contract['additional_random_samples_drawn_by_export'] == generation['random_samples_generated'] == 0, 'Exporter generated samples')
    check(generation['status'] == 'EXPORTED_PENDING_INDEPENDENT_S01_REVIEW' and not generation['full_S01_pass'], 'Exporter self-certified the scientific gate')
    check(not generation['state_modified'] and not generation['source_artifacts_modified'], 'Exporter claims source mutation')
    check(contract['development_root'] == 20260911 and contract['static_road_root'] == 20260904, 'Wrong generator seeds')
    check(contract['formal_seed_roots_metadata_only'] == {'planning': 20260901, 'calibration': 20260902, 'oos': 20260903}, 'Wrong formal seed metadata')
    check(tuple(contract['feature_order']) == FEATURES, 'Wrong feature order')
    check(contract['axes'] == {'targets':8,'weather_grid_points':441,'closed_lines':32,'road_feature_arcs':80,
        'routing_arcs':88,'replay_slots':100,'replay_hours':'[-1,24) step 0.25','operation_hours':24,
        'buses':33,'planning_per_target':160,'calibration_per_target':80}, 'Wrong declared scientific axes')
    check(set(contract['random_mechanisms'].values()) == set(range(1,11)), 'Missing mechanism identities')
    for item in contract['contracts']:
        reference(item)
    reference(contract['paper'])
    reference(contract['exporter'])
    snapshot = reference(generation['state_snapshot'], pack / 'state_snapshot.json')
    check(snapshot != Path(generation.get('original_mutable_state_path','/')).resolve(), 'State snapshot points to mutable live state')
    i08_path = a.i08_acceptance.resolve()
    i08 = read(i08_path)
    check(i08['increment'] == 'S01-I08' and i08['status'] == 'PASS_INCREMENT', 'Anchor I08 not accepted')
    source = historical_path(i08['evidence']).resolve()
    ancestry = {}
    for row in contract['accepted_increment_ancestry']:
        path = reference(row['acceptance'])
        card = read(path)
        check(card['increment'] == row['increment'] and card['status'] == row['status'] == 'PASS_INCREMENT', 'Bad increment ancestry')
        check(historical_path(row['evidence_directory']).resolve() == historical_path(card['evidence']).resolve(), 'Wrong ancestry evidence directory')
        for ref in row['evidence_manifests']:
            reference(ref)
        ancestry[row['increment']] = (path, card)
    check(set(ancestry) == {f'S01-I{i:02d}' for i in range(1,9)}, 'Missing I01-I08 acceptance ancestry')
    check(ancestry['S01-I08'][0] == i08_path, 'Pack anchored to another I08')
    routes = historical_path(ancestry['S01-I06'][1]['evidence']).resolve()
    check(generation['accepted_increment_ancestry'] == contract['accepted_increment_ancestry'], 'Generation/contract ancestry differs')
    gates = {g['gate']:g for g in generation['gates']}
    check(set(gates) == set(range(1,7)), 'Expected six S01 gates')
    expected_gates = {1:(1,8),2:(3,4,6,7),3:(1,2,6,8),4:(5,7,8),5:(8,),6:(6,8)}
    for gate, increments in expected_gates.items():
        row = gates[gate]
        check(row['status'] == 'ACCEPTED_INCREMENT_EVIDENCE_LINKED', 'Gate status falsely claims new acceptance')
        expected_paths = [ancestry[f'S01-I{i:02d}'][0] for i in increments]
        check([reference(ref) for ref in row['acceptance_references']] == expected_paths, 'Incorrect gate evidence mapping')
        check([historical_path(x).resolve() for x in row['evidence_directories']] ==
              [historical_path(ancestry[f'S01-I{i:02d}'][1]['evidence']).resolve() for i in increments], 'Gate evidence directories mismatch')
    check(sha(pack/'features.csv') == sha(source/'features.csv'), 'Feature CSV changed')
    check(sha(pack/'scenario_ledger.csv') == sha(source/'scenario_ledger.csv'), 'Scenario ledger CSV changed')
    summary = read(source/'increment_validation.json')
    events = [r['event'] for r in summary['targets']]
    check(contract['target_order'] == events and len(events) == 8, 'Target order changed')
    upstream = read(source/'upstream_validation.json')
    check(contract['input_hashes'] == upstream['input_hashes'], 'Raw input hash contract changed')
    case = get_paths().data_root
    for name, digest in contract['input_hashes'].items():
        check(sha(case/name) == digest, 'Raw input bytes changed')
    for name, digest in contract['accepted_source_hashes'].items():
        check(sha(frozen_source_file(name)) == digest, 'Accepted source changed')
    task_table, arc_table, moment_table = (rows(pack/name) for name in ('tasks.csv','task_route_arcs.csv','moment_bounds.csv'))
    diagnostics_table = rows(pack/'target_diagnostics.csv')
    check(len(task_table) == 18720 and len(moment_table) == 160 and len(diagnostics_table) == 8, 'Wrong export row counts')
    replay_index = read(pack/'task_scenario_replay_index.json')
    check(replay_index['format'] == 'lossless partitioned NPZ index' and not replay_index['merging_required'], 'Wrong replay artifact substitution')
    check(replay_index['units'] == contract['units'], 'Index/contract units differ')
    check(contract['units']['road_travel_time_min'] == 'minutes', 'Road time unit changed')
    check('DC' in contract['units']['task_usable_energy_dc_kwh'] and 'efficiency applied later' in contract['units']['task_usable_energy_dc_kwh'], 'Usable energy is not battery DC')
    check(contract['units']['load_p_kw'] == 'kW' and 'Q1' in contract['units']['load_q_kvar'], 'Power/Q1 units changed')
    reference(replay_index['static_axes'], source/'static_axes.npz')
    entries = {(r['target_event'],r['split']):r for r in replay_index['archives']}
    check(len(entries) == len(replay_index['archives']) == 16, 'Duplicate or missing replay partitions')
    check(set(entries) == {(e,s) for e in events for s in ('planning','calibration')}, 'Wrong partition IDs')
    task_refs = {r['target_event']:r for r in contract['frozen_task_sources']}
    check(set(task_refs) == set(events), 'Frozen task reference targets incomplete')
    check({reference(ref) for ref in contract['moment_sources']} == {source/f'{e}.moments.json' for e in events}, 'Moment source references incomplete')
    tpos = apos = mpos = 0
    output_records, all_scenarios = [], set()
    for event in events:
        bank_path = routes/f'{event}.tasks.json'
        bank = read(bank_path);tasks = bank['tasks']
        check(bank['certified'] and bank['status'] == 'COMPLETE', 'Uncertified H')
        task_index_path = source/f'{event}.task_index.json';task_index=read(task_index_path)
        reference(task_refs[event]['frozen_tasks'], bank_path)
        reference(task_refs[event]['task_index'], task_index_path)
        check(task_refs[event]['task_sha256'] == task_index['task_sha256'] == bank['task_sha256'], 'Frozen H identity changed')
        ids = [t['task_id'] for t in tasks]
        check(ids == task_index['task_ids'], 'Raw task index changed')
        for position, task in enumerate(tasks):
            row=task_table[tpos];tpos+=1
            check(row['target_event']==event and int(row['task_index'])==position and row['task_sha256']==bank['task_sha256'], 'Task CSV row identity changed')
            check(set(row)==set(task)|{'target_event','task_index','task_sha256'}, 'Task fields lost or added')
            for key, expected in task.items():
                check(parse_cell(row[key],expected)==expected, 'Task cell differs: '+event+'/'+task['task_id']+'/'+key)
            ids0=task['arc_ids']
            for arc_pos,arc_id in enumerate(ids0 or [None],1):
                arc=arc_table[apos];apos+=1
                check(arc['target_event']==event and int(arc['task_index'])==position and arc['task_id']==task['task_id'], 'Route row identity differs')
                check(arc['vehicle_id']==task['vehicle_id'] and arc['station_id']==task['station_id'], 'Route ownership differs')
                check(parse_cell(arc['route_is_empty'],not bool(ids0)) == (not bool(ids0)), 'Empty route marker wrong')
                check(int(arc['route_arc_count'])==len(ids0), 'Route length wrong')
                check((arc['arc_id']=='' and arc['arc_position_one_based']=='') if not ids0 else
                      (int(arc['arc_id'])==arc_id and int(arc['arc_position_one_based'])==arc_pos), 'Arc order/empty encoding changed')
                number(arc['route_distance_km'],task['distance_km'],'Route distance changed')
                number(arc['route_freeflow_h'],task['freeflow_h'],'Route free-flow changed')
        moments_path=source/f'{event}.moments.json';moment=read(moments_path)
        for rho in (.5,1.,1.5,2.):
            bound=moment['bounds_by_rho'][str(rho)]
            for i,feature in enumerate(FEATURES):
                row=moment_table[mpos];mpos+=1
                check(row['target_event']==event and float(row['rho'])==rho and row['feature']==feature and int(row['feature_index'])==i, 'Moment row identity changed')
                comparisons={'center':moment['center'][i],'original_base_halfwidth':moment['original_halfwidth'][i],
                    'repaired_base_halfwidth':moment['repaired_halfwidth'][i],'lower':bound['lower'][i],'upper':bound['upper'][i],
                    'common_normalized_delta':moment['delta'],'optimal_delta':moment['optimal_delta'],
                    'numeric_margin_delta':moment['numeric_margin_delta'],
                    'witness_maximum_violation':bound['feasibility']['primal_residuals']['maximum_violation']}
                for key,value in comparisons.items():number(row[key],value,'Moment CSV value changed: '+key)
                check(historical_path(row['source_moments_path']).resolve()==moments_path and row['source_moments_sha256']==sha(moments_path), 'Moment CSV source binding wrong')
        counters={'blocked':0,'movable':0,'road_cells':0,'success':0,'task_cells':0,'phiD':0.};qualified=set()
        for split,count in [('planning',160),('calibration',80)]:
            entry=entries[event,split]
            path=reference(entry['file'],source/f'{event}.{split}.npz')
            reference(entry['task_index'],task_index_path)
            ledger_path=reference(entry['scenario_ledger'],source/f'{event}.{split}.ledger.json')
            ledger=read(ledger_path)
            check(entry['development'] and entry['regime']=='R1' and entry['scenario_count']==count and entry['task_count']==len(tasks), 'Partition scope/count changed')
            check(entry['task_sha256']==bank['task_sha256'], 'Partition H hash differs')
            check(entry['array_schema']==schema(path), 'NPZ field/schema dropped or changed')
            with np.load(path,allow_pickle=False) as raw:
                check(raw['task_ids'].tolist()==ids, 'Partition task order differs')
                sids=raw['scenario_ids'].tolist()
                check(sids==entry['scenario_ids']==[r['scenario_id'] for r in ledger['scenarios']], 'Partition scenario order differs')
                check(len(sids)==count and len(set(sids))==count and not set(sids)&all_scenarios, 'Scenario missing or duplicated')
                all_scenarios.update(sids)
                check(np.array_equal(raw['times_h'],np.arange(-1.,24.,.25)), 'Partition time axis differs')
                blocked=raw['road_blocked'][:,:,:80];movable=raw['road_movable'][:,:,:80];success=raw['task_varsigma'];features=raw['features']
                check(blocked.shape==(count,100,80) and success.shape==(count,len(tasks)), 'Diagnostic source shape changed')
                counters['blocked']+=int(blocked.sum());counters['movable']+=int(movable.sum());counters['road_cells']+=blocked.size
                counters['success']+=int(success.sum());counters['task_cells']+=success.size;counters['phiD']+=float(features[:,3].sum())
                for pos in np.flatnonzero(np.any(success,axis=0)):qualified.add(tasks[int(pos)]['vehicle_id'])
        diag=diagnostics_table[len(output_records)]
        check(diag['target_event']==event,'Diagnostic target order changed')
        exact={'task_count':len(tasks),'moving_tasks':sum(t['distance_km']>0 for t in tasks),
            'zero_distance_tasks':sum(t['distance_km']==0 for t in tasks),'forecast_qualified_vehicles_with_H':len({t['vehicle_id'] for t in tasks}),
            'vehicles_total':len(bank['vehicle_summaries']),'vehicles_with_empty_H':sum(v['empty_H'] for v in bank['vehicle_summaries']),
            'realized_qualified_vehicles_at_least_once':len(qualified),'development_scenarios':240}
        for key,value in exact.items():number(diag[key],value,'Diagnostic count differs: '+key)
        fractions={'canonical80_blocked_fraction':counters['blocked']/counters['road_cells'],
            'canonical80_movable_fraction':counters['movable']/counters['road_cells'],
            'qualified_task_scenario_fraction':counters['success']/counters['task_cells'], 'mean_Phi_D':counters['phiD']/240}
        for key,value in fractions.items():number(diag[key],value,'Diagnostic formula differs: '+key,1e-13)
        check(diag['task_sha256']==bank['task_sha256'] and diag['assignment_state_sha256']==task_index['assignment_state_sha256'], 'Diagnostic provenance changed')
        output_records.append({'target_event':event,**exact,**fractions})
    check(tpos==len(task_table) and apos==len(arc_table) and mpos==len(moment_table),'Trailing/unconsumed export rows')
    check(len(all_scenarios)==1920,'Incomplete exported scenario identity universe')
    counts={'targets':8,'scenarios':1920,'tasks':tpos,'task_scenario_rows':tpos*240,'moment_bound_rows':mpos,
            'route_arc_rows_including_empty_markers':apos,'npz_partitions':16}
    check(generation['counts']==counts,'Generation audit counts disagree with raw row audit')
    result={'status':'PASS_INDEPENDENT_S01_PACK_AUDIT','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'pack':str(pack),'i08_acceptance':str(i08_path),'i08_acceptance_sha256':sha(i08_path),
        'export_manifest_sha256':sha(pack/'artifact_manifest.json'),'auditor_sha256':sha(Path(__file__).resolve()),
        'counts':counts,'target_diagnostics_recomputed':output_records,'verified_unique_files':len(hashes),
        'immutable_state_snapshot':str(snapshot),'formal_outcomes_generated':False,'random_samples_generated':0,
        'full_S01_signoff_scope':'This validates lossless derived delivery. Final scientific S01 acceptance remains root responsibility.',
        'comparison':'Task/moment CSV numeric round trips exact; diagnostics independently counted from raw arrays within1e-13; NPZ schema/path/hash and all H/scenario IDs exact'}
    result.update(source_provenance())
    with (audit/'s01_pack_audit.json').open('x') as target:
        json.dump(result,target,indent=2,allow_nan=False);target.write('\n')
    print(json.dumps({'status':result['status'],'counts':counts,'audit_output':str(audit)},allow_nan=False),flush=True)


if __name__=='__main__':main()
