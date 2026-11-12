#!/usr/bin/env python3
"""Independent scalar arc replay and vector task-availability reconstruction.

Reads accepted I04/I06 evidence. Does not import the I07 implementation,
forecast route replay, its cache or its task-deliverability functions.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import sys
import numpy as np
from evcs_experiments.config import get_paths
from evcs_experiments.evidence import (frozen_runner_root, frozen_source_file, historical_path, legacy_transfer_manifest, LEGACY_CASE_NAME, source_provenance, verify_scientific_kernel_identity)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_route(task, state, hardware, arc_lookup, times, travel, departure):
    node, arrival, distance, freeflow = state.road_node, departure, 0., 0.
    visited = {node}
    for aid in task['arc_ids']:
        i, arc = arc_lookup[aid]
        assert int(arc['m_road']) == node and int(arc['n_road']) not in visited
        node = int(arc['n_road']); visited.add(node)
        if arrival < times[0] or arrival >= 24.:
            return (math.inf, False, 0., 0., 0., 'ENTRY_OUTSIDE_HORIZON', aid)
        slot = int(np.searchsorted(times, arrival, side='right')-1)
        minutes = float(travel[slot,i])
        if not math.isfinite(minutes):
            return (math.inf, False, 0., 0., 0., 'IMPASSABLE_ARC', aid)
        arrival += minutes/60.
        distance += float(arc['D_a_km'])
        freeflow += float(arc['T0_a_min'])/60.
    route_time = arrival-departure
    energy = float(hardware['E_km_e_kWh_per_km'])*distance + float(hardware['P_aux_e_kW'])*max(route_time-freeflow,0.)
    soc = max(state.soc-energy/float(hardware['E_cap_e_kWh']),0.)
    return (arrival, True, route_time, energy, soc, 'COMPLETE', -1)


def main():
    p=argparse.ArgumentParser()
    for name in ['case-root','road-evidence','route-evidence','output']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();case=a.case_root.resolve();out=a.output.resolve()

    from evcs_experiments.case_inputs import load_case_inputs
    from evcs_experiments.road_connectors import derive_road_connectors
    source=load_case_inputs(case)
    with (case/'04_transportation/A__directed_road_arcs.csv').open(newline='',encoding='utf-8-sig') as f:
        arcs=sorted(list(csv.DictReader(f))+list(derive_road_connectors(case).arc_rows),key=lambda r:int(r['a']))
    lookup={int(r['a']):(i,r) for i,r in enumerate(arcs)}
    road=np.load(a.road_evidence/'development_roads.npz',allow_pickle=False)
    road_times=road['baseline_replay_times_h'];road_travel=road['travel_time_min']
    road_ledger=json.loads((a.road_evidence/'development_road_ledger.json').read_text())
    regimes=list(map(str,road['regimes']))
    assert road_travel.shape==(128,3,100,88) and np.array_equal(road_times,np.arange(-4,96)/4)
    assert list(road['arc_ids'])==list(range(1,89))
    source_keys={(r['target_event'],r['base_weather_event'],r['variant']):i for i,r in enumerate(road_ledger)}
    assert len(source_keys)==128
    stats=[]
    for event in sorted(source.event_ids):
        target=source.target(event,lead_h=1.)
        states={s.vehicle_id:s for s in target.vehicle_states}
        bank=json.loads((a.route_evidence/f'{event}.tasks.json').read_text())
        tasks=bank['tasks'];n=len(tasks);tau=bank['config']['tau_blk_h']
        metadata=json.loads((out/f'{event}.replay.json').read_text())
        with np.load(out/f'{event}.replay.npz',allow_pickle=False) as saved:
            generated={key:saved[key] for key in saved.files}
        # These names are fixed by the I07 validator's explicit export schema.
        assert metadata['task_ids']==[t['task_id'] for t in tasks]
        assert metadata['task_sha256']==bank['task_sha256']
        assert metadata['methods']==['EA','DSP']
        assert metadata['method_departure_h']=={'EA':-1.,'DSP':0.}
        scenarios=metadata['scenario_ledger']
        assert len(scenarios)==48
        np.testing.assert_array_equal(generated['task_ids'],metadata['task_ids'])
        np.testing.assert_array_equal(generated['methods'],metadata['methods'])
        np.testing.assert_array_equal(generated['slot_starts_h'],road_times)
        np.testing.assert_array_equal(generated['departure_hours'],[-1.,0.])
        np.testing.assert_array_equal(generated['scenario_ids'],[s['scenario_id'] for s in scenarios])
        actual_set={(row['base_weather_event'],row['variant'],row['regime']) for row in scenarios}
        assert actual_set=={(b,v,r) for b in source.event_ids for v in (1,2) for r in regimes}
        summary={'event':event,'methods':{},'source_task_sha256':bank['task_sha256'],'tasks':n,'scenarios_per_method':48,
                 'independent_physical_route_replays':0,'max_numeric_error':0.}
        starts=np.array([t['start_period']-1 for t in tasks],dtype=float)
        capacities=np.array([float(source.vehicle_parameters[t['vehicle_id']]['E_cap_e_kWh']) for t in tasks])
        dep_soc=np.array([float(source.vehicle_parameters[t['vehicle_id']]['SOC_dep_under_e']) for t in tasks])
        quals=np.array([max(bank['config']['basic_soc'], float(source.vehicle_parameters[t['vehicle_id']]['SOC_dep_under_e'])+
                            float(source.vehicle_parameters[t['vehicle_id']]['P_blk_under_e_kW'])*tau/
                            (float(source.vehicle_parameters[t['vehicle_id']]['eta_dis_e'])*float(source.vehicle_parameters[t['vehicle_id']]['E_cap_e_kWh']))) for t in tasks])
        for mi,(method,departure) in enumerate([('EA',-1.),('DSP',0.)]):
            complete_total=success_total=late_qualified=full_total=0
            for si,row in enumerate(scenarios):
                key=(event,row['base_weather_event'],row['variant'])
                ri=source_keys[key];regime_index=regimes.index(row['regime'])
                original=road_ledger[ri]
                assert row['target_event']==event and original['residual_donor_event']!=event
                assert row['road_ledger_index']==ri
                # Cache only identical vehicle+route within this fixed source
                # scenario and departure. It never shares task eligibility.
                cache={};physical=[]
                for task in tasks:
                    route_key=(task['vehicle_id'],tuple(task['arc_ids']))
                    if route_key not in cache:
                        cache[route_key]=raw_route(task,states[task['vehicle_id']],source.vehicle_parameters[task['vehicle_id']],
                                                   lookup,road_times,road_travel[ri,regime_index],departure)
                    physical.append(cache[route_key])
                summary['independent_physical_route_replays']+=len(cache)
                arrival=np.array([r[0] for r in physical]);gamma=np.array([r[1] for r in physical],dtype=bool)
                route_time=np.array([r[2] for r in physical]);energy=np.array([r[3] for r in physical]);soc=np.array([r[4] for r in physical])
                effective=np.maximum(arrival,starts)
                success=gamma & (soc>=quals) & (24-effective>=tau)
                mu=success[:,None] & (effective[:,None]<=np.arange(24)[None,:]) & (np.arange(24)[None,:]+tau<=24)
                usable=np.where(success,np.maximum(soc-dep_soc,0)*capacities,0.)
                failed=np.array([r[6] for r in physical],dtype=np.int64)
                status=np.array([('ENTRY_IMPASSABLE' if r[5]=='IMPASSABLE_ARC' else r[5]) if not gamma[i]
                                 else 'TASK_QUALIFIED' if success[i] else 'SOC_UNQUALIFIED' if soc[i]<quals[i]
                                 else 'INSUFFICIENT_SERVICE_TIME' for i,r in enumerate(physical)])
                np.testing.assert_array_equal(generated['failed_arc_ids'][mi,si],failed)
                np.testing.assert_array_equal(generated['status'][mi,si],status)
                expected={'arrival_h':arrival,'gamma_route':gamma,'route_time_h':route_time,
                          'travel_energy_kwh':energy,'arrival_soc':soc,'varsigma':success,
                          'mu_full':mu,'usable_energy_dc_kwh':usable}
                for field,value in expected.items():
                    actual=generated[field][mi,si]
                    assert actual.shape==value.shape,(event,field,actual.shape,value.shape)
                    if value.dtype==bool:
                        np.testing.assert_array_equal(actual,value,err_msg=f'{event} {method} {si} {field}')
                    else:
                        np.testing.assert_array_equal(np.isinf(actual),np.isinf(value))
                        finite=np.isfinite(value)
                        error=float(np.max(np.abs(actual[finite]-value[finite]),initial=0.))
                        summary['max_numeric_error']=max(summary['max_numeric_error'],error)
                        np.testing.assert_allclose(actual,value,rtol=1e-13,atol=1e-12,err_msg=f'{event} {method} {si} {field}')
                complete_total+=int(gamma.sum());success_total+=int(success.sum())
                late_qualified+=int((success & (arrival>starts)).sum());full_total+=int(mu.sum())
            summary['methods'][method]={'task_scenario_rows':48*n,'completed_routes':complete_total,
                                        'qualified_tasks':success_total,'late_but_qualified_tasks':late_qualified,'full_task_hours':full_total}
        stats.append(summary);print(json.dumps(summary),flush=True)
    result={'status':'PASS_INDEPENDENT_AUDIT','targets':stats,'source_sha256':sha(Path(__file__)),
            'task_scenario_rows':sum(2*48*t['tasks'] for t in stats),
            'independent_physical_route_replays':sum(t['independent_physical_route_replays'] for t in stats),
            'max_numeric_error':max(t['max_numeric_error'] for t in stats),'formal_OOS_outcomes_generated':False,
            'method':'Scalar original-minute arc-entry replay, source assignment/hardware; independently vectorized task eligibility'}
    result.update(source_provenance())
    with (out/'independent_replay_audit.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
