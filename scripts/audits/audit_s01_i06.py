#!/usr/bin/env python3
"""Independent path-label search and H reconstruction of saved I06 evidence.

Uses a heap of complete simple-path prefixes without node-label dominance or
the implementation's reverse-Dijkstra lower bounds. All saved pair queries
are checked, with memoization by full explicit argument tuple.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import heapq
import json
import math
import sys
import time
import numpy as np
from evcs_experiments.config import get_paths
from evcs_experiments.evidence import (frozen_runner_root, frozen_source_file, historical_path, legacy_transfer_manifest, LEGACY_CASE_NAME, source_provenance, verify_scientific_kernel_identity)


def oracle(arcs, times, travel, q):
    outgoing = {}
    for i, a in enumerate(arcs):
        outgoing.setdefault(int(a['m_road']), []).append((i, a))
    # No lower-bound heuristic and no dominance across distinct prefixes.
    origin, destination = q['origin'], q['destination']
    heap = [(q['departure_h'], 0., (), (origin,), 0., ())]
    best, expanded = None, 0
    started = time.monotonic()
    while heap:
        arrival, distance, ids, nodes, ff, entries = heapq.heappop(heap)
        if best is not None and arrival > best[0]:
            break
        if arrival > q['latest_arrival_h']:
            continue
        energy = q['energy_kwh_per_km']*distance + q['auxiliary_kw']*max(0., arrival-q['departure_h']-ff)
        soc = max(0., q['initial_soc']-energy/q['capacity_kwh'])
        # Prefix energy is monotone. Slack here only avoids roundoff pruning.
        if soc < q['qualification_soc']-1e-12:
            continue
        expanded += 1
        if expanded > 2000000 or time.monotonic()-started > 15.:
            raise RuntimeError('Independent oracle cap; cannot certify this query')
        if nodes[-1] == destination:
            if soc >= q['qualification_soc']:
                candidate = (arrival, distance, ids, ff, energy, soc, entries)
                if best is None or candidate[:3] < best[:3]:
                    best = candidate
            continue
        if arrival < times[0] or arrival >= 24.:
            continue
        slot = int(np.searchsorted(times, arrival, side='right')-1)
        for i, a in outgoing.get(nodes[-1], ()):
            nxt = int(a['n_road'])
            if nxt in nodes or not math.isfinite(travel[slot, i]):
                continue
            heapq.heappush(heap, (arrival+float(travel[slot, i]), distance+float(a['D_a_km']),
                                 ids+(int(a['a']),), nodes+(nxt,), ff+float(a['T0_a_min'])/60., entries+(arrival,)))
    return best, expanded


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case-root', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    case, out = args.case_root.resolve(), args.output.resolve()

    # The already accepted connector adapter supplies source-owned connector
    # attributes. No I06 search, task builder or replay function is imported.
    from evcs_experiments.road_connectors import derive_road_connectors
    from evcs_experiments.case_inputs import load_case_inputs
    from evcs_experiments.road_disruptions import load_road_model
    from evcs_experiments.wind_field import forecast_vector_field
    source_case = load_case_inputs(case)
    source_road = load_road_model(case, include_connectors=True)
    def rows(path):
        with path.open(newline='', encoding='utf-8-sig') as f:
            return list(csv.DictReader(f))
    arcs = sorted(rows(case/'04_transportation/A__directed_road_arcs.csv')+
                  list(derive_road_connectors(case).arc_rows), key=lambda a:int(a['a']))
    hardware = {r['e']:r for r in rows(case/'06_fleet/E__vehicle_parameters.csv')}
    t0 = np.array([float(a['T0_a_min'])/60 for a in arcs])
    ktt = np.array([float(a['kappa_tt_a']) for a in arcs])
    wmov = np.array([float(a['W_mov_type_a_m_s']) for a in arcs])
    all_stats = []
    for file in sorted(out.glob('*.tasks.json')):
        bank = json.loads(file.read_text())
        assert bank['certified'] and len(bank['queries']) == 640 and len(bank['vehicle_summaries']) == 80
        event = file.name[:-len('.tasks.json')]
        target = source_case.target(event, lead_h=1.)
        states = {s.vehicle_id:s for s in target.vehicle_states}
        pair_set = {(q['vehicle_id'],q['station_id']) for q in bank['queries']}
        assert pair_set == {(v,c) for v in states for c in target.station_ids}
        z = np.load(out/f'{event}.forecast.npz', allow_pickle=False)
        times, speed, travel = z['slot_starts_h'], z['forecast_speed_m_s'], z['forecast_travel_hours']
        np.testing.assert_array_equal(times, target.replay_times_h)
        source_speed = np.linalg.norm(forecast_vector_field(target.forecast).sample(times, source_road.points_xy_km),axis=-1)
        np.testing.assert_array_equal(speed, source_speed)
        expected = t0[None,:]*(1+ktt[None,:]*np.clip((speed-10)/15,0,1))
        expected[speed>wmov[None,:]] = np.inf
        assert np.array_equal(np.isinf(travel), np.isinf(expected))
        np.testing.assert_allclose(travel, expected, atol=2e-15, rtol=1e-14)
        cache, station_routes, expanded = {}, {}, 0
        for query in bank['queries']:
            assert query['status']=='COMPLETE' and query['certificate']['complete']
            q = query['arguments']
            vehicle = query['vehicle_id']; hp = hardware[vehicle]; state = states[vehicle]
            expected_q = {'origin':state.road_node,
                          'destination':int(source_case.station_parameters[query['station_id']]['road_node_c']),
                          'departure_h':-1.,'initial_soc':state.soc,
                          'capacity_kwh':float(hp['E_cap_e_kWh']),
                          'energy_kwh_per_km':float(hp['E_km_e_kWh_per_km']),
                          'auxiliary_kw':float(hp['P_aux_e_kW']),
                          'qualification_soc':max(.3,float(hp['SOC_dep_under_e'])+float(hp['P_blk_under_e_kW'])/(float(hp['eta_dis_e'])*float(hp['E_cap_e_kWh']))),
                          'latest_arrival_h':23.}
            assert q == expected_q
            key = json.dumps(q, sort_keys=True)
            if key not in cache:
                cache[key], n = oracle(arcs, times, travel, q)
                expanded += n
            best = cache[key]
            route = query['route']
            if best is None:
                assert route is None
                continue
            assert route is not None and tuple(route['arc_ids'])==best[2]
            np.testing.assert_allclose([route['arrival_h'],route['distance_km'],route['freeflow_h'],
                                      route['travel_energy_kwh'],route['arrival_soc']],
                                     [best[0],best[1],best[3],best[4],best[5]], atol=1e-12,rtol=1e-13)
            np.testing.assert_allclose(route['entry_times_h'], best[6], atol=1e-12,rtol=1e-13)
            station_routes.setdefault(query['vehicle_id'],[]).append((best[0],best[1],query['station_id'],route,q))
        expected_tasks = {}
        tau = int(bank['config']['tau_blk_h'])
        for vehicle, candidates in station_routes.items():
            for _, _, station, route, q in sorted(candidates,key=lambda r:r[:3])[:3]:
                vp=hardware[vehicle]
                qual=max(bank['config']['basic_soc'],float(vp['SOC_dep_under_e'])+float(vp['P_blk_under_e_kW'])*tau/(float(vp['eta_dis_e'])*float(vp['E_cap_e_kWh'])))
                assert qual == q['qualification_soc']
                for k in range(1,25):
                    block=list(range(k,k+tau))
                    reserve=[t>=k and t-1+tau<=24 for t in range(1,25)]
                    if (route['arrival_h']<=k-1 and route['arrival_soc']>=qual and 24-(k-1)>=tau
                        and all(t<=24 and reserve[t-1] for t in block)):
                        expected_tasks[(vehicle,station,k)]=(route,qual,reserve,block)
        assert len(expected_tasks)==len(bank['tasks'])
        actualkeys=set()
        for task in bank['tasks']:
            key=(task['vehicle_id'],task['station_id'],task['start_period'])
            assert key not in actualkeys
            actualkeys.add(key)
            route,qual,reserve,block=expected_tasks[key]
            assert task['arc_ids']==route['arc_ids'] and task['reserve']==reserve and task['block_periods']==block
            assert task['occupancy']==[t>=key[2] for t in range(1,25)]
            assert task['qualification_soc']==qual
            for name in ('arrival_h','distance_km','freeflow_h','travel_energy_kwh','arrival_soc'):
                assert task[name]==route[name]
            usable=max(0,route['arrival_soc']-float(hardware[key[0]]['SOC_dep_under_e']))*float(hardware[key[0]]['E_cap_e_kWh'])
            assert task['usable_energy_kwh']==usable and task['travel_time_h']==route['arrival_h']-route['departure_h']
        assert actualkeys==set(expected_tasks)
        all_stats.append({'event':event,'queries':len(bank['queries']),'unique_independent_searches':len(cache),
                          'oracle_expansions':expanded,'tasks':len(bank['tasks']),
                          'zero_distance_tasks':sum(t['distance_km']==0 for t in bank['tasks']),
                          'moving_tasks':sum(t['distance_km']>0 for t in bank['tasks']),
                          'empty_H_vehicles':sum(v['empty_H'] for v in bank['vehicle_summaries'])})
        print(json.dumps(all_stats[-1]),flush=True)
    assert len(all_stats)==8
    result={'status':'PASS_INDEPENDENT_AUDIT','targets':all_stats,
            'oracle':'all simple-path labels in arrival-priority queue; no static bounds or node dominance',
            'formal_OOS_outcomes_generated':False,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    result.update(source_provenance())
    target=out/'independent_route_audit.json'
    with target.open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
