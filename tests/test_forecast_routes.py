"""Independent exhaustive oracle for forecast node-simple non-FIFO routes."""
from __future__ import annotations

import math
import json
import unittest
from types import SimpleNamespace

import numpy as np

from evcs_experiments.forecast_routes import Arc, ForecastNetwork, replay_route, search_route


def network(arcs, travel=None, horizon=2.):
    slots = np.arange(0., horizon, .25)
    if travel is None:
        travel = np.tile([a.freeflow_h for a in arcs], (len(slots), 1))
    return ForecastNetwork(tuple(arcs), slots, np.asarray(travel, dtype=float), horizon_end_h=horizon)


def enumerate_reference(net, origin, destination, departure=0., initial_soc=.8,
                        capacity=10., per_km=.1, aux=1., qual=.3, deadline=1.9):
    """Enumerate every simple path and replay directly without production helpers."""
    arcs = net.arcs
    outgoing = {}
    for index, arc in enumerate(arcs):
        outgoing.setdefault(arc.source, []).append((index, arc))
    candidates = []

    def visit(node, visited, indices, arrival, distance, freeflow):
        if node == destination:
            energy = per_km * distance + aux * max(0., arrival - departure - freeflow)
            soc = max(0., initial_soc - energy / capacity)
            if soc >= qual and arrival <= deadline:
                candidates.append((arrival, distance, tuple(arcs[i].a for i in indices), energy, soc))
            return
        if arrival < net.slot_starts_h[0] or arrival >= net.horizon_end_h:
            return
        slot = int(np.searchsorted(net.slot_starts_h, arrival, side='right') - 1)
        for index, arc in outgoing.get(node, []):
            if arc.target in visited:
                continue
            travel = net.travel_hours[slot, index]
            if not math.isfinite(travel):
                continue
            visit(arc.target, visited | {arc.target}, indices + [index], arrival + travel,
                  distance + arc.distance_km, freeflow + arc.freeflow_h)

    visit(origin, {origin}, [], departure, 0., 0.)
    return min(candidates, key=lambda row: row[:3]) if candidates else None


class ForecastRouteTests(unittest.TestCase):
    def search(self, net, origin=0, destination=3, **kwargs):
        parameters = dict(departure_h=0., initial_soc=.8, capacity_kwh=10.,
                          energy_kwh_per_km=.1, auxiliary_kw=1., qualification_soc=.3,
                          latest_arrival_h=1.9)
        parameters.update(kwargs)
        return search_route(net, origin, destination, **parameters)

    def replay(self, net, ids, **kwargs):
        parameters = dict(departure_h=0., initial_soc=.8, capacity_kwh=10.,
                          energy_kwh_per_km=.1, auxiliary_kw=1.)
        parameters.update(kwargs)
        return replay_route(net, ids, **parameters)

    def test_exhaustive_small_graph_oracle_multiple_time_varying_cases(self):
        arcs = [Arc(i + 1, u, v, d, t) for i, (u, v, d, t) in enumerate([
            (0, 1, .1, .05), (0, 2, .2, .1), (1, 2, .15, .05),
            (2, 1, .2, .1), (1, 3, .3, .1), (2, 3, .1, .05),
            (1, 4, .1, .05), (4, 2, .1, .05), (4, 3, .1, .1)])]
        rng = np.random.default_rng(314159)
        for instance in range(30):
            travel = np.tile([a.freeflow_h for a in arcs], (8, 1)) + rng.integers(0, 5, (8, 9)) * .1
            travel[rng.random((8, 9)) < .12] = np.inf
            net = network(arcs, travel)
            result = self.search(net)
            expected = enumerate_reference(net, 0, 3)
            self.assertEqual(result.status, 'COMPLETE', instance)
            if expected is None:
                self.assertIsNone(result.route, instance)
            else:
                self.assertIsNotNone(result.route, instance)
                self.assertAlmostEqual(result.route.arrival_h, expected[0], delta=1e-12)
                self.assertAlmostEqual(result.route.distance_km, expected[1], delta=1e-12)
                self.assertEqual(result.route.arc_ids, expected[2])
                self.assertAlmostEqual(result.route.travel_energy_kwh, expected[3], delta=1e-12)
                self.assertAlmostEqual(result.route.arrival_soc, expected[4], delta=1e-12)

    def test_later_prefix_can_win_nonfifo_and_must_not_be_dominated(self):
        arcs = [Arc(1, 0, 1, .1, .1), Arc(2, 0, 2, .1, .15),
                Arc(3, 2, 1, .1, .15), Arc(4, 1, 3, .1, .1)]
        travel = np.tile([a.freeflow_h for a in arcs], (8, 1))
        travel[0, 3] = 1.
        net = network(arcs, travel)
        result = self.search(net)
        self.assertEqual(result.status, 'COMPLETE')
        self.assertEqual(result.route.arc_ids, (2, 3, 4))
        self.assertAlmostEqual(result.route.arrival_h, .4, delta=1e-14)
        early = self.replay(net, (1, 4))
        self.assertAlmostEqual(early.arrival_h, 1.1)

    def test_zero_route_and_low_soc_have_exact_complete_answers(self):
        net = network([Arc(1, 0, 3, 1., .1)])
        result = self.search(net, origin=0, destination=0)
        self.assertEqual(result.status, 'COMPLETE')
        self.assertEqual(result.route.arc_ids, ())
        self.assertEqual(result.route.node_ids, (0,))
        self.assertEqual(result.route.arrival_h, 0.)
        self.assertEqual(result.route.travel_energy_kwh, 0.)
        poor = self.search(net, origin=0, destination=0, initial_soc=.2)
        self.assertEqual(poor.status, 'COMPLETE')
        self.assertIsNone(poor.route)

    def test_closed_entry_cannot_wait_for_later_opening(self):
        arcs = [Arc(1, 0, 3, 1., .1)]
        travel = np.full((8, 1), .1)
        travel[0, 0] = np.inf
        net = network(arcs, travel)
        self.assertIsNone(self.search(net).route)
        self.assertIsNone(self.replay(net, (1,)))
        later = self.search(net, departure_h=.25)
        self.assertAlmostEqual(later.route.arrival_h, .35)

    def test_entry_boundary_uses_next_slot_and_traversal_is_frozen(self):
        arcs = [Arc(1, 0, 1, .1, .25), Arc(2, 1, 3, .1, .1)]
        travel = np.tile([.25, .1], (8, 1))
        travel[0, 1] = np.inf
        net = network(arcs, travel)
        self.assertEqual(net.slot_index(0.), 0)
        self.assertEqual(net.slot_index(.25), 1)
        self.assertIsNone(net.slot_index(2.))
        result = self.replay(net, (1, 2))
        self.assertAlmostEqual(result.arrival_h, .35)
        self.assertEqual(result.entry_times_h, (0., .25))
        one = network([Arc(1, 0, 3, .1, .1)], [[.6], [np.inf], [np.inf], [.1], [.1], [.1], [.1], [.1]])
        self.assertAlmostEqual(self.replay(one, (1,)).arrival_h, .6)

    def test_travel_energy_counts_only_auxiliary_delay_above_freeflow(self):
        net = network([Arc(1, 0, 3, .3, .1)], np.full((8, 1), .6))
        route = self.replay(net, (1,), initial_soc=.7, energy_kwh_per_km=.4, auxiliary_kw=2.)
        self.assertAlmostEqual(route.travel_energy_kwh, .3 * .4 + 2 * (.6 - .1))
        self.assertAlmostEqual(route.arrival_soc, .7 - 1.12 / 10)

    def test_fast_energy_infeasible_route_does_not_hide_slower_qualified_route(self):
        net = network([Arc(1, 0, 3, 2., .1), Arc(2, 0, 1, .1, .2), Arc(3, 1, 3, .1, .2)])
        result = self.search(net, initial_soc=.5, qualification_soc=.4, energy_kwh_per_km=1.)
        self.assertEqual(result.status, 'COMPLETE')
        self.assertEqual(result.route.arc_ids, (2, 3))
        self.assertAlmostEqual(result.route.arrival_soc, .48)
        self.assertIsNone(self.search(net, initial_soc=.5, qualification_soc=.4,
                                     energy_kwh_per_km=1., latest_arrival_h=.3).route)
        self.assertEqual(self.search(net, initial_soc=.5, qualification_soc=.4,
                                    energy_kwh_per_km=1., latest_arrival_h=.4).route.arc_ids, (2, 3))

    def test_deterministic_distance_then_arc_id_ties(self):
        arcs = [Arc(4, 0, 1, .2, .1), Arc(5, 1, 3, .2, .1),
                Arc(2, 0, 2, .1, .1), Arc(3, 2, 3, .1, .1)]
        self.assertEqual(self.search(network(arcs)).route.arc_ids, (2, 3))
        equal = [Arc(a.a, a.source, a.target, .1, a.freeflow_h) for a in arcs]
        self.assertEqual(self.search(network(equal)).route.arc_ids, (2, 3))
        self.assertEqual(self.search(network(list(reversed(equal)))).route.arc_ids, (2, 3))

    def test_simple_domain_no_cycle_as_waiting_and_invalid_route_rejected(self):
        net = network([Arc(1, 0, 1, .1, .1), Arc(2, 1, 0, .1, .1), Arc(3, 1, 3, .1, .1)])
        with self.assertRaises(ValueError):
            self.replay(net, (1, 2, 1, 3))
        with self.assertRaises(ValueError):
            self.replay(net, (3, 1))
        self.assertEqual(self.search(net).route.node_ids, (0, 1, 3))

    def test_search_cap_cannot_certify_complete_infeasibility(self):
        net = network([Arc(1, 0, 1, .1, .1), Arc(2, 1, 3, .1, .1)])
        result = self.search(net, max_expansions=0)
        self.assertEqual(result.status, 'SEARCH_INCOMPLETE')
        self.assertIsNone(result.route)
        self.assertTrue(result.certificate['termination_reason'])

    def test_qualification_and_two_hour_block_service_boundary(self):
        from evcs_experiments.forecast_tasks import qualification_soc, service_periods
        self.assertAlmostEqual(qualification_soc(.3, .2, 15., 1., .95, 80.), .3973684210526316)
        self.assertAlmostEqual(qualification_soc(.3, .2, 15., 2., .95, 80.), .5947368421052632)
        self.assertEqual(qualification_soc(.5, .2, 15., 1., .95, 80.), .5)
        occupancy, reserve, block, compatible = service_periods(24, 1.)
        self.assertTrue(compatible)
        self.assertEqual(block, (24,))
        self.assertTrue(reserve[-1])
        self.assertEqual(len(occupancy), 24)
        _, _, block, compatible = service_periods(23, 2.)
        self.assertEqual(block, (23, 24))
        self.assertFalse(compatible)
        self.assertTrue(service_periods(22, 2.)[3])

    def test_task_builder_station_ties_serialization_and_incomplete_gate(self):
        from evcs_experiments.forecast_tasks import ForecastTaskInputs, build_candidate_tasks
        net = network([Arc(i, 0, i, .1, .1) for i in range(1, 5)])
        hardware = {'SOC_dep_under_e': .2, 'P_blk_under_e_kW': 15.,
                    'eta_dis_e': .95, 'E_cap_e_kWh': 80.,
                    'E_km_e_kWh_per_km': .3, 'P_aux_e_kW': 1.}
        case = SimpleNamespace(vehicle_parameters={'EV': hardware},
                               station_parameters={f'CS{i}': {'road_node_c': i} for i in range(1, 5)})
        state = SimpleNamespace(vehicle_id='EV', road_node=0, soc=.8)
        target = SimpleNamespace(vehicle_states=(state,),
                                 station_ids=('CS4', 'CS2', 'CS3', 'CS1'), lead_h=0.)
        inputs = ForecastTaskInputs(case, target, net, np.zeros((8, 4)), net.travel_hours, {}, 'toy-source')
        result = build_candidate_tasks(inputs)
        self.assertTrue(result.certified)
        self.assertEqual(result.status, 'COMPLETE')
        self.assertEqual(result.vehicle_summaries[0]['retained_station_ids'], ['CS1', 'CS2', 'CS3'])
        self.assertEqual(len(result.tasks), 3 * 23)
        self.assertEqual({t.start_period for t in result.tasks}, set(range(2, 25)))
        self.assertEqual(len(json.loads(json.dumps(result.as_dict()))['queries']), 4)
        incomplete = build_candidate_tasks(inputs, max_expansions=0)
        self.assertFalse(incomplete.certified)
        self.assertEqual(incomplete.status, 'SEARCH_INCOMPLETE')
        self.assertTrue(incomplete.as_dict()['tasks_provisional'])


if __name__ == '__main__':
    unittest.main()
