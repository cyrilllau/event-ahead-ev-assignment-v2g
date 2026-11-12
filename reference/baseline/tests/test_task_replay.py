"""Hand-computed fixed-route and downstream delivery checks, independent of kernel helpers."""
from __future__ import annotations

import copy
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from evcs_experiments.forecast_routes import Arc, ForecastNetwork
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.task_replay import replay_task_bank, task_deliverability


PARAMETERS = {'EV': {'SOC_dep_under_e': .2, 'P_blk_under_e_kW': 15.,
                     'eta_dis_e': .95, 'E_cap_e_kWh': 80.,
                     'E_km_e_kWh_per_km': .3, 'P_aux_e_kW': 1.}}
STATES = {'EV': SimpleNamespace(road_node=0, soc=.8)}


def make_network(arcs=None, travel=None):
    if arcs is None:
        arcs = [Arc(1, 0, 1, 1., .1), Arc(2, 1, 2, 2., .2)]
    times = np.arange(-1., 24., .25)
    if travel is None:
        travel = np.tile([a.freeflow_h for a in arcs], (len(times), 1))
    return ForecastNetwork(tuple(arcs), times, travel)


def fixture_bank(arcs=None, periods=(1, 24), tau=1., departure=-1.):
    """Manually author a frozen forecast bank; do not invoke task construction."""
    if arcs is None:
        arcs = [Arc(1, 0, 1, 1., .1), Arc(2, 1, 2, 2., .2)]
    ids = [a.a for a in arcs]
    destination = arcs[-1].target if arcs else 0
    distance = sum(a.distance_km for a in arcs)
    ff = sum(a.freeflow_h for a in arcs)
    energy = .3 * distance
    soc = .8 - energy / 80.
    qualification = max(.3, .2 + 15 * tau / (.95 * 80))
    tasks = []
    for k in periods:
        tasks.append(dict(task_id=f'EV__CS__k{k:02d}', vehicle_id='EV', station_id='CS',
                          start_period=k, arc_ids=ids, arrival_h=departure + ff,
                          distance_km=distance, freeflow_h=ff, travel_time_h=ff,
                          travel_energy_kwh=energy, arrival_soc=soc,
                          usable_energy_kwh=(soc - .2) * 80, qualification_soc=qualification,
                          occupancy=[t >= k for t in range(1, 25)],
                          reserve=[t >= k and t - 1 + tau <= 24 for t in range(1, 25)],
                          block_periods=list(range(k, k + int(tau)))))
    config = {'basic_soc': .3, 'tau_blk_h': tau, 'nearest_stations': 3}
    dependency = 'a' * 64
    digest = canonical_hash(dict(dependency_sha256=dependency, config=config, certified=True, tasks=tasks))
    arguments = dict(origin=0, destination=destination, departure_h=departure, initial_soc=.8,
                     capacity_kwh=80., energy_kwh_per_km=.3, auxiliary_kw=1.,
                     qualification_soc=qualification, latest_arrival_h=23. if tau == 1 else 21.)
    entries = []
    entry = departure
    for arc in arcs:
        entries.append(entry)
        entry += arc.freeflow_h
    route = dict(arc_ids=ids, node_ids=[0] + [a.target for a in arcs], departure_h=departure,
                 arrival_h=departure + ff, distance_km=distance, freeflow_h=ff,
                 travel_energy_kwh=energy, arrival_soc=soc, entry_times_h=entries)
    return dict(status='COMPLETE', certified=True, tasks=tasks, task_sha256=digest,
                dependency_sha256=dependency, config=config,
                queries=[dict(vehicle_id='EV', station_id='CS', status='COMPLETE',
                              arguments=arguments, route=route)], vehicle_summaries=[])


class TaskReplayTests(unittest.TestCase):
    def run_bank(self, net=None, bank=None, **kwargs):
        return replay_task_bank(net or make_network(), bank or fixture_bank(), PARAMETERS, STATES,
                                development=True, **kwargs)

    def deliver(self, **kwargs):
        values = dict(arrival_h=.5, gamma_route=True, arrival_soc=.6, start_period=1,
                      capacity_kwh=80., departure_soc=.2, qualification_soc=.3973684210526316,
                      tau_blk_h=1.)
        values.update(kwargs)
        return task_deliverability(**values)

    def test_late_arrival_still_succeeds_and_energy_remains_battery_side(self):
        success, available, energy = self.deliver()
        self.assertTrue(success)
        np.testing.assert_array_equal(available, [False] + [True] * 23)
        self.assertAlmostEqual(energy, 32.)
        self.assertNotAlmostEqual(energy, 32 * .95)
        self.assertFalse(available.flags.writeable)

    def test_exact_soc_and_terminal_time_boundaries(self):
        qual = .2 + 15 / (.95 * 80)
        success, available, energy = self.deliver(arrival_soc=qual, qualification_soc=qual, arrival_h=23.)
        self.assertTrue(success)
        np.testing.assert_array_equal(available, [False] * 23 + [True])
        self.assertAlmostEqual(energy, 15 / .95)
        self.assertFalse(self.deliver(arrival_soc=np.nextafter(qual, -np.inf), qualification_soc=qual)[0])
        self.assertFalse(self.deliver(arrival_h=np.nextafter(23., np.inf))[0])
        self.assertFalse(self.deliver(arrival_h=24.)[0])

    def test_two_hour_availability_and_per_start_qualification(self):
        success, available, energy = self.deliver(arrival_h=22., tau_blk_h=2., qualification_soc=.59)
        self.assertTrue(success)
        np.testing.assert_array_equal(available, [False] * 22 + [True, False])
        self.assertAlmostEqual(energy, 32.)
        # k=24 is deliberately outside valid tau=2 H. The scalar rule must
        # still enforce per-start remaining time; valid H can differ in mu.
        self.assertFalse(self.deliver(arrival_h=22., start_period=24, tau_blk_h=2., qualification_soc=.59)[0])
        self.assertFalse(self.deliver(arrival_h=22.00001, tau_blk_h=2., qualification_soc=.59)[0])

    def test_zero_route_preserves_state_and_distinct_start_availability(self):
        result = self.run_bank(bank=fixture_bank(arcs=[]))
        np.testing.assert_array_equal(result.gamma_route, [True, True])
        np.testing.assert_array_equal(result.arrival_h, [-1., -1.])
        np.testing.assert_array_equal(result.travel_energy_kwh, [0., 0.])
        np.testing.assert_array_equal(result.route_time_h, [0., 0.])
        np.testing.assert_array_equal(result.arrival_soc, [.8, .8])
        np.testing.assert_array_equal(result.varsigma, [True, True])
        np.testing.assert_array_equal(result.mu_full[0], np.ones(24, dtype=bool))
        np.testing.assert_array_equal(result.mu_full[1], [False] * 23 + [True])
        np.testing.assert_allclose(result.usable_energy_dc_kwh, [48., 48.], atol=1e-14)
        self.assertEqual(result.unique_routes_replayed, 1)
        empty = fixture_bank(periods=())
        empty_result = self.run_bank(bank=empty)
        self.assertEqual(empty_result.task_ids, ())
        self.assertEqual(empty_result.arrival_h.shape, (0,))
        self.assertEqual(empty_result.mu_full.shape, (0, 24))
        self.assertEqual(empty_result.unique_routes_replayed, 0)

    def test_first_and_later_blockage_use_paper_failure_bookkeeping(self):
        for failed_id in (1, 2):
            net = make_network()
            travel = net.travel_hours.copy()
            travel[:, failed_id - 1] = np.inf
            blocked = make_network(travel=travel)
            result = self.run_bank(blocked)
            self.assertTrue(np.isinf(result.arrival_h).all())
            self.assertFalse(np.any(result.gamma_route))
            np.testing.assert_array_equal(result.failed_arc_ids, [failed_id, failed_id])
            for array in (result.route_time_h, result.travel_energy_kwh, result.arrival_soc,
                          result.usable_energy_dc_kwh):
                np.testing.assert_array_equal(array, [0., 0.])
            self.assertFalse(np.any(result.varsigma))
            self.assertFalse(np.any(result.mu_full))

    def test_entry_boundary_and_closure_after_entry_do_not_interrupt(self):
        arcs = [Arc(1, 0, 1, 1., .25), Arc(2, 1, 2, 2., .1)]
        travel = make_network(arcs).travel_hours.copy()
        travel[4, 1] = np.inf  # At zero second arc closed, at .25 open.
        travel[5:, 0] = np.inf  # First arc closes after vehicle entered.
        result = self.run_bank(make_network(arcs, travel), fixture_bank(arcs), departure_h=0.)
        np.testing.assert_allclose(result.arrival_h, [.35, .35], atol=1e-15)
        self.assertTrue(result.gamma_route.all())
        np.testing.assert_array_equal(result.mu_full[0], [False] + [True] * 23)

    def test_cross_slot_first_arc_finishes_but_later_closed_arc_fails(self):
        net = make_network()
        travel = net.travel_hours.copy()
        travel[:, 0] = .6
        travel[6, 1] = np.inf  # .6 is in [.5,.75).
        result = self.run_bank(make_network(travel=travel), departure_h=0.)
        self.assertFalse(result.gamma_route.any())
        np.testing.assert_array_equal(result.failed_arc_ids, [2, 2])

    def test_last_arc_can_arrive_after_horizon_but_next_arc_cannot_enter(self):
        one = [Arc(1, 0, 2, 1., .1)]
        travel = make_network(one).travel_hours.copy()
        travel[:, 0] = 25.
        result = self.run_bank(make_network(one, travel), fixture_bank(one))
        np.testing.assert_array_equal(result.arrival_h, [24., 24.])
        self.assertTrue(result.gamma_route.all())
        self.assertFalse(result.varsigma.any())
        two = make_network().travel_hours.copy()
        two[:, 0] = 25.
        failed = self.run_bank(make_network(travel=two))
        self.assertTrue(np.isinf(failed.arrival_h).all())
        np.testing.assert_array_equal(failed.failed_arc_ids, [2, 2])

    def test_auxiliary_delay_soc_exact_formula_and_dsp_no_wait_energy(self):
        travel = make_network().travel_hours.copy()
        travel[:, 0] = .6
        net = make_network(travel=travel)
        early = self.run_bank(net)
        deferred = self.run_bank(net, departure_h=0.)
        # D=3km, base energy=.9; route=.8h and freeflow=.3h gives .5 auxiliary.
        np.testing.assert_allclose(early.travel_energy_kwh, [1.4, 1.4], atol=1e-14)
        np.testing.assert_allclose(deferred.travel_energy_kwh, [1.4, 1.4], atol=1e-14)
        np.testing.assert_allclose(early.arrival_soc, [.8 - 1.4 / 80] * 2, atol=1e-15)
        np.testing.assert_allclose(deferred.arrival_h - early.arrival_h, [1., 1.], atol=1e-15)
        self.assertEqual(early.task_sha256, deferred.task_sha256)
        self.assertNotEqual(early.replay_sha256, deferred.replay_sha256)

    def test_realized_replay_never_searches_or_mutates_frozen_bank(self):
        bank = fixture_bank()
        before = copy.deepcopy(bank)
        with patch('evcs_experiments.forecast_routes.search_route', side_effect=AssertionError('No rerouting')):
            with patch('evcs_experiments.forecast_tasks.build_candidate_tasks', side_effect=AssertionError('No rebuilding H')):
                result = self.run_bank(bank=bank)
        self.assertEqual(bank, before)
        self.assertEqual(result.task_ids, tuple(t['task_id'] for t in bank['tasks']))
        self.assertEqual(result.task_sha256, bank['task_sha256'])
        for array in (result.arrival_h, result.gamma_route, result.route_time_h,
                      result.travel_energy_kwh, result.arrival_soc, result.varsigma,
                      result.mu_full, result.usable_energy_dc_kwh):
            self.assertFalse(array.flags.writeable)

    def test_identity_incomplete_forgery_state_and_formal_oos_guards(self):
        bank = fixture_bank()
        with self.assertRaises(ValueError):
            replay_task_bank(make_network(), bank, PARAMETERS, STATES, split='oos')
        bad = copy.deepcopy(bank)
        bad['certified'] = False
        with self.assertRaises(ValueError):
            self.run_bank(bank=bad)
        bad = copy.deepcopy(bank)
        bad['tasks'][0]['arc_ids'] = [1]
        with self.assertRaises(ValueError):
            self.run_bank(bank=bad)
        bad = copy.deepcopy(bank)
        bad['task_sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            self.run_bank(bank=bad)
        with self.assertRaises(ValueError):
            replay_task_bank(make_network(), bank, PARAMETERS,
                             {'EV': SimpleNamespace(road_node=0, soc=.7)}, development=True)
        with self.assertRaises(ValueError):
            self.run_bank(make_network([Arc(1, 0, 1, 2., .1), Arc(2, 1, 2, 2., .2)]))


if __name__ == '__main__':
    unittest.main()
