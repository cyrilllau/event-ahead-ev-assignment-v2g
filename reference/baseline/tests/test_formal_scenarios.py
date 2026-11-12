"""Actual lower-kernel namespace and development regression; never OOS."""
from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from evcs_experiments.case_inputs import InputValidationError
from evcs_experiments.config import get_paths
from evcs_experiments.forecast_tasks import qualification_soc, service_periods
from evcs_experiments.formal_scenarios import PlanningContext
from evcs_experiments.joint_scenarios import JointContext
from evcs_experiments.random_streams import canonical_hash, stream
from evcs_experiments.validation.validate_s01_replay import network_arcs


def hand_frozen_bank(context, event):
    """One fixed zero-length route using an actual vehicle's frozen state.

    This is explicitly a hand-authored development fixture, not a formal H or a
    replacement for the full bank. All weather/line/road/load kernels are real.
    """
    target = context.case.target(event, lead_h=1.)
    for state in target.vehicle_states:
        raw = context.case.vehicle_parameters[state.vehicle_id]
        hardware = {key: float(raw[key]) for key in ('SOC_dep_under_e', 'P_blk_under_e_kW',
            'eta_dis_e', 'E_cap_e_kWh', 'E_km_e_kWh_per_km', 'P_aux_e_kW')}
        qual = qualification_soc(.3, hardware['SOC_dep_under_e'], hardware['P_blk_under_e_kW'],
            1., hardware['eta_dis_e'], hardware['E_cap_e_kWh'])
        if state.soc >= qual:
            break
    else:
        raise AssertionError('Frozen case has no vehicle for the one-task hand fixture')
    vehicle, station = state.vehicle_id, 'HAND_ZERO_ROUTE'
    occupancy, reserve, block, _ = service_periods(1, 1.)
    task = dict(task_id=f'{vehicle}__{station}__k01', vehicle_id=vehicle, station_id=station,
        start_period=1, arc_ids=[], arrival_h=-1., distance_km=0., freeflow_h=0.,
        travel_time_h=0., travel_energy_kwh=0., arrival_soc=state.soc,
        usable_energy_kwh=(state.soc - hardware['SOC_dep_under_e']) * hardware['E_cap_e_kWh'],
        qualification_soc=qual, occupancy=list(occupancy), reserve=list(reserve), block_periods=list(block))
    arguments = dict(origin=state.road_node, destination=state.road_node, departure_h=-1.,
        initial_soc=state.soc, capacity_kwh=hardware['E_cap_e_kWh'],
        energy_kwh_per_km=hardware['E_km_e_kWh_per_km'], auxiliary_kw=hardware['P_aux_e_kW'],
        qualification_soc=qual)
    bank = dict(status='COMPLETE', certified=True, dependency_sha256=canonical_hash(
        dict(fixture='one fixed zero route', state_sha256=target.state_sha256)),
        config=dict(basic_soc=.3, tau_blk_h=1., nearest_stations=3), tasks=[task],
        queries=[dict(vehicle_id=vehicle, station_id=station, status='COMPLETE',
            arguments=arguments, route=dict(arc_ids=[]))], vehicle_summaries=[])
    bank['task_sha256'] = canonical_hash({key: bank[key] for key in
        ('dependency_sha256', 'config', 'certified', 'tasks')})
    return bank


class FormalScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = get_paths().data_root
        arcs = network_arcs(root)
        cls.context = PlanningContext(root, arcs)
        cls.old = JointContext(root, arcs)
        cls.event, cls.base = cls.context.events[0], cls.context.events[1]
        cls.bank = hand_frozen_bank(cls.context, cls.event)
        cls.prepared = cls.context.prepare_target(cls.event, cls.bank)
        cls.old_prepared = cls.old.prepare_target(cls.event, cls.bank)
        cls.old_sample = cls.old.generate(cls.old_prepared, cls.base, 0, 'planning')
        cls.development = cls.context.generate(cls.prepared, cls.base, 0, 'planning', domain='development')
        cls.formal = cls.context.generate(cls.prepared, cls.base, 0, 'planning', domain='formal')
        cls.calibration = cls.context.generate(cls.prepared, cls.base, 0, 'calibration', domain='formal')

    def test_development_baseline_matches_every_old_physical_array_and_identity(self):
        self.assertEqual(set(self.development.arrays), set(self.old_sample.arrays))
        for name in self.old_sample.arrays:
            with self.subTest(name=name):
                np.testing.assert_array_equal(self.development.arrays[name], self.old_sample.arrays[name])
        self.assertEqual(dict(self.development.ledger), dict(self.old_sample.ledger))
        self.assertEqual(dict(self.development.diagnostics), dict(self.old_sample.diagnostics))
        self.assertEqual(self.development.scenario_sha256, self.old_sample.scenario_sha256)

    def test_formal_arrays_come_from_original_formal_streams_not_relabelled_development(self):
        formal = self.formal
        self.assertFalse(formal.ledger['development'])
        self.assertIn('/formal/planning/', formal.ledger['scenario_id'])
        expected = stream('planning', self.context.events.index(self.base), 0,
            'line_gaussian', development=False)
        self.assertEqual(formal.ledger['line_stream_id'], expected.stream_id)
        for name in ('line_standard_normals', 'load_standard_innovation', 'road_corridor_uniforms'):
            with self.subTest(name=name):
                self.assertFalse(np.array_equal(formal.arrays[name], self.development.arrays[name]))
        self.assertNotEqual(formal.ledger['residual_donor_event'], self.event)
        self.assertEqual(formal.ledger['task_sha256'], self.development.ledger['task_sha256'])
        self.assertEqual(formal.ledger['assignment_state_sha256'], self.development.ledger['assignment_state_sha256'])

    def test_formal_planning_calibration_and_development_cache_domains_are_separate(self):
        self.assertNotEqual(self.formal.ledger['line_stream_id'], self.calibration.ledger['line_stream_id'])
        self.assertFalse(np.array_equal(self.formal.arrays['load_standard_innovation'],
            self.calibration.arrays['load_standard_innovation']))
        again = self.context.generate(self.prepared, self.base, 0, 'planning', domain='development')
        self.assertEqual(again.scenario_sha256, self.development.scenario_sha256)
        repeated = self.context.generate(self.prepared, self.base, 0, 'planning', domain='formal')
        self.assertEqual(repeated.scenario_sha256, self.formal.scenario_sha256)
        for name, values in repeated.arrays.items():
            np.testing.assert_array_equal(values, self.formal.arrays[name])
        for domain, split in (('development', 'planning'), ('formal', 'planning'), ('formal', 'calibration')):
            self.assertIn((domain, split, self.base, 0), self.context.primitives)

    def test_oos_and_unsupported_coordinates_are_rejected_before_any_primitive(self):
        with patch('evcs_experiments.line_outages.LineModel.draw_thresholds', side_effect=AssertionError('Primitive started')):
            for changes in (dict(split='oos'), dict(domain='unknown'), dict(variant_index=True),
                    dict(variant_index=20), dict(split='calibration', variant_index=10),
                    dict(regime='R2'), dict(base_event='unknown')):
                arguments = dict(base_event=self.base, variant_index=0, split='planning', domain='formal')
                arguments.update(changes)
                with self.subTest(changes=changes), self.assertRaises(InputValidationError):
                    self.context.generate(self.prepared, **arguments)

    def test_preparation_preserves_caller_bank_and_rejects_changed_epoch_state_or_tasks(self):
        original = deepcopy(self.bank)
        prepared = self.context.prepare_target(self.event, self.bank)
        self.assertEqual(self.bank, original)
        prepared.bank['queries'][0]['arguments']['initial_soc'] = -1
        self.assertEqual(self.bank, original)
        for lead in (0., .5, 2., True):
            with self.subTest(lead=lead), self.assertRaises(InputValidationError):
                self.context.prepare_target(self.event, self.bank, lead_h=lead)
        changed = deepcopy(self.bank); changed['tasks'][0]['arrival_soc'] += .01
        with self.assertRaisesRegex(InputValidationError, 'semantic hash'):
            self.context.prepare_target(self.event, changed)
        changed = deepcopy(self.bank); changed['queries'][0]['arguments']['initial_soc'] += .01
        with self.assertRaisesRegex(InputValidationError, 'state/hardware'):
            self.context.prepare_target(self.event, changed)
        foreign = replace(self.prepared, target=replace(self.prepared.target, state_sha256='0' * 64))
        with self.assertRaisesRegex(InputValidationError, 'target state'):
            self.context.generate(foreign, self.base, 0, 'planning')


if __name__ == '__main__':
    unittest.main()
