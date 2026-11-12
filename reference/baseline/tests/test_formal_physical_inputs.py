"""One real-kernel hand atom, never a production bank or OOS experiment."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from evcs_experiments.config import get_paths
from evcs_experiments.evidence import sha256
from evcs_experiments.forecast_tasks import qualification_soc, service_periods
from evcs_experiments.formal_physical_inputs import FormalPhysicalBank, EXPOSED
from evcs_experiments.formal_scenarios import PlanningContext
from evcs_experiments.joint_scenarios import array_hash
from evcs_experiments.physical_inputs import load_physical_case, load_frozen_tasks
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.scenario_features import generator_contract
from evcs_experiments.validation.validate_s01_replay import network_arcs


def write_json(path, data):
    path.write_text(json.dumps(data, sort_keys=True, indent=2) + '\n')
    return dict(path=str(path.resolve()), sha256=sha256(path), size_bytes=path.stat().st_size)


class FormalPhysicalInputsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = get_paths().data_root
        cls.case = load_physical_case()
        cls.context = PlanningContext(root, network_arcs(root))
        cls.target, cls.base = 'michael_2018', cls.context.events[0]
        state = next(x for x in cls.context.case.target(cls.target).vehicle_states if x.vehicle_id == 'D1-EV001')
        hardware = cls.case.vehicle_parameters[state.vehicle_id]
        qual = qualification_soc(.3, hardware['SOC_dep_under_e'], hardware['P_blk_under_e_kW'],
            1., hardware['eta_dis_e'], hardware['E_cap_e_kWh'])
        occ, reserve, block, _ = service_periods(1, 1.)
        task = dict(task_id=state.vehicle_id+'__CS1__k01', vehicle_id=state.vehicle_id, station_id='CS1',
            start_period=1, arc_ids=[], arrival_h=-1., distance_km=0., freeflow_h=0., travel_time_h=0.,
            travel_energy_kwh=0., arrival_soc=state.soc,
            usable_energy_kwh=(state.soc-hardware['SOC_dep_under_e'])*hardware['E_cap_e_kWh'],
            qualification_soc=qual, occupancy=list(occ), reserve=list(reserve), block_periods=list(block))
        args = dict(origin=state.road_node, destination=cls.case.station_parameters['CS1']['road_node'],
            departure_h=-1., initial_soc=state.soc, capacity_kwh=hardware['E_cap_e_kWh'],
            energy_kwh_per_km=hardware['E_km_e_kWh_per_km'], auxiliary_kw=hardware['P_aux_e_kW'], qualification_soc=qual)
        assert args['origin'] == args['destination']
        bank = dict(status='COMPLETE', certified=True, tasks_provisional=False, task_count=1,
            dependency_sha256=canonical_hash(dict(fixture='one actual-state zero route')),
            config=dict(basic_soc=.3, tau_blk_h=1., nearest_stations=3), tasks=[task],
            queries=[dict(vehicle_id=state.vehicle_id, station_id='CS1', status='COMPLETE', arguments=args,
                route={k:task[k] for k in ('arc_ids','arrival_h','distance_km','freeflow_h','travel_energy_kwh','arrival_soc')})],
            vehicle_summaries=[])
        bank['task_sha256'] = canonical_hash({k:bank[k] for k in ('dependency_sha256','config','certified','tasks')})
        cls.bank = bank
        cls.prepared = cls.context.prepare_target(cls.target, bank)
        cls.sample = cls.context.generate(cls.prepared, cls.base, 0, 'planning', domain='formal')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.task_ref = write_json(self.root/'tasks.json', self.bank)
        self.tasks = load_frozen_tasks(self.root/'tasks.json', self.case, self.task_ref['sha256'])
        self.arrays = {k:v.copy() for k,v in self.sample.arrays.items()}
        self.ledger = deepcopy(dict(self.sample.ledger))
        self.diagnostics = deepcopy(dict(self.sample.diagnostics))
        self.selected_index = 0
        self.publish()

    def publish(self):
        """Re-sign a synthetic fixture to test semantics beyond byte validation."""
        self.ledger['array_hashes'] = {k:array_hash(v) for k,v in self.arrays.items()}
        self.ledger['scenario_sha256'] = canonical_hash({k:v for k,v in self.ledger.items() if k!='scenario_sha256'})
        np.savez_compressed(self.root/'arrays.npz', **self.arrays)
        p=self.root/'arrays.npz'
        files=dict(arrays=dict(path=str(p), sha256=sha256(p), size_bytes=p.stat().st_size),
            ledger=write_json(self.root/'ledger.json',self.ledger),
            diagnostics=write_json(self.root/'diagnostics.json',self.diagnostics))
        manifest=write_json(self.root/'manifest.json',dict(scenario_id=self.ledger['scenario_id'],
            scenario_sha256=self.ledger['scenario_sha256'], files=files))
        rows=[dict(scenario_id=f'{self.target}/formal/planning/{base}/v{v+1:03d}/R1',
            scenario_sha256='0'*64, manifest=dict(path=str(self.root/'unmaterialized.json'),sha256='0'*64,size_bytes=0))
            for v in range(20) for base in self.context.events]
        rows[self.selected_index].update(scenario_sha256=self.ledger['scenario_sha256'],manifest=manifest)
        self.index=dict(target_event=self.target, split='planning', development=False, regime='R1',lead_h=1.,
            task_source=self.task_ref,task_sha256=self.tasks.task_sha256,task_ids=list(self.tasks.task_ids),
            assignment_state_sha256=self.prepared.target.state_sha256,times_h=(np.arange(-4,96)/4.).tolist(),scenarios=rows)
        self.write_index()

    def write_index(self):
        self.index_ref=write_json(self.root/'index.json',self.index)

    def adapter(self, **kwargs):
        return FormalPhysicalBank(self.root/'index.json',self.case,self.tasks,
            expected_index_sha256=self.index_ref['sha256'],**kwargs)

    def test_all_exposed_arrays_and_full_semantic_identity_preserved(self):
        result=self.adapter().load(0,expected_scenario_sha256=self.sample.scenario_sha256)
        self.assertFalse(result.development)
        self.assertEqual(result.scenario_sha256,self.sample.scenario_sha256)
        self.assertEqual(result.task_ids,self.tasks.task_ids)
        self.assertEqual(result.metadata['verified_array_count'],26)
        for key in EXPOSED:
            np.testing.assert_array_equal(getattr(result,key),self.arrays[key])
            with self.assertRaises(ValueError): getattr(result,key).setflags(write=True)
        self.assertTrue(np.isposinf(self.arrays['road_travel_time_min']).any())

    def test_each_use_revalidates_index_task_and_atom_bytes(self):
        for name in ('index.json','tasks.json','manifest.json','arrays.npz','ledger.json','diagnostics.json'):
            with self.subTest(name=name):
                adapter=self.adapter();path=self.root/name;data=path.read_bytes();path.write_bytes(data+b' ')
                with self.assertRaises(ValueError):adapter.load(0)
                path.write_bytes(data)

    def test_oos_development_and_unknown_schema_reject_before_primitives(self):
        original=deepcopy(self.index)
        for changed in (dict(split='oos'),dict(development=True),dict(lead_h=0),dict(regime='R2'),dict(unknown_schema=1)):
            self.index={**original,**changed};self.write_index()
            with self.subTest(changed=changed),patch('evcs_experiments.formal_physical_inputs.PlanningContext',side_effect=AssertionError('primitive started')):
                with self.assertRaises(ValueError):self.adapter()

    def test_support_order_state_and_fixed_h_cannot_change(self):
        original=deepcopy(self.index)
        for changed in (dict(task_ids=[]),dict(assignment_state_sha256='0'*64),
                dict(scenarios=list(reversed(original['scenarios']))),dict(times_h=list(reversed(original['times_h'])))):
            self.index={**original,**changed};self.write_index()
            with self.subTest(changed=list(changed)),self.assertRaises(ValueError):self.adapter()

    def test_rehashed_wrong_q_dc_energy_and_availability_rejected(self):
        original={k:v.copy() for k,v in self.arrays.items()}
        for key in ('load_q_kvar','task_usable_energy_dc_kwh','task_arrival_soc','task_gamma_route','task_mu_full'):
            self.arrays={k:v.copy() for k,v in original.items()}
            self.arrays[key].flat[0] = not self.arrays[key].flat[0] if self.arrays[key].dtype==bool else self.arrays[key].flat[0]+.01
            self.publish()
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'physical replay differs'):self.adapter().load(0)

    def test_full_array_hash_schema_types_and_nonfinite_rejected(self):
        original={k:v.copy() for k,v in self.arrays.items()}
        mutations=[('missing',lambda a:a.pop('features')),('shape',lambda a:a.update(features=np.zeros(6))),
            ('boolean',lambda a:a.update(line_outages=a['line_outages'].astype(int))),
            ('negative_inf',lambda a:a['road_travel_time_min'].__setitem__((0,0),-np.inf)),
            ('nan',lambda a:a['load_p_kw'].__setitem__((0,0),np.nan))]
        for name,mutate in mutations:
            self.arrays={k:v.copy() for k,v in original.items()};mutate(self.arrays);self.publish()
            with self.subTest(name=name),self.assertRaises(ValueError):self.adapter().load(0)

    def test_rehashed_wrong_original_primitive_or_replay_identity_rejected(self):
        for key in ('line_stream_id','task_replay_sha256','feature_sha256'):
            self.ledger=deepcopy(dict(self.sample.ledger));self.ledger[key]='0'*64;self.publish()
            with self.subTest(key=key),self.assertRaises(ValueError):self.adapter().load(0)

    def test_actual_zeta_variant11_strided_grid_survives_contiguous_storage(self):
        # The real failed atom's weather coordinates, with the small hand H.
        # This exercises the original physical kernels without any optimizer/OOS.
        sample=self.context.generate(self.prepared, 'zeta_2020', 10, 'planning', domain='formal')
        grid=sample.arrays['grid_hazard'];contiguous=np.ascontiguousarray(grid)
        self.assertEqual(grid.shape,(100,441));self.assertFalse(grid.flags.c_contiguous)
        np.testing.assert_array_equal(grid,contiguous)
        self.assertNotEqual(float(grid.mean()),float(contiguous.mean()))
        self.assertLess(abs(float(grid.mean()-contiguous.mean())),1e-12)
        self.assertEqual(float(grid.mean()),float(sample.arrays['features'][0]))
        self.arrays={k:v.copy() for k,v in sample.arrays.items()}
        self.ledger=deepcopy(dict(sample.ledger));self.diagnostics=deepcopy(dict(sample.diagnostics))
        self.selected_index=87;self.publish()
        result=self.adapter().load(87,expected_scenario_sha256=sample.scenario_sha256)
        self.assertEqual(result.scenario_sha256,sample.scenario_sha256)
        self.assertEqual(result.metadata['scenario_provenance']['feature_sha256'],sample.ledger['feature_sha256'])

    def test_subtolerance_saved_feature_change_still_breaks_original_semantic_hash(self):
        self.arrays['features'][0]+=1e-13
        self.publish()
        with self.assertRaisesRegex(ValueError,'feature semantic identity differs'):
            self.adapter().load(0)

    def test_resigned_wrong_feature_and_diagnostics_still_rejected(self):
        self.arrays['features'][0]+=.0001
        # Even internally re-signing every fixture byte must not excuse a
        # physically wrong feature value under the unchanged numerical check.
        self.ledger['feature_sha256']=canonical_hash(dict(values=self.arrays['features'].tolist(),
            diagnostics=self.diagnostics['feature_diagnostics'],contract=generator_contract()))
        self.publish()
        with self.assertRaisesRegex(ValueError,'physical replay differs: features'):self.adapter().load(0)
        self.arrays={k:v.copy() for k,v in self.sample.arrays.items()};self.ledger=deepcopy(dict(self.sample.ledger))
        self.diagnostics['feature_diagnostics']['road_feature_arc_count']=88
        self.publish()
        with self.assertRaisesRegex(ValueError,'feature diagnostics differ'):self.adapter().load(0)

    def test_explicit_relocation_retains_original_bytes_and_hashes(self):
        remote='/original/server/tasks.json'
        self.index['task_source']['path']=remote;self.write_index()
        result=self.adapter(path_overrides={remote:self.root/'tasks.json'}).load(0)
        self.assertEqual(result.scenario_sha256,self.sample.scenario_sha256)
        with self.assertRaises(FileNotFoundError):self.adapter()


if __name__=='__main__':unittest.main()
