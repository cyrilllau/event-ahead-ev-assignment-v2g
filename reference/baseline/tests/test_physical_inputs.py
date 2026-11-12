"""Hand-built adapter checks; no optimization or scenario generation."""
import copy
from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from evcs_experiments.physical_inputs import (
    load_physical_case, load_frozen_tasks, load_scenario_physical_inputs)
from evcs_experiments.random_streams import canonical_hash


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ah(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str((a.shape,a.dtype.str)).encode()+a.tobytes()).hexdigest()


class PhysicalInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = load_physical_case()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def bank(self, tau=2, empty=False):
        p = self.case.vehicle_parameters['D1-EV001']
        q = max(.3,.2+15*tau/(.95*80))
        route = dict(arc_ids=[],arrival_h=-1.,distance_km=0.,freeflow_h=0.,travel_energy_kwh=0.,arrival_soc=.8)
        args = dict(capacity_kwh=p['E_cap_e_kWh'],energy_kwh_per_km=.3,auxiliary_kw=1.,
                    qualification_soc=q,destination=1,departure_h=-1.)
        tasks=[]
        for k in ([] if empty else [3,4]):
            task=dict(task_id=f'D1-EV001__CS1__k{k:02d}',vehicle_id='D1-EV001',station_id='CS1',start_period=k,
                      **route,travel_time_h=0.,usable_energy_kwh=(.8-.2)*80,qualification_soc=q,
                      occupancy=[t>=k for t in range(1,25)],
                      reserve=[t>=k and t-1+tau<=24 for t in range(1,25)],block_periods=list(range(k,k+tau)))
            tasks.append(task)
        b=dict(status='COMPLETE',certified=True,tasks_provisional=False,tasks=tasks,task_count=len(tasks),
               queries=[dict(vehicle_id='D1-EV001',station_id='CS1',status='COMPLETE',arguments=args,route=route)],
               dependency_sha256='a'*64,config=dict(basic_soc=.3,tau_blk_h=float(tau)))
        self.rehash(b)
        return b

    @staticmethod
    def rehash(b):
        b['task_sha256']=canonical_hash({k:b[k] for k in ('dependency_sha256','config','certified','tasks')})

    def write_bank(self,b):
        p=self.root/'fixture.tasks.json';p.write_text(json.dumps(b));return p

    def load(self,b):
        p=self.write_bank(b)
        return load_frozen_tasks(p,self.case,sha(p))

    def scenario(self,tasks,edit=None):
        h=len(tasks.task_ids)
        arrival=np.array([3.,np.inf]);soc=np.array([.8,0.]);success=np.array([True,False])
        mu=np.zeros((h,24),dtype=bool);mu[0,3:23]=True
        arrays=dict(line_operating_outages=np.zeros((24,32),dtype=bool),load_p_kw=self.case.baseline_p_kw,
                    load_q_kvar=self.case.baseline_q_kvar,load_archived_q_kvar=self.case.archived_baseline_q_kvar,
                    load_epsilon=np.zeros(24),task_mu_full=mu,task_varsigma=success,task_arrival_soc=soc,
                    task_arrival_h=arrival,task_gamma_route=success,task_usable_energy_dc_kwh=np.array([(.8-.2)*80,0.]),
                    task_route_time_h=np.array([4.,0.]),task_travel_energy_kwh=np.zeros(h))
        if edit is not None: edit(arrays)
        row=dict(scenario_id='fixture/development/planning/base/v001/R1',target_event='fixture',split='planning',
                 base_weather_event='base',variant=1,variant_index=0,development=True,regime='R1',lead_h=1.,
                 task_sha256=tasks.task_sha256,array_hashes={k:ah(v) for k,v in arrays.items()})
        row['scenario_sha256']=canonical_hash(row)
        bank={k:v[None,...] for k,v in arrays.items()}
        bank.update(task_ids=np.array(tasks.task_ids),scenario_ids=np.array([row['scenario_id']]),times_h=np.arange(-4,96)/4.)
        ledger=dict(scenarios=[row],array_payload_hashes={k:ah(v) for k,v in bank.items()})
        p=self.root/'fixture.npz';l=self.root/'fixture.ledger.json'
        np.savez(p,**bank);l.write_text(json.dumps(ledger))
        return p,l,row

    def test_case_units_topology_and_exact_q1(self):
        c=self.case
        self.assertEqual(c.periods,tuple(range(1,25)))
        self.assertEqual(c.line_ids,tuple(range(1,33)))
        self.assertEqual(c.fixed_open_ties,tuple(range(33,38)))
        self.assertEqual(c.s_base_kva,10000.)
        self.assertEqual(c.station_bus_ids[0],21)
        np.testing.assert_array_equal(c.station_plugs,20)
        np.testing.assert_array_equal(c.station_interface_kw,300.)
        np.testing.assert_array_equal(c.station_rating_kva,375.)
        self.assertAlmostEqual(c.u_min,.9025)
        self.assertAlmostEqual(c.max_archived_q_adjustment_kvar,.0008,places=12)
        np.testing.assert_array_equal(c.baseline_q_kvar,c.baseline_p_kw*c.tan_phi_load)
        self.assertNotEqual(c.tan_phi_load[2],.4444444444)
        self.assertEqual(len(c.input_hashes),74)

    def test_arrays_and_nested_hardware_are_immutable(self):
        with self.assertRaises(ValueError): self.case.station_plugs[0]=0
        with self.assertRaises(ValueError): self.case.station_plugs.setflags(write=True)
        with self.assertRaises(TypeError): self.case.vehicle_parameters['D1-EV001']['eta_dis_e']=.5
        with self.assertRaises(FrozenInstanceError): self.case.dt_h=2
        tasks=self.load(self.bank())
        with self.assertRaises(ValueError): tasks.chi_occ.setflags(write=True)

    def test_tau_two_masks_full_keys_and_cost_units(self):
        b=self.bank();t=self.load(b)
        self.assertEqual(t.task_ids,('D1-EV001__CS1__k03','D1-EV001__CS1__k04'))
        self.assertEqual(t.vehicle_task_indices['D1-EV001'],(0,1))
        self.assertEqual(t.bus_task_indices[21],(0,1))
        self.assertTrue(t.chi_occ[0,23]);self.assertFalse(t.chi_res[0,23])
        self.assertEqual(np.flatnonzero(t.block_mask[0]).tolist(),[2,3])
        np.testing.assert_array_equal(t.e_forecast_dc_kwh,(.8-.2)*80)
        np.testing.assert_array_equal(t.c_act_usd,25.)
        np.testing.assert_array_equal(t.c_cap_usd_per_kwh,.02)
        self.assertEqual(t.config_hash,canonical_hash(b['config']))
        self.assertNotEqual(t.config_hash,t.task_sha256)

    def test_empty_h_keeps_explicit_axes(self):
        t=self.load(self.bank(empty=True))
        self.assertEqual(t.chi_occ.shape,(0,24));self.assertEqual(t.c_cap_usd_per_kwh.shape,(0,24))
        self.assertEqual(t.vehicle_task_indices,{})

    def test_file_config_and_semantic_hashes_are_separate_guards(self):
        b=self.bank();p=self.write_bank(b)
        with self.assertRaisesRegex(ValueError,'file SHA256'): load_frozen_tasks(p,self.case,'0'*64)
        with self.assertRaisesRegex(ValueError,'config hash'): load_frozen_tasks(p,self.case,sha(p),'0'*64)
        b['tasks'][0]['arrival_soc']=.7;p=self.write_bank(b)
        with self.assertRaisesRegex(ValueError,'semantic hash'): load_frozen_tasks(p,self.case,sha(p))

    def test_rehashed_bad_occupancy_and_unknown_vehicle_rejected(self):
        b=self.bank();b['tasks'][0]['occupancy'][23]=False;self.rehash(b)
        with self.assertRaisesRegex(ValueError,'occupancy'): self.load(b)
        b=self.bank();b['queries'][0]['vehicle_id']='UNKNOWN'
        with self.assertRaisesRegex(ValueError,'Unknown'): self.load(b)

    def test_rehashed_forecast_dc_scaling_error_rejected(self):
        b=self.bank();b['tasks'][0]['usable_energy_kwh']*=.95;self.rehash(b)
        with self.assertRaisesRegex(ValueError,'forecast physical'): self.load(b)

    def test_late_qualified_and_failed_scenario_preserved(self):
        t=self.load(self.bank());p,l,row=self.scenario(t)
        s=load_scenario_physical_inputs(p,l,0,self.case,t,expected_npz_sha256=sha(p),expected_ledger_sha256=sha(l),
                                        expected_scenario_sha256=row['scenario_sha256'])
        self.assertTrue(s.task_varsigma[0]);self.assertFalse(s.task_mu_full[0,2]);self.assertTrue(s.task_mu_full[0,3])
        self.assertTrue(np.isinf(s.task_arrival_h[1]));self.assertEqual(s.task_arrival_soc[1],0)
        self.assertEqual(s.task_usable_energy_dc_kwh[0],(.8-.2)*80)
        self.assertEqual(s.config_hash,t.config_hash)
        with self.assertRaises(ValueError): s.load_p_kw.setflags(write=True)

    def test_scenario_wrong_q_even_with_rehashed_payload_rejected(self):
        t=self.load(self.bank())
        def bad(a): a['load_q_kvar']=a['load_archived_q_kvar'].copy()
        p,l,_=self.scenario(t,bad)
        with self.assertRaisesRegex(ValueError,'Saved load'):
            load_scenario_physical_inputs(p,l,0,self.case,t,expected_npz_sha256=sha(p),expected_ledger_sha256=sha(l))

    def test_scenario_hash_and_task_order_are_bound(self):
        t=self.load(self.bank());p,l,_=self.scenario(t)
        with self.assertRaisesRegex(ValueError,'file SHA256'):
            load_scenario_physical_inputs(p,l,0,self.case,t,expected_npz_sha256='0'*64,expected_ledger_sha256=sha(l))
        with np.load(p) as z: values={k:z[k] for k in z.files}
        values['task_ids']=values['task_ids'][::-1];np.savez(p,**values)
        ledger=json.loads(l.read_text());ledger['array_payload_hashes']['task_ids']=ah(values['task_ids']);l.write_text(json.dumps(ledger))
        with self.assertRaisesRegex(ValueError,'task order'):
            load_scenario_physical_inputs(p,l,0,self.case,t,expected_npz_sha256=sha(p),expected_ledger_sha256=sha(l))


if __name__=='__main__':
    unittest.main()
