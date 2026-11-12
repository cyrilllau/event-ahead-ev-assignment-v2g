"""One saved real onset atom, complete ID inventory but NOT a full saved bank."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from evcs_experiments.config import get_paths
from evcs_experiments.evidence import sha256
from evcs_experiments.forecast_tasks import load_forecast_task_inputs
from evcs_experiments.formal_physical_inputs import FormalPhysicalBank, EXPOSED
from evcs_experiments.onset_physical_inputs import OnsetPhysicalBank
from evcs_experiments.onset_planning import OnsetPlanningContext
from evcs_experiments.joint_scenarios import array_hash
from evcs_experiments.physical_inputs import load_physical_case, load_frozen_tasks
from evcs_experiments.random_streams import canonical_hash


def write_json(path,data):
    path.write_text(json.dumps(data,sort_keys=True,indent=2)+'\n')
    return dict(path=str(path.resolve()),sha256=sha256(path),size_bytes=path.stat().st_size)


class OnsetPhysicalInputsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case=load_physical_case()
        root=get_paths().data_root
        cls.target='michael_2018'
        inputs=load_forecast_task_inputs(root,cls.target,0.)
        cls.context=OnsetPlanningContext(root,inputs.network.arcs)
        cls.prepared=cls.context.prepare_target(cls.target,overall_time_limit_seconds=10.)
        cls.bank=cls.prepared.bank
        cls.sample=cls.context.generate(cls.prepared,cls.context.events[0],0,'planning')

    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name)
        self.task_ref=write_json(self.root/'tasks.json',self.bank)
        self.tasks=load_frozen_tasks(self.root/'tasks.json',self.case,self.task_ref['sha256'])
        self.arrays={k:v.copy() for k,v in self.sample.arrays.items()}
        self.ledger=deepcopy(dict(self.sample.ledger))
        self.publish()

    def publish(self):
        self.ledger['array_hashes']={k:array_hash(v) for k,v in self.arrays.items()}
        self.ledger['scenario_sha256']=canonical_hash({k:v for k,v in self.ledger.items() if k!='scenario_sha256'})
        p=self.root/'arrays.npz';np.savez_compressed(p,**self.arrays)
        files=dict(arrays=dict(path=str(p),sha256=sha256(p),size_bytes=p.stat().st_size),
            ledger=write_json(self.root/'ledger.json',self.ledger),
            diagnostics=write_json(self.root/'diagnostics.json',dict(self.sample.diagnostics)))
        manifest=write_json(self.root/'manifest.json',dict(scenario_id=self.ledger['scenario_id'],
            scenario_sha256=self.ledger['scenario_sha256'],files=files))
        rows=[dict(scenario_id=f'{self.target}/formal/planning/{b}/v{v+1:03d}/R1',
            scenario_sha256='0'*64,manifest=dict(path=str(self.root/'NOT_GENERATED.json'),sha256='0'*64,size_bytes=0))
            for v in range(20) for b in self.context.events]
        rows[0].update(scenario_sha256=self.ledger['scenario_sha256'],manifest=manifest)
        self.index=dict(target_event=self.target,split='planning',development=False,regime='R1',lead_h=0.,
            task_source=self.task_ref,task_sha256=self.tasks.task_sha256,task_ids=list(self.tasks.task_ids),
            assignment_state_sha256=self.prepared.target.state_sha256,times_h=(np.arange(96)/4).tolist(),scenarios=rows)
        self.index_ref=write_json(self.root/'index.json',self.index)

    def adapter(self,reader=OnsetPhysicalBank):
        return reader(self.root/'index.json',self.case,self.tasks,expected_index_sha256=self.index_ref['sha256'])

    def test_saved_real_H_all_exposed_arrays_and_original_identity(self):
        result=self.adapter().load(0,expected_scenario_sha256=self.sample.scenario_sha256)
        self.assertEqual(len(result.task_ids),1128)
        self.assertEqual(len(result.times_h),96)
        self.assertEqual(result.metadata['verified_array_count'],26)
        for name in EXPOSED:
            np.testing.assert_array_equal(getattr(result,name),self.sample.arrays[name])
        self.assertEqual(result.metadata['scenario_provenance']['lead_h'],0.)
        with self.assertRaises(FileNotFoundError):self.adapter().load(1)

    def test_baseline_reader_cannot_admit_onset(self):
        with self.assertRaisesRegex(ValueError,'reader lead'):self.adapter(FormalPhysicalBank)

    def test_resealed_physical_Q_energy_and_initial_state_tampering_rejected(self):
        original={k:v.copy() for k,v in self.arrays.items()}
        for name in ('load_q_kvar','task_usable_energy_dc_kwh','task_arrival_soc'):
            self.arrays={k:v.copy() for k,v in original.items()};self.arrays[name].flat[0]+=.01
            self.publish()
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'physical replay differs'):
                self.adapter().load(0)

    def test_resealed_information_contract_and_stream_tampering_rejected(self):
        for name in ('forecast_dependency_sha256','information_contract','line_stream_id'):
            self.ledger=deepcopy(dict(self.sample.ledger));self.ledger[name]='0'*64;self.publish()
            with self.subTest(name=name),self.assertRaises(ValueError):self.adapter().load(0)

    def test_index_epoch_order_and_source_bytes_rejected(self):
        for mutation in ('epoch','state','times','order'):
            self.publish()
            if mutation=='epoch':self.index['lead_h']=1.
            if mutation=='state':self.index['assignment_state_sha256']='0'*64
            if mutation=='times':self.index['times_h']=(np.arange(-4,96)/4).tolist()
            if mutation=='order':self.index['scenarios'].reverse()
            self.index_ref=write_json(self.root/'index.json',self.index)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.adapter()
        self.publish();reader=self.adapter();p=self.root/'arrays.npz';p.write_bytes(p.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'bytes/hash changed'):reader.load(0)
