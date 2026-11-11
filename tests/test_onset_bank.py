"""Reduced orchestration fixtures; real onset physics is tested separately."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import ExitStack
from test_formal_bank import FakeContext
from evcs_experiments import formal_bank as shared
from evcs_experiments import onset_bank as module
from evcs_experiments.random_streams import canonical_hash


class FakeOnsetContext(FakeContext):
    def __init__(self,*args):
        super().__init__(*args);self.case_root=args[0]
    def prepare_target(self,target,bank):
        prepared=super().prepare_target(target,bank,lead_h=1.)
        prepared.bank=bank;prepared.target.lead_h=0.
        return prepared


class OnsetBankTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.out=self.root/'out'
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.request=dict(target_ids=['t0','t1'],domain='formal',lead_h=0,regime='R1',bank_kind='onset_v1',
            split_sizes={'planning':1,'calibration':1},source_manifest={},formal_protocol_lock={})
        FakeOnsetContext.calls=[];FakeOnsetContext.interrupt_next=False;FakeOnsetContext.wrong_domain=False;FakeOnsetContext.bad_float=None
        self.stack.enter_context(patch.object(module.OnsetBankAdapter,'context_type',FakeOnsetContext))
        self.stack.enter_context(patch.object(shared,'network_arcs',return_value=()))
        self.stack.enter_context(patch.object(shared,'get_paths',return_value=SimpleNamespace(data_root=self.root)))
        self.stack.enter_context(patch.object(module.OnsetBankAdapter,'sources',return_value=({'references':{'physical_inputs':[]}},{})))
        self.stack.enter_context(patch.object(module,'load_forecast_task_inputs',side_effect=lambda root,target,**kw:target))
        self.stack.enter_context(patch.object(module,'build_candidate_tasks',side_effect=lambda target,**kw:
            SimpleNamespace(as_dict=lambda:dict(certified=True,task_sha256=canonical_hash(target+'onset')))))

    def run_bank(self,resume=False):
        return module.generate_onset_bank(self.request,wall_seconds=60,outputroot=self.out,resume=resume)

    def test_complete_recomputes_moments_and_pins_new_task_sources(self):
        result=self.run_bank()
        self.assertEqual(result['status'],module.COMPLETE)
        self.assertNotIn('BASELINE',result['status'])
        self.assertEqual(result['atom_count'],8)
        cp=json.loads((self.out/'checkpoint.json').read_text())
        self.assertEqual(cp['task_sources'],result['task_sources'])
        for target in self.request['target_ids']:
            index=json.loads((self.out/target/'planning.index.json').read_text())
            moment=json.loads((self.out/target/'moments.json').read_text())
            self.assertEqual(index['lead_h'],0.)
            self.assertEqual(set(moment['bounds_by_rho']),{'0.5','1.0','1.5','2.0'})
            self.assertFalse(moment['bootstrap']['stream_identity']['development'])

    def test_resume_does_not_rebuild_H_or_regenerate_saved_atom(self):
        FakeOnsetContext.interrupt_next=True
        with self.assertRaisesRegex(RuntimeError,'interruption'):self.run_bank()
        cp=json.loads((self.out/'checkpoint.json').read_text())
        first=next(iter(cp['completed_atoms']))
        before=FakeOnsetContext.calls.count(first)
        with patch.object(module,'build_candidate_tasks',side_effect=AssertionError('Must reuse saved H')):
            # t1 has not been prepared yet; stop only after resuming first target.
            with patch.object(shared,'bootstrap_moments',side_effect=RuntimeError('end bounded resume')):
                with self.assertRaisesRegex(RuntimeError,'end bounded resume'):self.run_bank(resume=True)
        self.assertEqual(FakeOnsetContext.calls.count(first),before)

    def test_changed_checkpointed_task_reference_fails_closed(self):
        FakeOnsetContext.interrupt_next=True
        with self.assertRaises(RuntimeError):self.run_bank()
        p=self.out/'t0'/'task_source.json';ref=json.loads(p.read_text());ref['sha256']='0'*64;p.write_text(json.dumps(ref))
        with self.assertRaisesRegex(ValueError,'Checkpointed onset task source changed'):self.run_bank(resume=True)


class OnsetRequestTests(unittest.TestCase):
    def test_requires_formal_full_sizes_explicit_onset_identity(self):
        valid=dict(bank_kind='onset_v1',lead_h=0,domain='formal',regime='R1',split_sizes={'planning':20,'calibration':10})
        with patch.object(module,'accepted_protocol_sources',return_value=({},{})):
            self.assertEqual(module.OnsetBankAdapter().sources(valid),({},{}))
            for delta in (dict(lead_h=1),dict(lead_h=False),dict(bank_kind='baseline'),dict(split_sizes={'planning':1,'calibration':1})):
                with self.subTest(delta=delta),self.assertRaises(ValueError):module.OnsetBankAdapter().sources({**valid,**delta})
