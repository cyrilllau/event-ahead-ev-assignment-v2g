"""Independent compact-attempt evidence gates; no optimizer or server work."""
import json,unittest
from unittest.mock import patch
from tests import test_case0_memory_repair_independent as prior
from evcs_experiments import case0_evaluation_runner as r,case0_campaign as c

class RepresentationEvidenceTests(unittest.TestCase):
 def setUp(self):
  f=prior.MemoryRepairEvidenceTests();f.setUp();self.addCleanup(f.doCleanups);self.f=f;f.memory_patch.stop()
  destination=f.base/'compact-metadata-1';f.attempt.rename(destination);f.attempt=destination
  f.transition['new_attempt_directory']=str(destination);f.state['configuration_attempt_roots']={'size0/EA_DRO':str(destination)}
  self.refresh()
  p=patch('evcs_experiments.case0_memory_repair.validate_representation',return_value=(f.config,f.transition),create=True);p.start();self.addCleanup(p.stop)
 def refresh(self):
  f=self.f;ref=r.file_reference(f.attempt/'solve/result.json');f.state['completed']['size0/EA_DRO'].update(path=ref['path'],sha256=ref['sha256']);(f.root/'state.json').write_text(json.dumps(f.state))
 def test_original_inputs_and_new_evidence_remain_separate(self):
  f=self.f;_,_,e=r.planning_evidence(f.root)
  self.assertEqual(e['EA_DRO']['result_reference']['path'],str(f.attempt/'solve/result.json'))
  self.assertEqual((f.attempt/'inputs').resolve(),f.base/'inputs')
 def test_unconfigured_compact_claim_rejected_after_rehash(self):
  f=self.f;p=f.attempt/'solve/result.json';v=r.read(p);v.update(descriptive_metadata_representation='lossless_categorical_rows_v1',descriptive_metadata_code_bytes=10);p.write_text(json.dumps(v));self.refresh()
  with self.assertRaisesRegex(ValueError,'unconfigured compact'):r.planning_evidence(f.root)
 def test_configured_compact_omission_rejected(self):
  f=self.f;p=f.root/'config.lock.json';v=r.read(p);v['solver_compact_recourses']=True;p.write_text(json.dumps(v));f.config.update(v);sha=r.file_reference(p)['sha256']
  for method in r.METHODS:
   d=f.attempt if method=='EA_DRO' else f.root/'planning/size0'/method;p=d/'start.json';v=r.read(p);v['config_sha256']=sha;p.write_text(json.dumps(v))
  with self.assertRaisesRegex(ValueError,'omitted configured compact'):r.planning_evidence(f.root)
 def test_nested_master_budget_reset_rejected(self):
  f=self.f;f.transition.update(prior_configuration_seconds=613.6470186206637,prior_master_seconds=613.165128)
  state=dict(f.state,configuration_spent_seconds={'size0/EA_DRO':613.6470186206637},configuration_master_spent_seconds={'size0/EA_DRO':613.165128})
  self.assertEqual(c.configuration_attempt(f.root,state,'size0/EA_DRO')[0],f.attempt)
  state['configuration_master_spent_seconds']['size0/EA_DRO']=0.
  with self.assertRaisesRegex(ValueError,'carryover ledger'):c.configuration_attempt(f.root,state,'size0/EA_DRO')

class RepresentationForwardingTests(unittest.TestCase):
 def setUp(self):
  from tests.test_case0_representation import RepresentationTests
  self.f=RepresentationTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
 def test_explicit_compact_and_old_remaining_caps_reach_solver(self):
  from types import SimpleNamespace
  f=self.f;state=f.prepare();state['active_configuration']='size0/EA_DRO';c.save_state(f.root,state)
  original=[object(),object()];compressed=[object(),object()]
  ctx=SimpleNamespace(first=object(),probability_set=object(),planning_ids=['a','b'],case=object(),tasks=object(),scenario=lambda i:object(),recourse=lambda i:original[i])
  with patch('evcs_experiments.case0_isolation.verify_current_cgroup',return_value={}),patch('evcs_experiments.case0_inputs.Case0Context',return_value=ctx),patch('evcs_experiments.compact_recourses.compact_recourse',side_effect=compressed) as compact,patch('evcs_experiments.case0_planning.run_case0_planning',return_value=(None,'result')) as solve:
   self.assertEqual(c.plan_one('EA_DRO',1000.),'result')
  self.assertEqual([x.args[0] for x in compact.call_args_list],original);self.assertEqual(solve.call_args.args[1],compressed)
  kw=solve.call_args.kwargs;self.assertIs(kw['compact_descriptive_metadata'],True);self.assertIs(kw['zero_presolve'],True)
  self.assertAlmostEqual(kw['master_seconds'],586.834872);self.assertAlmostEqual(kw['deadline'],1000+1800-613.6470186206637-.125)
 def test_global_fee_reset_after_prepare_rejected(self):
  from evcs_experiments import case0_memory_repair as m
  f=self.f;state=f.prepare();state['compute_seconds']=0
  with self.assertRaisesRegex(ValueError,'budget reset'):m.validate_representation(f.root,state)
