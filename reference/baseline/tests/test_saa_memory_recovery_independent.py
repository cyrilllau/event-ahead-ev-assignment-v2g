"""Independent resource-only delta guards; no optimizer."""
from unittest.mock import patch
from tests.test_saa_memory_recovery import MemoryRecoveryTests, memory, native, worker, pipeline

class IndependentMemoryTests(MemoryRecoveryTests):
 def test_scientific_parameters_cannot_change_with_memory(self):
  def capture(**kw):return kw
  capture.__module__=pipeline.__name__
  with patch.object(worker,'run_saa_numerical_phases',capture):
   restore,_=memory.install(self.ref,self.ad)
   try:
    for kw in ({'time_limit':601,'threads':4,'mip_rel_gap':.001},{'time_limit':600,'threads':4,'mip_rel_gap':.1}):
     with self.subTest(kw=kw),self.assertRaises(ValueError):native._parameters(kw)
    with self.assertRaises(ValueError):worker.run_saa_numerical_phases(output_directory=self.p['original_phase_root'],resume_checkpoint_reference={'path':'fake'},model_manifest_reference=self.p['model_manifest_reference'])
   finally:restore()
 def test_resealed_memory_observation_cannot_remove_headroom(self):
  low=self.write('low.json',dict(available_memory_kib=64*1024*1024,native_soft_limit_gb=64.,not_a_process_rss_cap=True))
  with self.assertRaisesRegex(ValueError,'headroom'):
   memory.verify_policy(self.write('lowpolicy.json',dict(self.p,resource_observation_reference=low)),self.ad)
