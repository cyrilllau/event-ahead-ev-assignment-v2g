import tempfile,unittest,copy
from pathlib import Path
from tests.test_checkpoint_streaming_proof_roles import StreamingProofTests
from evcs_experiments import checkpoint_closure as m
from evcs_experiments.campaign_batch import file_reference

class PhaseProofTests(unittest.TestCase):
 def fixture(self,root):
  helper=StreamingProofTests();model,cp,current,review=helper.fixture(root);write=helper.write;direct=file_reference(current)
  manifest=file_reference(root/'manifest.json')
  admission=write(root/'admission.json',dict(kind='SAA_FORMAL_PHASE_ADMISSION_V1',domain='formal',model_manifest_reference=manifest,planning_source_references=[direct]))
  p=dict(kind='FORMAL_SAA_NUMERICAL_PHASES_V1',model_manifest_reference=manifest,sources=[direct],helper_reference=direct,input_binding=model['input_binding'],scenario_ids=model['scenario_ids'],scenario_sha256s=model['scenario_sha256s'],template_references=[direct]*160,case_sha256=model['input_binding']['physical_case_sha256'],task_sha256=model['input_binding']['task_sha256'],cumulative_budget_seconds=7200,master_seconds=600,audit_reserve_seconds=600,seal_reserve_seconds=10,formal_admission_reference=admission,final_certificate_reserve_seconds=1790,master_threads=4,recourse_workers=1,master_mip_gap=.001)
  return p,current,write
 def test_current_phase_history_role_retains_all_live_refs(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);p,current,write=self.fixture(root)
   m.verify_checkpoint_closure(write(root/'phase.json',p))
   current.write_bytes(b'changed current numerical/source input')
   with self.assertRaises(ValueError):m.verify_checkpoint_closure(file_reference(root/'phase.json'))
 def test_resealed_scope_and_missing_refs_fail(self):
  for mode in ('sources','templates','input','model','budget','extra'):
   with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
    root=Path(d);p,current,write=self.fixture(root);p=copy.deepcopy(p)
    if mode=='sources':p['sources']=[]
    elif mode=='templates':p['template_references'][0]=None
    elif mode=='input':p['input_binding']['task_sha256']='wrong'
    elif mode=='model':p['model_manifest_reference']=None
    elif mode=='budget':p['master_seconds']=True
    else:p['extra_history']=p['model_manifest_reference']
    with self.assertRaises(ValueError):m.verify_checkpoint_closure(write(root/'phase.json',p))
