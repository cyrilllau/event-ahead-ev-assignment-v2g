"""Independent current-dependency completeness negatives for streaming proof roles."""
import tempfile
from pathlib import Path
import unittest
from evcs_experiments import checkpoint_closure as m
from evcs_experiments.random_streams import canonical_hash
from tests.test_checkpoint_streaming_proof_roles import StreamingProofTests

class IndependentStreamingProofRoleTests(unittest.TestCase):
    def test_resealed_streaming_current_dependency_omissions_rejected(self):
        helper=StreamingProofTests()
        for mode in ('numeric_none','numeric_malformed','scenarios_none','source_none'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
                root=Path(d);model,cp,current,review=helper.fixture(root)
                if mode=='numeric_none':model['numeric_artifacts']=None
                elif mode=='numeric_malformed':model['numeric_artifacts']=[dict(model['numeric_artifacts'][0],unexpected=True)]
                elif mode=='scenarios_none':model['scenarios']=None
                else:model['input_binding']['planning_source']=None
                model['manifest_sha256']=canonical_hash({k:v for k,v in model.items() if k!='manifest_sha256'})
                with self.assertRaises(ValueError):m.verify_checkpoint_closure(helper.write(root/'manifest.json',model))

if __name__=='__main__':unittest.main()
