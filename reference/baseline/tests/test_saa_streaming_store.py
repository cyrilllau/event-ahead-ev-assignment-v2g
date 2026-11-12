from pathlib import Path
import tempfile,time,unittest,json,shutil
from copy import deepcopy
from evcs_experiments.random_streams import canonical_hash
import numpy as np
from evcs_experiments.validation.s03_toys import build_golden_cases
from evcs_experiments.first_stage import assemble_first_stage
from evcs_experiments.recourse import assemble_recourse
from evcs_experiments.saa_streaming import assemble_streaming_saa
from evcs_experiments.saa_streaming_store import load_streaming_saa
from evcs_experiments.evaluation_lock import file_reference

class SAAStoreTests(unittest.TestCase):
    def test_reload_identical_numeric_without_provider_and_reject_changed_source_or_bytes(self):
        f=build_golden_cases()[0];first=assemble_first_stage(f.case,f.tasks)
        ts=tuple(assemble_recourse(f.case,f.tasks,s) for s in f.scenarios)
        with tempfile.TemporaryDirectory() as temp:
            model=assemble_streaming_saa(first,ts,scenario_ids=f.scenario_ids,
                scenario_sha256s=[t.scenario_sha256 for t in ts],empirical_probabilities=np.full(3,1/3),
                output_directory=Path(temp)/'model',max_coefficient_bytes=1000000,input_binding={'source':'original'})
            ref=file_reference(model.root/'manifest.json')
            args=dict(first_stage=first,scenario_ids=f.scenario_ids,scenario_sha256s=[t.scenario_sha256 for t in ts],
                expected_input_binding={'source':'original'},deadline=time.monotonic()+10)
            restored=load_streaming_saa(ref,**args)
            self.assertEqual((restored.A!=model.A).nnz,0)
            np.testing.assert_array_equal(restored.c,model.c)
            self.assertFalse(restored.c.flags.writeable)
            with self.assertRaisesRegex(ValueError,'binding'):
                load_streaming_saa(ref,**{**args,'expected_input_binding':{'source':'changed'}})
            forged=deepcopy(model.manifest)
            alternate=model.root/'shards'/'c.npy';shutil.copyfile(model.root/'c.npy',alternate)
            forged['numeric_artifacts']=[file_reference(alternate) if Path(r['path']).name=='c.npy' else r for r in forged['numeric_artifacts']]
            forged['manifest_sha256']=canonical_hash({k:v for k,v in forged.items() if k!='manifest_sha256'})
            (model.root/'manifest.json').write_text(json.dumps(forged))
            with self.assertRaisesRegex(ValueError,'numeric path'):
                load_streaming_saa(file_reference(model.root/'manifest.json'),**args)
            (model.root/'manifest.json').write_text(json.dumps(model.manifest))
            ref=file_reference(model.root/'manifest.json')
            path=model.root/'c.npy';raw=bytearray(path.read_bytes());raw[-1]^=1;path.write_bytes(raw)
            with self.assertRaises(ValueError):load_streaming_saa(ref,**args)

if __name__=='__main__':unittest.main()
