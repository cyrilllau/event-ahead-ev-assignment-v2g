from dataclasses import replace
from pathlib import Path
import tempfile,time,unittest,json
from copy import deepcopy
from evcs_experiments.random_streams import canonical_hash
import numpy as np
from evcs_experiments.validation.s03_toys import build_golden_cases
from evcs_experiments.first_stage import assemble_first_stage
from evcs_experiments.recourse import assemble_recourse
from evcs_experiments.saa_streaming import assemble_streaming_saa
from evcs_experiments.saa_streaming_audit import audit_streaming_saa

class SAAAssemblyAuditTests(unittest.TestCase):
    def test_original_coefficients_and_negative_objective_tamper(self):
        f=build_golden_cases()[0];first=assemble_first_stage(f.case,f.tasks)
        ts=tuple(assemble_recourse(f.case,f.tasks,s) for s in f.scenarios)
        with tempfile.TemporaryDirectory() as tmp:
            model=assemble_streaming_saa(first,ts,scenario_ids=f.scenario_ids,
                scenario_sha256s=tuple(t.scenario_sha256 for t in ts),empirical_probabilities=np.full(len(ts),1/len(ts)),
                output_directory=Path(tmp)/'model',max_coefficient_bytes=10000000)
            result=audit_streaming_saa(model,ts,deadline=time.monotonic()+10)
            self.assertEqual(result['scenario_count'],3);self.assertFalse(result['solver_bound_audited'])
            forged=deepcopy(model.manifest);forged['domain']='formal'
            forged['manifest_sha256']=canonical_hash({k:v for k,v in forged.items() if k!='manifest_sha256'})
            (model.root/'manifest.json').write_text(json.dumps(forged))
            with self.assertRaisesRegex(ValueError,'uniform160'):
                audit_streaming_saa(replace(model,manifest=forged),ts,deadline=time.monotonic()+10)
            (model.root/'manifest.json').write_text(json.dumps(model.manifest))
            objective=model.c.copy();objective[-1]+=1
            with self.assertRaisesRegex(ValueError,'objective'):
                audit_streaming_saa(replace(model,c=objective),ts,deadline=time.monotonic()+10)
            matrix=model.A.copy();matrix.data[-1]*=2
            with self.assertRaisesRegex(ValueError,'coefficients'):
                audit_streaming_saa(replace(model,A=matrix),ts,deadline=time.monotonic()+10)

if __name__=='__main__':unittest.main()
