"""Saved-certificate audit regressions; no scientific assembler or solve."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from evcs_experiments.config import get_paths

spec=importlib.util.spec_from_file_location('audit_s03_i02_test',get_paths().project_root/'scripts/audits/audit_s03_i02.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


class SavedI02AuditTests(unittest.TestCase):
    def fixture(self,root,mass=.25):
        phi=np.array([[0.,0.,0.,0.,1.04],[1.,0.,0.,0.,1.04]])
        lo=hi=np.array([mass,0.,0.,0.,1.04]);p=np.array([1-mass,mass]);ids=['atom_a','atom_b']
        ah=audit.canonical({'support_ids':ids,'feature_ids':list(audit.old.FEATURES),'features':phi.tolist(),
            'lower':lo.tolist(),'upper':hi.tolist(),'feature_orientation':'scenario_by_feature','dual_convention':'alpha_lower_beta_upper'})
        r={'support_ids':ids,'feature_ids':list(audit.old.FEATURES),'ambiguity_sha256':ah,'probabilities':p.tolist(),
           'initial_support_indices':[0,1],'initial_support_ids':ids,'initial_probabilities':p.tolist(),
           'witness_sha256':audit.canonical({'ambiguity_sha256':ah,'support_ids':ids,'probabilities':p.tolist()}),
           'feasible':True,'basis_verified':True,'status':'FEASIBLE_BASIC',
           'certificate':{'basis':{'full_active_rank':2},'support_threshold':0.,'probabilities_clipped_or_renormalized':False},
           'solver':{'status':0,'success':True,'raw_solution':p.tolist(),'raw_objective':0.,'options':{'threads':1},
             'ineqlin_marginals':[0.]*10,'eqlin_marginals':[0.],'lower_marginals':[0.,0.],'upper_marginals':[0.,0.]}}
        np.savez_compressed(root/'inputs.npz',features=phi,lower=lo,upper=hi,support_ids=ids,feature_ids=audit.old.FEATURES)
        (root/'solution.json').write_text(json.dumps(r));return r,(phi,lo,hi,ids)

    def test_handwritten_full_witness_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);_,expected=self.fixture(p);r=audit.phase1(p,expected)
            self.assertEqual(r['initial_support'],2);self.assertEqual(r['full_active_rank'],2)

    def test_tiny_positive_is_required_even_below_numeric_tolerance(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);r,expected=self.fixture(p,1e-12);audit.phase1(p,expected)
            r['initial_support_indices']=[0];r['initial_support_ids']=['atom_a'];r['initial_probabilities']=[1.]
            (p/'solution.json').write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,'dropped nonzero'):audit.phase1(p,expected)

    def test_changed_source_feature_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);_,expected=self.fixture(p);changed=expected[0].copy();changed[0,0]=.1
            with self.assertRaisesRegex(ValueError,'accepted input changed'):audit.phase1(p,(changed,*expected[1:]))

    def test_false_rank_claim_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);r,expected=self.fixture(p);r['certificate']['basis']['full_active_rank']=1
            (p/'solution.json').write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,'certificate claims'):audit.phase1(p,expected)

    def test_zero_objective_cannot_hide_bad_raw_kkt(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);r,expected=self.fixture(p);r['solver']['lower_marginals'][0]=1.
            (p/'solution.json').write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,'numeric certificate'):audit.phase1(p,expected)


if __name__=='__main__':unittest.main()
