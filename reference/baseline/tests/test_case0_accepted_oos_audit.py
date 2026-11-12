import tempfile,unittest,json
from pathlib import Path
from evcs_experiments.case0_accepted_oos_audit import check_sources
from evcs_experiments.campaign_batch import file_reference
class SourceAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.p=self.root/'case0_evaluation_audit.py';self.p.write_text('# checker fixture')
        self.ref=file_reference(self.p);self.m=self.root/'manifest.json';self.m.write_text(json.dumps({'files':[dict(self.ref,path=self.p.name)]}));self.manifest=file_reference(self.m)
    def test_exact_inherited_checker(self):check_sources([self.ref],self.manifest,self.ref)
    def test_changed_coherently_rehashed_checker_rejected(self):
        self.p.write_text('# changed checker');r=file_reference(self.p)
        with self.assertRaisesRegex(ValueError,'immutable manifest'):check_sources([r],self.manifest,r)
    def test_unbound_checker_rejected(self):
        with self.assertRaisesRegex(ValueError,'not inherited'):check_sources([],self.manifest,self.ref)
if __name__=='__main__':unittest.main()
