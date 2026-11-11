"""Independent schema negatives and direct-dependency checks; no solver."""
import json
from pathlib import Path
import tempfile
import unittest
from evcs_experiments import checkpoint_closure as m
from evcs_experiments.campaign_batch import file_reference
from tests import test_checkpoint_association_history as author_fixture


class AssociationHistoryIndependentTests(unittest.TestCase):
    def fixture(self,root):
        return author_fixture.AssociationHistoryTests().fixture(root)

    def save(self,path,value):
        path.write_text(json.dumps(value));return file_reference(path)

    def test_mandatory_current_reference_fields_cannot_be_null_or_malformed(self):
        for field in ('index_reference','acceptance_reference','prior_review_reference','source_manifest','rows_reference'):
            for malformed in (None, {'path':'unused','sha256':'a'*64,'size_bytes':1,'quoted':True}):
                with self.subTest(field=field,malformed=malformed), tempfile.TemporaryDirectory() as d:
                    root=Path(d).resolve();ref,index,current,old,cert=self.fixture(root)
                    cert[field]=malformed;ref=self.save(Path(ref['path']),cert)
                    with self.assertRaises(ValueError):m.verify_checkpoint_closure(ref)

    def test_counts_are_actual_nonnegative_integer_measurements(self):
        for field,value in [('verified_template_count',160.),('verified_template_artifact_count',1280.),
                            ('hashed_unique_files',3204.),('solver_calls',False),('recourse_reassembly_calls',False),
                            ('hashed_bytes_including_repeats',-1),('elapsed_seconds',-1.)]:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as d:
                root=Path(d).resolve();ref,index,current,old,cert=self.fixture(root)
                cert[field]=value;ref=self.save(Path(ref['path']),cert)
                with self.assertRaises(ValueError):m.verify_checkpoint_closure(ref)

    def test_direct_list_and_same_index_after_certificate_are_not_cached_as_verified(self):
        for ordinary_index in (False,True):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d).resolve();ref,index,current,old,cert=self.fixture(root)
                direct=file_reference(current);current.write_text('tampered current source')
                # Remove this current source from the certificate's direct fields
                # using a fresh valid reference; the ordinary list must still fail.
                current_good=file_reference(current)
                for key in ('index_reference','acceptance_reference','prior_review_reference','source_manifest','rows_reference'):
                    cert[key]=current_good
                cert['proof_references']={k:current_good for k in cert['proof_references']}
                ref=self.save(Path(ref['path']),cert)
                combined=self.save(root/'current_request.json',{'prior_certificate':ref,'direct_current_list':[index if ordinary_index else direct]})
                with self.assertRaises((ValueError,FileNotFoundError)):m.verify_checkpoint_closure(combined)

    def test_other_filename_never_gains_historical_index_role(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();ref,index,current,old,cert=self.fixture(root)
            ref=self.save(root/'ordinary_association_document.json',cert)
            with self.assertRaises((ValueError,FileNotFoundError)):m.verify_checkpoint_closure(ref)


if __name__=='__main__':unittest.main()
