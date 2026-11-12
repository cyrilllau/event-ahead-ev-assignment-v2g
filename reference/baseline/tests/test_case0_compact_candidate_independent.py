"""Independent native provenance corruption checks; tiny fixtures only."""
import time
from tests.test_case0_compact_candidate import CompactCandidateTests,audit,test_native_result

class CompactIndependentTests(CompactCandidateTests):
    def test_changed_row_map_cannot_certify_native_bound(self):
        model,check,first,start=self.prepare();solved=test_native_result(model)
        solved.native_record['row_map_summary']['eq']['source_rows_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'mapping'):
            audit.audit_case0_native_bound(model,check,solved.native_record,native_status=solved.status,
                native_bound=solved.mip_dual_bound,incumbent=solved.x,
                expected_coefficient_sha256=audit.coefficient_sha256(model),fixed_upper_bound=self.ub['upper_bound_usd'],deadline=time.monotonic()+5)
    def test_false_original_optimal_status_cannot_hide_missing_incumbent(self):
        model,check,first,start=self.prepare();solved=test_native_result(model)
        with self.assertRaisesRegex(ValueError,'missing its incumbent'):
            audit.audit_case0_native_bound(model,check,solved.native_record,native_status=solved.status,
                native_bound=solved.mip_dual_bound,incumbent=None,
                expected_coefficient_sha256=audit.coefficient_sha256(model),fixed_upper_bound=self.ub['upper_bound_usd'],deadline=time.monotonic()+5)
