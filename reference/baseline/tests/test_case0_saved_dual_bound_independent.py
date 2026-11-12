"""Independent corruption cases for saved original extensive rows."""
import numpy as np
from tests.test_case0_saved_dual_bound import SavedDualBoundTests

class SavedDualIndependentTests(SavedDualBoundTests):
    def test_original_risk_objective_changed_is_rejected(self):
        row=next(b for b in self.header['block_map'] if b['kind']=='risk_epigraph')['row_start']
        a,b=self.header['variable_slices']['recourses'][self.f.scenario_ids[0]]
        mat=self.A.tolil(); cols=np.flatnonzero(self.A[row,a:b].toarray().ravel())
        self.assertGreater(len(cols),0); mat[row,a+int(cols[0])]+=1.; self.A=mat.tocsr()
        with self.assertRaises(ValueError):self.derive()

    def test_original_coupling_sign_corruption_rejected(self):
        nf=self.header['variable_slices']['first_stage'][1]
        row=next(b for b in self.header['block_map'] if b['kind']=='recourse_ub')
        block=self.A[row['row_start']:row['row_stop'],:nf].tocoo()
        self.assertGreater(block.nnz,0)
        mat=self.A.tolil(); r=row['row_start']+int(block.row[0]);c=int(block.col[0]);mat[r,c]=-mat[r,c];self.A=mat.tocsr()
        with self.assertRaises(ValueError):self.derive()
