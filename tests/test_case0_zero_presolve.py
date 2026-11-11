"""Same-model Case0 exact-zero integration and independent proof rejection."""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from evcs_experiments.campaign_batch import file_reference
from evcs_experiments.case0_planning import run_case0_planning
from evcs_experiments.case0_planning_audit import audit_case0_method, audit_case0_zero_reduction
from evcs_experiments.first_stage import assemble_first_stage
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.recourse import assemble_recourse
from evcs_experiments.validation.s03_toys import build_golden_cases


class Case0ZeroPresolveTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.f=build_golden_cases()[1]
        for scene in self.f.scenarios:scene.split='planning'
        self.first=assemble_first_stage(self.f.case,self.f.tasks)
        self.rec=[assemble_recourse(self.f.case,self.f.tasks,s) for s in self.f.scenarios]

    def produce(self,method='EA_DRO',zero=True):
        out=self.root/(method+('-zero' if zero else '-original'));now=time.monotonic()
        _,result=run_case0_planning(self.first,self.rec,self.f.scenario_ids,self.f.probability_set,
            case=self.f.case,tasks=self.f.tasks,scenes=self.f.scenarios,method_id=method,
            output_dir=out,backend='highs',deadline=now+1800,configuration_started_at=now,
            master_seconds=5,empirical_probabilities=np.full(len(self.rec),1/len(self.rec)),
            zero_presolve=zero)
        return out,result

    def audit(self,out):
        with patch('evcs_experiments.extensive.solve_extensive',side_effect=AssertionError('No optimization in audit')),\
             patch('evcs_experiments.zero_presolve.reduce_zero_columns',side_effect=AssertionError('No producer presolver in audit')):
            return audit_case0_method(result_reference=file_reference(out/'result.json'),case=self.f.case,
                tasks=self.f.tasks,scenes=self.f.scenarios,probability_set=self.f.probability_set,
                deadline=time.monotonic()+20)

    def edit(self,out,name,fn):
        path=out/name;record=json.loads(path.read_text());fn(record);path.write_text(json.dumps(record))

    def test_four_real_forms_preserve_original_model_bounds_and_physics(self):
        for method in ('EA_DRO','OR_DRO','SAA','ARO'):
            original,old=self.produce(method,False);out,new=self.produce(method,True)
            self.assertEqual(old['model_sha256'],new['model_sha256'])
            self.assertEqual(old['scenario_ids'],new['scenario_ids'])
            for key in ('lower_bound_usd','upper_bound_usd','risk_value_usd'):
                self.assertAlmostEqual(old[key],new[key],places=7)
            self.assertNotIn('zero_presolve',old)
            self.assertFalse((original/'zero_reduction_audit.json').exists())
            audit=self.audit(out);self.assertTrue(audit['passed'])
            self.assertTrue(audit['zero_reduction_audit']['original_zero_lift_checked'])
            self.assertGreater(audit['zero_reduction_audit']['zero_columns'],0)
            self.assertEqual(audit['zero_reduction_audit']['proof_sha256'],new['zero_reduction_proof_sha256'])
            self.assertEqual(json.loads((out/'master.json').read_text())['raw_solver']['options']['threads'],1)

    def test_resealed_nonredundant_row_deletion_rejected(self):
        out,_=self.produce()
        def change(record):
            p=record['raw_solver']['zero_reduction'];p['retained_rows']=p['retained_rows'][1:]
            p.pop('sha256');p['sha256']=canonical_hash(p)
        self.edit(out,'master.json',change)
        with self.assertRaisesRegex(ValueError,'nonredundant'):self.audit(out)

    def test_native_proof_cannot_be_removed_or_added_without_result_binding(self):
        out,_=self.produce()
        self.edit(out,'master.json',lambda r:r['raw_solver'].pop('zero_reduction'))
        with self.assertRaisesRegex(ValueError,'lacks native proof'):self.audit(out)
        out,_=self.produce('SAA')
        self.edit(out,'result.json',lambda r:r.pop('zero_presolve'))
        with self.assertRaisesRegex(ValueError,'Unrequested'):self.audit(out)

    def test_changed_original_coefficients_or_saved_proof_binding_rejected(self):
        out,_=self.produce()
        self.edit(out,'result.json',lambda r:r.__setitem__('zero_reduction_proof_sha256','0'*64))
        with self.assertRaisesRegex(ValueError,'proof binding'):self.audit(out)
        out,_=self.produce('SAA')
        self.edit(out,'zero_reduction_audit.json',lambda r:r.__setitem__('eliminated_rows',r['eliminated_rows']+1))
        with self.assertRaisesRegex(ValueError,'Saved exact reduction'):self.audit(out)

    def test_explicit_boolean_required_and_master_budget_unchanged(self):
        with self.assertRaisesRegex(ValueError,'explicit boolean'):self.produce(zero=1)
        with patch('evcs_experiments.case0_planning.solve_extensive',side_effect=RuntimeError('boundary')) as solver:
            with self.assertRaisesRegex(RuntimeError,'boundary'):self.produce('SAA')
            self.assertIs(solver.call_args.kwargs['zero_presolve'],True)
            self.assertEqual(solver.call_args.kwargs['time_limit_seconds'],5)
            self.assertEqual(solver.call_args.kwargs['mip_relative_gap'],.005)
            self.assertEqual(solver.call_args.kwargs['optimizer_threads'],1)

    def test_native_reduced_dimensions_and_row_map_are_bound(self):
        import hashlib
        from scipy.sparse import load_npz
        out,_=self.produce();raw=json.loads((out/'master.json').read_text())['raw_solver']
        proof=raw['zero_reduction'];A=load_npz(out/'matrix.npz')
        with np.load(out/'model_arrays.npz') as z:arrays={k:z[k] for k in z.files}
        with np.load(out/'plan_and_master_vector.npz') as z:vector=z['vector']
        rows=np.asarray(proof['retained_rows']);lo=arrays['row_lower'][rows];hi=arrays['row_upper'][rows]
        equal=np.isfinite(lo)&np.isfinite(hi)&(lo==hi)
        mapping={'eq':np.flatnonzero(equal),'le':np.flatnonzero(np.isfinite(hi)&~equal),'ge':np.flatnonzero(np.isfinite(lo)&~equal)}
        rowmap={k:dict(count=len(v),source_rows_sha256=hashlib.sha256(v.astype('<i8').tobytes()).hexdigest()) for k,v in mapping.items()}
        rowmap.update(original_rows=len(rows),translated_rows=sum(len(v) for v in mapping.values()),
            free_rows=int(np.count_nonzero(np.isneginf(lo)&np.isposinf(hi))),
            ranged_rows=int(np.count_nonzero(np.isfinite(lo)&np.isfinite(hi)&(lo<hi))))
        native=dict(original_matrix_shape=proof['reduced_shape'],original_matrix_nnz=proof['reduced_nnz'],
            row_map_summary=rowmap,model_columns=proof['reduced_shape'][1],model_rows=rowmap['translated_rows'])
        self.assertTrue(audit_case0_zero_reduction(A,arrays,proof,vector,native)['passed'])
        for key,value in [('original_matrix_nnz',A.nnz+1),('model_columns',A.shape[1]+1),
                          ('row_map_summary',dict(rowmap,translated_rows=0))]:
            changed=copy.deepcopy(native);changed[key]=value
            with self.assertRaisesRegex(ValueError,'Native reduced matrix'):
                audit_case0_zero_reduction(A,arrays,proof,vector,changed)


if __name__=='__main__':unittest.main()
