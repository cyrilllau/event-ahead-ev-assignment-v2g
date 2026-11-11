"""Independent original-array and exact-column association checks; no server work."""
import copy,importlib.util,json,tempfile,time,unittest,runpy,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy.sparse import save_npz
from evcs_experiments.config import get_paths
from evcs_experiments.campaign_batch import file_reference

root=get_paths().project_root
sp=importlib.util.spec_from_file_location('closure_independent_author_fixture',root/'tests/test_case0_extensive_closure.py')
fixture=importlib.util.module_from_spec(sp);sp.loader.exec_module(fixture)
closure=fixture.closure

class IndependentClosureTests(unittest.TestCase):
    def setUp(self):
        fixture.ExtensiveClosureTests.setUp(self)
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        hp=self.root/'model.json';hp.write_text(json.dumps(self.header));save_npz(self.root/'matrix.npz',self.A)
        np.savez_compressed(self.root/'model_arrays.npz',**self.arrays)
        self.ub.update(kind='CASE0_SAVED_CANDIDATE_UB_AUDIT_V1',first_stage_physical_passed=True,
            full_support_fixed_Q_and_KKT_passed=True,probability_primal_dual_passed=True,source_binding={'model':file_reference(hp)})
        entries=[]
        for i,sid in enumerate(self.header['scenario_ids']):
            directory=self.root/f'scene_{i}';directory.mkdir();np.savez_compressed(directory/'primal.npz',fixed=self.primals[sid])
            (directory/'fixed.json').write_text(json.dumps({'fixture':'real physical primal from author toy Q'}))
            (directory/'checks.json').write_text(json.dumps({'fixture':'assembly audit consumes strong original references, not these flags'}))
            entries.append(dict(index=i,scenario_id=sid,template_sha256=self.header['recourse_template_sha256s'][i],files={
                'primal':file_reference(directory/'primal.npz'),'fixed_solver':file_reference(directory/'fixed.json'),'checks':file_reference(directory/'checks.json')}))
        self.ub['completed_scenario_references']=entries
        (self.root/'result.json').write_text(json.dumps(self.ub));(self.root/'probability.json').write_text(json.dumps(self.P))
        self.refs={k:file_reference(self.root/n) for k,n in [('model','model.json'),('matrix','matrix.npz'),('model_arrays','model_arrays.npz'),('ub_result','result.json'),('probability','probability.json')]}
        self.refs['model_hash_checker']=file_reference(root/'scripts/audits/audit_case0_saved_lp_bound.py')
    def run_closure(self):
        with (patch('evcs_experiments.recourse.solve_fixed_plan',side_effect=AssertionError('No Q')),
              patch('evcs_experiments.extensive.solve_extensive',side_effect=AssertionError('No MIP')),
              patch('evcs_experiments.probability.solve_worst_distribution',side_effect=AssertionError('No P solve'))):
            return closure.numeric_closure(self.refs,self.root/'assembled',time.monotonic()+5)
    def test_actual_saved_arrays_and_original_primal_no_solver(self):
        result=self.run_closure();self.assertTrue(result['passed'])
        with np.load(result['assembled_primal_reference']['path'],allow_pickle=False) as f:
            self.assertEqual(f.files,['primal']);self.assertTrue(np.isfinite(f['primal']).all())
        self.assertFalse(result['original_native_primal_recovered'] or result['planning_acceptance'])
        self.assertEqual(result['new_Q_calls']+result['new_master_calls'],0)
    def test_resealed_modified_original_model_arrays_rejected(self):
        modified=dict(self.arrays);modified['c']=modified['c'].copy();modified['c'][0]+=.25
        p=self.root/'model_arrays.npz';np.savez_compressed(p,**modified);self.refs['model_arrays']=file_reference(p)
        with self.assertRaisesRegex(ValueError,'semantic/array hash'):self.run_closure()
    def test_resealed_wrong_physical_primal_is_not_rescued_by_ub_flags(self):
        u=copy.deepcopy(self.ub);entry=u['completed_scenario_references'][0];p=Path(entry['files']['primal']['path'])
        x=self.primals[entry['scenario_id']].copy();x[:]=-100.;np.savez_compressed(p,fixed=x);entry['files']['primal']=file_reference(p)
        result=self.root/'result.json';result.write_text(json.dumps(u));self.refs['ub_result']=file_reference(result)
        with self.assertRaisesRegex(ValueError,'original extensive'):self.run_closure()
    def test_overlapping_original_columns_rejected(self):
        header=copy.deepcopy(self.header);header['variable_slices']['r']=header['variable_slices']['x']
        with self.assertRaisesRegex(ValueError,'slice'):closure.assemble_witness(header,self.ub,self.P,self.primals)
    def test_cli_rejects_nonfinite_budget_before_any_numerical_io(self):
        p=self.root/'request.json';p.write_text(json.dumps(dict(kind='CASE0_ORIGINAL_EXTENSIVE_CLOSURE_REQUEST_V1',
            references=self.refs,output_directory=str(self.root/'unused'),budget_seconds=float('nan'),review_reference={})))
        ref=file_reference(p)
        with patch.object(sys,'argv',[str(root/'scripts/audits/audit_case0_extensive_closure.py'),'--request',str(p),'--expected-sha256',ref['sha256']]):
            with self.assertRaisesRegex(ValueError,'bounded closure time'):
                runpy.run_path(str(root/'scripts/audits/audit_case0_extensive_closure.py'),run_name='__main__')
        self.assertFalse((self.root/'unused').exists())
    def test_cli_resolves_output_before_enforcing_operations_scope(self):
        campaign=self.root/'ieee33-v54-q1-case0-expanded';operations=campaign/'operations';operations.mkdir(parents=True)
        base=dict(kind='CASE0_ORIGINAL_EXTENSIVE_CLOSURE_REQUEST_V1',references=self.refs,
            output_directory=str(operations/'..'/'escaped'),budget_seconds=20)
        source=root/'scripts/audits/audit_case0_extensive_closure.py';review=self.root/'review.json'
        review.write_text(json.dumps(dict(kind='CASE0_ORIGINAL_EXTENSIVE_CLOSURE_REVIEW_V1',passed=True,
            request_without_review=base,auditor_sha256=file_reference(source)['sha256'])))
        req=dict(base,review_reference=file_reference(review));p=self.root/'request.json';p.write_text(json.dumps(req));ref=file_reference(p)
        with (patch.object(sys,'argv',[str(source),'--request',str(p),'--expected-sha256',ref['sha256']]),
              patch('evcs_experiments.config.get_paths',return_value=SimpleNamespace(campaign_root=campaign)),
              patch('evcs_experiments.case0_isolation.verify_current_cgroup',return_value=None)):
            with self.assertRaisesRegex(ValueError,'fresh expanded operations'):
                runpy.run_path(str(source),run_name='__main__')
        self.assertFalse((campaign/'escaped').exists())
    def test_resealed_wrong_template_association_rejected(self):
        u=copy.deepcopy(self.ub);u['completed_scenario_references'][0]['template_sha256']='0'*64
        p=self.root/'result.json';p.write_text(json.dumps(u));self.refs['ub_result']=file_reference(p)
        with self.assertRaisesRegex(ValueError,'association'):self.run_closure()

if __name__=='__main__':unittest.main()
