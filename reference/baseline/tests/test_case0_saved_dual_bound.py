"""Saved physical Q duals -> original first-stage LP, with no new Q solves."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from scipy.sparse import load_npz
from evcs_experiments.config import get_paths
from evcs_experiments.first_stage import assemble_first_stage
from evcs_experiments.recourse import assemble_recourse
from evcs_experiments.case0_planning import run_case0_planning
from evcs_experiments.validation.s03_toys import build_golden_cases
from evcs_experiments.campaign_batch import file_reference
from evcs_experiments.probability import solve_worst_distribution
from evcs_experiments.random_streams import canonical_hash


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

root=get_paths().project_root
audit=module('case0_saved_dual_bound_test',root/'scripts/audits/audit_case0_saved_dual_bound.py')
candidate=module('case0_saved_dual_candidate_test',root/'scripts/audits/audit_case0_saved_candidate.py')
hashcheck=module('case0_saved_dual_hash_test',root/'scripts/audits/audit_case0_saved_lp_bound.py')

class SavedDualBoundTests(unittest.TestCase):
    def setUp(self):
        from types import SimpleNamespace
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        self.f=build_golden_cases()[0]
        for scene in self.f.scenarios:scene.split='planning'
        first=assemble_first_stage(self.f.case,self.f.tasks);rec=[assemble_recourse(self.f.case,self.f.tasks,s) for s in self.f.scenarios]
        self.producer=self.root/'producer';now=time.monotonic()
        _,r=run_case0_planning(first,rec,self.f.scenario_ids,self.f.probability_set,case=self.f.case,tasks=self.f.tasks,scenes=self.f.scenarios,
            method_id='EA_DRO',output_dir=self.producer,backend='highs',deadline=now+1800,configuration_started_at=now,master_seconds=5)
        master=json.loads((self.producer/'master.json').read_text());master.update(status='TIME_LIMIT',lower_bound_usd=None,mip_gap=None)
        self.header=json.loads((self.producer/'model.json').read_text());self.A=load_npz(self.producer/'matrix.npz')
        with np.load(self.producer/'model_arrays.npz',allow_pickle=False) as data:self.arrays={k:data[k].copy() for k in data.files}
        self.saved=self.root/'saved';self.saved.mkdir()
        ctx=SimpleNamespace(first=first,case=self.f.case,tasks=self.f.tasks,planning_ids=self.f.scenario_ids,
            scenario=lambda i:self.f.scenarios[i],probability_set=self.f.probability_set)
        self.ub=candidate.audit_candidate(master=master,header=self.header,context=ctx,output_dir=self.saved,deadline=time.monotonic()+10,binding={})
        self.prob=json.loads((self.saved/'probability.json').read_text());self.P=self.f.probability_set
    def derive(self,ub=None,header=None,prob=None):
        out=self.root/'derived';out.mkdir()
        with patch('evcs_experiments.recourse.solve_fixed_plan',side_effect=AssertionError('No repeated Q')),\
             patch('evcs_experiments.recourse.assemble_recourse',side_effect=AssertionError('No template rebuild')):
            return audit.derive_saved_cuts(self.A,self.arrays,self.header if header is None else header,self.ub if ub is None else ub,
                self.P,self.prob if prob is None else prob,output=out,deadline=time.monotonic()+5,binding={'ub':file_reference(self.saved/'result.json')})
    def test_actual_saved_duals_first_stage_lp_has_valid_bound(self):
        model,check=self.derive();r,w=audit.solve_relaxation_bound(model,time_limit_seconds=5,threads=1)
        self.assertTrue(r['certificate']['passed']);self.assertLessEqual(r['lower_bound_usd'],self.ub['upper_bound_usd']+1e-6)
        self.assertTrue(check['original_first_stage_A_bounds_preserved']);self.assertEqual(check['support_ids'],list(self.f.scenario_ids))
        self.assertEqual(model['matrix'].shape[1],len(self.ub['plan']['x'])+np.asarray(self.ub['plan']['r']).size)
        self.assertTrue(r['integrality_relaxed_only']);self.assertGreater(r['original_integer_columns'],0)
        self.assertEqual(r['master_milp_calls'],0)
    def test_entire_numeric_entry_uses_original_calibration_and_saved_arrays(self):
        identity={'fixture':'immutable original scientific input'}
        ip=self.root/'identity.json';ip.write_text(json.dumps(identity))
        fp=self.root/'features.npz';np.savez_compressed(fp,planning=self.P.features)
        mp=self.root/'moments.json';mp.write_text(json.dumps({'moments':{'bounds_by_rho':{'1.0':{'lower':self.P.lower.tolist(),'upper':self.P.upper.tolist()}}}}))
        cp=self.root/'calibration_manifest.json';cp.write_text(json.dumps({'context_sha256':canonical_hash(identity),'arrays':file_reference(fp),'moments':file_reference(mp)}))
        refs={k:file_reference(self.producer/v) for k,v in [('model','model.json'),('matrix','matrix.npz'),('model_arrays','model_arrays.npz')]}
        refs.update(ub_result=file_reference(self.saved/'result.json'),probability=file_reference(self.saved/'probability.json'),
            input_identity=file_reference(ip),calibration_manifest=file_reference(cp))
        out=self.root/'full';out.mkdir()
        with patch.object(audit,'FEATURE_NAMES',self.P.feature_ids),patch('evcs_experiments.recourse.solve_fixed_plan',side_effect=AssertionError('No new Q')):
            r=audit.numeric_run(refs,out,time.monotonic()+10,5,hashcheck)
        self.assertEqual(r['status'],'AUDITED_NUMERICAL_AUXILIARY_BOUND');self.assertLessEqual(r['lower_bound_usd'],r['upper_bound_usd']+1e-6)
        self.assertFalse(r['gap_passed'] or r['planning_acceptance'] or r['formal_OOS']);self.assertEqual(r['master_milp_calls'],0)
        self.assertEqual(r['actual_solve_type'],'continuous_first_stage_LP_relaxation')
    def test_resealed_wrong_saved_dual_rejected(self):
        u=copy.deepcopy(self.ub);entry=u['completed_scenario_references'][0];p=Path(entry['files']['fixed_solver']['path']);s=json.loads(p.read_text())
        s['certificate']['dual_lower_marginals'][0]+=1.;p.write_text(json.dumps(s));entry['files']['fixed_solver']=file_reference(p)
        with self.assertRaisesRegex(ValueError,'dual/KKT/anchor'):self.derive(u)
    def test_wrong_pairing_and_probability_cannot_change_bound(self):
        u=copy.deepcopy(self.ub);u['completed_scenario_references'][0]['scenario_id']='wrong'
        with self.assertRaisesRegex(ValueError,'pairing'):self.derive(u)
    def test_probability_not_normalized_or_clipped(self):
        p=copy.deepcopy(self.prob);p['probabilities'][0]+=.1
        with self.assertRaisesRegex(ValueError,'probability'):self.derive(prob=p)
    def test_foreign_recourse_coupling_or_sign_detected(self):
        h=copy.deepcopy(self.header);h['block_map'][1]['source_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'source'):self.derive(header=h)
    def test_wrong_anchor_identity_rejected_before_dual(self):
        u=copy.deepcopy(self.ub);u['plan']['r'][0][0]+=.1
        with self.assertRaisesRegex(ValueError,'anchor identity'):self.derive(u)
    def test_independent_saved_lp_witness_detects_tamper(self):
        model,_=self.derive();r,w=audit.solve_relaxation_bound(model,time_limit_seconds=5,threads=1)
        w['variable_lower_marginals'][0]+=1.
        check=audit.check_relaxation_certificate(model,w,solver_status=0,raw_objective_without_constant_usd=r['raw_solver']['objective_without_constant_usd'])
        self.assertFalse(check['passed']);self.assertIsNone(check['certified_lower_bound_usd'])

if __name__=='__main__':unittest.main()
