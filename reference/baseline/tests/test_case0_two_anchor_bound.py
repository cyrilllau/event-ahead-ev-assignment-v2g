"""Actual tiny physical two-anchor cuts; no repeated Q during bound construction."""
import copy,json,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from evcs_experiments.config import get_paths
from evcs_experiments.campaign_batch import file_reference
from evcs_experiments.first_stage import assemble_first_stage
from evcs_experiments.random_streams import canonical_hash
from tests.test_case0_compact_candidate import load,test_native_result

root=get_paths().project_root
fixture=load('two_anchor_physical_fixture',root/'tests/test_case0_compact_candidate.py')
audit=load('two_anchor_bound_test',root/'scripts/audits/audit_case0_two_anchor_bound.py')

class TwoAnchorTests(unittest.TestCase):
    def setUp(self):
        fixture.CompactCandidateTests.setUp(self)
        plan=copy.deepcopy(self.ub['plan']);plan['r']=(np.asarray(plan['r'])*.5).tolist()
        self.assertNotEqual(canonical_hash(plan),self.ub['plan_sha256'])
        proposal=dict(plan=plan,plan_sha256=canonical_hash(plan),model_sha256=self.header['model_sha256'],
            changed_plan=True,first_stage_numeric_passed=True,planning_acceptance=False)
        ctx=SimpleNamespace(first=assemble_first_stage(self.f.case,self.f.tasks),case=self.f.case,tasks=self.f.tasks,
            planning_ids=self.f.scenario_ids,scenario=lambda i:self.f.scenarios[i],probability_set=self.P)
        self.newfolder=self.root/'new_q';self.newfolder.mkdir()
        self.new=fixture.base.candidate.audit_candidate(master=None,proposal=proposal,header=self.header,context=ctx,
            output_dir=self.newfolder,deadline=time.monotonic()+10,binding={})
        self.newprob=json.loads((self.newfolder/'probability.json').read_text())
        self.binding['new_ub_result']=file_reference(self.newfolder/'result.json')
        self.prior=dict(kind='CASE0_COMPACT_DRO_CANDIDATE_RESULT_V1',model_sha256=self.header['model_sha256'],
            prior_plan_sha256=self.ub['plan_sha256'],plan_sha256=self.new['plan_sha256'],planning_acceptance=False,
            lower_bound_usd=self.weighted_result['lower_bound_usd'],retained_original_lower_bound_usd=self.weighted_result['lower_bound_usd'])
    def prepare(self,new=None,prob=None):
        out=self.root/'two';out.mkdir()
        with (patch('evcs_experiments.recourse.solve_fixed_plan',side_effect=AssertionError('No new Q')),
             patch('evcs_experiments.recourse.assemble_recourse',side_effect=AssertionError('No new scene/template'))):
            return audit.prepare_two_anchor(self.A,self.arrays,self.header,self.ub,self.new if new is None else new,
                self.P,self.prob,self.newprob if prob is None else prob,self.weighted_result,self.weighted,
                out=out,deadline=time.monotonic()+10,binding=self.binding,compact=fixture.audit,dual=fixture.base.audit)
    def test_two_full_axes_preserve_old_domain_and_bound(self):
        model,check,start=self.prepare();nf=self.header['variable_slices']['first_stage'][1]
        nrows=self.header['block_map'][0]['row_stop'];S=len(self.P.support_ids)
        self.assertEqual(model['matrix'].shape,(nrows+3*S,nf+1+2*len(self.P.feature_ids)))
        self.assertTrue(check['passed']);self.assertGreater(model['integrality'].sum(),0)
        np.testing.assert_array_equal(start[:nf],np.r_[self.ub['plan']['x'],np.asarray(self.ub['plan']['r']).ravel()])
        solved=test_native_result(model);out=self.root/'result';out.mkdir()
        r=audit.evaluate_bound(model,check,self.ub,self.new,self.prior,solved,out=out,deadline=time.monotonic()+5,
            binding=self.binding,compact=fixture.audit)
        self.assertEqual(r['plan_sha256'],self.ub['plan_sha256']);self.assertEqual(r['upper_bound_usd'],self.ub['upper_bound_usd'])
        self.assertEqual(r['status'],'AUDITED_TWO_ANCHOR_AUXILIARY_BOUND');self.assertLessEqual(r['lower_bound_usd'],self.ub['upper_bound_usd']+1e-6)
        self.assertGreaterEqual(r['lower_bound_usd'],self.prior['lower_bound_usd'])
        self.assertFalse(r['new_plan_accepted'] or r['planning_acceptance'] or r['gap_passed']);self.assertEqual(r['new_Q_solve_count'],0)
    def test_numeric_entry_preserves_full_solver_policy_and_never_calls_q(self):
        identity={'fixture':'original immutable physical inputs'};ip=self.root/'identity.json';ip.write_text(json.dumps(identity))
        fp=self.root/'features.npz';np.savez_compressed(fp,planning=self.P.features)
        mp=self.root/'moments.json';mp.write_text(json.dumps({'moments':{'bounds_by_rho':{'1.0':{'lower':self.P.lower.tolist(),'upper':self.P.upper.tolist()}}}}))
        cp=self.root/'calibration_manifest.json';cp.write_text(json.dumps({'context_sha256':canonical_hash(identity),'arrays':file_reference(fp),'moments':file_reference(mp)}))
        priorp=self.root/'prior.json';priorp.write_text(json.dumps(self.prior))
        refs={k:file_reference(self.producer/v) for k,v in [('model','model.json'),('matrix','matrix.npz'),('model_arrays','model_arrays.npz')]}
        refs.update(self.binding,probability=file_reference(self.saved/'probability.json'),new_probability=file_reference(self.newfolder/'probability.json'),
            input_identity=file_reference(ip),calibration_manifest=file_reference(cp),compact_result=file_reference(priorp))
        def solve(c,integer,lo,hi,A,rl,ru,*,options,log_path,start):
            self.assertEqual(options['solver_feasibility_tolerance'],1e-9);self.assertEqual(options['threads'],1)
            self.assertEqual(options['mip_rel_gap'],.005);self.assertLessEqual(options['time_limit'],3.)
            self.assertGreater(np.sum(integer),0)
            return test_native_result(dict(c=c,integrality=integer,lower_bounds=lo,upper_bounds=hi,matrix=A,row_lower=rl,row_upper=ru,objective_constant_usd=0.))
        out=self.root/'full';out.mkdir()
        with (patch.object(audit,'FEATURE_NAMES',self.P.feature_ids),patch.object(audit,'solve_milp',side_effect=solve),
              patch('evcs_experiments.recourse.solve_fixed_plan',side_effect=AssertionError('No Q')),
              patch('evcs_experiments.recourse.assemble_recourse',side_effect=AssertionError('No template'))):
            result=audit.numeric_run(refs,out,time.monotonic()+10,3,fixture.base.hashcheck,fixture.base.audit,fixture.audit)
        self.assertEqual(result['status'],'AUDITED_TWO_ANCHOR_AUXILIARY_BOUND')
        self.assertEqual(result['plan_sha256'],self.ub['plan_sha256']);self.assertFalse(result['new_plan_accepted'])
    def test_wrong_new_scenario_pairing_rejected(self):
        new=copy.deepcopy(self.new);new['completed_scenario_references'][0]['scenario_id']='wrong'
        with self.assertRaisesRegex(ValueError,'pairing'):self.prepare(new)
    def test_duplicate_anchor_rejected(self):
        new=copy.deepcopy(self.ub);new['kind']='CASE0_PROPOSED_CANDIDATE_UB_AUDIT_V1'
        with self.assertRaisesRegex(ValueError,'Distinct'):self.prepare(new)
    def test_wrong_new_support_and_risk_witness_rejected(self):
        p=copy.deepcopy(self.newprob);p['probabilities'][0]+=.1
        with self.assertRaisesRegex(ValueError,'probability'):self.prepare(prob=p)
    def test_resealed_new_dual_is_not_trusted(self):
        new=copy.deepcopy(self.new);entry=new['completed_scenario_references'][0];p=Path(entry['files']['fixed_solver']['path'])
        saved=json.loads(p.read_text());saved['certificate']['dual_lower_marginals'][0]+=1.;p.write_text(json.dumps(saved))
        entry['files']['fixed_solver']=file_reference(p)
        with self.assertRaisesRegex(ValueError,'dual/KKT/anchor'):self.prepare(new)
    def test_no_native_bound_preserves_old_bound_and_best_plan(self):
        model,check,_=self.prepare();solved=test_native_result(model);solved.mip_dual_bound=None
        self.prior['lower_bound_usd']-=1. # Earlier retained best must survive a weaker last native bound.
        out=self.root/'rejected';out.mkdir()
        r=audit.evaluate_bound(model,check,self.ub,self.new,self.prior,solved,out=out,deadline=time.monotonic()+5,binding=self.binding,compact=fixture.audit)
        self.assertEqual(r['status'],'NO_NEW_BOUND');self.assertEqual(r['lower_bound_usd'],self.prior['retained_original_lower_bound_usd'])
        self.assertEqual(r['upper_bound_usd'],self.ub['upper_bound_usd']);self.assertIsNone(r['new_native_lower_bound_usd'])
        self.assertFalse(r['planning_acceptance'])

if __name__=='__main__':unittest.main()
