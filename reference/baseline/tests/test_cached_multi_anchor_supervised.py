"""Cached admission is stubbed; real relaxation, independent audits and supervision run."""
import importlib.util
import json
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
import unittest
import time
from scipy.sparse import csr_matrix
from evcs_experiments import cached_bound_model, bounded_row_relaxation, bounded_row_relaxation_audit
from evcs_experiments import candidate_dual_adapter,native_gurobi
from evcs_experiments.config import get_paths
from evcs_experiments.evaluation_lock import file_reference
from evcs_experiments.dual_cut_bound import build_dual_cut
from evcs_experiments.multi_anchor_epigraph import build_multi_anchor_epigraph_master
from evcs_experiments.multi_anchor_bound_audit import audit_multi_anchor_model
from evcs_experiments.relaxation_bound import coefficient_sha256
import test_s05_multi_anchor_supervised as base
import test_cached_bound_model as budget

spec=importlib.util.spec_from_file_location('cached_supervised_driver',get_paths().project_root/'scripts/diagnostics/s05_cached_multi_anchor_bound.py')
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)

class CachedSupervisedTests(unittest.TestCase):
    def fixture(self,stack):
        helper=base.SupervisedBoundTests(methodName='runTest')
        stack.enter_context(patch.object(base,'d',d))
        tmp,req,rr=helper.fixture(stack)
        ctx=candidate_dual_adapter.CandidateDualContext.return_value
        old=candidate_dual_adapter.load_old_anchor.return_value
        cuts={};oldcuts={}
        from evcs_experiments.epigraph_master import certify_box_floor
        for i,sid in enumerate(ctx.ids):
            template=dict(A_ub=csr_matrix([[-1.],[-1.]]),A_eq=csr_matrix((0,1)),
                Bx_ub=csr_matrix([[4.],[-1e-14]]),Bx_eq=csr_matrix((0,1)),
                Br_ub=csr_matrix((2,0)),Br_eq=csr_matrix((0,0)),b_ub=[-4.,-4.],b_eq=[],c=[1.],
                lower_bounds=[0.],upper_bounds=[10.],scenario_id=sid,template_sha256='tiny'+sid)
            cuts[i]=build_dual_cut(template,dict(inequality=[0.,-1.],equality=[],lower=[0.],upper=[0.]),
                scenario_constant_usd=0.,anchor_first_stage=[1.],anchor_q_usd=4.+1e-14)
            oldcuts[i]=build_dual_cut(template,dict(inequality=[-1.,0.],equality=[],lower=[0.],upper=[0.]),scenario_constant_usd=0.,anchor_first_stage=[0.],anchor_q_usd=4.)
            ctx.floors[i]=certify_box_floor(template,scenario_constant_usd=0.)
        old=dict(old,cuts=oldcuts)
        new=dict(anchor_first_stage=[1.],cuts=cuts,source_references=old['source_references'])
        model,assembly=build_multi_anchor_epigraph_master(ctx.first_stage_problem,ctx.ids,[old,new],
            probability_set=ctx.probability_set,floors=ctx.floors)
        ma=audit_multi_anchor_model(model,ctx.first_stage_problem,ctx.ids,[old,new],
            probability_set=ctx.probability_set,floors=ctx.floors,deadline=time.monotonic()+5)
        stack.enter_context(patch.object(cached_bound_model,'load_cached_model',return_value=(model,ma,assembly,{'fixture_cached_admission_only':True})))
        def write(path,value):path.write_text(json.dumps(value));return file_reference(path)
        policy,settlement,review=budget.CachedBoundBudgetTests().fixture()
        for key,value in policy.items():
            if key!='parent_job':req[key]=value
        sr=write(tmp/'settlement.json',settlement)
        ar=write(tmp/'review.json',review)
        req['references']['settlement']=sr;req['references']['previous_result_review']=ar
        req['adapter_references']['candidate_review']=ar
        first=assembly['anchor_sets'][0]['construction'];prefix=assembly['retained_prefix_rows']
        req['references']['bound_witnesses']=write(tmp/'witness.json',dict(original_coefficient_sha256=coefficient_sha256(model),
            prefix_rows=prefix,first_stage_columns=first['original_first_stage_columns'],
            new_cut_row_range_half_open=[prefix,model['matrix'].shape[0]],column_to_reserve_upper_row={}))
        req['references']['cached_model_review']=write(tmp/'cache_review.json',{'fixture':True})
        rr=write(tmp/'request.json',req)
        job=Path(req['parent_job']['directory']);jm=json.loads((job/'manifest.json').read_text())
        jm['argv'][-1]=rr['sha256'];write(job/'manifest.json',jm)
        return helper,req,rr

    def test_success_time_limit_original_relaxed_identity_and_no_double_attempt(self):
        with ExitStack() as stack:
            helper,req,rr=self.fixture(stack)
            solve=stack.enter_context(patch.object(native_gurobi,'solve_milp',side_effect=helper.fake_native))
            d.execute(rr['path'],rr['sha256'])
            out=Path(req['output_root']);result=json.loads((out/'result.json').read_text())
            proof=json.loads((out/'relaxation_audit.json').read_text());native=json.loads((out/'native_bound_audit.json').read_text())
            self.assertEqual(proof['removed_entry_count'],2)
            self.assertNotEqual(proof['original_coefficient_sha256'],proof['relaxed_coefficient_sha256'])
            self.assertEqual(native['relaxation_audit_sha256'],proof['audit_sha256'])
            self.assertEqual(native['native_status'],'TIME_LIMIT')
            self.assertEqual(result['opening_cumulative_seconds_decimal'],str(cached_bound_model.OPENING))
            self.assertTrue(result['cumulative_seconds_provisional']);self.assertFalse(result['scientific_acceptance'])
            self.assertEqual(result['status'],'AUDITED_BOUND_PENDING_TERMINAL_PARENT_SETTLEMENT')
            with self.assertRaises(FileExistsError):d.execute(rr['path'],rr['sha256'])
            self.assertEqual(solve.call_count,1)

    def test_invalid_bound_retains_raw_evidence_and_reservation(self):
        with ExitStack() as stack:
            helper,req,rr=self.fixture(stack)
            def invalid(*args,**kwargs):
                answer=helper.fake_native(*args,**kwargs);answer.native_record['solver_bound_without_constant_usd']=999.;return answer
            solve=stack.enter_context(patch.object(native_gurobi,'solve_milp',side_effect=invalid))
            with self.assertRaisesRegex(ValueError,'bound differs'):d.execute(rr['path'],rr['sha256'])
            out=Path(req['output_root'])
            self.assertTrue((out/'native_return.json').exists());self.assertTrue((out/'relaxation_audit.json').exists())
            self.assertTrue((out/'failure_artifact_manifest.json').exists());self.assertFalse((out/'result.json').exists())
            self.assertEqual(json.loads((out/'failure.json').read_text())['phase'],'independent_native_bound_audit')
            with self.assertRaises(FileExistsError):d.execute(rr['path'],rr['sha256'])
            self.assertEqual(solve.call_count,1)
