"""Narrow actual-schema budget, provenance and candidate boundary tests."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
import numpy as np
from recourse_fixtures import make_fixture
from unittest.mock import patch
from evcs_experiments.config import get_paths

_path=get_paths().project_root/'scripts/diagnostics/run_s05_epigraph_master.py'
_spec=importlib.util.spec_from_file_location('epigraph_probe_driver',_path)
d=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(d)


class EpigraphBudgetTests(unittest.TestCase):
    def setUp(self):
        self.req=dict(kind=d.KIND,plan_id='p',wall_seconds=590,solver_seconds=120,parent_job=dict(timeout_seconds=600))
        self.settled=dict(kind='S05_DUAL_CUT_BOUND_PARENT_ACCOUNTING_V1',status='SETTLED_PENDING_BOUND_REVIEW',plan_id='p',
            accounting_PASS=True,nested_seconds_added=False,cumulative_seconds_provisional=False,
            opening_budget=dict(opening_cumulative_seconds=3944.848443),terminal_parent_seconds=3.570023,
            cumulative_accounted_seconds=3948.418466,remaining_seconds=3251.581534,cumulative_cap_seconds=7200)

    def test_real_settlement_schema_and_decimal_parent_once(self):
        self.assertEqual(d.budget_guard(self.req,self.settled),3948.418466)

    def test_cost_reset_nested_provisional_and_wrong_identity_rejected(self):
        for key,value in [('plan_id','different'),('nested_seconds_added',True),('cumulative_seconds_provisional',True),
                          ('accounting_PASS',False),('cumulative_accounted_seconds',0.),('remaining_seconds',7200),
                          ('terminal_parent_seconds',-1.),('cumulative_cap_seconds',8000)]:
            bad=copy.deepcopy(self.settled);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):d.budget_guard(self.req,bad)

    def test_retains_1790_plus_shutdown_under_parent_cap(self):
        bad=copy.deepcopy(self.settled);bad['opening_budget']['opening_cumulative_seconds']=5000.
        bad.update(cumulative_accounted_seconds=5003.570023,remaining_seconds=2196.429977)
        with self.assertRaisesRegex(ValueError,'reserve'):d.budget_guard(self.req,bad)

    def test_no_expand_solver_or_parent_budget(self):
        for key,value in [('solver_seconds',550),('wall_seconds',7200),('kind','other')]:
            bad=copy.deepcopy(self.req);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):d.budget_guard(bad,self.settled)
        bad=copy.deepcopy(self.req);bad['parent_job']['timeout_seconds']=7200
        with self.assertRaises(ValueError):d.budget_guard(bad,self.settled)


class EpigraphReferenceTests(unittest.TestCase):
    def test_same_size_input_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'source.py';p.write_text('abc\n');r=d.ref(p);self.assertEqual(d.verify(r),p.resolve())
            p.write_text('abd\n')
            with self.assertRaises(ValueError):d.verify(r)

    def test_wrong_request_hash_before_parent_or_solver(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'request.json';p.write_text('{}')
            with patch.object(d,'parent') as parent,patch.object(d,'solve_milp') as solve:
                with self.assertRaisesRegex(ValueError,'Changed request'):d.execute(p,'0'*64)
                parent.assert_not_called();solve.assert_not_called()

    def test_unbound_cli_cannot_reach_solver(self):
        req=dict(parent_job=dict(job_id='none',directory=str(get_paths().campaign_root/'jobs/none')))
        with patch.dict(d.os.environ,{'EVCS_JOB_ID':'wrong','EVCS_JOB_DIRECTORY':'/wrong'}),patch.object(d,'solve_milp') as solve:
            with self.assertRaises(ValueError):d.parent(req,{})
            solve.assert_not_called()

    def test_source_escape_fails_before_data_load_or_solver(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();release=root/'release';release.mkdir();campaign=root/'campaign';campaign.mkdir()
            outside=root/'outside.py';outside.write_text('pass\n');r=d.ref(outside)
            mf=release/'RELEASE_MANIFEST.json';mf.write_text(json.dumps(dict(files=[dict(path='../outside.py',sha256=r['sha256'],size_bytes=r['size_bytes'])])))
            q=dict(references=dict(source_manifest=d.ref(mf),script=r,settlement=r),output_root=str(campaign/'epigraph_probes/j'))
            request=root/'request.json';request.write_text(json.dumps(q));rr=d.ref(request)
            paths=types.SimpleNamespace(project_root=release,campaign_root=campaign)
            with (patch.object(d,'get_paths',return_value=paths),patch.object(d,'parent',return_value=dict(job_id='j')),
                 patch.object(d,'solve_milp') as solve,patch.object(d,'load_physical_case') as physics):
                with self.assertRaisesRegex(ValueError,'Source escape'):d.execute(request,rr['sha256'])
                solve.assert_not_called();physics.assert_not_called()
                failure=d.read(campaign/'epigraph_probes/j/failure.json')
                self.assertFalse(failure['scientific_acceptance']);self.assertTrue(failure['requires_terminal_parent_settlement'])

    def test_outputs_append_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'result.json';d.write(p,dict(lockable=False))
            with self.assertRaises(FileExistsError):d.write(p,dict(lockable=True))
            self.assertFalse(d.read(p)['lockable'])


class EpigraphCandidateTests(unittest.TestCase):
    def test_original_integer_plan_and_auxiliary_tail_preserved(self):
        f=make_fixture();first=d.assemble_first_stage(f.case,f.tasks);v=np.r_[f.plan.x,f.plan.r.ravel(),123.,0.,0.]
        before=v.copy();plan,check,physical=d.candidate_plan(first,f.case,f.tasks,v)
        self.assertTrue(check.feasible);self.assertTrue(physical['passed'])
        np.testing.assert_array_equal(plan.x,f.plan.x);np.testing.assert_array_equal(plan.r,f.plan.r)
        np.testing.assert_array_equal(v,before);self.assertEqual(plan.task_sha256,f.tasks.task_sha256)

    def test_fractional_infeasible_short_and_nonfinite_candidates_fail(self):
        f=make_fixture();first=d.assemble_first_stage(f.case,f.tasks)
        for v in ([.5,10.],[1.,100000.],[],[1.,float('nan')]):
            with self.subTest(v=v),self.assertRaises(ValueError):d.candidate_plan(first,f.case,f.tasks,v)

    def test_zero_plan_is_not_forced_into_nonzero_and_not_clipped(self):
        f=make_fixture();first=d.assemble_first_stage(f.case,f.tasks)
        plan,check,physical=d.candidate_plan(first,f.case,f.tasks,np.zeros(len(first.c)))
        self.assertTrue(check.feasible);self.assertTrue(physical['passed'])
        self.assertEqual(np.count_nonzero(plan.x),0);self.assertEqual(np.count_nonzero(plan.r),0)


if __name__=='__main__':unittest.main()
