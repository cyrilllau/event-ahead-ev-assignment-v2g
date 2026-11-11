"""Source/budget/parent contracts for the saved-dual bound wrapper; no solver."""
import copy
from datetime import datetime,timezone,timedelta
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from evcs_experiments.config import get_paths
from evcs_experiments.first_stage import assemble_first_stage

path=get_paths().project_root/'scripts/diagnostics/run_s05_dual_cut_bound.py'
spec=importlib.util.spec_from_file_location('dual_cut_driver_tested',path)
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)


class DualBoundBudgetTests(unittest.TestCase):
    def setUp(self):
        self.q=dict(kind=d.KIND,plan_id='mechanism.michael_2018.EA_DRO.baseline',wall_seconds=590,solver_seconds=120,parent_job=dict(timeout_seconds=600))
        self.s=dict(kind='S05_SAVED_RMP_BOUND_PROBE_PARENT_ACCOUNTING_V1',status='SETTLED_NO_BOUND_PENDING_NEXT_DECISION',plan_id=self.q['plan_id'],accounting_PASS=True,cumulative_seconds_provisional=False,nested_seconds_added=False,opening_budget=dict(opening_cumulative_seconds=3390.457493),terminal_parent_seconds=554.39095,cumulative_accounted_seconds=3944.848443,remaining_seconds=3255.151557,cumulative_cap_seconds=7200)

    def test_actual_settled_predecessor_keeps_all_parent_cost_and_reserve(self):
        self.assertEqual(d.budget_guard(self.q,self.s),3944.848443)
        self.assertGreater(self.s['remaining_seconds']-600,1790+10)

    def test_predecessor_kind_status_and_accounting_cannot_be_substituted(self):
        for key,value in [('kind','generic-settlement'),('status','SETTLED_UB_ONLY_PENDING_INDEPENDENT_REVIEW'),('accounting_PASS',False),('cumulative_seconds_provisional',True),('nested_seconds_added',True),('plan_id','other-plan')]:
            with self.subTest(key=key):
                other=copy.deepcopy(self.s);other[key]=value
                with self.assertRaises(ValueError):d.budget_guard(self.q,other)

    def test_reset_nonfinite_and_double_charge_rejected(self):
        for key,value in [('cumulative_accounted_seconds',0),('remaining_seconds',7200),('cumulative_cap_seconds',8000),('terminal_parent_seconds',float('nan'))]:
            with self.subTest(key=key):
                other=copy.deepcopy(self.s);other[key]=value
                with self.assertRaises(ValueError):d.budget_guard(self.q,other)

    def test_parent_budget_and_final_evidence_reserve_cannot_expand(self):
        for key,value in [('wall_seconds',7200),('solver_seconds',550),('parent_job',dict(timeout_seconds=601))]:
            with self.subTest(key=key):
                other=copy.deepcopy(self.q);other[key]=value
                with self.assertRaises(ValueError):d.budget_guard(other,self.s)
        other=copy.deepcopy(self.s);other.update(opening_budget=dict(opening_cumulative_seconds=5000),terminal_parent_seconds=500,cumulative_accounted_seconds=5500,remaining_seconds=1700)
        with self.assertRaisesRegex(ValueError,'reserve'):d.budget_guard(self.q,other)


class FirstDigestAndReferenceTests(unittest.TestCase):
    def test_production_first_stage_schema_digest_and_changed_price(self):
        case=SimpleNamespace(periods=(1,2),station_ids=('CS1',),station_bus_ids=(2,),station_plugs=np.array([1]),station_interface_kw=np.array([10.]),dt_h=.5)
        tasks=SimpleNamespace(task_ids=('task',),vehicle_ids=('ev',),station_ids=('CS1',),chi_occ=np.ones((1,2)),chi_res=np.ones((1,2)),block_mask=np.ones((1,2)),p_dis_kw=np.array([4.]),p_block_kw=np.array([2.]),eta_dis=np.array([.8]),e_forecast_dc_kwh=np.array([3.]),c_act_usd=np.array([5.]),c_cap_usd_per_kwh=np.array([[.1,.2]]),task_sha256='tasks',config_hash='config')
        f=assemble_first_stage(case,tasks)
        meta={k:getattr(f,k) for k in ('task_sha256','config_hash','task_ids','periods')}
        meta.update(variable_map=[dict(x) for x in f.variable_map],constraint_map=[dict(x) for x in f.constraint_map],units=dict(f.units))
        arrays={k:np.array(getattr(f,k),copy=True) for k in ('row_lower','row_upper','lower_bounds','upper_bounds','integrality','c')}
        self.assertNotIn('objective_constant_usd',arrays)
        self.assertEqual(d.first_digest(meta,f.A,arrays),f.matrix_sha256)
        arrays['c'][0]+=1
        self.assertNotEqual(d.first_digest(meta,f.A,arrays),f.matrix_sha256)

    def test_semantic_metadata_is_bound_with_coefficients(self):
        from scipy.sparse import csr_matrix
        meta=dict(task_sha256='t',config_hash='c',task_ids=['t'],periods=[1],variable_map=[dict(unit='kW_AC')],constraint_map=[],units={'objective':'USD'})
        a=csr_matrix([[1.]])
        v=dict(row_lower=np.array([-np.inf]),row_upper=np.array([2.]),lower_bounds=np.array([0.]),upper_bounds=np.array([1.]),integrality=np.array([1]),c=np.array([3.]))
        original=d.first_digest(meta,a,v);meta['variable_map'][0]['unit']='kW_DC'
        self.assertNotEqual(d.first_digest(meta,a,v),original)

    def test_reference_tamper_and_append_only_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'evidence.json';r=d.write(p,{'value':1})
            self.assertEqual(d.verify(r),p.resolve())
            with self.assertRaises(FileExistsError):d.write(p,{'value':2})
            p.write_text('{"value":3}')
            with self.assertRaises(ValueError):d.verify(r)


class DualBoundParentTests(unittest.TestCase):
    def fixture(self,tmp):
        root=Path(tmp).resolve();project=root/'release';campaign=root/'campaign';folder=campaign/'jobs'/'bound-job';folder.mkdir(parents=True)
        controller=project/'src/evcs_experiments/runtime/jobctl.py';controller.parent.mkdir(parents=True);controller.write_text('# fixture controller\n')
        rr=dict(path=str(root/'request.json'),sha256='fixed-request',size_bytes=1)
        p=dict(directory=str(folder),job_id='bound-job',threads=1,timeout_seconds=600,grace_seconds=5,controller_reference=d.ref(controller))
        manifest=dict(argv=[d.sys.executable,'-B',str(Path(d.__file__).resolve()),'--request',rr['path'],'--expected-request-sha256',rr['sha256']],cwd=str(project),outroot=str(folder.parent),job_id=p['job_id'],spec_id='S05',threads=1,timeout_seconds=600,grace_seconds=5,controller_path=str(controller),created_at_utc=datetime.now(timezone.utc).isoformat())
        state=dict(state='RUNNING',command_pid=os.getpid(),supervisor_sid=os.getsid(0))
        (folder/'manifest.json').write_text(json.dumps(manifest));(folder/'status.json').write_text(json.dumps(state))
        env={k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}
        env.update(EVCS_JOB_ID='bound-job',EVCS_JOB_DIRECTORY=str(folder))
        return dict(parent_job=p),rr,SimpleNamespace(project_root=project,campaign_root=campaign),manifest,state,env,folder

    def test_exact_live_parent_reaches_clock_without_a_solver(self):
        with tempfile.TemporaryDirectory() as tmp:
            q,rr,paths,m,s,env,folder=self.fixture(tmp)
            with patch.object(d,'get_paths',return_value=paths),patch.object(d,'observation',return_value={'supervisor_alive':True}),patch.dict(os.environ,env,clear=True):
                result=d.parent(q,rr)
            self.assertEqual(result['job_id'],'bound-job');self.assertEqual(result['manifest_reference'],d.ref(folder/'manifest.json'))

    def test_changed_argv_resources_or_expired_parent_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            q,rr,paths,m,s,env,folder=self.fixture(tmp)
            changes=[('argv',m['argv']+['--extra']),('threads',2),('timeout_seconds',601),('created_at_utc',(datetime.now(timezone.utc)-timedelta(seconds=601)).isoformat())]
            for key,value in changes:
                with self.subTest(key=key):
                    bad=copy.deepcopy(m);bad[key]=value;(folder/'manifest.json').write_text(json.dumps(bad))
                    with patch.object(d,'get_paths',return_value=paths),patch.object(d,'observation',return_value={'supervisor_alive':True}),patch.dict(os.environ,env,clear=True):
                        with self.assertRaises(ValueError):d.parent(q,rr)

    def test_inherited_or_forged_parent_environment_is_not_authorization(self):
        with tempfile.TemporaryDirectory() as tmp:
            q,rr,paths,m,s,env,folder=self.fixture(tmp)
            for update in ({'EVCS_JOB_ID':'other'},{'MKL_NUM_THREADS':'2'}):
                with self.subTest(update=update):
                    with patch.object(d,'get_paths',return_value=paths),patch.object(d,'observation',return_value={'supervisor_alive':True}),patch.dict(os.environ,{**env,**update},clear=True):
                        with self.assertRaises(ValueError):d.parent(q,rr)
            with patch.object(d,'get_paths',return_value=paths),patch.object(d,'observation',return_value={'supervisor_alive':False}),patch.dict(os.environ,env,clear=True):
                with self.assertRaises(ValueError):d.parent(q,rr)

if __name__=='__main__':unittest.main()
