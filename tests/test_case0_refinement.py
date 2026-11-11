"""Actual failed-native schema and cumulative precision-retry control, no solve."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch,Mock
import subprocess

from evcs_experiments import case0_campaign as c
from evcs_experiments.case0_refinement import prepare_numerical_refinement
from evcs_experiments.campaign_batch import file_reference


class Case0RefinementTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name);self.root=self.base/c.CAMPAIGN_NAME;self.root.mkdir()
        self.oldrelease=self.base/'oldrelease';self.oldrelease.mkdir();(self.oldrelease/'old.py').write_text('original')
        (self.root/'MANIFEST_SHA256.csv').write_text('case0-fixture')
        self.mapping={'old.py':c.digest(self.oldrelease/'old.py')}
        for patcher in (patch.object(c,'get_paths',return_value=SimpleNamespace(campaign_root=self.root,data_root=self.root)),
                        patch.object(c,'sources',side_effect=lambda:self.mapping)):
            patcher.start();self.addCleanup(patcher.stop)
        state=c.prepare('gurobi');self.original_config=(self.root/'config.lock.json').read_bytes()
        self.old=self.root/'planning/size1/OR_DRO';(self.old/'solve').mkdir(parents=True)
        self.ea=self.root/'planning/size1/EA_DRO/solve/result.json';self.ea.parent.mkdir(parents=True)
        self.put(self.ea,dict(method_id='EA_DRO',backend='gurobi',numerical_valid=True,gap_passed=True,gap=0.))
        state.update(active_job_id='oldjob',size_index=1,status='FAILURE_REQUIRES_REVIEW',compute_seconds=2274.9822266940027,
            completed={'size1/EA_DRO':dict(path=str(self.ea),sha256=c.digest(self.ea),gap=0.,numerical_valid=True,gap_passed=True)})
        c.save_state(self.root,state)
        self.put(self.old/'accounting.json',dict(configuration_wall_seconds=114.04157119244337,cumulative_case0_compute_seconds=state['compute_seconds']))
        self.put(self.old/'failure.json',dict(type='RuntimeError',error='child exited 1'))
        self.job=self.root/'jobs/oldjob';self.job.mkdir(parents=True)
        self.put(self.job/'manifest.json',{})
        self.put(self.job/'status.json',dict(state='SUCCEEDED'))
        self.put(self.job/'result.json',dict(job_id='oldjob',state='SUCCEEDED',complete=True))
        self.put(self.oldrelease/'RELEASE_MANIFEST.json',dict(files=[dict(path=k,sha256=v) for k,v in self.mapping.items()]))
        self.put(self.old/'solve/master.json',dict(solver_status=4,status='SOLVER_ERROR',wall_seconds=97.56526647135615,
            certificate={},raw_solver=dict(native_interface=dict(raw_model_status=2,original_certificate=dict(max_row_violation=1.02076e-8,max_integrality_violation=0.)))))
        self.mapping={'old.py':self.mapping['old.py'],'new.py':'a'*64}
        self.make_request()

    def put(self,p,row):p.write_text(json.dumps(row)+'\n')
    def make_request(self):
        refs={k:file_reference(p) for k,p in dict(config=self.root/'config.lock.json',state=self.root/'state.json',
            accounting=self.old/'accounting.json',failure=self.old/'failure.json',master=self.old/'solve/master.json',completed_ea_result=self.ea,last_parent_result=self.job/'result.json',original_source_manifest=self.oldrelease/'RELEASE_MANIFEST.json').items()}
        review=self.base/'review.json';self.put(review,dict(kind='CASE0_NUMERICAL_REFINEMENT_REVIEW_V1',passed=True,references=refs,new_source_sha256=self.mapping))
        refs['review']=file_reference(review);self.req=self.base/'request.json'
        self.put(self.req,dict(kind='CASE0_NUMERICAL_REFINEMENT_REQUEST_V1',references=refs,new_source_sha256=self.mapping,original_release_root=str(self.oldrelease)))
    def prepare(self):
        with patch('evcs_experiments.case0_refinement.time.monotonic',side_effect=[100.,101.]):
            return prepare_numerical_refinement(self.req,c.digest(self.req))

    def test_actual_rejected_native_schema_preserves_all_old_bytes_and_accounting(self):
        old={p:p.read_bytes() for p in self.old.rglob('*') if p.is_file()};eabytes=self.ea.read_bytes()
        state=self.prepare();transition=c.read(state['numerical_refinement']['path'])
        self.assertEqual((self.root/'config.lock.json').read_bytes(),self.original_config)
        self.assertEqual(self.ea.read_bytes(),eabytes)
        for p,data in old.items():self.assertEqual(p.read_bytes(),data)
        self.assertEqual(state['compute_seconds'],2275.9822266940027)
        self.assertEqual(transition['admission_seconds'],1.)
        self.assertEqual(transition['original_failed_configuration_seconds'],114.04157119244337)
        self.assertAlmostEqual(transition['remaining_configuration_seconds'],1684.9584288075566)
        self.assertAlmostEqual(transition['remaining_master_seconds'],1102.4347335286439)
        self.assertEqual(c.active_config(self.root)['solver_feasibility_tolerance'],1e-9)
        with self.assertRaises(ValueError):self.prepare()

    def test_changed_evidence_OOS_or_active_state_refused(self):
        for key,value in [('evaluation_opened',True),('active_configuration','size1/OR_DRO')]:
            state=c.read(self.root/'state.json');old=state[key];state[key]=value;c.save_state(self.root,state);self.make_request()
            with self.assertRaises(ValueError):self.prepare()
            state[key]=old;c.save_state(self.root,state)
        self.make_request();self.ea.write_text('changed')
        with self.assertRaises(ValueError):self.prepare()

    def test_live_Case0_job_refuses_transition(self):
        with patch('evcs_experiments.runtime.jobctl.observation',return_value=dict(blocks_new_launch=True,supervisor_alive=True,command_alive=True,live_group_members=[123])):
            with self.assertRaisesRegex(ValueError,'still active'):self.prepare()
        self.assertFalse((self.root/'revisions').exists())

    def test_missing_refinement_ledger_refused(self):
        state=self.prepare();state['configuration_spent_seconds']={}
        with self.assertRaisesRegex(ValueError,'carryover'):c.configuration_attempt(self.root,state,'size1/OR_DRO')

    def test_child_receives_remaining_total_and_master_budgets(self):
        state=self.prepare();state['active_configuration']='size1/OR_DRO';c.save_state(self.root,state)
        ctx=SimpleNamespace(probability_set=object(),planning_ids=('s',),scenario=lambda i:object(),recourse=lambda i:object(),first=object(),case=object(),tasks=object())
        with patch('evcs_experiments.case0_isolation.verify_current_cgroup'),patch('evcs_experiments.case0_inputs.Case0Context',return_value=ctx),patch('evcs_experiments.case0_planning.run_case0_planning',return_value=(None,{'fixture':True})) as run:
            c.plan_one('OR_DRO',1000.)
        kw=run.call_args.kwargs
        self.assertAlmostEqual(kw['deadline'],2684.9584288075566)
        self.assertAlmostEqual(kw['configuration_started_at'],884.9584288075566)
        self.assertAlmostEqual(kw['master_seconds'],1102.4347335286439)
        self.assertEqual(kw['solver_feasibility_tolerance'],1e-9)
        self.assertEqual(kw['output_dir'],self.old/'refinement-1/solve')

    def test_refinement_hard_timeout_subtracts_prior_work(self):
        folder=self.base/'child';folder.mkdir();child=Mock(pid=456);child.wait.side_effect=[subprocess.TimeoutExpired('x',1),0]
        with patch.object(c.subprocess,'Popen',return_value=child),patch.object(c.os,'killpg'),patch.object(c.time,'monotonic',return_value=110.):
            with self.assertRaises(TimeoutError):c.bounded_configuration('OR_DRO',100.,folder,prior_spent=114.04157119244337)
        self.assertAlmostEqual(child.wait.call_args_list[0].kwargs['timeout'],1675.9584288075566)

    def test_batch_preserves_EA_and_continues_OR_SAA_ARO_without_reset(self):
        self.prepare();clock=[100.];calls=[]
        def run(method,started,folder,**kw):
            calls.append((method,kw));clock[0]+=10.;(folder/'solve').mkdir()
            result=dict(method_id=method,backend='gurobi',numerical_valid=True,gap_passed=True,gap=0.)
            self.put(folder/'solve/result.json',result);return result
        with patch('evcs_experiments.case0_isolation.verify_current_cgroup',return_value={}),patch.object(c,'bounded_configuration',side_effect=run),patch.object(c.time,'monotonic',side_effect=lambda:clock[0]):
            state=c.plan_batch()
        self.assertEqual([m for m,_ in calls],['OR_DRO','SAA','ARO'])
        self.assertAlmostEqual(calls[0][1]['prior_spent'],115.04157119244337)
        self.assertAlmostEqual(state['compute_seconds'],2305.9822266940027)
        self.assertAlmostEqual(c.read(self.old/'refinement-1/accounting.json')['cumulative_configuration_seconds'],125.04157119244337)
        self.assertEqual(state['status'],'PLANNING_REVIEW_REQUIRED')


if __name__=='__main__':unittest.main()
