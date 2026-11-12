"""Pure metadata memory-repair admission; no server/cgroup/solver execution."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evcs_experiments import case0_memory_repair as m, case0_campaign as c, case0_interruption as old
from evcs_experiments.config import get_paths


class MemoryRepairTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.base=Path(tmp.name).resolve()
        self.root=self.base/'ieee33-v54-q1-case0-expanded';self.root.mkdir();self.release=self.base/'R2';self.release.mkdir()
        self.write(self.root/'MANIFEST_SHA256.csv',{'frozen':'data'})
        self.old_sources={}
        for name in sorted(m.CHANGED_SOURCES-{m.MODULE}|{'src/evcs_experiments/recourse.py'}):
            p=self.release/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('R2 original '+name)
            self.old_sources[name]=c.digest(p)
        self.new_sources=dict(self.old_sources);self.new_sources[m.MODULE]='d'*64
        self.new_sources.update({'src/evcs_experiments/'+k:v for k,v in m.NUMERICAL.items()})
        self.paths=SimpleNamespace(campaign_root=self.root,data_root=self.root)
        for obj,name,kwargs in ((c,'get_paths',dict(return_value=self.paths)),(c,'sources',dict(side_effect=lambda:dict(self.old_sources))),
                               (m,'linger_enabled',dict(return_value=True)),(old,'observation',dict(return_value=dict(blocks_new_launch=False,supervisor_alive=False,command_alive=False,live_group_members=[])))):
            p=patch.object(obj,name,**kwargs);p.start();self.addCleanup(p.stop)
        c.prepare('gurobi','expanded48-v1');self.original=self.ref(self.root/'config.lock.json')
        config=c.read(self.root/'config.lock.json');self.config=self.write(self.root/'revisions/interruption-1/config.json',config)
        self.prior=dict(kind='CASE0_INTERRUPTION_TRANSITION_V1',active_config_reference=self.config,
                        prior_configuration_seconds=204.2339227415018,prior_master_seconds=204.025118)
        self.prior_ref=self.write(self.root/'revisions/interruption-1/transition.json',self.prior)
        p=patch.object(old,'validate_transition',side_effect=lambda root,state:(c.read(self.config['path']),self.prior));p.start();self.addCleanup(p.stop)
        self.attempt=self.root/'planning'/m.KEY;self.failed=self.attempt/'continuation-1'
        self.write(self.attempt/'inputs/identity.json',{'frozen':'identity'});self.write(self.attempt/'inputs/tasks.json',{'frozen':'tasks'})
        self.original_inventory=self.write(self.base/'original_inventory.json',dict(root=str(self.attempt),files=[self.ref(p) for p in sorted(self.attempt.rglob('*')) if p.is_file()]))
        self.failed.mkdir();(self.failed/'inputs').symlink_to(self.attempt/'inputs',target_is_directory=True)
        self.write(self.failed/'solve/model.json',{'matrix':'original'})
        self.failed_inventory=self.write(self.base/'failed_inventory.json',dict(root=str(self.failed),files=[self.ref(p) for p in sorted(self.failed.rglob('*')) if p.is_file()]))
        job=self.root/'jobs/failed';job.mkdir(parents=True)
        self.parent=self.write(job/'result.json',dict(job_id='failed',complete=True,state='FAILED',returncode=1,elapsed_seconds=202.64269))
        self.manifest=self.write(job/'manifest.json',dict(job_id='failed'));self.write(job/'status.json',dict(job_id='failed',state='FAILED'))
        self.launch=self.write(self.base/'launch.json',dict(opening_compute_seconds=228.31222964370178))
        self.state=c.read(self.root/'state.json');self.state.update(status='PLANNING',active_job_id='failed',active_configuration=m.KEY,
            active_config_reference=self.config,operational_interruption=self.prior_ref,
            compute_seconds=228.32323574753033,planning_launch_admission_reference=self.launch)
        before=self.write(self.base/'state_failed.json',self.state)
        resource=self.write(self.base/'resource.json',dict(unit_properties={'Result':'oom-kill','ExecMainStatus':'9'}))
        guard=self.write(self.base/'guard.json',{'passed':False})
        terminal=self.write(self.base/'terminal.json',dict(kind='CASE0_EXPANDED_RECOVERY_TERMINAL_INDEPENDENT_REVIEW_V1',passed=True,
            references=dict(parent_result=self.parent,parent_manifest=self.manifest,resource_observation=resource,guard_failure=guard)))
        self.settlement=dict(kind='CASE0_OOM_PARENT_SETTLEMENT_V1',scientific_pass=False,OOS_opened=False,automatic_retry=False,
            configuration_key=m.KEY,parent_reference=self.parent,parent_manifest_reference=self.manifest,
            prior_interruption_transition_reference=self.prior_ref,state_before_reference=before,launch_admission_reference=self.launch,
            terminal_review_reference=terminal,resource_observation_reference=resource,guard_failure_reference=guard,
            original_input_identity_reference=self.ref(self.attempt/'inputs/identity.json'),original_attempt_manifest_reference=self.original_inventory,
            failed_attempt_manifest_reference=self.failed_inventory,parent_elapsed_seconds=202.64269,
            opening_compute_seconds=228.31222964370178,previously_accounted_compute_seconds=228.32323574753033,
            already_charged_parent_seconds=.011006103828549385,added_compute_seconds=202.63168389617144,
            closing_compute_seconds=430.95491964370177,prior_configuration_seconds=self.prior['prior_configuration_seconds'],
            prior_master_seconds=self.prior['prior_master_seconds'],closing_configuration_seconds=406.8766127415018,
            closing_master_seconds=406.667808,remaining_configuration_seconds=1393.1233872584983,remaining_master_seconds=793.332192)
        self.setref=self.write(self.base/'settlement.json',self.settlement)
        sr=self.write(self.release/'RELEASE_MANIFEST.json',dict(files=[dict(path=k,sha256=v) for k,v in self.old_sources.items()]))
        self.state.update(status='MEMORY_REPAIR_REVIEW_REQUIRED',active_job_id=None,active_configuration=None,
            compute_seconds=self.settlement['closing_compute_seconds'],planning_parent_pending_settlement=False,
            planning_parent_settlement_reference=self.setref,memory_failure_settlement_reference=self.setref,
            configuration_spent_seconds={m.KEY:self.settlement['closing_configuration_seconds']},
            configuration_master_spent_seconds={m.KEY:self.settlement['closing_master_seconds']},
            source_release_reference=sr,release_root=str(self.release))
        c.save_state(self.root,self.state)
        prepared=self.base/'prepared';prepared.mkdir();files={}
        source_refs={}
        for name in m.NUMERICAL:
            p=prepared/name;p.write_bytes((get_paths().project_root/'provenance/case0/expanded48_memory_20260910'/name).read_bytes())
            files[name]=self.ref(p);source_refs[name]=self.ref(p)
        files['component_review.json']=self.write(prepared/'component_review.json',dict(kind='CASE0_EXACT_ZERO_COMPONENT_INDEPENDENT_REVIEW_V1',passed=True,source_references=source_refs))
        p=prepared/'zero_reduction_diagnostic.jsonl';p.write_text('\n'.join(json.dumps(row) for row in [dict(stage='loaded'),dict(stage='reduced'),dict(stage='audited',no_optimizer=True,audit={'status':'PASS_EXACT_ZERO_PROOF'},elapsed_seconds=m.DIAGNOSTIC_SECONDS)]))
        files[p.name]=self.ref(p)
        self.prepared=dict(kind='CASE0_MEMORY_REPAIR_PREPARED_COMPONENT_V1',component_pass=True,controller_wiring_complete=False,
            deployed=False,launched=False,production_memory_pass=False,files=files,
            diagnostic_measured_seconds_pending_preparation_accounting=m.DIAGNOSTIC_SECONDS)
        prepref=self.write(self.base/'prepared.json',self.prepared)
        self.state['memory_repair_prepared_reference']=self.state['pending_diagnostic_accounting_reference']=prepref
        c.save_state(self.root,self.state)
        self.refs=dict(original_config=self.original,config=self.config,state=self.ref(self.root/'state.json'),
            original_source_manifest=sr,parent_manifest=self.manifest,parent_result=self.parent,settlement=self.setref,prepared=prepref)
        p=patch.object(c,'sources',side_effect=lambda:dict(self.new_sources));p.start();self.addCleanup(p.stop)
        self.seal_request()

    def ref(self,p):return m.file_reference(p)
    def write(self,p,value):
        p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return self.ref(p)
    def seal_request(self):
        self.refs.pop('review',None);self.refs['review']=self.write(self.base/'review.json',dict(kind='CASE0_MEMORY_REPAIR_REVIEW_V1',passed=True,references=dict(self.refs),new_source_sha256=self.new_sources))
        self.request=self.write(self.base/'request.json',dict(kind='CASE0_MEMORY_REPAIR_REQUEST_V1',references=self.refs,
            original_release_root=str(self.release),new_source_sha256=self.new_sources))
    def prepare(self):
        with patch.object(m.time,'monotonic',side_effect=[10.,10.125]):return m.prepare_memory_repair(self.request['path'],self.request['sha256'])
    def resettle(self):
        self.refs['settlement']=self.write(self.base/'settlement.json',self.settlement)
        state=c.read(self.root/'state.json');state['planning_parent_settlement_reference']=state['memory_failure_settlement_reference']=self.refs['settlement']
        c.save_state(self.root,state);self.refs['state']=self.ref(self.root/'state.json');self.seal_request()

    def test_exact_single_claim_and_same_science_and_all_three_budgets(self):
        old={str(p):p.read_bytes() for p in self.attempt.rglob('*') if p.is_file()}
        state=self.prepare();config,t=m.validate_transition(self.root,state)
        self.assertIs(config['solver_zero_presolve'],True)
        original=c.read(self.config['path']);self.assertEqual({k:v for k,v in config.items() if k not in ('source_sha256','solver_zero_presolve')},{k:v for k,v in original.items() if k!='source_sha256'})
        self.assertEqual(state['compute_seconds'],430.95491964370177+m.DIAGNOSTIC_SECONDS+.125)
        self.assertEqual(t['prior_configuration_seconds'],406.8766127415018+.125);self.assertEqual(t['prior_master_seconds'],406.667808)
        self.assertEqual(t['new_attempt_directory'],str(self.attempt/'memory-repair-1'))
        state.update(status='PLANNING',compute_seconds=state['compute_seconds']+1);c.save_state(self.root,state)
        self.assertEqual(m.validate_transition(self.root,state)[0],config)
        for p,v in old.items():self.assertEqual(Path(p).read_bytes(),v)
        with self.assertRaises(FileExistsError):self.prepare()

    def test_changed_mathematics_rejected_and_failed_admission_claim_retained(self):
        self.new_sources['src/evcs_experiments/recourse.py']='a'*64;self.seal_request()
        with self.assertRaisesRegex(ValueError,'mathematical'):self.prepare()
        self.assertTrue((self.root/'revisions/memory-repair-1/reservation.json').exists())
        with self.assertRaises(FileExistsError):self.prepare()
    def test_nested_master_reset_rejected_even_with_resealed_refs(self):
        self.settlement['closing_master_seconds']=0.;self.resettle()
        with self.assertRaisesRegex(ValueError,'Master expense reset'):self.prepare()
    def test_pending_diagnostic_cannot_change_or_be_double_charged(self):
        self.prepared['diagnostic_measured_seconds_pending_preparation_accounting']=0.
        self.refs['prepared']=self.write(self.base/'prepared.json',self.prepared)
        state=c.read(self.root/'state.json');state['memory_repair_prepared_reference']=state['pending_diagnostic_accounting_reference']=self.refs['prepared']
        c.save_state(self.root,state);self.refs['state']=self.ref(self.root/'state.json');self.seal_request()
        with self.assertRaisesRegex(ValueError,'diagnostic'):self.prepare()
    def test_old_input_mutation_and_retargeted_symlink_fail(self):
        (self.failed/'inputs').unlink();(self.failed/'inputs').symlink_to(self.base,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'input path'):self.prepare()
    def test_live_cumulative_budget_reset_is_rejected(self):
        state=self.prepare();state['compute_seconds']-=.01
        with self.assertRaisesRegex(ValueError,'cumulative budget reset'):m.validate_transition(self.root,state)

    def test_live_parent_is_rejected(self):
        with patch.object(old,'observation',return_value=dict(blocks_new_launch=True,supervisor_alive=True,command_alive=False,live_group_members=[])):
            with self.assertRaisesRegex(ValueError,'live'):self.prepare()
    def test_wrong_numerical_component_is_rejected(self):
        self.new_sources['src/evcs_experiments/case0_planning.py']='a'*64;self.seal_request()
        with self.assertRaisesRegex(ValueError,'numerical component'):self.prepare()


if __name__=='__main__':unittest.main()
