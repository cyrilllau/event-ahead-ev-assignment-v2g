"""Representation admission: real refs/arithmetic/inventories; old ancestry boundary mocked."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from evcs_experiments import case0_memory_repair as m, case0_campaign as c, case0_interruption as old

class RepresentationTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.base=Path(tmp.name).resolve()
        self.root=self.base/'ieee33-v54-q1-case0-expanded';self.root.mkdir();self.release=self.base/'R3';self.release.mkdir()
        self.write(self.root/'MANIFEST_SHA256.csv',{'data':'frozen'})
        self.old_sources={}
        for name in sorted(m.REPRESENTATION_SOURCES-{m.COMPACT_MODULE}|{'src/evcs_experiments/case0_planning_audit.py','src/evcs_experiments/recourse.py'}):
            path=self.release/name;path.parent.mkdir(parents=True,exist_ok=True)
            if name.endswith('case0_planning_audit.py'):path.write_bytes(Path('src/evcs_experiments/case0_planning_audit.py').read_bytes())
            else:path.write_text('old source '+name)
            self.old_sources[name]=c.digest(path)
        self.new_sources=dict(self.old_sources);self.new_sources[m.COMPACT_MODULE]='c'*64;self.new_sources['src/evcs_experiments/case0_planning.py']='d'*64
        paths=SimpleNamespace(campaign_root=self.root,data_root=self.root)
        for obj,name,kw in ((c,'get_paths',dict(return_value=paths)),(c,'sources',dict(side_effect=lambda:dict(self.old_sources))),
            (m,'linger_enabled',dict(return_value=True)),(old,'observation',dict(return_value=dict(blocks_new_launch=False,supervisor_alive=False,command_alive=False,live_group_members=[])))):
            p=patch.object(obj,name,**kw);p.start();self.addCleanup(p.stop)
        c.prepare('gurobi','expanded48-v1');config=c.read(self.root/'config.lock.json');config['solver_zero_presolve']=True
        self.config=self.write(self.root/'revisions/memory-repair-1/config.json',config)
        oldsettled=self.write(self.base/'old_settlement.json',{'original':'settled'})
        self.prior=dict(active_config_reference=self.config,settlement_reference=oldsettled,prior_configuration_seconds=407.14969862066365,prior_master_seconds=406.667808)
        self.transition=self.write(self.root/'revisions/memory-repair-1/transition.json',self.prior)
        p=patch.object(m,'validate_transition',side_effect=lambda root,state:(c.read(self.config['path']),self.prior));p.start();self.addCleanup(p.stop)
        self.attempt=self.root/'planning'/m.KEY
        self.write(self.attempt/'inputs/identity.json',{'identity':'frozen'})
        inventory={}
        for name,key in (('','original_attempt_manifest_reference'),('continuation-1','prior_failed_attempt_manifest_reference'),('memory-repair-1','failed_attempt_manifest_reference')):
            folder=self.attempt/name
            if name:
                folder.mkdir();(folder/'inputs').symlink_to(self.attempt/'inputs',target_is_directory=True)
                self.write(folder/'solve/model.json',{'partial':name})
            inventory[key]=self.write(self.base/(key+'.json'),dict(root=str(folder),files=[self.ref(p) for p in folder.rglob('*') if p.is_file()]))
        job=self.root/'jobs/failed';job.mkdir(parents=True)
        self.parent=self.write(job/'result.json',dict(job_id='failed',complete=True,state='FAILED',returncode=1,elapsed_seconds=206.49732))
        self.manifest=self.write(job/'manifest.json',dict(job_id='failed'));self.write(job/'status.json',dict(job_id='failed',state='FAILED'))
        self.launch=self.write(self.base/'launch.json',dict(opening_compute_seconds=453.0476121399568,transition_reference=self.transition))
        self.state=c.read(self.root/'state.json');self.state.update(status='PLANNING',active_job_id='failed',active_configuration=m.KEY,
            active_config_reference=self.config,memory_repair=self.transition,compute_seconds=453.1476121399568,
            planning_launch_admission_reference=self.launch)
        before=self.write(self.base/'before.json',self.state)
        resources=self.write(self.base/'resource.json',{'cause':'unknown'});guard=self.write(self.base/'guard.json',{'failed':True})
        logpath=self.base/'master.log';logpath.write_text('Gurobi Optimizer: Optimize a model with rows\n');log=self.ref(logpath)
        obs=self.write(self.base/'obs.json',{'dead':True})
        terminal=self.write(self.base/'terminal.json',dict(kind='CASE0_R3_TERMINAL_INDEPENDENT_REVIEW_V1',passed=True,
            observation_reference=obs,references=dict(parent_result=self.parent,parent_manifest=self.manifest,launch_admission=self.launch,resources=resources,guard_failure=guard,master_log=log)))
        self.settlement=dict(kind='CASE0_REPRESENTATION_PARENT_SETTLEMENT_V1',scientific_pass=False,OOS_opened=False,automatic_retry=False,
            configuration_key=m.KEY,parent_reference=self.parent,parent_manifest_reference=self.manifest,representation_transition_reference=self.transition,
            prior_parent_settlement_reference=oldsettled,state_before_reference=before,launch_admission_reference=self.launch,
            terminal_review_reference=terminal,terminal_observation_reference=obs,resource_observation_reference=resources,guard_failure_reference=guard,
            master_log_reference=log,original_input_identity_reference=self.ref(self.attempt/'inputs/identity.json'),**inventory,
            parent_elapsed_seconds=206.49732,opening_compute_seconds=453.0476121399568,previously_accounted_compute_seconds=453.1476121399568,
            already_charged_parent_seconds=.1,added_compute_seconds=206.39732,closing_compute_seconds=659.5449321399568,
            prior_configuration_seconds=self.prior['prior_configuration_seconds'],prior_master_seconds=self.prior['prior_master_seconds'],
            closing_configuration_seconds=613.6470186206637,closing_master_seconds=613.165128,
            remaining_configuration_seconds=1186.3529813793364,remaining_master_seconds=586.834872)
        sr=self.write(self.release/'RELEASE_MANIFEST.json',dict(files=[dict(path=k,sha256=v) for k,v in self.old_sources.items()]))
        self.state.update(status='REPRESENTATION_FAILURE_REVIEW_REQUIRED',active_job_id=None,active_configuration=None,
            planning_parent_pending_settlement=False,release_root=str(self.release),source_release_reference=sr)
        comp=self.write(self.base/'component.json',dict(kind='CASE0_COMPACT_RECOURSE_COMPONENT_REVIEW_V1',passed=True,exact_model_unchanged=True,
            source_sha256={k:self.new_sources[k] for k in (m.COMPACT_MODULE,'src/evcs_experiments/case0_planning.py')}))
        self.refs=dict(config=self.config,source_manifest=sr,parent_manifest=self.manifest,parent_result=self.parent,component=comp)
        p=patch.object(c,'sources',side_effect=lambda:dict(self.new_sources));p.start();self.addCleanup(p.stop)
        self.resettle()
    def ref(self,p):return m.file_reference(p)
    def write(self,p,v):
        p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v));return self.ref(p)
    def reseal(self):
        self.refs.pop('review',None)
        self.refs['review']=self.write(self.base/'review.json',dict(kind='CASE0_REPRESENTATION_REVIEW_V1',passed=True,revision=m.REPRESENTATION,
            references=dict(self.refs),new_source_sha256=self.new_sources))
        self.request=self.write(self.base/'request.json',dict(kind='CASE0_REPRESENTATION_REQUEST_V1',revision=m.REPRESENTATION,
            references=self.refs,new_source_sha256=self.new_sources))
    def resettle(self):
        self.refs['settlement']=self.write(self.base/'settlement.json',self.settlement)
        self.state.update(planning_parent_settlement_reference=self.refs['settlement'],representation_failure_settlement_reference=self.refs['settlement'],
            compute_seconds=self.settlement['closing_compute_seconds'],configuration_spent_seconds={m.KEY:self.settlement['closing_configuration_seconds']},
            configuration_master_spent_seconds={m.KEY:self.settlement['closing_master_seconds']})
        c.save_state(self.root,self.state);self.refs['state']=self.ref(self.root/'state.json');self.reseal()
    def prepare(self):
        with patch.object(m.time,'monotonic',side_effect=[10.,10.125]):return m.prepare_representation(self.request['path'],self.request['sha256'])
    def test_same_science_all_budgets_and_unique_claim_after_live_advance(self):
        originals={str(p):p.read_bytes() for p in self.attempt.rglob('*') if p.is_file()}
        state=self.prepare();config,t=m.validate_representation(self.root,state)
        self.assertIs(config['solver_compact_recourses'],True);self.assertIs(config['solver_zero_presolve'],True)
        self.assertEqual(state['compute_seconds'],659.5449321399568+.125)
        self.assertEqual(t['prior_configuration_seconds'],613.6470186206637+.125);self.assertEqual(t['prior_master_seconds'],613.165128)
        self.assertEqual(t['new_attempt_directory'],str(self.attempt/m.REPRESENTATION))
        state['compute_seconds']+=1;c.save_state(self.root,state);m.validate_representation(self.root,state)
        for p,b in originals.items():self.assertEqual(Path(p).read_bytes(),b)
        with self.assertRaises(FileExistsError):self.prepare()
    def test_resealed_nested_master_reset_rejected(self):
        self.settlement['closing_master_seconds']=0;self.resettle()
        with self.assertRaisesRegex(ValueError,'Master expense reset'):self.prepare()
    def test_math_source_change_rejected_with_reservation(self):
        self.new_sources['src/evcs_experiments/recourse.py']='0'*64;self.reseal()
        with self.assertRaisesRegex(ValueError,'mathematical'):self.prepare()
        self.assertTrue((self.root/'revisions'/m.REPRESENTATION/'reservation.json').exists())
    def test_live_parent_rejected(self):
        with patch.object(old,'observation',return_value=dict(blocks_new_launch=True,supervisor_alive=True,command_alive=False,live_group_members=[])):
            with self.assertRaisesRegex(ValueError,'live'):self.prepare()
    def test_retargeted_input_symlink_rejected(self):
        p=self.attempt/'memory-repair-1/inputs';p.unlink();p.symlink_to(self.base,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'symlink'):self.prepare()
    def test_review_component_pin_rejected(self):
        self.new_sources[m.COMPACT_MODULE]='0'*64;self.reseal()
        with self.assertRaisesRegex(ValueError,'component/source'):self.prepare()
    def test_live_budget_reset_rejected(self):
        state=self.prepare();state['configuration_master_spent_seconds'][m.KEY]=0
        with self.assertRaisesRegex(ValueError,'budget reset'):m.validate_representation(self.root,state)

if __name__=='__main__':unittest.main()
