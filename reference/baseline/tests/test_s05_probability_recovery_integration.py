"""Complete synthetic provenance + real 160-variable LP; temporary installed release."""
from dataclasses import replace
import importlib.util
from importlib.metadata import distribution, PackageNotFoundError
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from evcs_experiments.config import get_paths
from evcs_experiments import failure_carryover as fc
from evcs_experiments.runtime import jobctl

ROOT=get_paths().project_root
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
base=module('carryover_fixture',ROOT/'tests/test_failure_carryover.py')
unit=module('recovery_fixture',ROOT/'tests/test_s05_probability_recovery.py')
m=unit.m

def fixture_process(argv):
    result=subprocess.run(argv,capture_output=True,text=True)
    if result.returncode:
        raise RuntimeError('Offline fixture command failed: '+repr(argv)+'\nSTDOUT:\n'+result.stdout+'\nSTDERR:\n'+result.stderr)
    return result


class Chain(base.CarryoverTests):
    @staticmethod
    def remap(value):
        if isinstance(value,dict):return {('michael-EA' if k=='p' else k):Chain.remap(v) for k,v in value.items()}
        if isinstance(value,list):return [Chain.remap(v) for v in value]
        return {'p':'michael-EA','target':'michael_2018'}.get(value,value) if isinstance(value,str) else value
    def put(self,p,x):return super().put(p,self.remap(x))
    def release(self,name,new=False):
        root=self.root/name
        shutil.copytree(ROOT/'src',root/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        prob=root/'src/evcs_experiments/probability.py'
        prob.write_text(prob.read_text().replace('SOLVER_FEASIBILITY_TOLERANCE = 1e-10','SOLVER_FEASIBILITY_TOLERANCE = 1e-8'))
        if not new:
            p=root/'src/evcs_experiments/formal_physical_inputs.py';p.write_text(p.read_text()+'\n# synthetic predecessor source segment\n')
        return self.put(root/'RELEASE_MANIFEST.json',dict(files=[dict(path=str(p.relative_to(root)),sha256=fc.reference(p)['sha256'],size_bytes=p.stat().st_size) for p in sorted((root/'src').rglob('*.py'))]))


class RecoveryEntryIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chain=Chain('runTest');cls.chain.setUp();f=cls.chain;cls.root=f.root;cls.c=f.c
        cls.r=cls.root/'recovery_release';shutil.copytree(ROOT/'src',cls.r/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        script=cls.r/'scripts/recovery/prepare_s05_probability_recovery.py';script.parent.mkdir(parents=True)
        shutil.copyfile(ROOT/'scripts/recovery/prepare_s05_probability_recovery.py',script)
        shutil.copyfile(ROOT/'scripts/recovery/settle_s05_probability_recovery.py',script.parent/'settle_s05_probability_recovery.py')
        # Installation is local, offline and isolated; there is no sys.path injection.
        shutil.copyfile(ROOT/'pyproject.toml',cls.r/'pyproject.toml')
        fixture_process([sys.executable,'-m','venv','--without-pip','--system-site-packages',str(cls.r/'.venv')])
        cls.python=str(cls.r/'.venv/bin/python')
        # Reuse the already installed, version-locked binary distributions
        # offline; the nested venv does not inherit its creator venv's packages.
        site=Path(subprocess.check_output([cls.python,'-c','import sysconfig;print(sysconfig.get_path("purelib"))'],text=True).strip())
        for name in ('numpy','scipy','networkx','setuptools','wheel','packaging'):
            try:installed=distribution(name)
            except PackageNotFoundError:
                if name=='packaging':continue  # older wheel vendors packaging
                raise
            # Include metadata, binary-library directories and backend .pth
            # files as well as package code; never import the creator's editable
            # application or inject its site directory into the child path.
            for top in {p.parts[0] for p in installed.files if p.parts[0]!='..'}:
                source=Path(installed.locate_file(top)).resolve();target=site/top
                if not target.exists():target.symlink_to(source,target_is_directory=source.is_dir())
        fixture_process([sys.executable,'-m','pip','--python',cls.python,'install','--no-index','--disable-pip-version-check','--no-build-isolation','--no-deps','-e',str(cls.r)])
        cls.config=cls.r/'config.json';cls.config.write_text(json.dumps(dict(paths=dict(project_root=str(cls.r),campaign_root=str(cls.c)))))
        cls.new_manifest=f.put(cls.r/'RELEASE_MANIFEST.json',dict(files=[dict(path=str(p.relative_to(cls.r)),sha256=m.reference(p)['sha256'],size_bytes=p.stat().st_size) for p in sorted(cls.r.rglob('*')) if p.is_file() and '.venv' not in p.parts and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.toml') and p.name!='RELEASE_MANIFEST.json']))
        oldworker=Chain.remap(f.new);cls.plan=Path(oldworker['output_root']);cls.plan.mkdir(parents=True)
        oldworker['failure_carryover_reference']=f.pin
        oldqueue=f.queue;oldqueue.mkdir(parents=True)
        # The prior tail claim is verified by the real existing implementation.
        with patch.object(fc,'observation',return_value={'blocks_new_launch':False}):fc.claim_failure_carryover(f.pin,oldworker,oldqueue)
        cls.refs={};put=lambda k,p,v:cls.refs.__setitem__(k,f.put(p,v))
        put('worker_request',cls.root/'current_worker.json',oldworker)
        queue=f.put(cls.root/'current_queue.json',dict(queue_root=str(oldqueue),items=[dict(plan_id='michael-EA',request_reference=cls.refs['worker_request'])]))
        jm=dict(job_id='failed-current',argv=['python','--request',queue['path']])
        put('job_manifest',cls.c/'jobs/failed-current/manifest.json',jm)
        put('job_result',cls.c/'jobs/failed-current/result.json',dict(job_id='failed-current',complete=True,state='FAILED',returncode=2,elapsed_seconds=3351.921871))
        inv=dict(request_reference=queue,parent_job=dict(job_id='failed-current',manifest_reference=cls.refs['job_manifest']),
            item_request_reference=cls.refs['worker_request'],plan_id='michael-EA',previous_checkpoint=None,previous_result_reference=None,
            failure_carryover_reference=f.pin,cumulative_plan_seconds=16.85686,opening_carryover_seconds=16.85686,plan_budget_seconds=7200)
        put('invocation',cls.root/'current_invocation.json',inv)
        state,qi,ai=unit.fixture();state['pending']['plan']['x']=[0.]*1008;state['pending']['plan']['r']=[[0.]*24 for _ in range(1008)]
        state['engine_sha256']=fc._release(f.newrelease)[1]
        put('raw_request',cls.plan/'raw/request.json',dict(problem_identity=state['identity'],domain='formal',formal_OOS=False))
        history=[]
        for i,event in enumerate(['phase1','rmp']+['recourse']*160):
            history.append(f.put(cls.plan/f'raw/events/{i:06d}_{event}/record.json',dict(event=event,iteration=0 if i==0 else 1,artifacts=[])))
        probdir=cls.plan/'raw/events/000162_probability'
        probrefs=[f.put(probdir/'q_input.json',dict(qi,recourse_evidence=history[2:])),f.put(probdir/'probability_input.json',dict(effective_ambiguity=ai)),f.put(probdir/'solver.json',state['pending']['last_incomplete_probability'])]
        history.append(f.put(probdir/'record.json',dict(event='probability',iteration=1,artifacts=probrefs)))
        put('history',cls.plan/'raw/history.json',history)
        put('checkpoint',cls.plan/'driver/checkpoint.json',dict(state=state,sha256=m.canonical_hash(state)))
        cls.refs['old_source_manifest']=f.newrelease;cls.refs['new_source_manifest']=cls.new_manifest
        cls.refs['recovery_script']=m.reference(script)
        cls.refs['settlement_script']=m.reference(script.parent/'settle_s05_probability_recovery.py')
        cls.refs['probability_review']=f.put(cls.root/'prob_review.json',dict(review='synthetic review binding; not actual scientific acceptance'))
        oldsrc=fc._release(f.newrelease)[0];newsrc=fc._release(cls.new_manifest)[0]
        cls.refs['probability_source_approval']=f.put(cls.root/'source_approval.json',dict(status='PASS_PROBABILITY_SOURCE_CHANGE_REVIEW',old_source_manifest=f.newrelease,new_source_manifest=cls.new_manifest,
            changed_sources={'probability.py':dict(old=oldsrc['probability.py'],new=newsrc['probability.py'])},review_reference=cls.refs['probability_review']))
        proof=unit.AuditedProofTests('runTest');proof.setUp();cls.proof=proof
        # Replace the synthetic proof's origin bundle by the complete test chain.
        origins={k:cls.refs[k] for k in ('job_result','checkpoint','history','raw_request')};origins['source_manifest']=f.newrelease
        a,q=m.verify_pending(state,qi,ai);sep=m.separate(state,a,q,0.,0.)
        residual=dict(support_ids=list(a.support_ids),active_indices=state['active_indices'],residuals_usd=sep['residuals_usd'])
        rp=proof.out/'raw_rmp_dual_residuals.json';rp.write_text(json.dumps(residual))
        audit=json.loads((proof.out/'audit.json').read_text());audit.update(included_dual_violation_usd=sep['included_maximum_usd'],full_support_dual_violation_usd=sep['full_support_maximum_usd'])
        (proof.out/'audit.json').write_text(json.dumps(audit))
        (proof.out/'input_references.json').write_text(json.dumps(list(origins.values())+history+probrefs))
        artifact_path=proof.out/'artifact_manifest.json';artifact_path.write_text(json.dumps([m.reference(p) for p in sorted(proof.out.iterdir()) if p.name!='artifact_manifest.json']))
        for key,name in [('full160_audit','audit.json'),('audit_artifact_manifest','artifact_manifest.json'),('audit_input_references','input_references.json')]:cls.refs[key]=m.reference(proof.out/name)
        review=dict(status=m.REVIEW_STATUS,references={**origins,**{k:cls.refs[k] for k in ('full160_audit','audit_artifact_manifest','audit_input_references')}},problem_sha256=state['problem_sha256'],complete_lp_count=160,probability_PASS=False,formal_OOS=False)
        cls.refs['independent_acceptance']=f.put(cls.root/'full160_review.json',review)
        cls.script=script
        cls.env=dict(os.environ,EVCS_CONFIG=str(cls.config),EVCS_PROJECT_ROOT=str(cls.r),PYTHONDONTWRITEBYTECODE='1')
        cls.env.pop('PYTHONPATH',None)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls,'proof'):cls.proof.doCleanups()
        cls.chain.doCleanups()

    def request(self,jobid):
        p=self.root/(jobid+'.json');req=dict(schema_version=1,kind=m.KIND,domain='formal',spec_id='S05',method_id='EA',target_id='michael_2018',plan_id='michael-EA',
            wall_seconds=20.,references=self.refs,output_root=str(self.c/'recoveries'/jobid),
            parent_job=dict(job_id=jobid,directory=str(self.c/'jobs'/jobid),threads=1,timeout_seconds=30.,grace_seconds=5.,controller_reference=m.reference(self.r/'src/evcs_experiments/runtime/jobctl.py')))
        p.write_text(json.dumps(req));return m.reference(p)

    def test_01_full_prepare_cli_offline_installed_package(self):
        rr=self.request('prepare-fixture')
        result=subprocess.run([self.python,'-B',str(self.script),'--request',rr['path'],'--expected-request-sha256',rr['sha256'],'--action','verify'],env=self.env,cwd=self.r,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertAlmostEqual(json.loads(result.stdout)['budget']['opening_cumulative_seconds'],3368.778731)

    def test_02_direct_cli_without_real_parent_rejected(self):
        rr=self.request('no-parent')
        result=subprocess.run([self.python,'-B',str(self.script),'--request',rr['path'],'--expected-request-sha256',rr['sha256'],'--action','run'],env=self.env,cwd=self.r,capture_output=True,text=True,timeout=30)
        self.assertNotEqual(result.returncode,0);self.assertIn('parent environment differs',result.stderr)

    def test_02b_wrong_worker_identity_and_opening_reset_rejected(self):
        for key,value,expected in [('plan_id','wrong-plan','worker identity'),('cumulative_plan_seconds',0.,'charged exactly once')]:
            name='worker_request' if key=='plan_id' else 'invocation'
            original=self.refs[name];newpath=self.root/(key+'_bad.json')
            data=json.loads(Path(original['path']).read_text());data[key]=value;newpath.write_text(json.dumps(data))
            rr=self.request('wrong-'+key);request=json.loads(Path(rr['path']).read_text());request['references'][name]=m.reference(newpath)
            # A wrong invocation carries the same worker ref, so it reaches the
            # actual old carryover arithmetic instead of an earlier SHA guard.
            Path(rr['path']).write_text(json.dumps(request));rr=m.reference(rr['path'])
            result=subprocess.run([self.python,'-B',str(self.script),'--request',rr['path'],'--expected-request-sha256',rr['sha256'],'--action','verify'],env=self.env,cwd=self.r,capture_output=True,text=True,timeout=30)
            self.assertNotEqual(result.returncode,0,result.stdout);self.assertIn(expected,result.stderr)

    def test_02c_source_member_tamper_rejected_before_probability(self):
        p=self.r/'src/evcs_experiments/recourse.py';original=p.read_bytes();p.write_bytes(original+b'\n# changed\n')
        try:
            rr=self.request('tampered-source')
            result=subprocess.run([self.python,'-B',str(self.script),'--request',rr['path'],'--expected-request-sha256',rr['sha256'],'--action','verify'],env=self.env,cwd=self.r,capture_output=True,text=True,timeout=30)
            self.assertNotEqual(result.returncode,0);self.assertIn('Evidence bytes changed',result.stderr)
        finally:p.write_bytes(original)

    def test_03_real_jobctl_parent_atomic_cli(self):
        rr=self.request('recovery-integration')
        args=[self.python,'-B',str(self.script),'--request',rr['path'],'--expected-request-sha256',rr['sha256'],'--action','run']
        request=dict(job_id='recovery-integration',argv=args,cwd=str(self.r),outroot=str(self.c/'jobs'),threads=1,timeout_seconds=30.,grace_seconds=5.,spec_id='S05',purpose='Synthetic provenance + real small probability LP integration only')
        # The historical test terminal jobs require status files for jobctl inventory.
        for d in (self.c/'jobs').iterdir():
            if (d/'manifest.json').exists() and (d/'result.json').exists() and not (d/'status.json').exists():shutil.copyfile(d/'result.json',d/'status.json')
        requestpath=self.root/'parent_request.json';requestpath.write_text(json.dumps(request))
        launch=subprocess.run([self.python,'-m','evcs_experiments.runtime.jobctl','launch','--request',str(requestpath)],env=self.env,cwd=self.r,capture_output=True,text=True,timeout=10)
        self.assertEqual(launch.returncode,0,launch.stderr)
        deadline=time.monotonic()+20;resultpath=self.c/'jobs/recovery-integration/result.json'
        while time.monotonic()<deadline:
            result=json.loads(resultpath.read_text())
            if result.get('complete'):break
            time.sleep(.05)
        self.assertTrue(result.get('complete'));self.assertEqual(result['state'],'SUCCEEDED',(self.c/'jobs/recovery-integration/stderr.log').read_text())
        out=json.loads((self.c/'recoveries/recovery-integration/result.json').read_text())
        self.assertEqual(out['status'],'VALID_INCUMBENT_BOUND_UNAVAILABLE');self.assertEqual(out['master_solver_calls'],0)
        self.assertEqual(out['parent_job']['job_id'],'recovery-integration');self.assertTrue(out['requires_final_parent_job_settlement'])
        pr=m.reference(resultpath)
        settlement_args=[self.python,'-B',str(self.r/'scripts/recovery/settle_s05_probability_recovery.py'),'--request',rr['path'],
            '--expected-request-sha256',rr['sha256'],'--parent-result',pr['path'],'--expected-parent-result-sha256',pr['sha256']]
        settled=subprocess.run(settlement_args,env=self.env,cwd=self.r,capture_output=True,text=True,timeout=30)
        self.assertEqual(settled.returncode,0,settled.stderr)
        record=json.loads(settled.stdout)['record']
        self.assertAlmostEqual(record['cumulative_accounted_seconds'],3368.778731+result['elapsed_seconds'])
        self.assertFalse(record['cumulative_seconds_provisional']);self.assertFalse(record['lockable'])
        duplicate=subprocess.run(settlement_args,env=self.env,cwd=self.r,capture_output=True,text=True,timeout=30)
        self.assertNotEqual(duplicate.returncode,0);self.assertIn('FileExistsError',duplicate.stderr)


def load_tests(loader,tests,pattern):
    # Chain is a fixture factory adapted from the separately tested carryover
    # suite; its inherited assertions refer to that original suite's labels.
    return loader.loadTestsFromTestCase(RecoveryEntryIntegrationTests)

if __name__=='__main__':unittest.main()
