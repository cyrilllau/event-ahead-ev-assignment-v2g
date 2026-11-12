"""Real two-worker publication inside the original queue sealing allocation."""
from copy import deepcopy
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import save_npz
from evcs_experiments import campaign_batch, recourse_store
from evcs_experiments.campaign_batch import file_reference, run_batch
from evcs_experiments.config import get_paths
from evcs_experiments.evaluation_lock import PROTOCOL_REFERENCE_KINDS, create_protocol_lock, lock_plan_batch, verify_plan_batch
from evcs_experiments.recourse import assemble_recourse
from evcs_experiments.validation.s03_toys import build_golden_cases
from evcs_experiments.core_request_adapter import _development_context, build_core_worker_request, build_core_queue
from tests.test_core_production_wiring import WORKER

POLICY='IN_ATTEMPT_FINAL_SEAL_V1'


def build_queue(root, policy=POLICY):
    project=get_paths().project_root
    def write(relative,value):
        p=root/relative
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
        return file_reference(p)
    worker=root/'fixture_worker.py';worker.write_text(WORKER)
    budget=dict(cumulative_wall_seconds=30.,single_master_seconds=3.,
                final_certificate_reserve_seconds=3.,shutdown_reserve_seconds=.1)
    config={'planning_budget':budget}
    configref=write('config.json',config)
    budgetref=write('budget.json',budget)
    # Freeze source before execution; the same closure is used by both workers.
    sources=[file_reference(p) for p in sorted((project/'src/evcs_experiments').rglob('*.py'))]
    goldens=build_golden_cases()[:2]
    identities=[dict(plan_id='mixed.'+method,target_id=f.name,
                     method_id=method,setting_id='baseline')
                for f,method in zip(goldens,('EA_DRO','SAA'))]
    protocol=dict(domain='development',model_version='IEEE33-v54-Q1',
        target_ids=[f.name for f in goldens],method_ids=['EA_DRO','SAA'],
        configuration=config,
        references={k:[configref] for k in PROTOCOL_REFERENCE_KINDS},
        batches=[dict(batch_id='mixed',expected_plans=identities,
            sample_ids_by_target={f.name:[f.name+'/not-opened'] for f in goldens})])
    protocol['references']['source']=sources
    protocol['references']['fixture']=[file_reference(worker)]
    protocolref=create_protocol_lock(root/'locks',protocol)
    context=_development_context(protocol_reference=protocolref,config_reference=configref,
                                 planning_source_references=sources)
    items=[];requests=[];input_refs=[]
    helper=file_reference(project/'scripts/audits/audit_s05_pending_recovery.py')
    numerical=dict(schema_version=1,postprocessing_reserve_seconds=.8,
        sources=dict(helper=helper,numerical_kernel=file_reference(project/'scripts/audits/audit_s05_candidate_recourse.py')))
    for f,identity in zip(goldens,identities):
        fixture=write(f.name+'/inputs.json',dict(task_sha256=f.tasks.task_sha256,
            config_hash=f.tasks.config_hash,fixture=f.name,task_ids=list(f.tasks.task_ids),
            scenario_ids=list(f.scenario_ids),scenario_sha256s=[s.scenario_sha256 for s in f.scenarios]))
        refs={k:fixture for k in ('tasks','routes','initial_state','planning_support')}
        input_refs.append(refs)
        options=None
        if identity['method_id']=='SAA':
            templates=[]
            for i,scene in enumerate(f.scenarios):
                t=assemble_recourse(f.case,f.tasks,scene)
                folder=root/f.name/f'template_{i}';folder.mkdir()
                for name in recourse_store.MATRICES:
                    save_npz(folder/(name+'.npz'),getattr(t,name))
                np.savez_compressed(folder/'arrays.npz',**{name:getattr(t,name) for name in recourse_store.ARRAYS})
                (folder/'identity.json').write_text(json.dumps(t.as_dict(),indent=2)+'\n')
                record=dict(kind='recourse_template',scenario_index=i,scenario_id=f.scenario_ids[i],
                            template_sha256=t.template_sha256,
                            artifacts=[file_reference(p) for p in sorted(folder.iterdir())])
                templates.append(write(str(folder.relative_to(root)/'record.json'),record))
            options=dict(model_input_mode='assemble',max_coefficient_bytes=50_000_000,
                         helper_reference=helper,template_references=templates)
        request=build_core_worker_request(context,identity=identity,canonical_batch_id='mixed',
            plan_input_references=refs,bank_acceptance_reference=fixture,summary_reference=fixture,
            audit_reference=fixture,output_root=str(root/('output_'+identity['method_id'])),
            numerical_audit=numerical if options is None else None,saa_options=options)
        if policy is not None:request['canonical_export_policy']=policy
        rr=write(identity['method_id']+'.json',request);requests.append(rr)
        items.append(dict(item_id=identity['method_id'],identity=identity,request_reference=rr))
    request=build_core_queue(context,batch_id='mixed',queue_root=str(root/'queue'),
        job_wall_seconds=60.,budget_reference=budgetref,items=items,executable=sys.executable,
        cwd=str(project),_development_argv=[sys.executable,str(worker)])
    qr=write('queue.json',request)
    return qr, identities, protocolref


class CanonicalInAttemptTests(unittest.TestCase):
    @contextmanager
    def workspace(self):
        retained=os.environ.get('EVCS_P2_TEST_ARTIFACTS_ROOT')
        if retained:
            root=Path(retained).resolve()/self._testMethodName
            root.mkdir(parents=True,exist_ok=False)
            yield root
        else:
            with tempfile.TemporaryDirectory() as temp:yield Path(temp).resolve()

    def read(self,path):return json.loads(Path(path).read_text())

    def test_mixed_queue_publishes_before_seal_deadline_and_reuses(self):
        from evcs_experiments import canonical_in_attempt as publication
        observed=[]
        original=publication.prepare_publication
        def timed_prepare(*args,**kwargs):
            started=time.monotonic()
            value=original(*args,**kwargs)
            observed.append((started,time.monotonic()))
            return value
        with self.workspace() as root:
            qr,identities,protocolref=build_queue(root)
            with patch.object(publication,'prepare_publication',side_effect=timed_prepare):
                result=run_batch(qr['path'],root/'queue')
            self.assertEqual(result['status'],'COMPLETE_BATCH',result)
            self.assertEqual(len(observed),2)
            state=self.read(root/'queue'/'state.json')
            receipts=state['canonical_publication_receipts']
            self.assertEqual(len(receipts),2)
            plans=[]
            for identity,ref,observed_times,expected in zip(identities,receipts,observed,(32.12,39.)):
                completion=self.read(ref['path'])
                attempt=root/'queue'/(identity['method_id']+'__attempt_0001')
                seal=self.read(attempt/'finalization.json')
                invocation=self.read(attempt/'invocation.json')
                numeric=self.read(seal['references']['solver_record']['path'])
                self.assertGreaterEqual(numeric['actual_attempt_seconds_at_cutoff'],observed_times[1]-invocation['attempt_started_monotonic'])
                self.assertGreaterEqual(numeric['post_receipt_administration_seconds_at_cutoff'],observed_times[1]-observed_times[0])
                self.assertEqual(completion['charged_attempt_seconds'],seal['charged_attempt_seconds'])
                self.assertLessEqual(completion['actual_attempt_seconds_before_confirmation_write'],completion['charged_attempt_seconds'])
                self.assertAlmostEqual(numeric['upper_bound_usd'],expected,places=7)
                record=self.read(completion['canonical_record_reference']['path'])
                self.assertEqual(record['references']['budget_seal'],file_reference(attempt/'finalization.json'))
                self.assertEqual(record['references']['solver_record'],seal['references']['solver_record'])
                record['canonical_record_reference']=completion['canonical_record_reference']
                record['publication_completion_reference']=ref
                verified=publication.verify_publication(ref,receipt_reference=numeric['receipt_reference'],
                    finalization_reference=file_reference(attempt/'finalization.json'),worker_request_reference=numeric['request_reference'])
                self.assertEqual(verified,completion)
                plans.append(record)
            protocol=self.read(protocolref['path'])['payload']
            for i,missing in enumerate(('canonical_record_reference','publication_completion_reference')):
                bad=deepcopy(plans);bad[0].pop(missing)
                lockroot=root/('incomplete_lock_'+str(i));create_protocol_lock(lockroot,protocol)
                with self.assertRaisesRegex(ValueError,'both original record and completion'):
                    lock_plan_batch(lockroot,'mixed',bad)
                self.assertFalse((lockroot/'batch-mixed.json').exists())
            changed=deepcopy(plans);changed[0]['canonical_record_reference']['sha256']='f'*64
            lockroot=root/'changed_hash_lock';create_protocol_lock(lockroot,protocol)
            with self.assertRaisesRegex(ValueError,'Evidence bytes'):
                lock_plan_batch(lockroot,'mixed',changed)
            self.assertFalse((lockroot/'batch-mixed.json').exists())
            lock_plan_batch(root/'locks','mixed',plans)
            self.assertEqual(verify_plan_batch(root/'locks','mixed')['batch']['plans'],plans)
            self.assertFalse((root/'locks'/'access-mixed.json').exists())
            with patch.object(campaign_batch.subprocess,'Popen',side_effect=AssertionError('No completed work replay')):
                reused=run_batch(qr['path'],root/'queue')
            self.assertEqual(reused['new_attempt_item_ids'],[])
            self.assertEqual(reused['cumulative_plan_seconds'],result['cumulative_plan_seconds'])
            (root/'test_evidence_summary.json').write_text(json.dumps(dict(status='PASS_DEVELOPMENT_CANONICAL_IN_ATTEMPT',result=result,reused=reused,completion_references=receipts,protocol_reference=protocolref,formal_OOS=False),indent=2))

    def test_missing_completion_tampered_record_and_removed_history_refused(self):
        with self.workspace() as root:
            qr,_,_=build_queue(root)
            result=run_batch(qr['path'],root/'queue')
            self.assertEqual(result['status'],'COMPLETE_BATCH',result)
            statepath=root/'queue'/'state.json';state=self.read(statepath)
            cref=state['canonical_publication_receipts'][0]
            cpath=Path(cref['path']);original=cpath.read_bytes()
            recordpath=Path(self.read(cpath)['canonical_record_reference']['path'])
            recordbytes=recordpath.read_bytes()
            with patch.object(campaign_batch.subprocess,'Popen',side_effect=AssertionError('Invalid history must stop')):
                cpath.unlink()
                with self.assertRaises((ValueError,FileNotFoundError)):run_batch(qr['path'],root/'queue')
                cpath.write_bytes(original)
                record=self.read(recordpath);record['plan_sha256']='f'*64;recordpath.write_text(json.dumps(record))
                with self.assertRaises(ValueError):run_batch(qr['path'],root/'queue')
                recordpath.write_bytes(recordbytes)
                originalstate=statepath.read_bytes();state['canonical_publication_receipts']=[]
                statepath.write_text(json.dumps(state))
                with self.assertRaises(ValueError):run_batch(qr['path'],root/'queue')
                statepath.write_bytes(originalstate)
                self.assertEqual(run_batch(qr['path'],root/'queue')['status'],'COMPLETE_BATCH')

    def test_publication_after_charged_deadline_halts_without_acceptance(self):
        from evcs_experiments import canonical_in_attempt as publication
        original=publication.complete_publication
        def expired(*args,**kwargs):
            exhausted=kwargs['attempt_started']+kwargs['charged_attempt_seconds']+.001
            with patch.object(publication.time,'monotonic',return_value=exhausted):
                return original(*args,**kwargs)
        with self.workspace() as root:
            qr,_,_=build_queue(root)
            with patch.object(publication,'complete_publication',side_effect=expired):
                result=run_batch(qr['path'],root/'queue')
            self.assertEqual(result['status'],'FAILED_VALIDITY_OR_WORKER',result)
            self.assertTrue(result['unsettled_plan_seconds'])
            self.assertIn('Canonical publication exceeded original seal allocation',result['reason'])
            state=self.read(root/'queue'/'state.json')
            self.assertTrue(state['halted'])
            self.assertEqual(state.get('canonical_publication_receipts',[]),[])
            self.assertFalse(list((root/'queue').glob('*/canonical_record/completion.json')))
            with self.assertRaises(RuntimeError):run_batch(qr['path'],root/'queue')

    def test_late_state_commit_preserves_failure_and_refuses_completion(self):
        from evcs_experiments import canonical_in_attempt as publication
        real_clock=time.monotonic
        real_write=campaign_batch.atomic_json
        forced=[None]
        def clock():return real_clock() if forced[0] is None else forced[0]
        def late_commit(path,value):
            real_write(path,value)
            if value.get('canonical_publication_receipts') and value.get('active') is None and not value.get('halted'):
                cref=value['canonical_publication_receipts'][-1]
                completion=self.read(cref['path'])
                receipt=self.read(completion['receipt_reference']['path'])
                invocation=self.read(receipt['invocation_reference']['path'])
                forced[0]=invocation['attempt_started_monotonic']+completion['charged_attempt_seconds']+.001
        with self.workspace() as root:
            qr,_,_=build_queue(root)
            with patch.object(campaign_batch,'atomic_json',side_effect=late_commit),patch.object(campaign_batch.time,'monotonic',side_effect=clock):
                result=run_batch(qr['path'],root/'queue')
            self.assertEqual(result['status'],'FAILED_VALIDITY_OR_WORKER',result)
            self.assertIn('Canonical publication state commit exceeded original seal allocation',result['reason'])
            state=self.read(root/'queue'/'state.json')
            self.assertTrue(state['halted'])
            self.assertEqual(len(state['canonical_publication_receipts']),1)
            cref=state['canonical_publication_receipts'][0];completion=self.read(cref['path'])
            self.assertTrue((Path(completion['receipt_reference']['path']).parent/'failure.json').is_file())
            with self.assertRaisesRegex(ValueError,'unresolved failure'):
                publication.verify_publication(cref,receipt_reference=completion['receipt_reference'],
                    finalization_reference=completion['finalization_reference'],worker_request_reference=completion['worker_request_reference'])
            with self.assertRaises(RuntimeError):run_batch(qr['path'],root/'queue')

    def test_legacy_without_policy_has_no_automatic_publication(self):
        with self.workspace() as root:
            qr,_,_=build_queue(root,policy=None)
            result=run_batch(qr['path'],root/'queue')
            self.assertEqual(result['status'],'COMPLETE_BATCH',result)
            state=self.read(root/'queue'/'state.json')
            self.assertEqual(state.get('canonical_publication_receipts',[]),[])
            self.assertFalse(list((root/'queue').glob('*/canonical_record')))
            self.assertEqual(run_batch(qr['path'],root/'queue')['cumulative_plan_seconds'],result['cumulative_plan_seconds'])


if __name__=='__main__':unittest.main()
