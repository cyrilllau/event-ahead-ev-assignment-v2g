"""Two physical goldens traverse the real mixed queue, seals and plan lock.

The fixture replaces only physical preparation and development resource limits;
master, recourse, numerical audit, queue accounting, exporter and lock are real.
It never calls an OOS access function or changes a production domain.
"""
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import save_npz

from evcs_experiments.campaign_batch import file_reference, run_batch
from evcs_experiments.config import get_paths
from evcs_experiments.evaluation_lock import (
    PROTOCOL_REFERENCE_KINDS, create_protocol_lock, lock_plan_batch, verify_plan_batch,
)
from evcs_experiments.recourse import assemble_recourse
from evcs_experiments import recourse_store
from evcs_experiments.validation.s03_toys import build_golden_cases


WORKER = r'''
import json, os
from pathlib import Path
from types import SimpleNamespace
from evcs_experiments.campaign_batch import file_reference
from evcs_experiments.formal_planning_worker import run_attempt as run_generic, _DevelopmentFixture as GenericFixture
from evcs_experiments.saa_planning_worker import run_attempt as run_saa, _DevelopmentFixture as SAAFixture
from evcs_experiments.formal_planning import RawPlanningEvidence
from evcs_experiments.ccg import CCGProblem
from evcs_experiments.recourse import assemble_recourse
from evcs_experiments.lazy_recourses import LazyRecourses
from evcs_experiments.validation.s03_toys import build_golden_cases
invref=file_reference(os.environ['EVCS_BATCH_ATTEMPT_REQUEST'])
inv=json.loads(Path(invref['path']).read_text())
req=json.loads(Path(inv['item_request_reference']['path']).read_text())
fixtures=build_golden_cases()
f=next(f for f in fixtures if f.name==req['target_id'])
templates=tuple(assemble_recourse(f.case,f.tasks,s) for s in f.scenarios)
first=templates[0].first_stage
binding={'fixture':f.name}
if req['method_id']=='SAA':
    class Recourses(tuple): pass
    rec=Recourses(templates)
    rec.scenario_ids=f.scenario_ids
    rec.scenario_sha256s=tuple(s.scenario_sha256 for s in f.scenarios)
    prepared=SimpleNamespace(physical=SimpleNamespace(first_stage=first,case=f.case,tasks=f.tasks,bank=SimpleNamespace(load=f.scenarios.__getitem__)),recourses=rec,input_binding=binding,empirical_probabilities=f.empirical_probabilities,scenario_cost_constants_usd=f.scenario_cost_constants_usd)
    fixture=SAAFixture(prepare=lambda **kw:prepared,budget=req['budget'],master_seconds=3.,audit_reserve_seconds=3.,seal_reserve_seconds=.1)
    # Cooperative LP cutoff injection exercises real partial-Q files and queue resume.
    # Every recorded solve is genuine; only the second attempt's acceptance is withheld.
    from dataclasses import replace
    from types import MappingProxyType
    from unittest.mock import patch
    from evcs_experiments.recourse import solve_fixed_plan
    interrupt_saa = False
    if interrupt_saa:
        trace=Path(req['output_root']).parent/'test_saa_lp_calls.jsonl'
        calls=[]
        def solve(template,plan,**kw):
            calls.append(template.scenario_sha256)
            with trace.open('a') as stream:
                stream.write(json.dumps({'scenario_sha256':template.scenario_sha256,'resumed':inv['previous_checkpoint'] is not None})+'\n')
            raw=solve_fixed_plan(template,plan,**kw)
            if inv['previous_checkpoint'] is None and len(calls)==2:
                return replace(raw,status='LIMIT_REACHED',solver_status=1,optimal=False,solution=None,objective_usd=None,
                               certificate=MappingProxyType({'optimal_certificate_valid':False}))
            return raw
        with patch('evcs_experiments.saa_fixed_plan.solve_fixed_plan',side_effect=solve):
            if inv['previous_checkpoint'] is not None:
                with patch('evcs_experiments.saa_streaming_solve.solve_streaming_saa',side_effect=AssertionError('Saved SAA master must not repeat')):
                    result=run_saa(invref,_development_fixture=fixture)
            else:
                result=run_saa(invref,_development_fixture=fixture)
    else:
        result=run_saa(invref,_development_fixture=fixture)
else:
    def prepare(**kw):
        cache=LazyRecourses(f.scenario_ids,[s.scenario_sha256 for s in f.scenarios],first.matrix_sha256,binding,templates.__getitem__,capacity=2)
        problem=CCGProblem(first,cache,f.probability_set,f.scenario_cost_constants_usd,'DRO')
        evidence=RawPlanningEvidence(kw['evidence_root'],f.case,f.tasks,problem,f.scenarios.__getitem__,domain='development',input_binding=binding,resume=kw['resume_evidence'],resume_snapshot_reference=kw['resume_snapshot_reference'])
        return SimpleNamespace(problem=problem,evidence=evidence,sink_factory=evidence.sink_factory,input_binding=binding)
    fixture=GenericFixture(prepare,req['budget'],dict(master_backend='highs',optimizer_threads=1,oracle_time_limit_seconds=2.,mip_relative_gap=1e-9,optimality_relative_gap=1e-8),.05)
    result=run_generic(invref,_development_fixture=fixture)
print(json.dumps({'status':result['status'],'method':req['method_id']}))
'''


class CoreProductionWiringTests(unittest.TestCase):
    def test_mixed_physical_queue_exports_locks_and_reuses_without_solving(self):
        self._run_mixed(interrupt_saa=False)

    def test_mixed_partial_saa_resume_preserves_ea_master_q_and_budget(self):
        self._run_mixed(interrupt_saa=True)

    def _run_mixed(self, *, interrupt_saa):
        from evcs_experiments.core_request_adapter import (
            _development_context, build_core_worker_request, build_core_queue,
        )
        from evcs_experiments.sealed_plan_record import prepare_sealed_plan_record

        retained=os.environ.get('EVCS_P1_TEST_ARTIFACTS_ROOT')
        if retained:
            destination=Path(retained).resolve()/self._testMethodName
            destination.mkdir(parents=True,exist_ok=False)
            workspace=nullcontext(str(destination))
        else:
            workspace=tempfile.TemporaryDirectory()
        with workspace as temp:
            root=Path(temp).resolve()
            project=get_paths().project_root
            def write(relative,value):
                p=root/relative
                p.parent.mkdir(parents=True,exist_ok=True)
                p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
                return file_reference(p)
            worker=root/'fixture_worker.py';worker.write_text(WORKER.replace('interrupt_saa = False','interrupt_saa = '+str(interrupt_saa)))
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
                rr=write(identity['method_id']+'.json',request);requests.append(rr)
                items.append(dict(item_id=identity['method_id'],identity=identity,request_reference=rr))
            request=build_core_queue(context,batch_id='mixed',queue_root=str(root/'queue'),
                job_wall_seconds=60.,budget_reference=budgetref,items=items,executable=sys.executable,
                cwd=str(project),_development_argv=[sys.executable,str(worker)])
            qr=write('queue.json',request)
            result=run_batch(qr['path'],root/'queue')
            first_result=result
            if interrupt_saa:
                self.assertEqual(result['status'],'TIME_LIMIT',result)
                self.assertEqual(result['completed_item_ids'],['EA_DRO'])
                self.assertEqual(result['pending_item_ids'],['SAA'])
                old_attempt=root/'queue'/'SAA__attempt_0001'
                first_checkpoint=file_reference(old_attempt/'worker_checkpoint.json')
                old_files={str(p):file_reference(p) for p in old_attempt.rglob('*') if p.is_file()}
                old_ea=file_reference(root/'queue'/'EA_DRO__attempt_0001'/'finalization.json')
                result=run_batch(qr['path'],root/'queue')
                self.assertEqual(result['new_attempt_item_ids'],['SAA'])
                self.assertEqual(result['reused_item_ids'],['EA_DRO'])
                self.assertEqual(result['cumulative_plan_seconds']['mixed.EA_DRO'],first_result['cumulative_plan_seconds']['mixed.EA_DRO'])
                self.assertGreater(result['cumulative_plan_seconds']['mixed.SAA'],first_result['cumulative_plan_seconds']['mixed.SAA'])
                self.assertEqual(file_reference(old_ea['path']),old_ea)
                self.assertEqual(file_reference(first_checkpoint['path']),first_checkpoint)
                for p,ref in old_files.items():self.assertEqual(file_reference(p),ref)
                calls=[json.loads(line) for line in (root/'test_saa_lp_calls.jsonl').read_text().splitlines()]
                first_sha=goldens[1].scenarios[0].scenario_sha256
                self.assertEqual(sum(row['scenario_sha256']==first_sha for row in calls),1)
                self.assertEqual(sum(row['resumed'] for row in calls),3)
                self.assertEqual(len(calls),5)
            diagnostics={str(p.relative_to(root)):p.read_text()[-5000:] for p in (root/'queue').rglob('*')
                         if p.is_file() and ('stderr' in p.name or p.name=='worker_failure.json')}
            self.assertEqual(result['status'],'COMPLETE_BATCH',(result,diagnostics))
            self.assertEqual(result['completed_item_ids'],['EA_DRO','SAA'])
            plans=[]
            for f,identity,rr,inputs,objective,selected in zip(goldens,identities,requests,input_refs,(32.12,39.),(3,1)):
                attempt=root/'queue'/(identity['method_id']+('__attempt_0002' if interrupt_saa and identity['method_id']=='SAA' else '__attempt_0001'))
                sealref=file_reference(attempt/'finalization.json')
                seal=json.loads(Path(sealref['path']).read_text())
                numeric=json.loads(Path(seal['references']['solver_record']['path']).read_text())
                self.assertTrue(numeric['full_support_verified'])
                self.assertTrue(numeric['final_budget_acceptance'])
                self.assertFalse(numeric['formal_OOS'])
                self.assertAlmostEqual(numeric['upper_bound_usd'],objective,places=7)
                self.assertLessEqual(seal['actual_attempt_seconds_before_confirmation_write'],seal['charged_attempt_seconds'])
                export_request=dict(kind='SEALED_PLAN_RECORD_REQUEST_V1',identity=identity,batch_id='mixed',
                    references=dict(protocol=protocolref,worker_request=rr,budget_seal=sealref),
                    plan_input_references=inputs,planning_source=sources)
                published=prepare_sealed_plan_record(export_request,output=root/'records'/identity['method_id'])
                record=json.loads(Path(published['plan_record_reference']['path']).read_text())
                plan=json.loads(Path(record['references']['plan']['path']).read_text())
                self.assertEqual(sum(plan['x']),selected)
                self.assertEqual(plan['task_sha256'],f.tasks.task_sha256)
                self.assertEqual(record['references']['solver_record'],seal['references']['solver_record'])
                self.assertEqual(record['references']['budget'],seal['references']['budget'])
                self.assertEqual(record['references']['budget_seal'],sealref)
                audit=json.loads(Path(numeric['numerical_audit_reference']['path']).read_text())
                if identity['method_id']=='SAA':
                    joint=json.loads(Path(audit['candidate_audit_reference']['path']).read_text())
                    self.assertEqual(joint['fixed_q_audit']['support_ids'],list(f.scenario_ids))
                    self.assertEqual(joint['risk_certificate_kind'],'fixed_empirical')
                else:
                    rec=json.loads(Path(audit['recourse_audit_reference']['path']).read_text())
                    self.assertEqual(rec['support_ids'],list(f.scenario_ids))
                plans.append(record)
            self.assertNotEqual(plans[0]['plan_sha256'],plans[1]['plan_sha256'])
            lock_plan_batch(root/'locks','mixed',plans)
            checked=verify_plan_batch(root/'locks','mixed')
            self.assertEqual(checked['batch']['plans'],plans)
            self.assertFalse((root/'locks'/'access-mixed.json').exists())
            # The second queue call must validate existing evidence without launching either worker.
            before={str(p):file_reference(p) for p in (root/'queue').rglob('*') if p.is_file() and p.name!='.batch.lock' and p.name!='state.json'}
            with patch('evcs_experiments.campaign_batch.subprocess.Popen',side_effect=AssertionError('Completed plans must not be rerun')):
                reused=run_batch(qr['path'],root/'queue')
            self.assertEqual(reused['new_attempt_item_ids'],[])
            self.assertEqual(reused['cumulative_plan_seconds'],result['cumulative_plan_seconds'])
            for p,reference in before.items():self.assertEqual(file_reference(p),reference)
            self.assertEqual(verify_plan_batch(root/'locks','mixed')['batch']['plans'],plans)
            self.assertFalse((root/'locks'/'access-mixed.json').exists())
            write('test_evidence_summary.json',dict(status='PASS_DEVELOPMENT_MIXED_CORE_WIRING',
                first_queue_result=first_result,complete_queue_result=result,reuse_result=reused,
                protocol_reference=protocolref,worker_fixture_reference=file_reference(worker),
                planning_source_references=sources,plan_records=plans,
                batch_lock_reference=file_reference(root/'locks'/'batch-mixed.json'),
                interruption_injected=interrupt_saa,formal_OOS=False,scientific_spec_pass=False))


if __name__=='__main__':
    unittest.main()
