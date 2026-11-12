"""Real small native solve -> same-plan full Q/all-master joint audit and pins."""
from copy import deepcopy
import json
import os
from pathlib import Path
import time
import unittest
from unittest.mock import patch
import numpy as np
from evcs_experiments import saa_l_shaped as kernel
from evcs_experiments import saa_l_shaped_candidate as candidate
from evcs_experiments import saa_l_shaped_master_evidence as masters
from evcs_experiments import saa_fixed_plan_audit as full_q
from evcs_experiments.campaign_batch import file_reference
import test_saa_l_shaped_audit_bridge as fixtures


class LShapedCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = fixtures.LShapedAuditBridgeTests('test_real_kernel_events_original_audit_same_plan_negative_constants_once')
        cls.fixture.setUp(); cls.addClassCleanup(cls.fixture.doCleanups)
        f=cls.fixture; cls.root=f.f.root.resolve(); cls.problem=f.problem
        cls.sources=list({r['path']:r for r in [file_reference(p) for p in (candidate.__file__,kernel.__file__,masters.__file__,full_q.__file__)]
            +f.args['source_references']+f.f.args['source_references']+list(masters._sources().values())+[f.f.args['helper_reference']]}.values())
        cls.ms=masters.DevelopmentMasterEvidence(cls.root/'masters'); path=cls.root/'kernel.json'
        cls.partial=kernel.run_saa_l_shaped(cls.problem,checkpoint_path=path,evidence_sink=f.sink,master_sink=cls.ms,max_iterations=1)
        cls.partial_cp=cls.root/'partial_kernel_copy.json';cls.partial_cp.write_bytes(path.read_bytes())
        cls.partial_pair=cls.build(cls.partial,cls.partial_cp,'partial')
        old_refs=[m['evidence_reference'] for m in cls.partial['master_history']]
        old_pin=file_reference(path)
        # A fresh exporter contains only new records. The final candidate must
        # instead collect old + new execution references from original history.
        cls.new_ms=masters.DevelopmentMasterEvidence(cls.root/'masters')
        cls.final=kernel.run_saa_l_shaped(cls.problem,checkpoint_path=path,resume=True,
            expected_checkpoint_sha256=old_pin['sha256'],evidence_sink=f.sink,master_sink=cls.new_ms)
        cls.final_pair=cls.build(cls.final,path,'final');cls.old_refs=old_refs
        cls.report=dict(partial=cls.partial_pair[2],final=cls.final_pair[2],old_master_count=len(old_refs),
            new_master_count=len(cls.new_ms.records),formal_admission=False)
        candidate._write(cls.root/'candidate_test_summary.json',cls.report)

    @classmethod
    def build(cls,state,kernel_path,name):
        f=cls.fixture;best=cls.problem.plan(state['best']['anchor'])
        pd=dict(x=best.x.tolist(),r=best.r.tolist(),task_sha256=best.task_sha256,config_hash=best.config_hash)
        digest=candidate.canonical_hash(pd)
        ix=next(i for i,s in f.sink.sweeps.items() if s['plan_sha256']==digest and len(s['q'])==3)
        cp=f.sink._checkpoints.get(ix) or f.sink.checkpoint(ix)
        qr=json.loads(Path(cp['path']).read_text())['request_reference']
        env=full_q.bind_saa_template_envelope(request_reference=qr,template_references=f.f.templates,
            first_stage_sha256=cls.problem.first_stage.matrix_sha256,source_references=f.args['source_references'],
            output_path=cls.root/(name+'_env.json'))
        evidence=dict(kernel_checkpoint_reference=file_reference(kernel_path),
            master_execution_references=[m['evidence_reference'] for m in state['master_history']],
            full_q_checkpoint_reference=cp,full_q_request_reference=qr,template_envelope_reference=env,source_references=cls.sources)
        ref=candidate.write_development_candidate(cls.problem,plan_id='development.SAA.'+name,
            **evidence,output_path=cls.root/(name+'_candidate.json'))
        with (patch.object(kernel,'milp',side_effect=AssertionError('No new native solve')),
              patch.object(kernel,'solve_fixed_plan',side_effect=AssertionError('No new recourse'))):
            checked=candidate.audit_development_candidate(cls.problem,candidate_reference=ref,
                expected_plan_id='development.SAA.'+name,expected_evidence=evidence,
                case=cls.problem.case,tasks=cls.problem.tasks,scene_loader=f.args['scene_loader'],
                helper_reference=f.f.args['helper_reference'],full_q_source_references=f.f.args['source_references'],
                deadline=time.monotonic()+30,output_path=cls.root/(name+'_joint.json'))
        return ref,evidence,checked

    def metadata(self,pair=None,**changes):
        ref,evidence,joint=pair or self.final_pair
        inner=json.loads(Path(ref['path']).read_text())
        request=dict(domain='development',method_id='SAA',model_input_mode='l_shaped_development',
            algorithm_id=kernel.ALGORITHM_ID,input_binding=inner['input_binding'],plan_id=inner['plan_id'],
            planning_source_references=self.sources)
        record=dict(candidate=inner,plan_id=inner['plan_id'],l_shaped_evidence=evidence,
            driver_status=inner['driver_status'],raw_engine_status=inner['driver_status'],
            method_id='SAA',algorithm_id=kernel.ALGORITHM_ID)
        if 'request' in changes:request.update(changes['request'])
        if 'record' in changes:record.update(changes['record'])
        return candidate.validate_development_joint(changes.get('joint',joint),request,record)

    def test_genuine_two_session_convergence_and_same_plan_original_audits(self):
        self.assertEqual(self.final['status'],'CONVERGED')
        result=self.metadata();joint=self.final_pair[2]
        self.assertAlmostEqual(result['upper_bound_usd'],-69.73333333333333,places=7)
        self.assertAlmostEqual(result['lower_bound_usd'],result['upper_bound_usd'],places=7)
        self.assertEqual(len(self.final['sessions']),2)
        self.assertGreater(len(self.final['master_history']),len(self.new_ms.records))
        self.assertTrue(joint['passed']);self.assertTrue(joint['convergence_claim'])
        self.assertEqual(joint['solver_calls'],0);self.assertFalse(joint['budget_settled'])
        self.assertFalse(joint['formal_admission']);self.assertFalse(joint['scientific_acceptance'])

    def test_real_iteration_limit_is_not_native_master_optimal_or_time_limit(self):
        result=self.metadata(self.partial_pair)
        self.assertEqual(result['raw_engine_status'],'ITERATION_LIMIT')
        self.assertEqual(result['native_master_status'],'OPTIMAL')
        self.assertFalse(self.partial_pair[2]['convergence_claim'])
        self.assertEqual(result['lower_bound_usd'],-98)
        self.assertEqual(result['upper_bound_usd'],-50)
        with self.assertRaisesRegex(ValueError,'stop'):
            self.metadata(self.partial_pair,record={'raw_engine_status':'TIME_LIMIT'})

    def test_omitted_old_master_wrong_order_or_foreign_pin_rejected(self):
        ref,evidence,_=self.final_pair
        for refs in (self.new_ms.records,list(reversed(evidence['master_execution_references'])),[]):
            bad={**evidence,'master_execution_references':refs}
            with self.assertRaises(ValueError):
                candidate.write_development_candidate(self.problem,plan_id='different',**bad,output_path=self.root/'never.json')
        altered=deepcopy(ref);altered['sha256']='0'*64
        with self.assertRaises(ValueError):candidate._read(altered)

    def test_cross_plan_full_q_is_rejected_even_if_every_individual_q_is_valid(self):
        final=self.final_pair[1];partial=self.partial_pair[1]
        self.assertNotEqual(self.partial_pair[2]['plan_sha256'],self.final_pair[2]['plan_sha256'])
        bad={**final,**{k:partial[k] for k in ('full_q_checkpoint_reference','full_q_request_reference','template_envelope_reference')}}
        with self.assertRaisesRegex(ValueError,'another best plan'):
            candidate.write_development_candidate(self.problem,plan_id='foreign',**bad,output_path=self.root/'never.json')

    def test_outer_method_algorithm_physical_sources_and_formal_scope_rejected(self):
        changes=({'request':{'domain':'formal'}},{'request':{'input_binding':{'foreign':True}}},
            {'request':{'planning_source_references':[]}},{'record':{'algorithm_id':'CCG'}},{'record':{'method_id':'ARO'}})
        for delta in changes:
            with self.subTest(delta=delta),self.assertRaises(ValueError):self.metadata(**delta)

    def test_joint_bound_summary_source_and_expected_pin_mutations_rejected(self):
        joint=self.final_pair[2]
        for mode in ('lb','pin','master','source'):
            changed=deepcopy(joint)
            if mode=='lb':changed['lower_bound_usd']-=1
            elif mode=='pin':changed['expected_evidence']['kernel_checkpoint_reference']['sha256']='f'*64
            elif mode=='master':changed['master_audits'].pop(0)
            else:changed['full_q_source_references']=[]
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.metadata(joint=changed)

    def test_deadline_and_existing_output_refused(self):
        ref,evidence,_=self.final_pair;f=self.fixture
        with self.assertRaises(TimeoutError):
            candidate.audit_development_candidate(self.problem,candidate_reference=ref,expected_plan_id='irrelevant',
                expected_evidence=evidence,case=self.problem.case,tasks=self.problem.tasks,scene_loader=f.args['scene_loader'],
                helper_reference=f.f.args['helper_reference'],full_q_source_references=f.f.args['source_references'],deadline=time.monotonic()-1)
        with self.assertRaises(FileExistsError):candidate._write(Path(ref['path']),{})

if __name__=='__main__':unittest.main()
