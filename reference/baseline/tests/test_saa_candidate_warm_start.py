import json
import tempfile
import time
import unittest
from pathlib import Path
import numpy as np
from evcs_experiments import campaign_batch as b
from evcs_experiments.saa_candidate_warm_start import build_candidate_warm_start, map_primal

class CandidateWarmStartTests(unittest.TestCase):
    def fixture(self, root):
        def write(name, value):
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return b.file_reference(p)
        plan=write('plan.json',dict(x=[1],r=[[2]],task_sha256='task',config_hash='config'))
        events=[];scenes=[];ids=[f'scene{i}' for i in range(160)]
        for i,sid in enumerate(ids):
            d=root/str(i);d.mkdir()
            np.savez(d/'primal_vector.npz',vector=np.array([3.,4.,5.]))
            np.save(d/'mapping.npy',np.array([2,0]))
            tr=write(f'{i}/template.json',dict(kind='recourse_template',scenario_index=i,scenario_id=sid,template_sha256=f't{i}'))
            ptr=write(f'{i}/template_reference.json',tr)
            ev=write(f'{i}/event.json',dict(event='recourse',domain='formal',scenario_index=i,scenario_id=sid,plan_sha256='plan',optimal_certificate_valid=True,artifacts=[ptr,plan,b.file_reference(d/'primal_vector.npz')]))
            events.append(ev);scenes.append(dict(index=i,scenario_id=sid,column_slice=[2+2*i,4+2*i],template_sha256=f't{i}',mapping_reference=b.file_reference(d/'mapping.npy'),original_recourse_columns=3))
        snapshot=write('snapshot.json',dict(events=events,formal_OOS=False))
        cp=write('cp.json',dict(support_ids=ids,completed_count=160,next_index=160,plan_reference=plan,plan_sha256='plan',recourse_event_references=events))
        audit=write('audit.json',dict(passed=True,status='PASS_INDEPENDENT_CANDIDATE_160_RECOURSE_AUDIT',checkpoint_reference=cp,raw_snapshot_reference=snapshot,support_ids=ids,complete_lp_count=160,plan_sha256='plan',original_first_stage_sha256='first'))
        review=write('review.json',dict(passed=True,status='PASS_INDEPENDENT_FULL_SUPPORT_CANDIDATE_NUMERICAL_RESULT_REVIEW',references=dict(recourse_audit=audit),plan_sha256='plan'))
        model=write('model.json',dict(domain='formal',scenario_ids=ids,scenarios=scenes,shape=[1,322],first_stage_sha256='first',input_binding=dict(task_sha256='task',config_hash='config')))
        return model,dict(candidate_review=review,candidate_audit=audit,candidate_checkpoint=cp,candidate_plan=plan,candidate_snapshot=snapshot)
    def test_real_arrays_all_160_and_reordered_mapping(self):
        with tempfile.TemporaryDirectory() as td:
            model,evidence=self.fixture(Path(td))
            v,r=build_candidate_warm_start(model_manifest_reference=model,evidence=evidence,deadline=time.monotonic()+10)
            np.testing.assert_array_equal(v,np.r_[1,2,np.tile([5.,3.],160)])
            self.assertEqual(len(r['artifact_references']),160);self.assertEqual(r['solver_calls'],0)
            self.assertFalse(r['new_scientific_acceptance']);self.assertTrue(r['original_model_feasibility_check_required'])
    def test_raw_tamper_and_resealed_wrong_template_rejected(self):
        for mode in ('bytes','template','order','task'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as td:
                model,evidence=self.fixture(Path(td));p=Path(td)
                if mode=='bytes':(p/'0/primal_vector.npz').write_bytes(b'bad')
                else:
                    body=b.read_json(model['path'])
                    if mode=='template':body['scenarios'][0]['template_sha256']='wrong'
                    elif mode=='order':body['scenarios'][0]['scenario_id']='wrong'
                    else:body['input_binding']['task_sha256']='wrong'
                    Path(model['path']).write_text(json.dumps(body));model=b.file_reference(model['path'])
                with self.assertRaises(ValueError):build_candidate_warm_start(model_manifest_reference=model,evidence=evidence,deadline=time.monotonic()+10)
    def test_mapping_and_nonfinite_rejected(self):
        for v,m in [([1,2],[0,0]),([1,2],[0,2]),([1,2],[0.,1.]),([1,float('nan')],[0,1])]:
            with self.assertRaises(ValueError):map_primal(v,m,2,2)
    def test_expired_deadline_no_read(self):
        with self.assertRaises(ValueError):build_candidate_warm_start(model_manifest_reference={},evidence={},deadline=time.monotonic()-1)
