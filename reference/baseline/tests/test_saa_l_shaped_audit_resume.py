"""Separate-process event inventory recovery against actual partial kernel state."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest.mock import patch
from dataclasses import replace
from evcs_experiments import saa_l_shaped as kernel
from evcs_experiments import saa_l_shaped_audit_bridge as bridge
from evcs_experiments import saa_fixed_plan_audit as audit
from evcs_experiments import saa_l_shaped_master_evidence as master_evidence
from evcs_experiments.campaign_batch import file_reference
from evcs_experiments.random_streams import canonical_hash
import test_saa_l_shaped_audit_bridge as fixtures

CHILD = r'''
import json,sys
from pathlib import Path
from evcs_experiments import saa_l_shaped as kernel
from evcs_experiments import saa_l_shaped_audit_bridge as bridge
from evcs_experiments.validation.s03_toys import build_golden_cases
cfg=json.loads(Path(sys.argv[1]).read_text());t=build_golden_cases()[0]
p=kernel.DevelopmentSAAProblem(t.case,t.tasks,t.scenarios,t.scenario_cost_constants_usd-100.)
sink=bridge.DevelopmentFixedQEvidence(case=t.case,tasks=t.tasks,scene_loader=t.scenarios.__getitem__,**cfg['args'],
 resume_inventory_reference=cfg['inventory_reference'],kernel_checkpoint_reference=cfg['kernel_reference'])
actual=kernel.solve_fixed_plan;calls=[]
def observed(template,plan,**kwargs):
 calls.append(template.scenario_sha256)
 return actual(template,plan,**kwargs)
kernel.solve_fixed_plan=observed
from evcs_experiments.saa_l_shaped_master_evidence import DevelopmentMasterEvidence
ms=DevelopmentMasterEvidence(output_root=cfg['master_evidence_root']) if cfg.get('master_evidence_root') else None
state=kernel.run_saa_l_shaped(p,evidence_sink=sink,master_sink=ms,resume=True,checkpoint_path=cfg['kernel_reference']['path'],
 expected_checkpoint_sha256=cfg['kernel_reference']['sha256'],max_iterations=1)
cp=sink.checkpoint(1)
from evcs_experiments.campaign_batch import file_reference
inv=sink.snapshot(file_reference(cfg['kernel_reference']['path']))
Path(sys.argv[2]).write_text(json.dumps(dict(pid=__import__('os').getpid(),calls=calls,state=state,
 checkpoint_reference=cp,inventory_reference=inv)))
'''


class EventInventoryResumeTests(unittest.TestCase):
    def setUp(self):
        self.b=fixtures.LShapedAuditBridgeTests('test_real_kernel_events_original_audit_same_plan_negative_constants_once')
        self.b.setUp();self.addCleanup(self.b.doCleanups)
        self.root=self.b.f.root.resolve();self.sink=self.b.sink;self.args=self.b.args;self.problem=self.b.problem

    def partial(self, *, with_master=False):
        actual=kernel.solve_fixed_plan;calls=[]
        def limited(template,plan,**kwargs):
            calls.append(template.scenario_sha256);raw=actual(template,plan,**kwargs)
            if len(calls)==2:return replace(raw,optimal=False,solver_status=1,status='LIMIT_REACHED',
                solution=None,objective_usd=None,certificate={'optimal_certificate_valid':False})
            return raw
        path=self.root/'partial_kernel.json'
        ms=master_evidence.DevelopmentMasterEvidence(output_root=self.root/'master_evidence') if with_master else None
        with patch.object(kernel,'solve_fixed_plan',side_effect=limited):
            state=kernel.run_saa_l_shaped(self.problem,evidence_sink=self.sink,master_sink=ms,checkpoint_path=path)
        self.assertEqual(state['status'],'TIME_LIMIT');self.assertEqual(len(state['pending']['atoms']),1)
        ref=file_reference(path);inv=self.sink.snapshot(ref)
        return state,ref,inv

    def restore(self, inv, kernel_ref, **delta):
        return bridge.DevelopmentFixedQEvidence(**{**self.args,**delta},
            resume_inventory_reference=inv,kernel_checkpoint_reference=kernel_ref)

    def test_real_second_process_skips_completed_q_and_preserves_full_audit_and_time(self):
        opening,kref,iref=self.partial(with_master=True);old_raw=list(self.sink.sweeps[1]['rows'])
        cfg_args={k:v for k,v in self.args.items() if k not in ('case','tasks','scene_loader')}
        cfg_args['output_root']=str(cfg_args['output_root'])
        cfg=self.root/'child_config.json';out=self.root/'child_result.json'
        cfg.write_text(json.dumps(dict(args=cfg_args,kernel_reference=kref,inventory_reference=iref,
            master_evidence_root=str(self.root/'master_evidence'))))
        result=subprocess.run([sys.executable,'-c',CHILD,str(cfg),str(out)],text=True,capture_output=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        final=json.loads(out.read_text());state=final['state']
        self.assertNotEqual(final['pid'],os.getpid())
        self.assertEqual(final['calls'],self.args['scenario_sha256s'][1:])
        self.assertEqual(state['status'],'ITERATION_LIMIT');self.assertEqual(state['iterations'],1)
        self.assertEqual(len(state['sessions']),2)
        self.assertEqual(state['sessions'][0],opening['sessions'][0])
        self.assertGreater(state['elapsed_seconds'],opening['elapsed_seconds'])
        self.assertAlmostEqual(state['elapsed_seconds'],sum(v['elapsed_seconds'] for v in state['sessions']))
        self.assertEqual(state['config'],opening['config'])
        for ref in old_raw:self.assertEqual(file_reference(ref['path']),ref)
        self.assertEqual(file_reference(iref['path']),iref)
        cp=final['checkpoint_reference'];raw=json.loads(Path(cp['path']).read_text())
        self.assertEqual(len(raw['attempts']),4);self.assertEqual(raw['completed_indices'],[0,1,2])
        env=audit.bind_saa_template_envelope(request_reference=raw['request_reference'],template_references=self.b.f.templates,
            first_stage_sha256=self.problem.first_stage.matrix_sha256,source_references=self.args['source_references'],
            output_path=self.root/'resumed_envelope.json')
        checked=audit.audit_saa_fixed_plan_q(**{**self.b.f.args,'checkpoint_reference':cp,
            'expected_request_reference':raw['request_reference'],'template_envelope_reference':env,'deadline':time.monotonic()+30})
        self.assertTrue(checked['passed']);self.assertEqual(checked['solver_calls'],0)
        self.assertEqual(checked['q_usd'],state['best']['q_usd'])
        self.assertAlmostEqual(checked['upper_bound_usd'],state['upper_bound_usd'])
        self.assertFalse(checked['budget_settled']);self.assertFalse(checked['master_bound_audited'])
        master_audits=[master_evidence.audit_master_evidence(self.problem,m['evidence_reference'],deadline=time.monotonic()+30) for m in state['master_history']]
        self.assertTrue(master_audits)
        self.assertTrue(all(a['passed'] for a in master_audits))
        audit._write(self.root/'resumed_master_audits.json',master_audits)
        audit._write(self.root/'resumed_original_audit.json',checked)
        audit._write(self.root/'resume_test_summary.json',dict(status='PASS_DEVELOPMENT_CROSS_PROCESS_Q_EVENT_RECOVERY',
            parent_pid=os.getpid(),child_pid=final['pid'],opening_elapsed_seconds=opening['elapsed_seconds'],
            closing_elapsed_seconds=state['elapsed_seconds'],resume_lp_scenario_hashes=final['calls'],
            prior_inventory_reference=iref,final_inventory_reference=final['inventory_reference'],
            master_audit_reference=file_reference(self.root/'resumed_master_audits.json'),
            child_result_reference=file_reference(out),audit_reference=file_reference(self.root/'resumed_original_audit.json'),
            source_references=self.args['source_references'],production_ready=False,budget_settled=False,formal_OOS=False))
        # A new inventory works, but an old snapshot cannot hide subsequent events.
        self.restore(final['inventory_reference'],file_reference(kref['path']))
        with self.assertRaisesRegex(ValueError,'inventory'):self.restore(iref,file_reference(kref['path']))

    def test_empty_sink_pair_missing_and_changed_kernel_or_inputs_rejected(self):
        state,kref,iref=self.partial()
        empty=bridge.DevelopmentFixedQEvidence(**{**self.args,'output_root':self.root/'empty'})
        with self.assertRaisesRegex(ValueError,'Empty'):empty.snapshot(kref)
        with self.assertRaisesRegex(ValueError,'paired'):
            bridge.DevelopmentFixedQEvidence(**self.args,resume_inventory_reference=iref)
        with self.assertRaisesRegex(ValueError,'identity'):
            self.restore(iref,kref,input_binding={'different':'physical source'})
        changed=json.loads(Path(kref['path']).read_text());changed['state']['config']['budget_seconds']=31.
        changed['sha256']=canonical_hash(changed['state']);path=self.root/'changed_budget.json';path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError,'Different kernel'):self.restore(iref,file_reference(path))

    def test_inventory_extra_missing_symlink_and_changed_raw_rejected(self):
        _,kref,iref=self.partial();root=self.args['output_root']
        extra=root/'unknown.json';extra.write_text('{}')
        with self.assertRaisesRegex(ValueError,'inventory'):self.sink.snapshot(kref)
        with self.assertRaisesRegex(ValueError,'inventory'):self.restore(iref,kref)
        extra.unlink();alias=root/'alias';alias.symlink_to(kref['path'])
        with self.assertRaisesRegex(ValueError,'Symlink'):self.restore(iref,kref)
        alias.unlink();raw=Path(self.sink.sweeps[1]['rows'][0]['path']);original=raw.read_bytes();raw.unlink()
        with self.assertRaisesRegex(ValueError,'inventory'):self.restore(iref,kref)
        raw.write_bytes(original+b' ')
        with self.assertRaisesRegex(ValueError,'inventory'):self.restore(iref,kref)

    def test_stale_or_foreign_inventory_and_completed_kernel_mapping_rejected(self):
        _,kref,iref=self.partial()
        other=self.root/'foreign_inventory.json';other.write_bytes(Path(iref['path']).read_bytes())
        with self.assertRaisesRegex(ValueError,'root'):self.restore(file_reference(other),kref)
        changed=json.loads(Path(kref['path']).read_text());changed['state']['pending']['atoms']=[]
        changed['sha256']=canonical_hash(changed['state']);p=self.root/'mismatched_kernel.json';p.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError,'completed Q'):self.sink.snapshot(file_reference(p))
        old=Path(iref['path']);old.write_bytes(old.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'changed'):self.restore(iref,kref)

if __name__=='__main__':unittest.main()
