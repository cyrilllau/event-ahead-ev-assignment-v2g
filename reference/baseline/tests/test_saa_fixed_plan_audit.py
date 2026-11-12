"""Independent no-solve audit of genuine fixed-plan SAA LP evidence."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import save_npz
from evcs_experiments import saa_fixed_plan_audit as m
from evcs_experiments.campaign_batch import file_reference
from evcs_experiments.config import get_paths
from evcs_experiments import recourse
import test_saa_fixed_plan as fixture_module


class SAAQAuditTests(unittest.TestCase):
    def setUp(self):
        self.f=fixture_module.FixedSAAQTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.root=self.f.root.parent;self.result=self.f.run_q();self.cp=self.result['checkpoint_reference']
        self.request=self.result['request_reference'];refs=[]
        for i,t in enumerate(self.f.templates):
            folder=self.root/f'template_{i}';folder.mkdir()
            for name in m.recourse_store.MATRICES:save_npz(folder/(name+'.npz'),getattr(t,name))
            np.savez_compressed(folder/'arrays.npz',**{k:getattr(t,k) for k in m.recourse_store.ARRAYS})
            m._write(folder/'identity.json',t.as_dict())
            refs.append(m._write(folder/'record.json',dict(kind='recourse_template',scenario_index=i,
                scenario_id=self.f.toy.scenario_ids[i],template_sha256=t.template_sha256,
                artifacts=[file_reference(p) for p in sorted(folder.iterdir())])))
        self.templates=refs;self.first=self.f.templates[0].first_stage
        self.env=m.bind_saa_template_envelope(request_reference=self.request,template_references=refs,
            first_stage_sha256=self.first.matrix_sha256,source_references=[file_reference(recourse.__file__)],
            output_path=self.root/'envelope.json')
        self.args=dict(checkpoint_reference=self.cp,expected_request_reference=self.request,
            template_envelope_reference=self.env,first_stage=self.first,case=self.f.toy.case,tasks=self.f.toy.tasks,
            scene_loader=self.f.toy.scenarios.__getitem__,empirical_probabilities=self.f.toy.empirical_probabilities,
            helper_reference=file_reference(get_paths().project_root/'scripts/audits/audit_s05_pending_recovery.py'),
            source_references=[file_reference(p) for p in (m.__file__,m.recourse_store.__file__,m.first_stage_check.__file__,m.recourse_check.__file__)],
            deadline=time.monotonic()+30)

    def audit(self,**kw):return m.audit_saa_fixed_plan_q(**{**self.args,**kw})

    def reseal_raw(self,name,change):
        cp=json.loads(Path(self.cp['path']).read_text());ref=cp['attempts'][0];row=json.loads(Path(ref['path']).read_text())
        artifact=next(r for r in row['artifacts'] if Path(r['path']).name==name);p=Path(artifact['path']);change(p)
        row['artifacts']=[file_reference(p) if r==artifact else r for r in row['artifacts']]
        Path(ref['path']).write_text(json.dumps(row));cp['attempts'][0]=file_reference(ref['path'])
        out=self.root/'resealed_checkpoint.json';out.write_text(json.dumps(cp))
        return file_reference(out)

    def test_real_original_kkt_and_uniform_expectation_no_solver(self):
        with patch('evcs_experiments.recourse.solve_fixed_plan',side_effect=AssertionError('No new LP')):
            result=self.audit()
        self.assertTrue(result['passed']);self.assertEqual(result['solver_calls'],0)
        self.assertAlmostEqual(result['upper_bound_usd'],np.mean(self.result['q_usd']))
        self.assertTrue(all(r['physical_passed'] for r in result['rows']))
        self.assertFalse(result['master_bound_audited']);self.assertFalse(result['formal_OOS'])

    def test_nonuniform_or_incomplete_support_refused(self):
        with self.assertRaisesRegex(ValueError,'uniform'):self.audit(empirical_probabilities=[.1,.2,.7])
        cp=json.loads(Path(self.cp['path']).read_text());cp['attempts'].pop();p=self.root/'missing.json';p.write_text(json.dumps(cp))
        with self.assertRaisesRegex(ValueError,'Missing'):self.audit(checkpoint_reference=file_reference(p))

    def test_resealed_raw_rhs_rejected_numerically(self):
        def change(p):
            with np.load(p) as z:v={k:z[k] for k in z.files}
            v['b_ub']=v['b_ub']+1;np.savez_compressed(p,**v)
        cp=self.reseal_raw('fixed_rhs.npz',change)
        with self.assertRaisesRegex(ValueError,'RHS'):self.audit(checkpoint_reference=cp)

    def test_resealed_raw_dual_rejected_numerically(self):
        def change(p):
            with np.load(p) as z:v={k:z[k] for k in z.files}
            v['lower']=v['lower']+1;np.savez_compressed(p,**v)
        cp=self.reseal_raw('dual_kkt.npz',change)
        with self.assertRaisesRegex(ValueError,'dual'):self.audit(checkpoint_reference=cp)

    def test_coherently_resealed_duals_still_fail_original_kkt(self):
        def change(p):
            with np.load(p) as z:v={k:z[k] for k in z.files}
            v['lower']=v['lower']+1;np.savez_compressed(p,**v)
        cp_ref=self.reseal_raw('dual_kkt.npz',change)
        cp=json.loads(Path(cp_ref['path']).read_text());rr=cp['attempts'][0];row=json.loads(Path(rr['path']).read_text())
        sr=next(r for r in row['artifacts'] if Path(r['path']).name=='solver.json')
        sp=Path(sr['path']);solver=json.loads(sp.read_text())
        solver['certificate']['dual_lower_marginals']=(np.asarray(solver['certificate']['dual_lower_marginals'])+1).tolist()
        sp.write_text(json.dumps(solver));row['artifacts']=[file_reference(sp) if r==sr else r for r in row['artifacts']]
        Path(rr['path']).write_text(json.dumps(row));cp['attempts'][0]=file_reference(rr['path'])
        Path(cp_ref['path']).write_text(json.dumps(cp))
        with self.assertRaisesRegex(ValueError,'KKT'):self.audit(checkpoint_reference=file_reference(cp_ref['path']))

    def test_resealed_request_kind_and_scene_hash_refused(self):
        for delta,pattern in (({'kind':'CCG'},'kind'),({'scenario_sha256s':['f'*64]*3},'hash')):
            with self.subTest(delta=delta):
                req=json.loads(Path(self.request['path']).read_text());req.update(delta)
                rp=self.root/'alternate_request.json';rp.write_text(json.dumps(req));rref=file_reference(rp)
                cp=json.loads(Path(self.cp['path']).read_text());cp['request_reference']=rref
                env=json.loads(Path(self.env['path']).read_text());env['request_reference']=rref
                for i,rr in enumerate(cp['attempts']):
                    row=json.loads(Path(rr['path']).read_text());row['request_reference']=rref
                    if 'scenario_sha256s' in delta:row['scenario_sha256']=delta['scenario_sha256s'][i]
                    ap=Path(rr['path']).parent/'alternate_record.json';ap.write_text(json.dumps(row));cp['attempts'][i]=file_reference(ap)
                ep=self.root/'alternate_env.json';ep.write_text(json.dumps(env));p=self.root/'alternate_cp.json';p.write_text(json.dumps(cp))
                with self.assertRaisesRegex(ValueError,pattern):
                    self.audit(expected_request_reference=rref,checkpoint_reference=file_reference(p),template_envelope_reference=file_reference(ep))

    def test_changed_original_matrix_hash_fails(self):
        p=Path(self.templates[0]['path']).parent/'arrays.npz'
        with np.load(p) as z:v={k:z[k] for k in z.files}
        v['c']=v['c']+1;np.savez_compressed(p,**v)
        with self.assertRaisesRegex(ValueError,'hash|size'):self.audit()

    def test_resealed_checkpoint_constant_double_count_rejected(self):
        cp=json.loads(Path(self.cp['path']).read_text());cp['q_usd'][0]+=self.f.kw['scenario_cost_constants_usd'][0]
        p=self.root/'wrongq.json';p.write_text(json.dumps(cp))
        with self.assertRaisesRegex(ValueError,'Checkpoint Q'):self.audit(checkpoint_reference=file_reference(p))

    def test_expired_audit_and_source_gate(self):
        with self.assertRaises(TimeoutError):self.audit(deadline=time.monotonic()-1)
        with self.assertRaisesRegex(ValueError,'source'):self.audit(source_references=[])

if __name__=='__main__':unittest.main()
