"""Small physical recourse cuts; no formal bank or full160 solver calls."""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from evcs_experiments import candidate_dual_adapter as m
from evcs_experiments.campaign_batch import file_reference
from evcs_experiments.first_stage import FirstStagePlan, assemble_first_stage
from evcs_experiments.recourse import assemble_recourse, solve_fixed_plan
from evcs_experiments.validation.s03_toys import build_golden_cases


class CandidateDualAdapterTests(unittest.TestCase):
    def setUp(self):
        self.toy = build_golden_cases()[0]
        f = self.toy; self.t = assemble_recourse(f.case,f.tasks,f.scenarios[0])
        self.first = assemble_first_stage(f.case,f.tasks)
        plan = FirstStagePlan(x=np.zeros(len(f.tasks.task_ids)),r=np.zeros((len(f.tasks.task_ids),len(f.case.periods))),
                              task_sha256=f.tasks.task_sha256,config_hash=f.tasks.config_hash)
        self.solved = solve_fixed_plan(self.t,plan,time_limit_seconds=3.)
        self.assertTrue(self.solved.optimal)
        self.plan = dict(x=plan.x.tolist(),r=plan.r.tolist(),task_sha256=plan.task_sha256,config_hash=plan.config_hash)
        self.dual = {key:np.asarray(self.solved.certificate['dual_'+key+'_marginals'])
                     for key in ('inequality','equality','lower','upper')}
        self.values = {key:getattr(self.t,key).copy() for key in
                       ('b_ub','b_eq','c','lower_bounds','upper_bounds','component_objectives')}
        self.matrices = {key:getattr(self.t,key) for key in m.MATRICES}
        self.fixed = {key:self.values[key].copy() for key in ('b_ub','b_eq')}
        self.args = dict(scenario_id=f.scenario_ids[0],template_sha256=self.t.template_sha256,
            first_stage_sha256=self.first.matrix_sha256,meta=self.t.as_dict(),matrices=self.matrices,
            values=self.values,dual=self.dual,fixed=self.fixed,solver=self.solved.as_dict(),
            plan=self.plan,constant=-7.,q=self.solved.objective_usd-7.)

    def test_real_physical_dual_affine_bound_and_constant_once(self):
        with patch('evcs_experiments.recourse.solve_fixed_plan',side_effect=AssertionError('No adapter LP')):
            cut,g=m.derive_atom_arrays(**self.args)
        self.assertTrue(cut['passed']);self.assertEqual(cut['recourse_calls'],0)
        self.assertAlmostEqual(cut['anchor_affine_value_usd'],self.solved.objective_usd-7.)
        self.assertEqual(len(g),len(self.first.c))
        # A distinct feasible physical plan is obtained from this tiny golden model.
        from evcs_experiments.extensive import assemble_extensive, solve_extensive
        model=assemble_extensive(self.first,[self.t],risk_model='SAA',
            scenario_ids=[self.toy.scenario_ids[0]],empirical_probabilities=[1.])
        plan=solve_extensive(model,time_limit_seconds=3.).solution.plan
        other=solve_fixed_plan(self.t,plan,time_limit_seconds=3.)
        self.assertTrue(other.optimal)
        self.assertLessEqual(cut['intercept_usd']+g@np.r_[plan.x,plan.r.ravel()],other.objective_usd-7.+1e-7)

    def test_coherently_changed_dual_is_rejected_numerically(self):
        args=copy.deepcopy(self.args);args['dual']['lower'][0]+=1.
        args['solver']['certificate']['dual_lower_marginals']=args['dual']['lower'].tolist()
        with self.assertRaisesRegex(ValueError,'minorant'):m.derive_atom_arrays(**args)

    def test_fixed_rhs_tamper_even_with_valid_dual_is_rejected(self):
        args=copy.deepcopy(self.args);args['fixed']['b_ub'][0]+=1.
        with self.assertRaisesRegex(ValueError,'RHS'):m.derive_atom_arrays(**args)

    def test_base_rhs_and_gradient_coupling_change_break_semantic_hash(self):
        for name in ('b_ub','Bx_ub'):
            args=copy.deepcopy(self.args)
            if name=='b_ub':args['values'][name][0]+=1.
            else:
                args['matrices'][name]=args['matrices'][name].tolil();args['matrices'][name][0,0]+=1.
                args['matrices'][name]=args['matrices'][name].tocsr()
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'semantic'):
                m.derive_atom_arrays(**args)

    def test_q_constant_solver_status_and_task_identity_rejected(self):
        for name in ('constant','status','task'):
            args=copy.deepcopy(self.args)
            if name=='constant':args['constant']+=1.
            elif name=='status':args['solver']['status']='TIME_LIMIT'
            else:args['plan']['task_sha256']='wrong-task'
            with self.subTest(name=name),self.assertRaises(ValueError):m.derive_atom_arrays(**args)

    def test_objective_not_covered_only_by_hash_flag(self):
        args=copy.deepcopy(self.args);args['values']['c'][0]+=1.
        with self.assertRaisesRegex(ValueError,'aggregation'):m.derive_atom_arrays(**args)

    def test_real_saved_atom_adapter_to_independent_auditor(self):
        from scipy.sparse import save_npz
        from evcs_experiments.candidate_dual_audit import audit_derived_atom
        from evcs_experiments.random_streams import canonical_hash
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();td=root/'template';ed=root/'candidate';od=root/'original'
            for p in (td,ed,od):p.mkdir()
            def write(p,value):
                p.write_text(json.dumps(value,indent=2));return file_reference(p)
            def record(folder,body):
                return write(folder/'record.json',dict(body,artifacts=[file_reference(p) for p in sorted(folder.iterdir())]))
            for key,value in self.matrices.items():save_npz(td/(key+'.npz'),value)
            np.savez_compressed(td/'arrays.npz',**self.values)
            write(td/'identity.json',self.t.as_dict())
            sid=self.toy.scenario_ids[0];sha=self.t.template_sha256
            tr=record(td,dict(kind='recourse_template',scenario_index=0,scenario_id=sid,template_sha256=sha))
            oldpointer=write(od/'template_reference.json',tr)
            oldevent=record(od,dict(event='recourse'))
            write(ed/'template_reference.json',tr);pr=write(ed/'plan.json',self.plan)
            write(ed/'solver.json',self.solved.as_dict())
            z=self.t.pack(self.solved.solution);np.savez_compressed(ed/'primal_vector.npz',vector=z)
            np.savez_compressed(ed/'fixed_rhs.npz',**self.fixed)
            y,eq,lo,hi=[self.dual[k] for k in ('inequality','equality','lower','upper')]
            lb=self.values['lower_bounds'];ub=self.values['upper_bounds'];fl=np.isfinite(lb);fu=np.isfinite(ub)
            slack=self.fixed['b_ub']-self.t.A_ub@z
            dual=dict(self.dual,stationarity=self.t.c-self.t.A_ub.T@y-self.t.A_eq.T@eq-lo-hi,
                inequality_slack=slack,equality_residual=self.t.A_eq@z-self.fixed['b_eq'],
                inequality_complementarity=y*slack,lower_complementarity=lo[fl]*(z-lb)[fl],
                upper_complementarity=hi[fu]*(ub-z)[fu])
            np.savez_compressed(ed/'dual_kkt.npz',**dual)
            er=record(ed,dict(event='recourse',domain='formal',iteration=1,scenario_index=0,scenario_id=sid,
                plan_sha256=canonical_hash(self.plan),optimal_certificate_valid=True,q_usd=self.args['q'],
                status='RAW_EVIDENCE_RETAINED_PENDING_AUDIT'))
            # This tiny object exercises the file adapter; full160 admission remains separate.
            ctx=object.__new__(m.CandidateDualContext);ctx.deadline=time.monotonic()+10;ctx.used={}
            ctx.ids=(sid,);ctx.events=[er];ctx.plan=self.plan;ctx.q=[self.args['q']];ctx.constants=[-7.]
            ctx.identity=dict(first_stage_sha256=self.first.matrix_sha256,scenario_sha256s=[self.t.scenario_sha256])
            ctx.scenario_hashes=[self.t.scenario_sha256];ctx.template_hashes=[sha];ctx.first_metadata=self.t.as_dict();ctx.sources=[file_reference(m.__file__)]
            ctx.refs={'candidate_plan':pr};ctx.binding={'plan_sha256':canonical_hash(self.plan)}
            ctx.binding['binding_sha256']=canonical_hash(ctx.binding)
            ctx.inventory=[dict(index=0,scenario_id=sid,template_reference=tr,template_sha256=sha,
                                event_reference=oldevent,template_pointer_reference=oldpointer)]
            result=m.derive_candidate_atom(ctx,0)
            review=audit_derived_atom(ctx,0,result,deadline=ctx.deadline)
            self.assertTrue(review['passed']);self.assertEqual(review['solver_calls'],0)
            # Change a bound raw byte after derivation; the next atom load must fail.
            (ed/'fixed_rhs.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'size changed'):m.derive_candidate_atom(ctx,0)

    def test_actual_lazy_identity_schema_and_changed_binding_rejected(self):
        raw={'input_binding':{'task_source':'source','target':'Michael'}}
        identity=dict(first_stage_sha256='first',scenario_ids=['s0'],lazy_input_binding=dict(
            kind='input_bound_lazy_v1',binding=copy.deepcopy(raw['input_binding']),
            first_stage_sha256='first',scenario_ids=['s0'],scenario_sha256s=['a'*64]))
        self.assertEqual(m.scenario_hash_axis(identity,raw),('a'*64,))
        for name in ('binding','first_stage_sha256','scenario_ids','kind'):
            changed=copy.deepcopy(identity);changed['lazy_input_binding'][name]='changed'
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'binding'):
                m.scenario_hash_axis(changed,raw)
        changed=copy.deepcopy(identity);changed['scenario_sha256s']=['b'*64]
        with self.assertRaisesRegex(ValueError,'disagree'):m.scenario_hash_axis(changed,raw)


    def test_context_deadline_schema_and_unbound_source_refusal(self):
        with self.assertRaises(TimeoutError):m.CandidateDualContext({},source_references=[],deadline=time.monotonic()-1)
        with self.assertRaisesRegex(ValueError,'schema'):
            m.CandidateDualContext({},source_references=[],deadline=time.monotonic()+3)
        refs={key:{} for key in m.REQUIRED}
        with self.assertRaisesRegex(ValueError,'Unbound executable'):
            m.CandidateDualContext(refs,source_references=[file_reference(__file__)],deadline=time.monotonic()+3)

    def test_stream_hash_rejects_mutation_and_deadline_in_chunk(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'blob';p.write_bytes(b'a'*(2*1024*1024));ref=file_reference(p)
            ctx=object.__new__(m.CandidateDualContext);ctx.used={};ctx.deadline=time.monotonic()+30
            self.assertEqual(ctx.verify(ref),p.resolve())
            p.write_bytes(b'b'*(2*1024*1024))
            with self.assertRaisesRegex(ValueError,'hash changed'):ctx.verify(ref)
            with patch.object(ctx,'check_deadline',side_effect=[None,None,TimeoutError('expired')]):
                with self.assertRaises(TimeoutError):ctx.verify(ref)

    def test_metadata_reads_header_without_giant_descriptions(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'identity.json';meta=self.t.as_dict();p.write_text(json.dumps(meta,indent=2))
            got=m.metadata_header(p)
            self.assertEqual(got['template_sha256'],meta['template_sha256']);self.assertNotIn('variable_map',got)


if __name__=='__main__':unittest.main()
