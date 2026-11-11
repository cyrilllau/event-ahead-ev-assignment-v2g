"""Selected-plan numerical audits against a hand-solvable physical CCG case."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from evcs_experiments.config import get_paths
from evcs_experiments.evaluation_lock import file_reference
from evcs_experiments.raw_plan_audit import audit_fixed_plan_recourses
from evcs_experiments.raw_plan_audit import audit_selected_probability
from evcs_experiments.ccg import run_ccg
from evcs_experiments.recourse import solve_fixed_plan
from evcs_experiments.probability import solve_worst_distribution
import test_raw_template_audit as envelope_fixtures


class RawPlanAuditTests(unittest.TestCase):
    def setUp(self):
        self.helper=envelope_fixtures.RawTemplateAuditTests(methodName='runTest')
        self.helper.setUp();self.addCleanup(self.helper.doCleanups)
        project=get_paths().project_root
        self.sources=dict(helper_source_reference=file_reference(project/'scripts/audits/audit_s05_pending_recovery.py'),
            numerical_kernel_source_reference=file_reference(project/'scripts/audits/audit_s05_candidate_recourse.py'))

    def prepare(self,provider=True):
        e,snapshot=self.helper.run_evidence(provider)
        candidates=[]
        for ref in e.events:
            event=json.loads(Path(ref['path']).read_text())
            if event['event']=='recourse':
                plan=json.loads((Path(ref['path']).parent/'plan.json').read_text())
                if sum(plan['x'])==3:candidates.append((event['iteration'],ref,plan))
        self.assertTrue(candidates)
        iteration,ref,plan=max(candidates,key=lambda item:item[0])
        return e,snapshot,iteration,ref,plan

    def audit(self,e,snapshot,iteration,plan,**options):
        f=self.helper.helper.fixture
        return audit_fixed_plan_recourses(snapshot_reference=snapshot,
            expected_request_reference=file_reference(e.root/'request.json'),plan_dict=plan,iteration=iteration,
            case=f.case,tasks=f.tasks,scene_loader=f.scenarios.__getitem__,**self.sources,**options)

    def test_genuine_legacy_and_provider_plan_q_and_first_cost_match_hand_values(self):
        for provider in (False,True):
            with self.subTest(provider=provider):
                e,snapshot,iteration,ref,plan=self.prepare(provider)
                result=self.audit(e,snapshot,iteration,plan)
                self.assertEqual(result['status'],'PASS_SELECTED_PLAN_FULL_SUPPORT_RECOURSE_NUMERICS')
                # 3 selected tasks, 2 periods: 3*(8 + .4*2) = 26.4.
                # Loads 4/8/12: 2*.2*load + constants 1/2/3.
                self.assertAlmostEqual(result['first_stage_cost_usd'],26.4,places=7)
                np.testing.assert_allclose(result['q_usd'],[2.6,5.2,7.8],atol=1e-8,rtol=0)
                self.assertTrue(result['full_support_recourse_verified'])
                for key in ('probability_audited','master_bound_audited','scientific_acceptance','formal_OOS'):
                    self.assertIs(result[key],False)
                self.assertEqual(result['solver_calls'],0)

    def tamper_and_audit(self,filename,key,error):
        e,snapshot,iteration,ref,plan=self.prepare()
        folder=Path(ref['path']).parent;path=folder/filename
        with np.load(path,allow_pickle=False) as source:values={k:source[k].copy() for k in source.files}
        values[key].flat[0]+=1.
        np.savez_compressed(path,**values)
        self.helper.reseal_record(folder)
        changed=self.helper.reseal_snapshot(e,snapshot)
        with self.assertRaisesRegex(ValueError,error):self.audit(e,changed,iteration,plan)

    def test_resealed_saved_fixed_rhs_numerically_rejected(self):
        self.tamper_and_audit('fixed_rhs.npz','b_ub','Fixed RHS ub changed')

    def test_resealed_saved_dual_numerically_rejected(self):
        self.tamper_and_audit('dual_kkt.npz','inequality','Saved dual differs from original solver')

    def test_nonexistent_selected_iteration_rejected(self):
        e,snapshot,iteration,ref,plan=self.prepare()
        with self.assertRaisesRegex(ValueError,'lacks unique full support'):
            self.audit(e,snapshot,iteration+1000,plan)

    def test_shared_deadline_expiry_before_return_cannot_grant_pass(self):
        e,snapshot,iteration,ref,plan=self.prepare()
        with patch('evcs_experiments.raw_plan_audit.time.monotonic',side_effect=[100.,100.,100.,100.,100.,101.]):
            with self.assertRaisesRegex(TimeoutError,'Shared numerical audit deadline reached'):
                self.audit(e,snapshot,iteration,plan,deadline=100.5)

    def probability_args(self,risk='DRO'):
        if risk=='DRO':
            e,snapshot,iteration,ref,plan=self.prepare()
        else:
            h=self.helper.helper;p,e=h.prepare_run('ARO')
            solved=run_ccg(p,evidence_sink=e.sink_factory(h.root/'aro_call'),wall_seconds=20,
                rmp_time_limit_seconds=3,oracle_time_limit_seconds=3,mip_relative_gap=1e-9,optimality_relative_gap=1e-8)
            self.assertEqual(solved.status,'CONVERGED')
            snapshot=e.snapshot(h.root/'aro_snapshot.json')
            rows=[(ref,json.loads(Path(ref['path']).read_text())) for ref in e.events]
            ref,event=max(((ref,row) for ref,row in rows if row['event']=='recourse'),key=lambda pair:pair[1]['iteration'])
            iteration=event['iteration'];plan=json.loads((Path(ref['path']).parent/'plan.json').read_text())
        audit=self.audit(e,snapshot,iteration,plan)
        return e,dict(recourse_audit=audit,snapshot_reference=snapshot,
            expected_request_reference=file_reference(e.root/'request.json'),plan_dict=plan)

    def test_real_dro_and_aro_probability_witnesses_pass_without_solver_or_master_acceptance(self):
        for risk in ('DRO','ARO'):
            with self.subTest(risk=risk):
                e,args=self.probability_args(risk)
                result=audit_selected_probability(**args)
                self.assertEqual(result['status'],'PASS_SELECTED_PLAN_PROBABILITY_WITNESS')
                self.assertTrue(result['check']['passed'])
                self.assertTrue(result['probability_audited'])
                self.assertFalse(result['master_bound_audited'])
                self.assertFalse(result['scientific_acceptance'])
                self.assertFalse(result['formal_OOS'])
                self.assertEqual(result['solver_calls'],0)

    def tamper_probability(self,kind):
        e,args=self.probability_args();iteration=args['recourse_audit']['iteration']
        folder=next(Path(ref['path']).parent for ref in e.events
            if (lambda row:row['event']=='probability' and row['iteration']==iteration)(json.loads(Path(ref['path']).read_text())))
        filename='solver.json' if kind=='theta' else 'q_input.json'
        path=folder/filename;value=json.loads(path.read_text())
        if kind=='q':value['q_usd'][0]+=1.
        elif kind=='plan':value['plan']['x'][0]=1-value['plan']['x'][0]
        else:value['theta0']-=100.
        self.helper.helper.write(str(path),value);self.helper.reseal_record(folder)
        snapshot=self.helper.reseal_snapshot(e,args['snapshot_reference'])
        args['snapshot_reference']=snapshot
        args['recourse_audit']=self.audit(e,snapshot,iteration,args['plan_dict'])
        with self.assertRaises(ValueError):audit_selected_probability(**args)

    def test_resealed_probability_changed_q_rejected(self):self.tamper_probability('q')

    def test_resealed_probability_changed_plan_rejected(self):self.tamper_probability('plan')

    def test_resealed_probability_infeasible_theta_rejected(self):self.tamper_probability('theta')

    def repeated_plan_probability(self,future=False):
        e,snapshot,iteration,ref,plan=self.prepare();h=self.helper.helper
        original_refs=[]
        for ref in e.events:
            row=json.loads(Path(ref['path']).read_text())
            if row['event']=='recourse' and row['iteration']==iteration:
                original_refs.append(ref)
        repeated=iteration+1;point=SimpleNamespace(**{**plan,'x':np.asarray(plan['x']),'r':np.asarray(plan['r'])})
        sink=e.sink_factory(h.root/'repeat_same_plan_call');q=[]
        for i in range(len(h.fixture.scenario_ids)):
            solved=solve_fixed_plan(e.problem.recourses[i],point,time_limit_seconds=3)
            self.assertTrue(solved.optimal)
            sink('recourse',repeated,index=i,plan=point,result=solved)
            q.append(float(solved.objective_usd+e.problem.scenario_cost_constants_usd[i]))
        probability=solve_worst_distribution(e.problem.ambiguity,q,q_support_ids=e.ids,time_limit_seconds=3)
        self.assertTrue(probability.certified_optimal)
        sink('probability',repeated,plan=point,result=probability)
        folder=Path(e.events[-1]['path']).parent
        value=json.loads((folder/'q_input.json').read_text())
        # Exporter normally cites its latest per-scenario result. A sealed
        # history citation is also valid when current Q was independently
        # recomputed for precisely the same plan and support.
        value['recourse_evidence']=original_refs
        if future:
            old_path=Path(original_refs[0]['path']);old=json.loads(old_path.read_text())
            old['iteration']=repeated+1
            value['recourse_evidence'][0]=h.write(str(old_path),old)
        h.write(str(folder/'q_input.json'),value);self.helper.reseal_record(folder)
        changed=self.helper.reseal_snapshot(e,snapshot)
        audit=self.audit(e,changed,repeated,plan)
        return dict(recourse_audit=audit,snapshot_reference=changed,
            expected_request_reference=file_reference(e.root/'request.json'),plan_dict=plan)

    def test_real_repeated_plan_can_cite_earlier_same_plan_q(self):
        result=audit_selected_probability(**self.repeated_plan_probability())
        self.assertEqual(result['status'],'PASS_SELECTED_PLAN_PROBABILITY_WITNESS')

    def test_repeated_plan_cannot_cite_future_iteration(self):
        with self.assertRaisesRegex(ValueError,'Probability Q source differs'):
            audit_selected_probability(**self.repeated_plan_probability(future=True))


if __name__=='__main__':unittest.main()
