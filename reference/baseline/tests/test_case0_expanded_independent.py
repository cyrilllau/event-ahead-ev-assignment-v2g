"""Independent expanded-study boundary checks; no model solve or OOS draw."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from evcs_experiments import case0_campaign as c
from evcs_experiments.case0_profiles import EXPANDED, CAMPAIGNS


class ExpandedIndependentTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        base=Path(self.temp.name);self.root=base/CAMPAIGNS[EXPANDED];self.root.mkdir()
        data=base/'inputs';data.mkdir();(data/'MANIFEST_SHA256.csv').write_text('frozen-fixture')
        self.paths=SimpleNamespace(campaign_root=self.root,data_root=data)
        for name,value in [('get_paths',lambda:self.paths),('sources',lambda:{'fixed-source':'a'*64})]:
            p=patch.object(c,name,side_effect=value);p.start();self.addCleanup(p.stop)
        c.prepare('gurobi',EXPANDED)

    def test_changed_scientific_scope_and_campaign_cannot_reopen(self):
        path=self.root/'config.lock.json';original=c.read(path)
        for key,value in [('target_event','michael_2018'),('horizon_hours',48),('network','IEEE69'),('variant_start',10000),('solver_feasibility_tolerance',1e-8)]:
            altered=dict(original,**{key:value});path.write_text(json.dumps(altered))
            with self.subTest(key=key),self.assertRaises(ValueError):c.active_config(self.root)
        stripped={k:v for k,v in original.items() if k not in ('profile','campaign')}
        path.write_text(json.dumps(stripped))
        with self.assertRaises(ValueError):c.active_config(self.root)
        path.write_text(json.dumps(original))
        other=self.root.parent/c.CAMPAIGN_NAME;other.mkdir();(other/'config.lock.json').write_text(json.dumps(original))
        with self.assertRaises(ValueError):c.active_config(other)

    def test_fixed_scale_gap_failure_retains_cost_and_cannot_fallback(self):
        state=c.read(self.root/'state.json');state['compute_seconds']=23.;c.save_state(self.root,state)
        clock=[100.]
        def failed_gap(method,started,folder):
            clock[0]=111.;result=dict(method_id=method,backend='gurobi',gap=.2,numerical_valid=True,gap_passed=False)
            (folder/'solve').mkdir();(folder/'solve/result.json').write_text(json.dumps(result));return result
        with patch('evcs_experiments.case0_isolation.verify_current_cgroup',return_value={}),patch.object(c,'bounded_configuration',side_effect=failed_gap),patch.object(c.time,'monotonic',side_effect=lambda:clock[0]):
            state=c.plan_batch()
        self.assertEqual(state['status'],'DIAGNOSIS_REQUIRED');self.assertEqual(state['compute_seconds'],34.)
        self.assertEqual(set(state['completed']),{'size0/EA_DRO'});self.assertFalse(state['evaluation_opened'])
        self.assertNotIn('use the single declared smaller',state['next_action'])
        before=(self.root/'state.json').read_bytes()
        with self.assertRaises(ValueError):c.fallback('Do not shrink enlarged study to old cohort')
        self.assertEqual(before,(self.root/'state.json').read_bytes())

    def test_plan_child_receives_locked_new_stream_and_profile(self):
        state=c.read(self.root/'state.json');state['active_configuration']='size0/OR_DRO';c.save_state(self.root,state)
        ctx=SimpleNamespace(first=object(),case=object(),tasks=object(),planning_ids=tuple(range(8)),probability_set=object(),scenario=lambda i:i,recourse=lambda i:i)
        with patch('evcs_experiments.case0_isolation.verify_current_cgroup'),patch('evcs_experiments.case0_inputs.Case0Context',return_value=ctx) as factory,patch('evcs_experiments.case0_planning.run_case0_planning',return_value=(None,{'fixture':True})) as solve:
            c.plan_one('OR_DRO',100.)
        self.assertEqual(factory.call_args.kwargs['profile'],EXPANDED)
        self.assertEqual(factory.call_args.kwargs['variant_start'],30000)
        self.assertEqual(factory.call_args.kwargs['fleet_per_depot'],12)
        self.assertEqual(factory.call_args.kwargs['lead_hours'],0)
        self.assertEqual(solve.call_args.kwargs['solver_feasibility_tolerance'],1e-9)
        self.assertEqual(solve.call_args.kwargs['master_seconds'],1200)
        self.assertEqual(solve.call_args.kwargs['deadline'],1900.)

    def test_evaluation_auditor_root_gate_allows_only_declared_case0_roots(self):
        import time
        from evcs_experiments import case0_evaluation_audit as auditor
        with patch.object(auditor,'_read',side_effect=RuntimeError('reached evidence gate')) as read:
            with self.assertRaisesRegex(ValueError,'Independent Case0 root'):
                auditor.audit_case0_evaluation(dict(root=str(self.root.parent/'ieee33-v54-q1')),time.monotonic()+10)
            read.assert_not_called()
            with self.assertRaisesRegex(RuntimeError,'reached evidence gate'):
                auditor.audit_case0_evaluation(dict(root=str(self.root),lock_reference={}),time.monotonic()+10)


class ExpandedPhysicalInputTests(unittest.TestCase):
    def test_real_debug_cohort_horizon_stream_and_reopen(self):
        from evcs_experiments.case0_inputs import Case0Context
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'debug40000'
            kwargs=dict(profile=EXPANDED,fleet_per_depot=12,planning_variants=1,variant_start=40000)
            context=Case0Context(folder,**kwargs)
            self.assertEqual(context.selected_vehicle_ids,tuple(f'D{d}-EV{i:03d}' for d in range(1,5) for i in range(1,13)))
            self.assertEqual(len(context.case.bus_ids),33)
            self.assertEqual(tuple(context.case.periods),tuple(range(1,25)))
            self.assertEqual(len(context.planning_ids),8)
            self.assertEqual(len(context.bank['queries']),48*8)
            self.assertEqual(context.identity['profile'],EXPANDED)
            self.assertEqual(context.target_id,'zeta_2020')
            self.assertEqual(context.identity['target'],'zeta_2020')
            self.assertTrue(all(s.startswith('zeta_2020/') for s in context.planning_ids))
            self.assertGreater(sum(bool(q['route'] and q['route']['arc_ids']) for q in context.bank['queries']),0)
            planning=context.sample(0);calibration=context.sample(0,split='calibration')
            self.assertEqual(planning.ledger['variant_index'],40000)
            self.assertEqual(planning.ledger['target_event'],'zeta_2020')
            self.assertEqual(planning.ledger['wind_parameters']['target_event'],'zeta_2020')
            self.assertNotEqual(planning.ledger['line_stream_id'],calibration.ledger['line_stream_id'])
            self.assertTrue(planning.physical.development)
            self.assertEqual(planning.physical.load_p_kw.shape,(24,33))
            with patch('evcs_experiments.case0_inputs.build_candidate_tasks',side_effect=AssertionError('Immutable inputs must reopen')):
                self.assertEqual(Case0Context(folder,**kwargs).identity,context.identity)
                with self.assertRaises(ValueError):Case0Context(folder,**dict(kwargs,variant_start=30000))

class ExpandedTargetLockTests(unittest.TestCase):
    def test_expanded_lock_rejects_missing_wrong_or_legacy_target(self):
        from evcs_experiments.case0_profiles import lock_target
        self.assertEqual(lock_target({}),'michael_2018')
        self.assertEqual(lock_target(dict(profile=EXPANDED,target_event='zeta_2020')),'zeta_2020')
        for value in [dict(profile=EXPANDED),dict(profile=EXPANDED,target_event='michael_2018'),dict(target_event='zeta_2020'),dict(profile='arbitrary',target_event='zeta_2020')]:
            with self.subTest(value=value),self.assertRaises(ValueError):lock_target(value)

    def test_tiny_physical_evaluation_uses_verified_expanded_target(self):
        # Synthetic one-bus fixture: no weather/OOS random sample is drawn.
        import time
        import numpy as np
        from evcs_experiments import case0_evaluation as e
        from evcs_experiments.validation.s02_toys import golden_single_bus
        from evcs_experiments.first_stage import FirstStagePlan
        from evcs_experiments.recourse import assemble_recourse
        f=golden_single_bus();f.scenario.target_event='zeta_2020';f.scenario.split='oos';f.scenario.regime='R1'
        f.scenario.metadata={'departure_h':-1};f.scenario.scenario_id='zeta_2020/case0/development/oos/toy/v30001/R1'
        plans={m:f.plan for m in e.METHODS['R1']};plans['ZERO']=FirstStagePlan(np.zeros_like(f.plan.x),np.zeros_like(f.plan.r),f.plan.task_sha256,f.plan.config_hash)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);lock=e._write(root/'lock.json',{'synthetic':True})
            checked=dict(profile=EXPANDED,target_event='zeta_2020',method_plan_sha256={m:e.plan_identity(p) for m,p in plans.items()},sample_ids_by_regime={r:[f.scenario.scenario_id] for r in e.METHODS})
            def provider(case,tasks,scene):
                t=assemble_recourse(case,tasks,scene);return t,e._write(root/'template.json',{'template_sha256':t.template_sha256})
            ctx=e.Case0EvaluationContext(lock_reference=lock,lock_validator=lambda ref:checked,plans=plans,physical_factory=lambda *args:(f.case,f.tasks,f.scenario),template_provider=provider,source_references=[e.file_reference(e.__file__)],samples_per_regime=1)
            self.assertEqual(ctx.target_id,'zeta_2020')
            item=next(x for x in ctx.items.values() if x['regime']=='R1' and x['method_id']=='EA_DRO')
            result=e.evaluate_case0_item(ctx,item,output_directory=root/'item',deadline=time.monotonic()+10)
            row=e._read(result['record_reference']);self.assertEqual(row['scenario_id'],f.scenario.scenario_id)
            self.assertFalse(row['scientific_acceptance'])

if __name__=='__main__':unittest.main()
