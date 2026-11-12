import tempfile,unittest
from types import SimpleNamespace
from unittest.mock import MagicMock,patch
from pathlib import Path
from dataclasses import replace
import numpy as np
from scipy.sparse import csr_matrix
from evcs_experiments.extensive import assemble_extensive,solve_extensive,full_shedding_start
from evcs_experiments.first_stage import assemble_first_stage
from evcs_experiments.recourse import assemble_recourse
from evcs_experiments.validation.s03_toys import build_golden_cases
from evcs_experiments.native_highs import solve_with_start

class NativeHighsTests(unittest.TestCase):
    def test_timeout_with_invalid_info_does_not_invent_zero_bound(self):
        from evcs_experiments import native_highs as native
        solver=MagicMock()
        solver.setOptionValue.return_value=native.h.HighsStatus.kOk
        solver.passModel.return_value=native.h.HighsStatus.kOk
        solver.setSolution.return_value=native.h.HighsStatus.kOk
        solver.run.return_value=native.h.HighsStatus.kOk
        solver.getModelStatus.return_value=native.h.HighsModelStatus.kTimeLimit
        solver.getInfo.return_value=SimpleNamespace(valid=False,primal_solution_status=0,mip_dual_bound=0.,mip_gap=0.,
            simplex_iteration_count=0,ipm_iteration_count=0)
        solver.getSolution.return_value=SimpleNamespace(value_valid=False)
        solver.getRunTime.return_value=0.;solver.version.return_value='test'
        solver.modelStatusToString.return_value='Time limit reached'
        with patch.object(native.h,'_Highs',return_value=solver):
            r=solve_with_start(np.array([1.]),np.array([1]),np.array([0.]),np.array([1.]),csr_matrix([[1.]]),
                np.array([-np.inf]),np.array([1.]),np.array([0.]),{'time_limit':1.})
        self.assertEqual(r.status,1);self.assertIsNone(r.x);self.assertIsNone(r.mip_dual_bound)
        self.assertIsNone(r.mip_gap)

    def test_all_nine_models_match_reference_and_hint_does_not_fix_decisions(self):
        improvement=False
        for f in build_golden_cases():
            first=assemble_first_stage(f.case,f.tasks);ts=tuple(assemble_recourse(f.case,f.tasks,s) for s in f.scenarios)
            for risk in ('SAA','DRO','ARO'):
                m=assemble_extensive(first,ts,risk_model=risk,scenario_ids=f.scenario_ids,
                    probability_set=f.probability_set,empirical_probabilities=f.empirical_probabilities,
                    scenario_cost_constants_usd=f.scenario_cost_constants_usd)
                seed=full_shedding_start(m)
                original=solve_extensive(m,time_limit_seconds=5)
                with tempfile.TemporaryDirectory() as tmp:
                    logfile=Path(tmp)/'solver.log'
                    native=solve_extensive(m,time_limit_seconds=5,zero_presolve=True,native_start=True,solver_log_path=logfile)
                    self.assertTrue(logfile.exists());self.assertIn('HiGHS',logfile.read_text())
                with self.subTest(case=f.name,risk=risk):
                    self.assertTrue(native.optimal,native.as_dict())
                    self.assertAlmostEqual(original.objective_usd,native.objective_usd,places=7)
                    self.assertAlmostEqual(original.lower_bound_usd,native.lower_bound_usd,places=7)
                    self.assertTrue(native.certificate['primal_feasible'])
                    self.assertTrue(native.raw_solver['native_interface']['start_is_not_fixed_decision'])
                    improvement |= native.objective_usd < float(m.c@seed+m.objective_constant_usd)-1e-6
        self.assertTrue(improvement,'Solver must be able to improve the all-zero assignment start')

    def test_bad_start_rejected_before_optimizer(self):
        with self.assertRaisesRegex(ValueError,'violates'):
            solve_with_start(np.array([1.]),np.array([1]),np.array([0.]),np.array([1.]),csr_matrix([[1.]]),
                np.array([-np.inf]),np.array([1.]),np.array([2.]),{'time_limit':1.})

    def test_native_checkpoint_binds_interface_and_preserves_time(self):
        from evcs_experiments.ccg import CCGProblem,run_ccg
        f=build_golden_cases()[0];first=assemble_first_stage(f.case,f.tasks)
        ts=tuple(assemble_recourse(f.case,f.tasks,s) for s in f.scenarios)
        p=CCGProblem(first,ts,f.probability_set,f.scenario_cost_constants_usd)
        with tempfile.TemporaryDirectory() as tmp:
            cp=Path(tmp)/'checkpoint.json'
            a=run_ccg(p,checkpoint_path=cp,wall_seconds=5,max_iterations=1,native_start=True,
                zero_presolve=True,mip_relative_gap=1e-9,optimality_relative_gap=1e-8)
            with self.assertRaisesRegex(ValueError,'source/config'):
                run_ccg(p,checkpoint_path=cp,resume=True,wall_seconds=5,zero_presolve=True,
                    mip_relative_gap=1e-9,optimality_relative_gap=1e-8)
            b=run_ccg(p,checkpoint_path=cp,resume=True,wall_seconds=5,native_start=True,
                zero_presolve=True,mip_relative_gap=1e-9,optimality_relative_gap=1e-8)
            self.assertEqual(b.status,'CONVERGED')
            self.assertAlmostEqual(b.state['upper_bound_usd'],32.12,places=7)
            self.assertGreater(b.state['elapsed_seconds'],a.state['elapsed_seconds'])

    def test_seed_satisfies_all_original_rows_and_is_not_cost_bound_claim(self):
        f=build_golden_cases()[0];first=assemble_first_stage(f.case,f.tasks)
        ts=tuple(assemble_recourse(f.case,f.tasks,s) for s in f.scenarios)
        m=assemble_extensive(first,ts,risk_model='DRO',scenario_ids=f.scenario_ids,probability_set=f.probability_set,
                             scenario_cost_constants_usd=f.scenario_cost_constants_usd)
        v=full_shedding_start(m)
        self.assertTrue(np.all(m.A@v<=m.row_upper+1e-8));self.assertTrue(np.all(m.A@v>=m.row_lower-1e-8))
        self.assertGreater(float(m.c@v),f.expected['DRO']['objective_usd'])

if __name__=='__main__':unittest.main()
