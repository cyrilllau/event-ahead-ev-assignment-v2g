"""Finite-support recourse-copy C&CG (paper equations 43--52).

The master incumbent is never a lower bound. Every candidate is evaluated by
all original fixed-plan LPs; its own RMP dual and the independently solved
probability dual are retained separately. Checkpoints resume between LPs, not
inside a HiGHS search tree; cold master restarts retain cumulative wall time.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .extensive import assemble_extensive, solve_extensive
from .first_stage import FirstStagePlan, SparseFirstStage
from .probability import ProbabilitySet, solve_worst_distribution
from .probability_phase1 import solve_phase1
from .random_streams import canonical_hash
from .recourse import solve_fixed_plan
from .runtime.jobctl import atomic_json
from .lazy_recourses import LazyRecourses


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _plan_dict(plan):
    return dict(x=plan.x.tolist(), r=plan.r.tolist(), task_sha256=plan.task_sha256,
                config_hash=plan.config_hash)


def _plan(value):
    return FirstStagePlan(**value)


def _engine_hash():
    names = ('ccg.py', 'extensive.py', 'first_stage.py', 'recourse.py',
             'probability.py', 'probability_phase1.py', 'random_streams.py',
             'lazy_recourses.py', 'zero_presolve.py', 'native_highs.py', 'native_gurobi.py')
    return canonical_hash({name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                           for name in names})


@dataclass(frozen=True)
class CCGProblem:
    first_stage: SparseFirstStage
    recourses: tuple
    probability_set: ProbabilitySet
    scenario_cost_constants_usd: np.ndarray | None = None
    risk_model: str = 'DRO'

    def __post_init__(self):
        _need(self.risk_model in ('DRO', 'ARO'), 'C&CG supports DRO/ARO; SAA uses its full reference')
        ids = self.probability_set.support_ids
        templates = self.recourses if isinstance(self.recourses, LazyRecourses) else tuple(self.recourses)
        _need(len(templates) == len(ids), 'Recourse/support length mismatch')
        if isinstance(templates, LazyRecourses):
            _need(templates.scenario_ids == ids and templates.first_stage_sha256 == self.first_stage.matrix_sha256,
                  'Lazy recourse/support/first-stage identity mismatch')
        else:
            for template in templates:
                _need(template.first_stage.matrix_sha256 == self.first_stage.matrix_sha256,
                      'Recourse first-stage identity mismatch')
        constants = np.asarray(np.zeros(len(ids)) if self.scenario_cost_constants_usd is None
                               else self.scenario_cost_constants_usd, dtype=float)
        _need(constants.shape == (len(ids),) and np.isfinite(constants).all(), 'Invalid scenario constants')
        immutable = np.frombuffer(constants.tobytes(), dtype=float)
        object.__setattr__(self, 'recourses', templates)
        object.__setattr__(self, 'scenario_cost_constants_usd', immutable)

    @property
    def ambiguity(self):
        a = self.probability_set
        if self.risk_model == 'ARO':
            return ProbabilitySet(a.support_ids, (), np.empty((len(a.support_ids), 0)), [], [])
        return a

    @property
    def identity(self):
        if isinstance(self.recourses, LazyRecourses):
            return {'first_stage_sha256': self.first_stage.matrix_sha256,
                    'scenario_ids': list(self.probability_set.support_ids),
                    'lazy_input_binding': self.recourses.identity,
                    'template_hashes_deferred_until_assembly': True,
                    'input_ambiguity_sha256': self.probability_set.ambiguity_sha256,
                    'effective_ambiguity_sha256': self.ambiguity.ambiguity_sha256,
                    'constants_usd': self.scenario_cost_constants_usd.tolist(), 'risk_model': self.risk_model}
        return {'first_stage_sha256': self.first_stage.matrix_sha256,
                'scenario_ids': list(self.probability_set.support_ids),
                'scenario_sha256s': [v.scenario_sha256 for v in self.recourses],
                'template_sha256s': [v.template_sha256 for v in self.recourses],
                'input_ambiguity_sha256': self.probability_set.ambiguity_sha256,
                'effective_ambiguity_sha256': self.ambiguity.ambiguity_sha256,
                'constants_usd': self.scenario_cost_constants_usd.tolist(), 'risk_model': self.risk_model}


@dataclass(frozen=True)
class CCGResult:
    state: dict

    @property
    def status(self):
        return self.state['status']

    @property
    def best_plan(self):
        return None if self.state['best'] is None else _plan(self.state['best']['plan'])

    def as_dict(self):
        return json.loads(json.dumps(self.state, allow_nan=False))


def run_ccg(problem, *, checkpoint_path=None, resume=False, wall_seconds=600,
            rmp_time_limit_seconds=600, oracle_time_limit_seconds=60,
            mip_relative_gap=.001, optimality_relative_gap=.005,
            max_iterations=1000, evidence_sink=None, zero_presolve=False,
            native_start=False, solver_log_directory=None, master_backend='highs', optimizer_threads=1):
    """Run a bounded invocation, preserving partial full-support evaluations.

    evidence_sink(event, iteration, **objects) must persist raw solver/matrix
    witnesses before checkpointing. A production evidence exporter is required
    for scientific acceptance; the lightweight checkpoint alone is not a proof.
    max_iterations limits this invocation, not the lifetime count. Each resume
    explicitly grants a new wall budget and retains all previous elapsed time.
    """
    started = time.monotonic()
    _need(isinstance(problem, CCGProblem), 'Expected CCGProblem')
    for value in (wall_seconds, rmp_time_limit_seconds, oracle_time_limit_seconds):
        _need(not isinstance(value, bool) and np.isfinite(value) and value > 0, 'Invalid time budget')
    for value in (mip_relative_gap, optimality_relative_gap):
        _need(not isinstance(value, bool) and np.isfinite(value) and value >= 0, 'Invalid gap')
    _need(isinstance(max_iterations, int) and not isinstance(max_iterations, bool) and max_iterations > 0,
          'Invalid iteration budget')
    _need(master_backend in ('highs', 'gurobi'), 'Unknown master backend')
    _need(isinstance(optimizer_threads,int) and not isinstance(optimizer_threads,bool)
          and 1 <= optimizer_threads <= 4, 'Optimizer threads must be 1..4')
    _need(master_backend != 'highs' or optimizer_threads == 1, 'HiGHS reference uses one thread')
    _need(master_backend != 'gurobi' or solver_log_directory is not None, 'Gurobi requires solver logs')
    path = None if checkpoint_path is None else Path(checkpoint_path)
    _need(not resume or path is not None, 'Resume requires checkpoint')
    config = {'mip_relative_gap': float(mip_relative_gap), 'optimality_relative_gap': float(optimality_relative_gap),
              'separation_absolute_usd': 1e-4, 'separation_relative': 1e-8}
    if master_backend != 'highs':
        config.update(master_backend=master_backend,optimizer_threads=optimizer_threads)
    if zero_presolve:
        config['zero_presolve'] = 'exact_zero_singleton_v1'
    if native_start:
        config['native_start'] = 'checked_full_shedding_hint_v1'
    identity = canonical_hash(problem.identity)
    if resume:
        envelope = json.loads(path.read_text())
        state = envelope['state']
        _need(envelope['sha256'] == canonical_hash(state), 'Checkpoint checksum mismatch')
        _need(state['problem_sha256'] == identity and state['engine_sha256'] == _engine_hash()
              and state['config'] == config, 'Checkpoint problem/source/config mismatch')
        _need(state['status'] not in ('INVALID_RMP', 'RMP_RESIDUAL_FAILED', 'BOUND_CONTRADICTION',
                                     'PHASE1_FAILED', 'RECOURSE_FAILED', 'PROBABILITY_FAILED', 'RMP_FAILED'),
              'Failed checkpoint requires diagnosis before resume')
        if state['status'] == 'CONVERGED':
            return CCGResult(state)
        _need(state['best'] is None or problem.first_stage.evaluate(_plan(state['best']['plan'])).feasible,
              'Checkpoint best plan is infeasible')
    else:
        _need(path is None or not path.exists(), 'Refusing to overwrite existing checkpoint')
        state = {'schema_version': 1, 'problem_sha256': identity, 'identity': problem.identity,
                 'engine_sha256': _engine_hash(), 'config': config, 'status': 'INITIALIZED',
                 'elapsed_seconds': 0., 'sessions': [], 'phase1': None, 'active_indices': [],
                 'trace': [], 'pending': None, 'lower_bound_usd': None, 'upper_bound_usd': None,
                 'best': None, 'next_iteration': 1, 'next_mip_gap': float(mip_relative_gap),
                 'checkpoint_kind': 'algorithm and completed LPs; no solver search-tree warm start'}
    previous_elapsed = state['elapsed_seconds']
    state['sessions'].append({'wall_budget_seconds': float(wall_seconds), 'elapsed_seconds': 0.})
    a = problem.ambiguity
    ids = a.support_ids
    sink = evidence_sink or (lambda *args, **kwargs: None)

    def remaining(limit=None):
        left = max(0., float(wall_seconds) - (time.monotonic()-started))
        return left if limit is None else min(left, float(limit))

    def save(status=None):
        if status is not None:
            state['status'] = status
        elapsed = time.monotonic()-started
        state['elapsed_seconds'] = previous_elapsed + elapsed
        state['sessions'][-1]['elapsed_seconds'] = elapsed
        if path is not None:
            atomic_json(path, {'state': state, 'sha256': canonical_hash(state)})
        return CCGResult(json.loads(json.dumps(state, allow_nan=False)))

    if state['phase1'] is None:
        if remaining() <= 0:
            return save('TIME_LIMIT')
        phase = solve_phase1(a, time_limit_seconds=remaining(oracle_time_limit_seconds))
        sink('phase1', 0, result=phase)
        state['phase1'] = phase.as_dict()
        if not phase.feasible or phase.probabilities is None or np.any(phase.probabilities < 0):
            state['last_incomplete_phase1'] = phase.as_dict()
            state['phase1'] = None
            return save('PHASE1_FAILED' if phase.status != 'TIME_LIMIT' else 'TIME_LIMIT')
        state['active_indices'] = list(phase.initial_support_indices)
        # The full nonzero numerical witness is retained even when nonbasic.
        _need(bool(state['active_indices']), 'Empty initial support')
        save('RUNNING')
    elif not state['active_indices']:
        # A phase-I timeout did not establish a usable initialization.
        state['phase1'] = None
        return save('PHASE1_FAILED')

    completed = 0
    while completed < max_iterations:
        if remaining() <= 0:
            return save('TIME_LIMIT')
        k = state['next_iteration']
        if state['pending'] is None:
            indices = state['active_indices']
            restricted = ProbabilitySet(tuple(ids[i] for i in indices), a.feature_ids,
                                        a.features[indices], a.lower, a.upper)
            model = assemble_extensive(problem.first_stage, tuple(problem.recourses[i] for i in indices),
                risk_model=problem.risk_model, scenario_ids=restricted.support_ids,
                probability_set=restricted, scenario_cost_constants_usd=problem.scenario_cost_constants_usd[indices])
            if remaining() <= 0:
                return save('TIME_LIMIT')
            extra = {'zero_presolve': True} if zero_presolve else {}
            if native_start:
                extra.update(native_start=True,solver_log_path=None if solver_log_directory is None else
                             Path(solver_log_directory)/f'rmp_{k:04d}.log')
            if master_backend != 'highs':
                extra.update(master_backend=master_backend,optimizer_threads=optimizer_threads,
                             solver_log_path=Path(solver_log_directory)/f'rmp_{k:04d}.log')
            master = solve_extensive(model, time_limit_seconds=remaining(rmp_time_limit_seconds),
                                     mip_relative_gap=state['next_mip_gap'],
                                     **extra)
            sink('rmp', k, model=model, result=master)
            cert = master.certificate
            if master.solver_status not in (0, 1):
                state['last_failure'] = master.as_dict()
                return save('RMP_FAILED')
            if (master.raw_solver.get('raw_lower_bound_without_constant_usd') is not None
                    and master.lower_bound_usd is None):
                state['last_failure'] = master.as_dict()
                return save('INVALID_RMP')
            if master.lower_bound_usd is not None and cert['solver_bound_usable']:
                old = state['lower_bound_usd']
                state['lower_bound_usd'] = master.lower_bound_usd if old is None else max(old, master.lower_bound_usd)
            if master.solution is None:
                state['last_incomplete_master'] = master.as_dict()
                # Keep a distinct iteration ID for a future cold search.
                state['next_iteration'] += 1
                return save(master.status if master.solver_status == 1 else 'INVALID_RMP')
            if not (cert['primal_feasible'] and cert['objective_matches_solver'] and
                    problem.first_stage.evaluate(master.solution.plan).feasible):
                state['last_failure'] = master.as_dict()
                return save('INVALID_RMP')
            sol = master.solution
            state['pending'] = {'iteration': k, 'active_indices': list(indices), 'plan': _plan_dict(sol.plan),
                'rmp_model_sha256': model.model_sha256, 'rmp': master.as_dict(),
                'raw_rmp_dual': {'theta0': sol.theta0, 'alpha': None if sol.alpha is None else sol.alpha.tolist(),
                                'beta': None if sol.beta is None else sol.beta.tolist(), 'threshold': sol.risk_threshold},
                'q_usd': [], 'recourse_certificates': []}
            save('RUNNING')
        pending = state['pending']
        plan = _plan(pending['plan'])
        for index in range(len(pending['q_usd']), len(ids)):
            if remaining() <= 0:
                return save('TIME_LIMIT')
            template = problem.recourses[index]
            oracle_budget = remaining(oracle_time_limit_seconds)
            if oracle_budget <= 0:
                return save('TIME_LIMIT')
            result = solve_fixed_plan(template, plan, time_limit_seconds=oracle_budget)
            sink('recourse', k, index=index, plan=plan, result=result)
            if not (result.optimal and result.certificate.get('optimal_certificate_valid', False)):
                pending['last_incomplete_recourse'] = {'index': index, 'result': result.as_dict()}
                return save('TIME_LIMIT' if result.solver_status == 1 else 'RECOURSE_FAILED')
            pending['q_usd'].append(float(result.objective_usd + problem.scenario_cost_constants_usd[index]))
            # Large raw marginal/vector arrays belong in the evidence sink.
            pending['recourse_certificates'].append({'index': index, 'scenario_id': ids[index],
                'template_sha256': problem.recourses[index].template_sha256,
                'solver_result_sha256': canonical_hash(result.as_dict()),
                'optimal_certificate_valid': True, 'objective_usd': result.objective_usd})
            save('RUNNING')
        if remaining() <= 0:
            return save('TIME_LIMIT')
        q = np.asarray(pending['q_usd'])
        probability = solve_worst_distribution(a, q, q_support_ids=ids,
                                               time_limit_seconds=remaining(oracle_time_limit_seconds))
        sink('probability', k, plan=plan, result=probability)
        if not probability.certified_optimal:
            pending['last_incomplete_probability'] = probability.as_dict()
            return save('TIME_LIMIT' if probability.status == 'TIME_LIMIT' else 'PROBABILITY_FAILED')
        first_cost = problem.first_stage.evaluate(plan).objective_usd
        candidate = float(first_cost + probability.upper_bound_usd)
        dual = pending['raw_rmp_dual']
        residual = q-dual['threshold'] if problem.risk_model == 'ARO' else (
            q-dual['theta0'] + a.features@(np.asarray(dual['alpha'])-np.asarray(dual['beta'])))
        scale = max(1., float(np.max(abs(q))))
        sep_tol = 1e-4 + 1e-8*scale
        included = pending['active_indices']
        omitted = [i for i in range(len(ids)) if i not in included]
        worst_new = None if not omitted else omitted[int(np.argmax(residual[omitted]))]
        maximum = max(0., float(np.max(residual)))
        included_violation = max(0., float(np.max(residual[included])))
        if state['upper_bound_usd'] is None or candidate < state['upper_bound_usd']:
            state['upper_bound_usd'] = candidate
            state['best'] = {'iteration': k, 'plan': pending['plan'], 'plan_sha256': canonical_hash(pending['plan']),
                'q_usd': q.tolist(), 'probability': probability.as_dict(), 'objective_usd': candidate,
                'Fpre_usd': first_cost, 'original_rmp_dual': dual, 'original_rmp_residual_usd': residual.tolist()}
        lb, ub = state['lower_bound_usd'], state['upper_bound_usd']
        gap = None if lb is None else max(0., ub-lb)/max(1., abs(ub))
        row = {'iteration': k, 'active_indices': list(included), 'plan_sha256': canonical_hash(pending['plan']),
            'rmp_model_sha256': pending['rmp_model_sha256'], 'rmp': pending['rmp'],
            'raw_rmp_dual': dual, 'q_usd': q.tolist(), 'residual_usd': residual.tolist(),
            'probability_audit': probability.as_dict(), 'candidate_upper_bound_usd': candidate,
            'lower_bound_usd': lb, 'upper_bound_usd': ub, 'best_iteration': state['best']['iteration'],
            'gap_relative': gap, 'max_full_residual_usd': maximum, 'max_included_residual_usd': included_violation,
            'max_new_residual_usd': None if worst_new is None else float(residual[worst_new]),
            'no_omitted_scenarios': not omitted, 'separation_scale_usd': scale,
            'separation_tolerance_usd': sep_tol, 'added_scenario_index': None,
            'elapsed_seconds': previous_elapsed + time.monotonic()-started}
        terminal = None
        if lb is not None and lb > ub + 1e-4 + 1e-7*max(1., abs(lb), abs(ub)):
            terminal = 'BOUND_CONTRADICTION'
        elif included_violation > sep_tol:
            terminal = 'RMP_RESIDUAL_FAILED'
        elif maximum <= sep_tol and gap is not None and gap <= optimality_relative_gap:
            terminal = pending['rmp']['status'] if pending['rmp']['solver_status'] == 1 else 'CONVERGED'
        elif worst_new is not None and residual[worst_new] > sep_tol:
            state['active_indices'].append(worst_new)
            row['added_scenario_index'] = worst_new
        elif pending['rmp']['solver_status'] == 1:
            terminal = pending['rmp']['status']
        elif state['next_mip_gap'] > 1e-9:
            state['next_mip_gap'] = max(1e-9, state['next_mip_gap']/10.)
        else:
            terminal = 'RMP_NEEDS_MORE_WORK'
        # Preserve native time/memory/interruption limits, even after adding.
        if terminal is None and pending['rmp']['solver_status'] == 1:
            terminal = pending['rmp']['status']
        if terminal == 'CONVERGED' and remaining() <= 0:
            terminal = 'TIME_LIMIT'
        state['trace'].append(row)
        sink('iteration', k, record=row)
        state['pending'] = None
        state['next_iteration'] += 1
        completed += 1
        save(terminal or 'RUNNING')
        if terminal is not None:
            return CCGResult(state)
    return save('ITERATION_LIMIT')
