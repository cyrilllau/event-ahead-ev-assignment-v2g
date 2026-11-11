"""Thin Gurobi adapter for the SAA first-stage/theta small master.

This adapter preserves the existing native solver's incumbent and bound gates.
It does not establish formal physical admission, replay a branch tree, evaluate
full Q, or claim that a development test executed a licensed solver. The caller
owns the preceding master-build budget and the subsequent scientific audit.
"""
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from . import native_gurobi
from .campaign_batch import file_reference, verify_reference
from .config import get_paths
from .relaxation_bound import coefficient_sha256


BLAS_ENVIRONMENT = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
EVIDENCE_KIND = 'SAA_L_SHAPED_GUROBI_ADAPTER_EXECUTION_V1'


@dataclass(frozen=True)
class NativeSmallMasterResult:
    raw: object
    evidence_reference: dict
    effective_options: dict
    native_elapsed_seconds: float
    elapsed_seconds: float


def _need(ok, message):
    if not bool(ok):
        raise ValueError(message)


def _number(value, name, *, positive=False):
    _need(type(value) in (int, float) and math.isfinite(value)
          and (value > 0 if positive else value >= 0), 'Invalid ' + name)
    return float(value)


def _sources():
    return {name: file_reference(Path(__file__).with_name(name)) for name in
            ('saa_l_shaped_native.py', 'native_gurobi.py',
             'relaxation_bound.py', 'campaign_batch.py')}


def _small_master(model):
    _need(isinstance(model, dict), 'Expected small-master model dictionary')
    for key in ('first_stage_columns', 'theta_columns', 'cut_rows', 'recourse_columns'):
        _need(type(model.get(key)) is int and model[key] >= 0,
              'Invalid small-master axis: ' + key)
    d, n = model['first_stage_columns'], model['theta_columns']
    _need(d > 0 and n > 0 and model['recourse_columns'] == 0,
          'Only first-stage/theta small masters are supported')
    c = np.asarray(model['c'])
    _need(c.shape == (d + n,) and model['matrix'].shape[1] == d + n
          and model['matrix'].shape[0] >= model['cut_rows'], 'Small-master axes differ')
    _need(np.array_equal(c[d:], np.full(n, 1 / n))
          and np.array_equal(np.asarray(model['integrality'])[d:], np.zeros(n))
          and np.isposinf(np.asarray(model['upper_bounds'])[d:]).all(),
          'Fixed empirical theta objective/domain differs')
    return coefficient_sha256(dict(model, objective_constant_usd=0.))


def solve_native_small_master(model, *, time_limit_seconds, mip_relative_gap,
                              threads=1, output_root):
    """Call the unchanged native adapter once, with no fallback or retries.

    ``time_limit_seconds`` is the remaining caller grant after assembling and
    saving the master, and may not exceed 600 seconds. This adapter subtracts
    its own preparation; native_gurobi subtracts its loading before optimize.
    Post-call evidence work is measured in ``elapsed_seconds`` for the caller's
    parent accounting. The parent deadline must also cover that work.

    ``output_root`` must not exist; a native log and immutable execution record
    are written there. Runtime environment must have been set before importing
    numerical libraries by the original job controller. Checking environment
    values here is a guard, not proof of an already initialized BLAS pool size.
    ``raw`` keeps normalized status and the distinct raw Gurobi record, including
    a missing finite lower bound. The caller must audit the exact source/model.
    """
    began = time.monotonic()
    seconds = _number(time_limit_seconds, 'master time grant', positive=True)
    gap = _number(mip_relative_gap, 'master relative gap')
    _need(seconds <= 600 and gap <= .001, 'Original master ceiling/tolerance exceeded')
    _need(type(threads) is int and 1 <= threads <= 4, 'Master threads must be 1 through 4')
    environment = {key: os.environ.get(key) for key in BLAS_ENVIRONMENT}
    _need(all(value == '1' for value in environment.values()),
          'Original BLAS/OpenMP environment must be single-threaded')
    root = Path(output_root)
    _need(root.is_absolute() and root.resolve() == root, 'Canonical absolute native output root required')
    paths = get_paths()
    for protected in (paths.data_root, paths.frozen_source_root, paths.project_root / 'src'):
        _need(not root.is_relative_to(Path(protected).resolve()), 'Native output overlaps frozen/source content')
    _need(not root.exists(), 'Refusing to overwrite native small-master evidence')
    coefficient = _small_master(model)
    sources = _sources()
    root.mkdir(parents=True)
    log_path = root / 'gurobi.log'
    grant = seconds - (time.monotonic() - began)
    if grant <= 0:
        raise TimeoutError('Native small-master preparation exhausted remaining grant')
    options = dict(time_limit=grant, mip_rel_gap=gap, threads=threads,
                   primal_feasibility_tolerance=1e-8, dual_feasibility_tolerance=1e-8)
    native_started = time.monotonic()
    raw = native_gurobi.solve_milp(model['c'], model['integrality'], model['lower_bounds'],
        model['upper_bounds'], model['matrix'], model['row_lower'], model['row_upper'],
        options=options, log_path=log_path)
    native_elapsed = time.monotonic() - native_started
    # Do not rename status, fabricate a bound from fun, or remove a failed raw
    # certificate. The original adapter already decides which values are usable.
    normalized = dict(status=int(raw.status), success=bool(raw.success), message=str(raw.message),
        x=None if raw.x is None else np.asarray(raw.x).tolist(), fun=raw.fun,
        mip_dual_bound=raw.mip_dual_bound, mip_gap=raw.mip_gap,
        mip_node_count=raw.mip_node_count, limit_status=raw.get('limit_status'))
    body = dict(kind=EVIDENCE_KIND, backend='gurobipy', gurobi_version=list(native_gurobi.GUROBI_VERSION),
        source_references=sources, submitted_coefficient_sha256=coefficient,
        first_stage_columns=model['first_stage_columns'], theta_columns=model['theta_columns'],
        recourse_columns=0, cut_rows=model['cut_rows'], matrix_shape=list(model['matrix'].shape),
        matrix_nnz=int(model['matrix'].nnz),
        requested_time_limit_seconds=seconds, effective_options=options,
        blas_environment=environment, normalized_result=normalized,
        native_record=raw.native_record, native_log_reference=file_reference(log_path),
        native_elapsed_seconds=native_elapsed, elapsed_before_evidence_seconds=time.monotonic() - began,
        native_branch_and_bound_proof_replayed=False, formal_admission=False,
        scientific_acceptance=False)
    record_path = root / 'native_execution.json'
    with record_path.open('x') as handle:
        json.dump(body, handle, indent=2, allow_nan=False)
        handle.write('\n')
    reference = file_reference(record_path)
    return NativeSmallMasterResult(raw=raw, evidence_reference=reference,
        effective_options=options, native_elapsed_seconds=native_elapsed,
        elapsed_seconds=time.monotonic() - began)


def verify_native_small_master_evidence(reference, *, expected_coefficient_sha256):
    """Verify retained native provenance without loading Gurobi or solving.

    The supplied execution pin and coefficient hash must already be trusted by
    the caller. A valid failure record is returned for diagnosis; this function
    does not turn a MEM_LIMIT, missing bound, or rejected incumbent into an
    accepted optimization result. Original-matrix arithmetic belongs to the
    enclosing independent small-master audit.
    """
    path = Path(reference['path'])
    _need(path.is_absolute() and path.resolve() == path
          and path.name == 'native_execution.json', 'Aliased or foreign native execution path')
    verify_reference(reference)
    body = json.loads(path.read_text())
    _need(body['kind'] == EVIDENCE_KIND and body['backend'] == 'gurobipy'
          and body['gurobi_version'] == list(native_gurobi.GUROBI_VERSION),
          'Native adapter kind/backend/version differs')
    _need(body['submitted_coefficient_sha256'] == expected_coefficient_sha256,
          'Native submitted coefficients differ')
    _need(body['source_references'] == _sources(), 'Native adapter source differs')
    for source in body['source_references'].values():
        verify_reference(source)
    log_reference = body['native_log_reference']
    _need(Path(log_reference['path']) == path.parent / 'gurobi.log', 'Foreign native log path')
    _need(Path(log_reference['path']).resolve() == Path(log_reference['path']), 'Aliased native log')
    verify_reference(log_reference)
    _need(body['formal_admission'] is False and body['scientific_acceptance'] is False
          and body['native_branch_and_bound_proof_replayed'] is False,
          'Native adapter scope changed')
    _need(body['blas_environment'] == {key: '1' for key in BLAS_ENVIRONMENT},
          'Retained BLAS/OpenMP environment differs')
    options = body['effective_options']
    _need(set(options) == {'time_limit', 'mip_rel_gap', 'threads',
                          'primal_feasibility_tolerance', 'dual_feasibility_tolerance'},
          'Native submitted options differ')
    seconds = _number(body['requested_time_limit_seconds'], 'retained master grant', positive=True)
    grant = _number(options['time_limit'], 'retained effective grant', positive=True)
    gap = _number(options['mip_rel_gap'], 'retained master gap')
    _need(grant <= seconds <= 600 and gap <= .001
          and type(options['threads']) is int and 1 <= options['threads'] <= 4
          and options['primal_feasibility_tolerance'] == options['dual_feasibility_tolerance'] == 1e-8,
          'Native retained resources/tolerances differ')
    native = body['native_record']
    _need(native['backend'] == 'gurobipy' and native['gurobi_version'] == list(native_gurobi.GUROBI_VERSION)
          and native['requested_time_limit_seconds'] == grant
          and native['log_path'] == log_reference['path'], 'Original native invocation differs')
    _need(native['original_matrix_shape'] == body['matrix_shape']
          and native['original_matrix_nnz'] == body['matrix_nnz'],
          'Original native matrix dimensions differ')
    _, original_parameters = native_gurobi._parameters(options)
    actual_parameters = native['effective_options']
    _need(all(actual_parameters.get(key) == value for key, value in original_parameters.items())
          and actual_parameters.get('LogFile') == log_reference['path'], 'Original native parameters differ')
    run_limit = _number(native['effective_run_time_limit_seconds'], 'native remaining run time')
    load_seconds = _number(native['load_seconds'], 'native load seconds')
    _need(run_limit == max(0., grant - load_seconds), 'Native loading time was not deducted')
    if run_limit > 0:
        _need(actual_parameters.get('TimeLimit') == run_limit, 'Original optimize grant differs')
    _need(set(actual_parameters) == set(original_parameters) | {'LogFile'}
          | ({'TimeLimit'} if run_limit > 0 else set()), 'Additional native parameters appeared')
    _number(body['native_elapsed_seconds'], 'native elapsed seconds')
    before_evidence = _number(body['elapsed_before_evidence_seconds'], 'adapter elapsed seconds')
    _need(before_evidence >= body['native_elapsed_seconds'], 'Native elapsed exceeds whole adapter time')
    normalized = body['normalized_result']
    _need(type(normalized['status']) is int and type(normalized['success']) is bool,
          'Invalid normalized native status')
    raw_status = native['raw_model_status']
    _need(type(raw_status) is int, 'Invalid raw Gurobi status')
    if native['status'] == 'MODEL_LOADING_TIME_LIMIT':
        _need(raw_status == 1 and native['solver_run_seconds'] == 0
              and native['valid_incumbent'] is False
              and run_limit == 0 and normalized['status'] == 1
              and normalized['limit_status'] == 'TIME_LIMIT'
              and normalized['x'] is None and normalized['fun'] is None
              and normalized['mip_dual_bound'] is None and normalized['success'] is False
              and normalized['mip_gap'] is None and normalized['mip_node_count'] is None
              and normalized['message'] == 'Gurobi native: model loading exhausted time budget',
              'Loading-time stop was reinterpreted')
    else:
        _need(native['status'] == native_gurobi._STATUSES.get(raw_status, 'STATUS_' + str(raw_status)),
              'Original raw Gurobi status name differs')
        status = (0 if raw_status == 2 else 1 if raw_status in native_gurobi._LIMIT_STATUSES
                  else 2 if raw_status == 3 else 3 if raw_status == 5 else 4)
        if native['certificate_failures']:
            status = 4
        _need(normalized['status'] == status, 'Normalized status reinterprets native result')
        limit = native['status'] if raw_status in native_gurobi._LIMIT_STATUSES else None
        _need(normalized['limit_status'] == limit, 'Original limit status was renamed')
        _need(normalized['mip_dual_bound'] == native['solver_bound_without_constant_usd'],
              'Original certified bound changed')
        valid = native['valid_incumbent']
        _need(type(valid) is bool and ((normalized['x'] is not None) == valid)
              and normalized['fun'] == (native['incumbent_without_constant_usd'] if valid else None),
              'Original incumbent availability changed')
        _need(normalized['success'] == (status == 0 and valid and normalized['mip_dual_bound'] is not None),
              'Original native success was reinterpreted')
        expected_message = 'Gurobi native: ' + native['status']
        if native['certificate_failures']:
            expected_message += '; certificate failed: ' + '; '.join(native['certificate_failures'])
        _need(normalized['message'] == expected_message, 'Original native message changed')
    for key in ('fun', 'mip_dual_bound', 'mip_gap', 'mip_node_count'):
        value = normalized[key]
        _need(value is None or type(value) in (int, float) and math.isfinite(value),
              'Nonfinite normalized native value: ' + key)
    return body
