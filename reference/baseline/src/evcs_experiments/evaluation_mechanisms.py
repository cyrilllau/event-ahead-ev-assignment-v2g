"""Fixed-plan mechanism summaries for the IEEE33 experiment protocol.

The caller supplies the planning-certified energy estimand and R1 sample
identities. Shared routes are matched through explicit physical-corridor IDs,
including reverse arcs; station identities never define a shared route.
"""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np

from .random_streams import canonical_hash


def _identifier(value, name):
    if isinstance(value, str) and value:
        return value
    if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    raise ValueError(f'{name} must be a nonempty string or integer ID')


def _ids(values, name):
    if isinstance(values, (str, bytes)):
        raise ValueError(f'{name} must be an ordered ID sequence')
    result = tuple(_identifier(v, name) for v in values)
    if len(set(result)) != len(result):
        raise ValueError(f'{name} contains duplicate IDs')
    return result


def _order(value):
    return (type(value).__name__, str(value))


def planning_optimism(planning_certified_kwh, r1_certified_kwh, *, sample_ids):
    """(Planning prediction - R1 empirical mean) / planning prediction.

The supplied planning prediction may come from a declared probability law or
another already-frozen planning estimand. This helper never substitutes
procured energy for certified energy, or chooses the planning law itself.
"""
    if (not isinstance(planning_certified_kwh, Real)
            or isinstance(planning_certified_kwh, (bool, np.bool_))
            or not np.isfinite(planning_certified_kwh) or planning_certified_kwh < 0):
        raise ValueError('Planning certified energy must be finite nonnegative kWh')
    predicted = float(planning_certified_kwh)
    samples = _ids(sample_ids, 'sample_ids')
    observed = np.asarray(r1_certified_kwh, dtype=float)
    if (observed.shape != (len(samples),) or not np.isfinite(observed).all()
            or np.any(observed < 0)):
        raise ValueError('R1 certified energies must match sample IDs and be finite nonnegative kWh')
    reasons = []
    if predicted == 0:
        reasons.append('zero planning-certified-energy denominator')
    if not samples:
        reasons.append('no R1 samples')
    mean = None if not samples else float(observed.mean())
    ratio = None if reasons else float((predicted - mean) / predicted)
    identity = dict(planning_certified_kwh=predicted, sample_ids=list(samples),
                    r1_certified_kwh=observed.tolist())
    return dict(status='NA' if reasons else 'DEFINED', na_reasons=reasons,
        optimism_ratio=ratio, optimism_percent=None if ratio is None else 100. * ratio,
        planning_certified_kwh=predicted, r1_mean_certified_kwh=mean,
        sample_count=len(samples), sample_ids=list(samples), input_sha256=canonical_hash(identity),
        definition='(planning certified kWh - R1 mean certified kWh) / planning certified kWh',
        planning_estimand_policy='caller must bind the declared planning prediction and its probability law')


def shared_corridor_cofailure(task_ids, selection, routes_by_task, arc_to_corridor,
                             failures, *, sample_ids, failure_definition='route_noncompletion'):
    """Summarize selected unordered task pairs sharing a physical corridor.

``failures[s, h]`` is a binary indicator on the declared common sample axis;
the default route-noncompletion interpretation is ``1 - Gamma[s, h]``.
Each pair is counted once even when its routes share multiple corridors.
Pair-specific joint failure and product of empirical marginal probabilities
use the same samples. Aggregate rates average the pair-specific quantities.
"""
    tasks = _ids(task_ids, 'task_ids')
    samples = _ids(sample_ids, 'sample_ids')
    x = np.asarray(selection, dtype=float)
    if (x.shape != (len(tasks),) or not np.isfinite(x).all()
            or np.any(x < -1e-8) or np.any(x > 1 + 1e-8)
            or np.max(np.abs(x - np.rint(x)), initial=0.) > 1e-8):
        raise ValueError('Selection must match task IDs and satisfy the existing binary tolerance')
    if not isinstance(failure_definition, str) or not failure_definition:
        raise ValueError('Failure definition must be an explicit nonempty label')
    if not isinstance(routes_by_task, Mapping) or set(routes_by_task) != set(tasks):
        raise ValueError('Fixed routes must map exactly the full task-ID axis')
    if not isinstance(arc_to_corridor, Mapping):
        raise ValueError('An explicit arc-to-physical-corridor mapping is required')
    mapping = {_identifier(a, 'arc ID'): _identifier(c, 'physical corridor ID')
               for a, c in arc_to_corridor.items()}
    routes, corridor_sets = {}, []
    for task in tasks:
        arcs = routes_by_task[task]
        if isinstance(arcs, (str, bytes)):
            raise ValueError('Each fixed route must be an ordered arc-ID sequence')
        arcs = tuple(_identifier(a, 'route arc ID') for a in arcs)
        if any(a not in mapping for a in arcs):
            raise ValueError('Every route arc must have a physical corridor mapping')
        routes[task] = arcs
        corridor_sets.append(set(mapping[a] for a in arcs))
    raw = np.asarray(failures)
    if (raw.shape != (len(samples), len(tasks))
            or not np.issubdtype(raw.dtype, np.number) and raw.dtype != np.bool_
            or not np.all((raw == 0) | (raw == 1))):
        raise ValueError('Failures must be a binary [sample, task] matrix on the declared axes')
    failed = raw.astype(bool)
    selected = np.flatnonzero(x > .5)
    pairs = []
    for position, i in enumerate(selected):
        for j in selected[position + 1:]:
            shared = corridor_sets[i] & corridor_sets[j]
            if not shared:
                continue
            count_i = int(failed[:, i].sum()); count_j = int(failed[:, j].sum())
            joint_count = int(np.sum(failed[:, i] & failed[:, j]))
            p_i = None if not samples else count_i / len(samples)
            p_j = None if not samples else count_j / len(samples)
            joint = None if not samples else joint_count / len(samples)
            product = None if not samples else p_i * p_j
            pairs.append(dict(task_i=tasks[i], task_j=tasks[j],
                shared_physical_corridor_ids=sorted(shared, key=_order),
                shared_physical_corridor_count=len(shared), scenario_count=len(samples),
                task_i_failure_count=count_i, task_j_failure_count=count_j,
                joint_failure_count=joint_count, marginal_i_probability=p_i,
                marginal_j_probability=p_j, joint_failure_probability=joint,
                marginal_product=product, excess_over_marginal_product=None if not samples else joint-product,
                status='DEFINED' if samples else 'NA',
                na_reason=None if samples else 'no scenarios'))
    reasons = []
    if not pairs:
        reasons.append('no selected task pairs sharing a physical corridor')
    if not samples:
        reasons.append('no scenarios')
    joint = None if reasons else float(np.mean([p['joint_failure_probability'] for p in pairs]))
    product = None if reasons else float(np.mean([p['marginal_product'] for p in pairs]))
    identity = dict(task_ids=list(tasks), selection=x.tolist(), sample_ids=list(samples),
        routes=[dict(task_id=task, arc_ids=list(routes[task])) for task in tasks],
        arc_to_physical_corridor=[dict(arc_id=a, physical_corridor_id=mapping[a])
                                 for a in sorted(mapping, key=_order)],
        failures=failed.astype(int).tolist(), failure_definition=failure_definition)
    return dict(status='NA' if reasons else 'DEFINED', na_reasons=reasons,
        selected_task_count=len(selected), selected_candidate_pair_count=len(selected)*(len(selected)-1)//2,
        shared_pair_count=len(pairs), scenario_count=len(samples),
        pair_scenario_count=len(pairs)*len(samples),
        joint_failure_count=sum(p['joint_failure_count'] for p in pairs),
        joint_failure_probability=joint, marginal_product=product,
        excess_over_marginal_product=None if reasons else joint-product,
        pairs=pairs, input_sha256=canonical_hash(identity), sample_ids=list(samples),
        failure_definition=failure_definition,
        aggregation='equal-weight mean over unordered selected shared-physical-corridor pairs; common samples within every pair',
        overlap_rule='physical corridor intersection; opposite directed arcs share their supplied physical ID; each task pair counted once')
