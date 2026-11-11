"""Exact user-accepted Case0.1 comparison admission; no normal-gap override.

This distinct admission joins existing numerical certificates, original physical
checks and a frozen evaluator inventory. It cannot confer normal planning PASS.
"""
import json
import sys
from pathlib import Path

from .campaign_batch import file_reference, verify_reference
from .case0_accepted_group import load_accepted_group
from .case0_campaign import active_config, METHODS
from .first_stage import FirstStagePlan
from .evaluation_metrics import plan_identity

KIND = 'CASE0_USER_ACCEPTED_COMPARISON_ADMISSION_V1'
OVERLAYS = ('case0_accepted_plan', 'case0_aro_methods', 'case0_accepted_group',
            'case0_accepted_evaluation', 'case0_evaluation_runner')


def need(ok, message):
    if not bool(ok):
        raise ValueError(message)


def read(ref):
    verify_reference(ref)
    return json.loads(Path(ref['path']).read_text())


def evaluation_sources():
    """Bind inherited installed sources AND every explicitly loaded overlay."""
    from . import case0_evaluation
    paths = set(Path(case0_evaluation.__file__).parent.rglob('*.py'))
    for name in OVERLAYS:
        module = sys.modules.get('evcs_experiments.' + name)
        need(module is not None, 'Required comparison module not loaded: ' + name)
        paths.add(Path(module.__file__))
    return [file_reference(p) for p in sorted(paths)]


def validate_view(method, row, physical, identity):
    result = row['comparison_result']
    need(result['method_id'] == method and result['accepted_for_comparison'] is True
         and result['gap_passed'] is False and result['normal_gap_target'] == .005,
         'User acceptance cannot relabel normal convergence')
    need(0 <= result['gap'] <= .01 and result['numerical_valid'] is True,
         'Comparison precision/numerical evidence differs')
    plan = FirstStagePlan(**result['plan'])
    need(plan_identity(plan) == result['plan_sha256'] == physical['plan_sha256'],
         'Accepted plan differs from original physical check')
    need(physical['first_stage_check']['passed'] is True
         and physical['input_identity_reference'] == row['input_identity_reference'],
         'Original first-stage check missing or foreign')
    need(plan.task_sha256 == identity['task_sha256']
         and identity['lead_h'] == (0 if method == 'OR_DRO' else 1)
         and identity['profile'] == 'expanded48-v1'
         and identity['target'] == 'zeta_2020'
         and len(set(identity['vehicle_ids'])) == len(identity['vehicle_ids']) == 48,
         'Accepted input information/fleet differs')
    return result


def validate_accepted_comparison(root, acceptance_reference, *, _attempt_reference=None):
    root = Path(root).resolve()
    state = json.loads((root / 'state.json').read_text())
    config = active_config(root)
    card = read(acceptance_reference)
    need(root.name == 'ieee33-v54-q1-case0-expanded' and state['size_index'] == 0
         and card['kind'] == KIND and card['accepted_for_comparison'] is True
         and card['normal_planning_passed'] is False, 'Exact Case0.1 comparison scope required')
    need(card['accepted_approximate_plans'] == state['accepted_approximate_plans']
         and card['config_reference'] == state['active_config_reference']
         and card['original_config_reference'] == file_reference(root / 'config.lock.json'),
         'Accepted source/config/user decisions changed')
    need(not state.get('audit_reservation_reference') and not state.get('accepted_group_physical_pending'),
         'Unsettled pre-OOS audit')
    pending = state.get('pending_evaluation_attempt')
    need((pending is None and state['active_configuration'] is None) or
         (_attempt_reference is not None and pending == _attempt_reference
          and state['active_configuration'] == 'CASE0_EVALUATION:PREPARATION'),
         'Unsettled planning/evaluation operation')
    need(state['status'] in ('FOUR_PLANS_USER_ACCEPTED_PENDING_COMPARISON_INTEGRATION',
         'CASE0_EVALUATION_IN_PROGRESS', 'CASE0_EVALUATION_COMPLETE_PENDING_REVIEW'),
         'Campaign not ready for accepted comparison')
    for key in ('configuration_spent_seconds', 'configuration_master_spent_seconds'):
        need(card[key] == state[key], 'Accepted planning budget was changed')
    sources = evaluation_sources()
    need(card['evaluation_source_references'] == sources, 'Evaluator source inventory changed')
    review = read(card['evaluation_review_reference'])
    need(review['passed'] is True and review['evaluation_source_references'] == sources
         and review['scope'] == 'Exact accepted comparison admission; unchanged fixed-plan evaluator',
         'Evaluator source review differs')
    group = read(card['group_reference'])
    need(group == load_accepted_group(root, state), 'Existing numerical/acceptance evidence changed')
    settlement = read(card['physical_settlement_reference'])
    need(state['accepted_group_physical_reference'] == card['physical_settlement_reference']
         and settlement['passed'] is True and settlement['owned_parent_dead'] is True
         and settlement['configuration_ledger_unchanged'] is True, 'Physical check not settled')
    physical = read(settlement['physical_reference'])
    need(physical['passed'] is True and physical['source_references']['group'] == card['group_reference']
         and physical['new_Q_calls'] == physical['realized_OOS_samples'] == 0,
         'Original pre-OOS physical scope differs')
    need(set(group['methods']) == set(physical['methods']) == set(METHODS), 'Four-method scope differs')
    results, identities = {}, {}
    for method in METHODS:
        row = group['methods'][method]
        need(card['input_identity_references'][method] == row['input_identity_reference'],
             'Foreign accepted input identity')
        identity = read(row['input_identity_reference'])
        read(identity['task_file'])
        results[method] = validate_view(method, row, physical['methods'][method], identity)
        identities[method] = identity
    need(len({tuple(i['vehicle_ids']) for i in identities.values()}) == 1, 'Paired fleet differs')
    need(len({identities[m]['task_sha256'] for m in ('EA_DRO', 'SAA', 'ARO')}) == 1,
         'Baselines changed common fixed routes/H')
    need(card['item_count'] == 768 and card['formal_OOS'] is False and card['Case1_acceptance'] is False,
         'Original development evaluation scope differs')
    return config, state, results, identities
