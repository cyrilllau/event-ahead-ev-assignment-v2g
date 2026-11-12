"""Explicit audited migration for the lazy-load budget-guard repair only.

This is never invoked implicitly by run_ccg. Callers preserve the old envelope,
seal and independently audit its raw evidence, and bind the resulting bridge.
"""
import copy
from .random_streams import canonical_hash

OLD_GUARD = '''            result = solve_fixed_plan(problem.recourses[index], plan,
                                      time_limit_seconds=remaining(oracle_time_limit_seconds))'''
NEW_GUARD = '''            template = problem.recourses[index]
            oracle_budget = remaining(oracle_time_limit_seconds)
            if oracle_budget <= 0:
                return save('TIME_LIMIT')
            result = solve_fixed_plan(template, plan, time_limit_seconds=oracle_budget)'''


def migrate_budget_guard(envelope, *, audit, old_source, new_source, new_engine,
                         new_source_manifest_sha256, audit_sha256):
    if old_source.count(OLD_GUARD)!=1 or old_source.replace(OLD_GUARD,NEW_GUARD)!=new_source:
        raise ValueError('Migration permits only the exact post-loading budget guard')
    state=envelope['state']
    if envelope['sha256']!=canonical_hash(state):raise ValueError('Original checkpoint checksum')
    if (audit['status']!='PASS_RECOVERABLE_PARTIAL_26_LP_NOT_S03' or audit['full_S03_pass']
            or audit['q_count']!=26 or state['status']!='RUNNING' or state['pending'] is None
            or len(state['pending']['q_usd'])!=26):
        raise ValueError('Requires independently audited interrupted 26-LP checkpoint')
    if canonical_hash(state['pending']['plan'])!=audit['pending_plan_sha256']:
        raise ValueError('Audited plan differs from checkpoint')
    if [v['objective_usd'] for v in audit['verified_LPs']]!=state['pending']['q_usd']:
        raise ValueError('Audited LP values differ from checkpoint')
    if state['elapsed_seconds']!=audit['checkpoint_elapsed_seconds']:
        raise ValueError('Checkpoint time differs from audit')
    if state['engine_sha256']!=audit['engine_sha256']:
        raise ValueError('Checkpoint engine differs from audit')
    identity=state['identity']
    if identity['lazy_input_binding']['binding']['source_manifest_sha256']!=audit['source_manifest_sha256']:
        raise ValueError('Old input binding source mismatch')
    if state['problem_sha256']!=canonical_hash(identity):raise ValueError('Original problem identity checksum')
    extra=float(audit['actual_job_seconds'])-float(state['elapsed_seconds'])
    if extra<0:raise ValueError('Negative failure overhead')
    value=copy.deepcopy(state)
    value['identity']['lazy_input_binding']['binding']['source_manifest_sha256']=new_source_manifest_sha256
    value['problem_sha256']=canonical_hash(value['identity']);value['engine_sha256']=new_engine;value['status']='TIME_LIMIT'
    value['checkpoint_migration']={'kind':'audited_post_load_budget_guard_v1','old_envelope_sha256':canonical_hash(envelope),
        'old_engine_sha256':state['engine_sha256'],'new_engine_sha256':new_engine,'audit_sha256':audit_sha256,
        'old_source_manifest_sha256':audit['source_manifest_sha256'],'new_source_manifest_sha256':new_source_manifest_sha256,
        'preserved_q_count':26,'preserved_pending_plan_sha256':audit['pending_plan_sha256'],
        'original_checkpoint_elapsed_seconds':state['elapsed_seconds'],'original_job_wall_seconds':audit['actual_job_seconds'],
        'additional_wall_seconds':extra,'overhead_added_separately_to_accounted_cumulative_time':True}
    return {'state':value,'sha256':canonical_hash(value)}
