"""Append a reviewed auxiliary bound to the same accepted formal plan record.

This metadata operation neither solves nor locks a paired group. Old native,
plan, full-support and source evidence keep their own original identities.
"""
from copy import deepcopy
from decimal import Decimal
import json
import math
from pathlib import Path

from .campaign_batch import verify_reference
from .config import get_paths
from .evaluation_lock import file_reference, verify_formal_numerical_records
from .random_streams import canonical_hash


def need(ok, message):
    if not ok:
        raise ValueError(message)


def build_revision(request, documents):
    d = documents; refs = request['references']
    old = d['old_numeric']; budget = d['old_budget']; record = d['old_plan_record']
    acceptance = d['old_acceptance']; result = d['bound_result']; review = d['bound_review']
    settlement = d['settlement']; failed = d['failed_settlement']; protocol = d['protocol']['payload']
    need(d['protocol']['payload_sha256'] == canonical_hash(protocol), 'Frozen protocol changed')
    need(acceptance['status'] == 'PASS_COMPONENT'
         and acceptance['component_id'] == 'INDIVIDUAL_FORMAL_NUMERICAL_RECORD'
         and acceptance['plan_record_reference'] == refs['old_plan_record'], 'Old formal record is not accepted')
    need(record['references']['solver_record'] == refs['old_numeric']
         and record['references']['budget'] == refs['old_budget'], 'Old record components differ')
    identity = {k:record[k] for k in ('plan_id','target_id','method_id','setting_id')}
    need(all(old[k] == v for k,v in identity.items()) and old['domain'] == 'formal'
         and old['status'] == 'TIME_LIMIT' and old['full_support_verified'] is True
         and old['support_count'] == 160, 'Original formal numerical scope differs')
    plan = d['old_plan']; semantic = {k:plan[k] for k in ('x','r','task_sha256','config_hash')}
    ph = canonical_hash(semantic)
    need(ph == old['plan_sha256'] == record['plan_sha256'] == result['plan_sha256'] == review['plan_sha256'],
         'Bound refinement belongs to a different plan')
    need(result['plan_id'] == review['plan_id'] == settlement['plan_id'] == failed['plan_id'] == record['plan_id'],
         'Bound refinement plan identifier differs')
    need(result['kind'] == 'S05_CACHED_MULTI_ANCHOR_BOUND_ONLY_V1'
         and review['kind'] == 'R23_INDEPENDENT_TERMINAL_AUXILIARY_BOUND_REVIEW_V1'
         and review['passed'] is True
         and review['status'] == 'PASS_INDEPENDENT_RELAXED_AUXILIARY_BOUND_NUMERICAL_REVIEW'
         and review['references']['result'] == refs['bound_result']
         and review['references']['request'] == result['request_reference'] == settlement['request_reference']
         and review['references']['parent_result'] == settlement['parent_result_reference'],
         'Independent auxiliary review does not bind the settled result')
    need(settlement['kind'] == 'S05_CACHED_MULTI_ANCHOR_BOUND_PARENT_ACCOUNTING_V1'
         and settlement['terminal_state'] == 'SUCCEEDED'
         and settlement['result_reference'] == refs['bound_result']
         and settlement['previous_settlement_reference'] == refs['failed_settlement'],
         'Auxiliary result has no successful terminal settlement')
    need(failed['kind'] == 'S05_MULTI_ANCHOR_BOUND_PARENT_ACCOUNTING_V1'
         and failed['terminal_state'] == 'FAILED'
         and failed['previous_settlement_reference'] == budget['accounting_reference'], 'Failed attempt charge omitted')
    total = Decimal(old['cumulative_wall_seconds_decimal'])
    need(total == Decimal(budget['cumulative_accounted_seconds_decimal']), 'Old numerical/budget balances differ')
    cap = Decimal(protocol['configuration']['planning_budget']['cumulative_wall_seconds'])
    need(budget['planning_budget'] == protocol['configuration']['planning_budget'], 'Frozen planning resources changed')
    for entry in (failed, settlement):
        need(entry['accounting_PASS'] is True and entry['parent_charged_once'] is True
             and entry['nested_seconds_added'] is False and entry['cumulative_seconds_provisional'] is False
             and entry['cumulative_cap_seconds'] == cap, 'Unsettled or altered original budget')
        need(Decimal(entry['opening_cumulative_seconds_decimal']) == total, 'Failure/parent charged twice or reset')
        actual = Decimal(entry['terminal_parent_seconds_decimal'])
        need(actual.is_finite() and actual >= 0, 'Invalid terminal duration')
        total += actual
        need(total == Decimal(entry['cumulative_accounted_seconds_decimal'])
             and cap-total == Decimal(entry['remaining_seconds_decimal']), 'Incorrect cumulative parent arithmetic')
    need(0 <= total <= cap and cap-total < Decimal(old['complete_iteration_reserve_seconds_decimal']),
         'Budget/ordinary complete-iteration stop reason differs')
    lb = result['lower_bound_usd']; ub = result['upper_bound_usd']
    need(all(type(x) in (int,float) and math.isfinite(x) for x in (lb,ub))
         and old['lower_bound_usd'] <= lb <= ub
         and ub == old['upper_bound_usd'] == review['upper_bound_usd']
         and lb == review['lower_bound_usd'], 'Refinement changed fixed UB or reversed bounds')
    gap = max(0.,ub-lb)/max(1.,abs(ub))
    need(gap == result['relative_gap'] == review['relative_gap'], 'Incorrect refined gap')
    need(old['original_native_master_status'] == 'TIME_LIMIT'
         and old['original_native_master_lower_bound_usd'] is None
         and old['full_CCG_converged'] is False and old['normal_ccg_checkpoint_compatible'] is False,
         'Original native/CCG provenance changed')
    numeric = deepcopy(old); new_budget = deepcopy(budget)
    numeric.update(lower_bound_usd=lb,gap_relative=gap,normal_gap_met=gap <= old['normal_gap_relative'],
        auxiliary_master_status=review['native_status'],cumulative_wall_seconds=float(total),
        cumulative_wall_seconds_decimal=str(total),remaining_wall_seconds_decimal=str(cap-total),
        lower_bound_origin='R23 independently audited outward relaxation and native auxiliary global lower bound',
        revision_provenance=dict(kind='AUXILIARY_BOUND_RECORD_REVISION_V1',references=refs,
                                 revision_sources=request['revision_sources'],original_evidence_references_preserved=True))
    new_budget.update(cumulative_accounted_seconds_decimal=str(total),remaining_seconds_decimal=str(cap-total),
        accounting_reference=refs['settlement'],previous_budget_reference=refs['old_budget'])
    return numeric,new_budget


def prepare_revision(request, *, output):
    def read(ref):
        verify_reference(ref)
        return json.loads(Path(ref['path']).read_text())
    refs = request['references']; docs = {k:read(v) for k,v in refs.items()}
    old_record = docs['old_plan_record']
    for key in ('plan','tasks','routes','initial_state','planning_support','solver_record','budget'):
        verify_reference(old_record['references'][key])
    for ref in old_record['references']['planning_source'] + request['revision_sources']:
        verify_reference(ref)
    need(file_reference(__file__) in request['revision_sources'], 'Revision implementation source is unbound')
    need(old_record['references']['plan'] == refs['old_plan'], 'Original plan bytes changed')
    for doc in (docs['bound_result'],docs['bound_review'],docs['settlement'],docs['failed_settlement']):
        def walk(value):
            if isinstance(value, dict):
                if set(value) == {'path','sha256','size_bytes'}: verify_reference(value)
                else:
                    for child in value.values(): walk(child)
            elif isinstance(value,list):
                for child in value: walk(child)
        walk(doc)
    numeric,budget = build_revision(request,docs)
    out = Path(output).resolve()
    paths = get_paths()
    protected = [paths.data_root.resolve(),paths.frozen_source_root.resolve()]
    protected += [Path(ref['path']).resolve().parent for ref in request['revision_sources']]
    protected.append(Path(refs['old_plan_record']['path']).resolve().parent)
    need(not any(out.is_relative_to(p) for p in protected), 'Revision output overlaps frozen/source evidence')
    need(not any(Path(ref['path']).resolve().is_relative_to(out) for ref in refs.values()),
         'Revision output overlaps old input evidence')
    out.mkdir(parents=True,exist_ok=False)
    def write(name,value):
        p = out/name
        with p.open('x') as f: json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
        return file_reference(p)
    nr = write('solver_record.json',numeric); br = write('planning_budget.json',budget)
    record = deepcopy(old_record); rr = record['references']; rr.update(solver_record=nr,budget=br)
    seen = {v['path'] for v in rr['planning_source']}
    for ref in request['revision_sources']:
        if ref['path'] not in seen: rr['planning_source'].append(ref);seen.add(ref['path'])
    rr['previous_plan_record'] = refs['old_plan_record']
    verify_formal_numerical_records(docs['protocol']['payload'],{'plans':[record]})
    pr = write('plan_record.json',record)
    ar = write('artifact_manifest.json',dict(files=[file_reference(p) for p in sorted(out.iterdir()) if p.is_file()]))
    return dict(kind='PREPARED_AUXILIARY_BOUND_FORMAL_RECORD_REVISION_V1',plan_record_reference=pr,
        solver_record_reference=nr,budget_reference=br,artifact_manifest_reference=ar,
        lower_bound_usd=numeric['lower_bound_usd'],upper_bound_usd=numeric['upper_bound_usd'],
        gap_relative=numeric['gap_relative'],old_record_modified=False,paired_group_locked=False,
        formal_OOS=False,scientific_acceptance=False,solver_calls=0)
