"""Append-only adoption of the reviewed Michael SAA same-plan auxiliary bound.

V1 deliberately names one accepted evidence chain. It does not solve or open OOS,
replace the original final seal, or turn auxiliary OPTIMAL into native OPTIMAL.
"""
import importlib.util
import json
import math
from pathlib import Path
from .evaluation_lock import _verify_reference, file_reference
from .random_streams import canonical_hash

ACCOUNTING_KIND = 'SAA_ORIGINAL_SEAL_PLUS_EXTERNAL_LEDGER_V1'
NUMERICAL_KIND = 'FORMAL_SAA_SAME_PLAN_AUXILIARY_NUMERICAL_RECORD_V1'
REQUIRED = ('protocol','original_worker_request','candidate','original_bridge','original_joint_audit',
            'original_solver','original_budget','original_finalization','original_join','original_terminal_review',
            'ledger_anchor','latest_settlement','auxiliary_result','auxiliary_review','auxiliary_acceptance',
            'adoption_decision','ledger_source','ledger_source_upgrade')
PINNED = {'original_terminal_review':'c609ddc236bf11ec4d0c9b7246ee5255a6770846db3fec6fcb4265e4b1fa46ec',
          'ledger_source':'c7ae4c5f0e08a5b9819bfd3ae0203e4abecfab72eba79a30daddcae1f580efe8',
          'auxiliary_acceptance':'d81ea53a121777dbebae97453f5938fa5047c1fdd0e571432cea96448d294047'}

def need(ok, message):
    if not bool(ok): raise ValueError(message)

def read(ref):
    _verify_reference(ref)
    return json.loads(Path(ref['path']).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))

def close(a,b):
    # Only accounting/arithmetic joins; no physical or scientific tolerance changes.
    return (type(a) in (int,float) and type(b) in (int,float) and math.isfinite(a) and math.isfinite(b)
            and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-9))

def normalized(ref):
    return dict(path=ref['path'],sha256=ref['sha256'],size_bytes=ref.get('size_bytes',ref.get('bytes')))

def build_saa_composite_record(request, documents):
    """Pure validated mapping; the public wrapper verifies bytes and live ledger."""
    d=documents; refs=request['references']; ident=request['identity']
    need(request['kind']=='SAA_SAME_PLAN_AUXILIARY_RECORD_REQUEST_V1','Wrong composite request')
    need(all(k in refs and k in d for k in REQUIRED),'Missing composite evidence')
    need(all(refs[k]['sha256']==v for k,v in PINNED.items()),'Unsupported original acceptance chain')
    env=d['protocol']; protocol=env['payload']; pb=protocol['configuration']['planning_budget']
    need(env['kind']=='protocol_lock' and env['payload_sha256']==canonical_hash(protocol) and protocol['domain']=='formal','Protocol envelope differs')
    need(set(ident)=={'plan_id','target_id','method_id','setting_id'} and ident['method_id']=='SAA'
         and ident['target_id']=='michael_2018' and ident['setting_id']=='risk_objective=baseline','Unsupported composite identity')
    need(sum(p==ident for b in protocol['batches'] if b['batch_id']==request['batch_id'] for p in b['expected_plans'])==1,'Identity not uniquely predeclared')
    worker=d['original_worker_request']; old=d['original_solver']; budget=d['original_budget']; seal=d['original_finalization']
    need(all(worker[k]==old[k]==ident[k] for k in ('plan_id','target_id','method_id')) and worker['domain']==old['domain']=='formal'
         and worker['budget']==pb and budget['planning_budget']==pb and worker['protocol_reference']==refs['protocol'],'Original worker identity/budget differs')
    plan=d['candidate']['candidate']['plan']; ph=canonical_hash(plan)
    need(set(plan)=={'x','r','task_sha256','config_hash'},'Original optimization plan fields differ')
    bridge=d['original_bridge']; joint=d['original_joint_audit']; terminal=d['original_terminal_review']; join=d['original_join']
    need(old['candidate_record_reference']==refs['candidate'] and old['request_reference']==refs['original_worker_request']
         and old['numerical_audit_reference']==refs['original_bridge'] and old['candidate_audit_reference']==refs['original_joint_audit'],'Original numeric references differ')
    need(bridge['candidate_record_reference']==refs['candidate'] and bridge['candidate_audit_reference']==refs['original_joint_audit']
         and bridge['request_reference']==refs['original_worker_request'],'Original bridge references differ')
    need(all(x['plan_sha256']==ph for x in (old,bridge,joint,d['candidate']['candidate'])),'Changed candidate plan')
    need(joint['status']=='PASS_SAA_CANDIDATE_NUMERICAL_AUDIT' and joint['full_support_recourse_verified'] is True
         and old['full_support_verified'] is True and old['risk_certificate_kind']=='fixed_empirical','Incomplete SAA evidence')
    need(seal['status']=='PASS_FINAL_RECORD_SEAL' and seal['references']==dict(solver_record=refs['original_solver'],budget=refs['original_budget'])
         and old['final_budget_acceptance'] is True and budget['final_budget_acceptance'] is True
         and old['requires_seal_confirmation'] is True,'Original seal differs')
    cutoff=old['actual_attempt_seconds_at_cutoff']; actual=seal['actual_attempt_seconds_before_confirmation_write']; charged=seal['charged_attempt_seconds']
    need(all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in (cutoff,actual,charged))
         and cutoff<=actual<=charged and charged==old['charged_attempt_seconds']==budget['charged_attempt_seconds']
         and close(charged-cutoff,old['final_seal_allocated_seconds']) and old['final_seal_allocated_seconds']==seal['seal_allocation_seconds']==10,'Original seal clock differs')
    mapping={'final_solver_record_reference':'original_solver','finalization_reference':'original_finalization',
             'candidate_audit_reference':'original_joint_audit','numerical_bridge_reference':'original_bridge'}
    need(all(terminal[k]==join[k]==refs[v] for k,v in mapping.items()),'Original independent seal review differs')
    need(terminal['passed'] is True and join['passed'] is True and terminal['settlement_reference']==join['settlement_reference'],'Original terminal/join differs')
    need(join['gate']['anchor_reference']==refs['ledger_anchor'] and join['gate']['candidate_reference']==refs['candidate'],'Original ledger anchor/candidate differs')
    a=d['auxiliary_acceptance']; ar=d['auxiliary_review']; aux=d['auxiliary_result']; latest=d['latest_settlement']
    need(a['kind']=='SAA_AUXILIARY_BOUND_ACCEPTANCE_V1' and a['passed'] is True and a['review_reference']==refs['auxiliary_review']
         and a['references']['result']==ar['references']['result']==refs['auxiliary_result']
         and a['references']['settlement']==ar['references']['settlement']==refs['latest_settlement']
         and a['references']['parent_result']==latest['terminal_job_result_reference'],'Auxiliary result/parent unreviewed')
    need(ar['passed'] is True and ar['original_physical_acceptance_inherited']==refs['original_joint_audit']
         and ar['all_160_raw_dual_stationarity_sign_gradient_intercept_checks'] is True and ar['original_integer_domain_retained'] is True
         and ar['full_support_count']==aux['full_support_count']==160,'Auxiliary numerical scope differs')
    need(a['plan_sha256']==ar['plan_sha256']==aux['original_plan_sha256']==ph,'Auxiliary plan differs')
    need(aux['source_binding']['candidate_audit']==refs['original_joint_audit'] and aux['source_binding']['final_record']==refs['original_solver']
         and aux['source_binding']['final_budget_join']==refs['original_join'] and aux['source_binding']['final_acceptance_review']==refs['original_terminal_review'],'Auxiliary original source binding differs')
    lb=a['lower_bound_usd']; ub=old['upper_bound_usd']; gap=max(0.,ub-lb)/max(1.,abs(ub))
    need(close(lb,aux['lower_bound_usd']) and close(lb,ar['lower_bound_usd']) and ub==a['upper_bound_usd']==aux['upper_bound_usd']==ar['upper_bound_usd']==joint['upper_bound_usd']
         and close(gap,a['gap_relative']) and close(gap,ar['gap']) and lb<=ub and gap>.005,'Auxiliary bounds or UB differ')
    need(old['status']==old['native_master_status']==old['raw_engine_status']==aux['original_native_status']==a['original_native_status']=='TIME_LIMIT'
         and aux['auxiliary_native_status']==ar['auxiliary_native_status']==a['auxiliary_native_status']=='OPTIMAL'
         and close(old['lower_bound_usd'],aux['previous_lower_bound_usd']) and close(old['gap_relative'],max(0.,ub-old['lower_bound_usd'])/max(1.,abs(ub))), 'Native status/bound relabelled')
    queue=old['cumulative_wall_seconds']; total=latest['effective_spent_seconds']; extra=latest['cumulative_extra_seconds']
    need(close(queue,join['original_queue_charged_seconds']) and close(queue,latest['closing_queue_seconds'])
         and close(queue+extra,total) and close(total,a['effective_spent_seconds']) and latest['budget_seconds']==pb['cumulative_wall_seconds']==7200
         and 0<=total<=7200 and close(7200-total,a['remaining_seconds']) and total>join['effective_spent_seconds'],'Stale/reset current budget')
    # The runtime wrapper independently replays every reservation and failed/successful parent.
    need(d['ledger_checked']['latest_settlement']==refs['latest_settlement'] and d['ledger_checked']['pending'] is False
         and close(d['ledger_checked']['extra'],extra),'Incomplete ledger replay')
    inputs=request['plan_input_references']; summary=d['summary']; target=[x for x in summary['targets'] if x['target_event']==ident['target_id']]
    need(len(target)==1 and set(inputs)=={'tasks','routes','initial_state','planning_support'},'Plan input scope differs')
    taskref=normalized(summary['task_sources'][ident['target_id']])
    need(all(inputs[k]==taskref for k in ('tasks','routes','initial_state')) and inputs['planning_support']==target[0]['planning_index'],'Foreign task/route/state/support source')
    need(d['tasks']['task_sha256']==plan['task_sha256'] and all(k in d['tasks'] for k in ('tasks','queries','vehicle_summaries')),'Task semantic identity differs')
    q=d['fixed_q_request']; model=d['model_manifest']; ids=model['scenario_ids']
    need(len(ids)==len(set(ids))==160 and q['scenario_ids']==joint['fixed_q_audit']['support_ids']==ids
         and q['scenario_sha256s']==model['scenario_sha256s'] and q['plan']==plan and q['plan_sha256']==ph
         and model['empirical_probabilities']==joint['fixed_q_audit']['empirical_probabilities']==[1/160]*160,'Original support/law/plan differs')
    support=d['planning_support']
    need(support['target_event']==ident['target_id'] and support['split']=='planning' and support['development'] is False
         and support['regime']=='R1' and support['lead_h']==1 and normalized(support['task_source'])==inputs['tasks']
         and support['task_sha256']==plan['task_sha256']
         and [row['scenario_id'] for row in support['scenarios']]==ids
         and [row['scenario_sha256'] for row in support['scenarios']]==q['scenario_sha256s'], 'Frozen planning support axis/state differs')
    decision=d['adoption_decision']
    need(decision['kind']=='SAA_COMPOSITE_ADOPTION_DECISION_V1' and decision['approved'] is True and decision['identity']==ident
         and decision['references']=={k:v for k,v in refs.items() if k!='adoption_decision'}
         and decision['planning_source']==request['planning_source'] and decision['adapter_sources']==request['adapter_sources']
         and decision['plan_input_references']==inputs and decision['no_new_Q'] is True and decision['formal_OOS'] is False
         and decision['administration_accounting']=='separate_measured_not_solver_time','Unreviewed composite adoption')
    need(request['planning_source']==worker['planning_source_references'],'Original numerical source relabelled')
    common=dict(accounting_kind=ACCOUNTING_KIND,scientific_acceptance=False,formal_OOS=False)
    numeric=dict(kind=NUMERICAL_KIND,domain='formal',**ident,plan_sha256=ph,status='TIME_LIMIT',full_support_verified=True,
        support_count=160,risk_certificate_kind='fixed_empirical',lower_bound_usd=lb,upper_bound_usd=ub,gap_relative=gap,
        normal_gap_met=False,normal_gap_relative=pb['normal_gap_relative'],cumulative_wall_seconds=total,
        original_native_record=old,native_record_reference=refs['original_solver'],auxiliary_native_status='OPTIMAL',
        bound_origin='independently_reviewed_saved_dual_uniform_auxiliary',references=refs,final_budget_acceptance=True,**common)
    newbudget=dict(planning_budget=pb,original_queue_seconds=queue,cumulative_extra_seconds=extra,cumulative_wall_seconds=total,
        effective_spent_seconds=total,remaining_seconds=7200-total,latest_settlement_reference=refs['latest_settlement'],
        original_seal_reference=refs['original_finalization'],final_budget_acceptance=True,**common)
    accounting=dict(kind=ACCOUNTING_KIND,references=refs,same_plan=True,no_new_Q=True,no_native_relabel=True,
                    no_double_charge=True,original_queue_seconds=queue,effective_spent_seconds=total,
                    administration_accounting='separate_measured_not_solver_time',formal_OOS=False)
    return dict(plan=dict(plan,execution_mode='EA',departure_h=1),solver_record=numeric,budget_record=newbudget,composite_accounting=accounting)

def _documents(request):
    refs=request['references']; need(all(k in refs for k in REQUIRED),'Missing composite reference')
    need(all(refs[k]['sha256']==v for k,v in PINNED.items()),'Unsupported original acceptance/source chain')
    d={k:read(refs[k]) for k in REQUIRED if k!='ledger_source'}
    d['ledger_source']=refs['ledger_source']
    for ref in request['planning_source']+request['adapter_sources']:_verify_reference(ref)
    from . import evaluation_lock
    need(all(file_reference(p) in request['adapter_sources'] for p in (__file__,evaluation_lock.__file__)),
         'Actual composite adapter/gateway source unbound')
    # Execute only the explicitly reviewed existing ledger checker, preserving its
    # source-upgrade binding; no dynamic solver or arbitrary callback is accepted.
    upgrade=d['ledger_source_upgrade']
    need(upgrade['anchor_reference']==refs['ledger_anchor'] and upgrade['upgraded_source_reference']==refs['ledger_source'], 'Ledger source upgrade differs')
    _verify_reference(refs['ledger_source'])
    spec=importlib.util.spec_from_file_location('evcs_experiments._saa_composite_ledger',refs['ledger_source']['path'])
    ledger=importlib.util.module_from_spec(spec); spec.loader.exec_module(ledger)
    ledger.SOURCE_UPGRADE_REFERENCE=refs['ledger_source_upgrade']
    anchor,extra,last,pending,_=ledger._chain(refs['ledger_anchor'])
    need(last==refs['latest_settlement'] and pending is None and anchor['plan_id']==request['identity']['plan_id'], 'Latest ledger not closed')
    d['ledger_checked']=dict(latest_settlement=last,pending=False,extra=extra)
    worker=d['original_worker_request']; d['summary']=read(worker['summary_reference'])
    inputs=request['plan_input_references']
    for ref in inputs.values():_verify_reference(ref)
    d['tasks']=read(inputs['tasks'])
    d['planning_support']=read(inputs['planning_support'])
    joint=d['original_joint_audit']; d['model_manifest']=read(joint['master_audit']['model_manifest'])
    d['fixed_q_request']=read(joint['fixed_q_audit']['request_reference'])
    return d


def validate_composite(record, protocol):
    """Central gateway: reconstruct new records while validating ORIGINAL seal."""
    refs=record['references']; request=read(refs['composite_request']); d=_documents(request)
    need(d['protocol']['payload']==protocol,'Composite frozen protocol differs')
    expected=build_saa_composite_record(request,d)
    for name,key in (('plan','plan'),('solver_record','solver_record'),('budget','budget_record')):
        need(read(refs[name])==expected[key],'Composite exported '+name+' differs')
    need(all(record[k]==v for k,v in request['identity'].items()) and record['plan_sha256']==expected['solver_record']['plan_sha256'],'Composite record identity differs')
    need(all(refs[k]==v for k,v in request['plan_input_references'].items())
         and refs['planning_source']==request['planning_source'] and refs['adapter_sources']==request['adapter_sources'],'Composite sources differ')
    accounting=read(refs['composite_accounting'])
    need(accounting==dict(expected['composite_accounting'],new_references={k:refs[k] for k in ('plan','solver_record','budget')},composite_request=refs['composite_request']), 'Composite adoption accounting differs')
    return expected


def prepare_saa_composite_record(request, *, output):
    """Exclusive immutable export. No lock, access marker, solver or old-file edit."""
    import time
    started=time.monotonic(); d=_documents(request); payload=build_saa_composite_record(request,d)
    out=Path(output).resolve()
    from .config import get_paths
    protected=[Path(get_paths().data_root).resolve(),Path(get_paths().frozen_source_root).resolve()]
    protected += [Path(r['path']).resolve().parent for r in request['planning_source']+request['adapter_sources']]
    need(not any(out.is_relative_to(p) for p in protected),'Composite output overlaps source/frozen root')
    need(not any(Path(r['path']).resolve().is_relative_to(out) for r in request['references'].values()),'Composite output contains original evidence')
    out.mkdir(parents=True,exist_ok=False)
    def write(name,value):
        path=out/name
        with path.open('x') as f: json.dump(value,f,indent=2,allow_nan=False); f.write('\n')
        return file_reference(path)
    requestref=write('request.json',request)
    refs={name:write(name+'.json',payload[key]) for name,key in (('plan','plan'),('solver_record','solver_record'),('budget','budget_record'))}
    accounting=write('composite_accounting.json',dict(payload['composite_accounting'],new_references=refs,composite_request=requestref))
    record=dict(**request['identity'],plan_sha256=payload['solver_record']['plan_sha256'],references=dict(refs,
        composite_request=requestref,composite_accounting=accounting,**request['plan_input_references'],
        planning_source=request['planning_source'],adapter_sources=request['adapter_sources']))
    rr=write('plan_record.json',record)
    return dict(status='PREPARED_PENDING_CANONICAL_GATE',plan_record_reference=rr,formal_OOS=False,scientific_acceptance=False,
        solver_calls=0,administration_seconds_before_receipt=time.monotonic()-started,administration_accounting='separate_measured_not_solver_time')
