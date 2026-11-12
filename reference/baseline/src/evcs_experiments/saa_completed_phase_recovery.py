"""Source-bound finalization of an already complete original SAA phase.

No master or recourse solve is admitted. Original phase restore, model loading,
worker bridge, queue receipt and external ledger gates remain in force.
"""
import ast,hashlib,inspect,types,importlib,math
from pathlib import Path
from . import campaign_batch as batch
KIND='SAA_COMPLETED_PHASE_RECOVERY_POLICY_V1'

def need(ok,message):
    if not ok:raise ValueError(message)

def read(ref):
    batch.verify_reference(ref);return batch.read_json(ref['path'])

def derived_binding(ref):
    batch.verify_reference(ref);text=Path(ref['path']).read_text();nodes=[n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='run_attempt']
    need(len(nodes)==1 and not nodes[0].decorator_list,'Original SAA worker function differs')
    n=nodes[0];source=''.join(text.splitlines(keepends=True)[n.lineno-1:n.end_lineno])
    old="phase_previous=None if previous is None else previous['phase_checkpoint_reference']"
    new=old+"\n        phase_previous=_explicit_completed_saa_phase(phase_previous,model_ref,reqref)"
    need(source.count(old)==1,'Original phase selection differs');derived=source.replace(old,new)
    return source,derived,dict(original_function_sha256=hashlib.sha256(source.encode()).hexdigest(),derived_function_sha256=hashlib.sha256(derived.encode()).hexdigest(),replacement_count=1)

def verify_policy(reference,admission):
    p=read(reference);need(p['kind']==KIND,'Wrong completed-phase policy')
    need(p['worker_request_reference']==admission['worker_request_reference'] and p['settlement_reference']==admission['external_ledger']['expected_last_settlement_reference'],'Completed phase request/head differs')
    req=read(p['worker_request_reference']);settle=read(p['settlement_reference']);cp=read(p['completed_phase_checkpoint_reference'])
    worker=importlib.import_module('evcs_experiments.saa_planning_worker');pipeline=importlib.import_module('evcs_experiments.saa_planning_pipeline')
    for key,module in [('original_worker_reference',worker),('original_pipeline_reference',pipeline)]:
        need(p[key]==batch.file_reference(module.__file__) and p[key] in req['planning_source_references'],'Original completion source differs')
    need(cp['status']=='COMPLETE_FORMAL_SAA_NUMERICAL_PHASES' and cp['formal_admission'] is True and cp['formal_OOS'] is False and cp['scientific_spec_pass'] is False,'Incomplete phase cannot be reused')
    phase=read(cp['request_reference']);model=read(p['model_manifest_reference'])
    need(phase['model_manifest_reference']==p['model_manifest_reference'] and phase['sources']==req['planning_source_references'] and phase['input_binding']==model['input_binding'] and phase['scenario_ids']==model['scenario_ids'] and phase['scenario_sha256s']==model['scenario_sha256s'] and len(model['scenario_ids'])==160,'Completed phase model/source/support differs')
    joined=read(cp['candidate_audit_reference']);q=read(cp['fixed_q_checkpoint_reference'])
    need(joined['status']=='PASS_SAA_CANDIDATE_NUMERICAL_AUDIT' and joined['passed'] is True and joined['full_support_recourse_verified'] is True and joined['risk_certificate_kind']=='fixed_empirical' and joined['formal_OOS'] is False,'Completed phase lacks numerical audit')
    need(joined['fixed_q_audit']['checkpoint_reference']==cp['fixed_q_checkpoint_reference'] and joined['master_audit']['result_reference']==cp['master_result_reference'] and q['completed_indices']==list(range(160)),'Completed numerical refs differ')
    terminal=read(p['terminal_result_reference']);need(settle['terminal_job_result_reference']==p['terminal_result_reference'] and terminal['complete'] is True and settle['new_queue_receipt_references']==[] and settle['budget_within_limit'] is True,'Original failed parent not fully settled')
    extra=settle['cumulative_extra_seconds'];need(type(extra) in (int,float) and math.isfinite(extra) and extra>=0 and settle['effective_spent_seconds']==settle['closing_queue_seconds']+extra,'Invalid settled phase budget offset')
    root=Path(req['output_root']);phase_root=root/'phases-warm-start-1'
    need(p['original_phase_root']==str(root/'phases') and p['completed_phase_root']==str(phase_root) and Path(cp['request_reference']['path'])==phase_root/'request.json' and Path(p['completed_phase_checkpoint_reference']['path']).parent==phase_root/'checkpoints','Completed phase location differs')
    # Original pipeline checks the active checkpoint at restore, then seals a
    # new immutable checkpoint. The execution receipt must still validate later.
    _,_,binding=derived_binding(p['original_worker_reference'])
    return dict(kind='SAA_COMPLETED_PHASE_RECOVERY_EXECUTION_V1',policy_reference=reference,module_reference=batch.file_reference(__file__),original_worker_reference=p['original_worker_reference'],original_pipeline_reference=p['original_pipeline_reference'],completed_phase_checkpoint_reference=p['completed_phase_checkpoint_reference'],completed_phase_root=str(phase_root),derived_worker_binding=binding,external_prior_offset_seconds=extra,solver_calls_authorized=0,recourse_solves_authorized=0,original_model_and_numerical_gates_retained=True)

def install(reference,admission):
    record=verify_policy(reference,admission);p=read(reference)
    worker=importlib.import_module('evcs_experiments.saa_planning_worker');pipeline=importlib.import_module('evcs_experiments.saa_planning_pipeline')
    original=worker.run_attempt;original_phases=worker.run_saa_numerical_phases
    source,derived,_=derived_binding(p['original_worker_reference'])
    need(original.__globals__ is worker.__dict__ and inspect.getsource(original)==source and original_phases.__module__==pipeline.__name__,'Original SAA worker already modified')
    key='_explicit_completed_saa_phase';need(key not in worker.__dict__,'Completed phase hook already present')
    def select(previous,model,request):
        need(previous is None and model==p['model_manifest_reference'] and request==p['worker_request_reference'],'Unexpected prior/model/request in completed-phase injection')
        return p['completed_phase_checkpoint_reference']
    def phases(**kw):
        need(Path(kw['output_directory'])==Path(p['original_phase_root']) and kw['resume_checkpoint_reference']==p['completed_phase_checkpoint_reference'] and kw['model_manifest_reference']==p['model_manifest_reference'],'Wrong completed-phase dispatch')
        cp=read(p['completed_phase_checkpoint_reference'])
        need(cp['candidate_audit_reference'] is not None and cp['status']=='COMPLETE_FORMAL_SAA_NUMERICAL_PHASES','Cannot solve in completion recovery')
        return original_phases(**dict(kw,output_directory=Path(p['completed_phase_root']),prior_accounted_seconds=kw['prior_accounted_seconds']+record['external_prior_offset_seconds']))
    namespace={};exec(compile(derived,__file__+'::derived_original_run_attempt','exec'),namespace)
    function=types.FunctionType(namespace['run_attempt'].__code__,worker.__dict__,original.__name__,original.__defaults__,original.__closure__)
    function.__kwdefaults__=original.__kwdefaults__
    worker.__dict__[key]=select;worker.run_attempt=function;worker.run_saa_numerical_phases=phases
    def restore():
        worker.run_attempt=original;worker.run_saa_numerical_phases=original_phases;worker.__dict__.pop(key,None)
    return restore,record
