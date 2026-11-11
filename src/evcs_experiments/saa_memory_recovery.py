"""Explicit native memory-resource retry over unchanged R26 mathematical code."""
import importlib
from pathlib import Path
from . import campaign_batch as batch

KIND='SAA_NATIVE_MEMORY_RECOVERY_POLICY_V1'

def need(ok,message):
    if not ok:raise ValueError(message)

def read(ref):
    batch.verify_reference(ref);return batch.read_json(ref['path'])

def verify_policy(reference,admission):
    p=read(reference)
    need(p['kind']==KIND and p['soft_memory_limit_gb']==64. and type(p['soft_memory_limit_gb']) in (float,int),'Exact reviewed64GB native policy required')
    need(p['worker_request_reference']==admission['worker_request_reference'] and p['settlement_reference']==admission['external_ledger']['expected_last_settlement_reference'],'Resource retry request/head differs')
    req=read(p['worker_request_reference']);settlement=read(p['settlement_reference'])
    native=importlib.import_module('evcs_experiments.native_gurobi')
    pipeline=importlib.import_module('evcs_experiments.saa_planning_pipeline')
    worker=importlib.import_module('evcs_experiments.saa_planning_worker')
    for key,module in [('original_native_reference',native),('original_pipeline_reference',pipeline),('original_worker_reference',worker)]:
        batch.verify_reference(p[key]);need(batch.file_reference(module.__file__)==p[key] and p[key] in req['planning_source_references'],'Original numerical source differs')
    fail=read(p['failed_master_reference']);raw=read(p['failed_native_reference']);cp=read(p['failed_phase_checkpoint_reference'])
    need(fail['raw_native_result']==p['failed_native_reference'] and fail['model_manifest']==p['model_manifest_reference'] and fail['status']=='MEM_LIMIT' and fail['primal'] is None and fail['lower_bound_usd'] is None and fail['full_support_verified'] is False,'Retry requires original no-incumbent MEM_LIMIT')
    ni=raw['native_interface'];need(ni['raw_model_status']==17 and ni['solution_count']==0 and ni['effective_options']['SoftMemLimit']==24. and ni['effective_options']['Threads']==4,'Original native memory failure differs')
    need(cp['status']=='FAILED' and cp['master_result_reference']==p['failed_master_reference'] and cp['fixed_q_checkpoint_reference'] is None and cp['candidate_audit_reference'] is None,'Failed phase contains unexpected numerical evidence')
    phase_request=read(cp['request_reference'])
    need(phase_request['model_manifest_reference']==p['model_manifest_reference'] and phase_request['sources']==req['planning_source_references'],'Failed phase/model/source differs')
    model=read(p['model_manifest_reference']);need(model['domain']=='formal' and model['scenario_ids']==phase_request['scenario_ids'],'Original full support differs')
    need(req['budget']['single_master_seconds']==600 and req['budget']['master_threads']==4 and req['budget']['normal_gap_relative']==.005,'Original scientific budgets differ')
    terminal=read(p['terminal_result_reference']);need(settlement['terminal_job_result_reference']==p['terminal_result_reference'] and terminal['complete'] is True and settlement['new_queue_receipt_references']==[] and settlement['budget_within_limit'] is True,'Memory failure parent costs not settled')
    root=Path(req['output_root']);old=root/'phases';new=root/'phases-memory-retry-1'
    need(p['original_phase_root']==str(old) and p['new_phase_root']==str(new) and Path(p['failed_master_reference']['path'])==old/'master/result.json' and Path(p['failed_phase_checkpoint_reference']['path']).parent==old/'checkpoints','Phase retry path differs')
    observation=read(p['resource_observation_reference'])
    need(observation['available_memory_kib']>=128*1024*1024 and observation['native_soft_limit_gb']==64. and observation['not_a_process_rss_cap'] is True,'Insufficient reviewed resource headroom')
    return dict(kind='SAA_NATIVE_MEMORY_RECOVERY_EXECUTION_V1',policy_reference=reference,module_reference=batch.file_reference(__file__),
        original_native_reference=p['original_native_reference'],original_pipeline_reference=p['original_pipeline_reference'],
        original_worker_reference=p['original_worker_reference'],old_soft_memory_limit_gb=24.,new_soft_memory_limit_gb=64.,
        original_phase_root=str(old),new_phase_root=str(new),mathematical_arrays_unchanged=True,original_tolerances_unchanged=True,
        original_master_seconds=600,original_master_threads=4)

def install(reference,admission):
    record=verify_policy(reference,admission);p=read(reference)
    native=importlib.import_module('evcs_experiments.native_gurobi');worker=importlib.import_module('evcs_experiments.saa_planning_worker')
    original_parameters=native._parameters;original_phases=worker.run_saa_numerical_phases
    need(original_parameters.__module__==native.__name__ and original_phases.__module__=='evcs_experiments.saa_planning_pipeline','Numerical resource hook already installed')
    def parameters(options):
        seconds,result=original_parameters(options)
        need(result['SoftMemLimit']==24. and result['Threads']==4 and result['MIPGap']==.001 and all(result[k]==1e-8 for k in ('FeasibilityTol','IntFeasTol','OptimalityTol')) and seconds<=600,'Unexpected native parameters in resource hook')
        return seconds,dict(result,SoftMemLimit=64.)
    def phases(**kwargs):
        need(str(Path(kwargs['output_directory']))==record['original_phase_root'] and kwargs['resume_checkpoint_reference'] is None and kwargs['model_manifest_reference']==p['model_manifest_reference'],'Unexpected original phase continuation')
        need(not Path(record['new_phase_root']).exists(),'Resource retry phase already exists; require new terminal review')
        return original_phases(**dict(kwargs,output_directory=Path(record['new_phase_root'])))
    native._parameters=parameters;worker.run_saa_numerical_phases=phases
    def restore():
        native._parameters=original_parameters;worker.run_saa_numerical_phases=original_phases
    return restore,record
