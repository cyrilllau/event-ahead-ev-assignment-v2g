"""Explicit, source-bound SAA validation recovery dispatch (no numerical replacement).

The old queue and installed numerical package remain the authority. This module
is executed by absolute filename and records its distinct parent/child argv.
Production use additionally requires independent terminal parent-cost coverage.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from evcs_experiments import campaign_batch as batch
from evcs_experiments.random_streams import canonical_hash

KIND='SAA_VALIDATOR_RECOVERY_ADMISSION_V1'
RECEIPT='SAA_VALIDATOR_RECOVERY_EXECUTION_V1'


def need(ok,msg):
    if not ok:raise ValueError(msg)


def read(ref):
    batch.verify_reference(ref);return batch.read_json(ref['path'])


def new(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    return batch.file_reference(path)


def actual_argv():
    path=Path('/proc/self/cmdline')
    return path.read_bytes().decode().rstrip('\0').split('\0') if path.exists() else [sys.executable,*sys.argv]


def load_external_ledger(value):
    """Explicit added source, loaded over the unchanged installed numerical package."""
    hook=value.get('external_ledger')
    need(isinstance(hook,dict) and set(hook) in ({'anchor_reference','module_reference','parent_job_request_path','expected_last_settlement_reference'},
          {'anchor_reference','module_reference','parent_job_request_path','expected_last_settlement_reference','source_upgrade_reference'}),
         'Formal recovery admission deferred: external terminal fee ledger is not implemented or admitted')
    need(Path(hook['parent_job_request_path']).is_absolute(),'Absolute parent request path required')
    batch.verify_reference(hook['module_reference'])
    # Validate authorization as data before executing any added module source.
    review=read(value['independent_review_reference'])
    need(review['kind']=='SAA_VALIDATOR_RECOVERY_SOURCE_REVIEW_V1' and review['passed'] is True and
         review['admission_payload_sha256']==canonical_hash({k:v for k,v in value.items() if k!='independent_review_reference'}) and
         review['adapter_reference']==value['adapter_reference']==batch.file_reference(__file__) and
         review['validator_reference']==value['validator_reference'] and
         review.get('external_ledger_anchor_reference')==hook['anchor_reference'] and
         review.get('external_ledger_module_reference')==hook['module_reference'],'External ledger source not independently authorized before loading')
    anchor_data=read(hook['anchor_reference']);anchor_review=read(anchor_data['independent_review_reference'])
    need(anchor_data['kind']=='SAA_RECOVERY_EXTERNAL_LEDGER_V1' and (anchor_data['ledger_source_reference']==hook['module_reference'] or 'source_upgrade_reference' in hook) and
         anchor_review['kind']=='SAA_RECOVERY_LEDGER_ANCHOR_REVIEW_V1' and anchor_review['passed'] is True and
         anchor_review['anchor_payload_sha256']==canonical_hash({k:v for k,v in anchor_data.items() if k!='independent_review_reference'}),
         'External ledger anchor not independently authorized before loading')
    name='evcs_experiments._explicit_recovery_parent_ledger'
    if name in sys.modules:
        module=sys.modules[name]
        need(batch.file_reference(module.__file__)==hook['module_reference'],'External ledger source changed in process')
    else:
        spec=importlib.util.spec_from_file_location(name,hook['module_reference']['path'])
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module
        try:spec.loader.exec_module(module)
        except BaseException:sys.modules.pop(name,None);raise
    need(module.batch is batch,'External ledger did not bind original queue package')
    if 'source_upgrade_reference' in hook:
        batch.verify_reference(hook['source_upgrade_reference']);module.SOURCE_UPGRADE_REFERENCE=hook['source_upgrade_reference']
    anchor=module._anchor(hook['anchor_reference'])
    need((anchor['ledger_source_reference']==hook['module_reference'] or 'source_upgrade_reference' in hook) and anchor['queue_reference']==value['queue_reference']
         and anchor['plan_id']==value['plan_id'],'External ledger anchor/source/queue differs')
    review=read(value['independent_review_reference'])
    need(review.get('external_ledger_anchor_reference')==hook['anchor_reference'] and
         review.get('external_ledger_module_reference')==hook['module_reference'],'External ledger entry was not independently reviewed')
    return module


def admission(ref):
    value=read(ref)
    if value.get('domain')=='formal' or 'external_ledger' in value:load_external_ledger(value)
    need(value['kind']==KIND and value['domain'] in ('formal','development'),'Wrong recovery admission domain')
    need(value['adapter_reference']==batch.file_reference(__file__),'Recovery adapter source differs')
    review=read(value['independent_review_reference'])
    payload={k:v for k,v in value.items() if k!='independent_review_reference'}
    need(review['kind']=='SAA_VALIDATOR_RECOVERY_SOURCE_REVIEW_V1' and review['passed'] is True and
         review['admission_payload_sha256']==canonical_hash(payload),'Recovery admission was not independently reviewed')
    need(review['adapter_reference']==value['adapter_reference'] and review['validator_reference']==value['validator_reference'],'Recovery reviewed source differs')
    for key in ('queue_reference','worker_request_reference','original_release_reference','validator_reference','baseline_ledger_reference'):
        batch.verify_reference(value[key])
    queue=read(value['queue_reference']);worker=read(value['worker_request_reference'])
    need(queue['domain']==value['domain'] and worker['domain']==value['domain'] and worker['method_id']=='SAA','Recovery is SAA only')
    need(Path(queue['queue_root']).resolve()==Path(value['queue_root']).resolve() and len(queue['items'])==1 and
         queue['items'][0]['request_reference']==value['worker_request_reference'],'Recovery queue/worker differs')
    need(queue['items'][0]['plan_id']==value['plan_id']==worker['plan_id'],'Recovery plan differs')
    release=read(value['original_release_reference']);root=Path(value['original_release_reference']['path']).parent
    inventory={row['path']:row['sha256'] for row in release['files']}
    from evcs_experiments import formal_planning_worker,saa_planning_worker,saa_record_bridge
    modules=dict(campaign_batch=batch,formal_planning_worker=formal_planning_worker,saa_planning_worker=saa_planning_worker,saa_record_bridge=saa_record_bridge)
    need(set(value['original_module_references'])==set(modules),'Original module source scope differs')
    for name,module in modules.items():
        source=value['original_module_references'][name];batch.verify_reference(source)
        need(source==batch.file_reference(module.__file__),'Installed original package differs')
        relative=Path(source['path']).relative_to(root).as_posix()
        need(inventory.get(relative)==source['sha256'],'Original module absent from frozen release')
    need(value['original_python']==sys.executable,'Original interpreter differs')
    load_queue_allocator(value)
    load_memory_recovery(value)
    return value,queue


def load_queue_allocator(value):
    ref=value.get('queue_allocator_reference')
    if ref is None:return None
    review=read(value['independent_review_reference'])
    need(review['admission_payload_sha256']==canonical_hash({k:v for k,v in value.items() if k!='independent_review_reference'}) and
         review.get('queue_allocator_reference')==ref and review['passed'] is True and
         review['kind']=='SAA_VALIDATOR_RECOVERY_SOURCE_REVIEW_V1','Queue allocator was not independently admitted')
    batch.verify_reference(ref)
    name='evcs_experiments._explicit_recovery_queue_allocator'
    if name in sys.modules:
        module=sys.modules[name];need(batch.file_reference(module.__file__)==ref,'Allocator source changed')
    else:
        spec=importlib.util.spec_from_file_location(name,ref['path']);module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module
        try:spec.loader.exec_module(module)
        except BaseException:sys.modules.pop(name,None);raise
    return module


def load_memory_recovery(value):
    hook=value.get('native_memory_recovery')
    if hook is None:return None
    need(isinstance(hook,dict) and set(hook)=={'module_reference','policy_reference'},'Wrong native memory recovery scope')
    review=read(value['independent_review_reference'])
    need(review['kind']=='SAA_VALIDATOR_RECOVERY_SOURCE_REVIEW_V1' and review['passed'] is True and
         review['admission_payload_sha256']==canonical_hash({k:v for k,v in value.items() if k!='independent_review_reference'}) and
         review.get('native_memory_recovery')==hook,'Native resource change was not independently reviewed')
    batch.verify_reference(hook['module_reference']);batch.verify_reference(hook['policy_reference'])
    name='evcs_experiments._explicit_native_memory_recovery'
    if name in sys.modules:
        module=sys.modules[name];need(batch.file_reference(module.__file__)==hook['module_reference'],'Native resource source changed')
    else:
        spec=importlib.util.spec_from_file_location(name,hook['module_reference']['path']);module=importlib.util.module_from_spec(spec);sys.modules[name]=module
        try:spec.loader.exec_module(module)
        except BaseException:sys.modules.pop(name,None);raise
    module.verify_policy(hook['policy_reference'],value)
    return module


def install_validator(value):
    """Only the three documented closure aliases are replaced, visibly."""
    from evcs_experiments import formal_planning_worker as fw,saa_planning_worker as sw,saa_record_bridge as bridge
    batch.verify_reference(value['validator_reference'])
    name='evcs_experiments._explicit_recovery_checkpoint_closure'
    need(name not in sys.modules,'Validator already installed in this process')
    spec=importlib.util.spec_from_file_location(name,value['validator_reference']['path'])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    try:spec.loader.exec_module(module)
    except BaseException:sys.modules.pop(name,None);raise
    need(module.file_reference is batch.file_reference and module.verify_reference is batch.verify_reference,
         'Validator did not bind original campaign helpers')
    old=(fw._closure,sw._closure,bridge._closure)
    fw._closure=sw._closure=module.verify_checkpoint_closure
    def value_closure(value,seen=None):
        seen=set() if seen is None else seen
        if isinstance(value,dict):
            if set(value)=={'path','sha256','size_bytes'}:module.verify_checkpoint_closure(value,seen)
            else:
                for child in value.values():value_closure(child,seen)
        elif isinstance(value,list):
            for child in value:value_closure(child,seen)
    bridge._closure=value_closure
    def restore():
        fw._closure,sw._closure,bridge._closure=old;sys.modules.pop(name,None)
    return restore


def validate_execution_result(result_ref,invocation_ref,*,baseline_result_references=()):
    result=read(result_ref);inv=read(invocation_ref)
    if result_ref in baseline_result_references:return result
    receipt_ref=result.get('validation_recovery_execution_reference')
    need(receipt_ref is not None and receipt_ref in result.get('artifacts',[]),'New worker omitted recovery execution receipt')
    receipt=read(receipt_ref);value,_=admission(receipt['admission_reference'])
    parent=read(receipt['parent_execution_reference'])
    need(parent['parent_job']['job_id']==inv['parent_job']['job_id'] and parent['parent_job']['manifest_reference']==inv['parent_job']['manifest_reference'],'Receipt parent job differs')
    need(receipt['observed_parent_identity']==parent['process_identity'] and parent['pid']==parent['process_identity']['pid'],'Receipt parent process identity differs')
    need(read(parent['parent_job']['manifest_reference'])['argv']==parent['actual_argv'],'Recorded parent manifest argv differs')
    need(receipt['kind']==RECEIPT and receipt['role']=='child' and parent['kind']==RECEIPT and parent['role']=='parent','Wrong execution receipt role')
    need(receipt['invocation_reference']==invocation_ref and receipt['parent_job']==inv['parent_job'] and
         receipt['declared_worker_argv']==inv['argv'],'Recovery invocation/parent differs')
    need(parent['admission_reference']==receipt['admission_reference'] and parent['queue_reference']==inv['request_reference']==value['queue_reference'], 'Recovery parent/admission/queue differs')
    need(parent['actual_argv']==[value['original_python'],value['adapter_reference']['path'],'parent','--admission',receipt['admission_reference']['path'],'--expected-sha256',receipt['admission_reference']['sha256']] and receipt['actual_argv']==child_argv(value,receipt['admission_reference'],receipt['parent_execution_reference']), 'Recovery actual child argv differs')
    need(receipt['adapter_reference']==value['adapter_reference'] and receipt['validator_reference']==value['validator_reference'] and
         receipt['original_module_references']==value['original_module_references'],'Recovery execution source differs')
    if 'native_memory_recovery' in value:
        memory=load_memory_recovery(value)
        need(receipt.get('native_memory_recovery_execution')==memory.verify_policy(value['native_memory_recovery']['policy_reference'],value),'Native memory execution receipt differs')
    if value.get('queue_allocator_reference') is not None:
        allocation=parent.get('queue_allocation_execution',{})
        allocator=load_queue_allocator(value)
        need(allocation==allocator.expected_binding_record(batch,value['original_module_references']['campaign_batch']),'Missing exact queue allocator execution provenance')
    if 'external_ledger' in value:
        ledger=load_external_ledger(value);reservation=parent.get('external_parent_reservation_reference')
        need(reservation is not None and receipt.get('external_parent_reservation_reference')==reservation,'Worker omitted external parent reservation')
        reserved=ledger.read(reservation)
        need(reserved['anchor_reference']==value['external_ledger']['anchor_reference'] and reserved['admission_reference']==receipt['admission_reference']
             and reserved['parent_manifest_reference']==inv['parent_job']['manifest_reference'],'Execution external ledger binding differs')
    return result


def child_argv(value,ref,parent_ref):
    entry=value.get('development_entry_reference',value['adapter_reference'])
    if 'development_entry_reference' in value:
        need(value['domain']=='development','Formal recovery cannot substitute a development worker');batch.verify_reference(entry)
    return [value['original_python'],entry['path'],'child','--admission',ref['path'],'--expected-sha256',ref['sha256'],
            '--parent-execution',parent_ref['path'],'--parent-sha256',parent_ref['sha256']]


def run_child(ref,parent_ref,*,_development_fixture=None):
    value,queue=admission(ref);parent=read(parent_ref)
    need(value['domain']=='development' or _development_fixture is None,'Formal recovery cannot inject physical fixtures')
    invref=batch.file_reference(os.environ['EVCS_BATCH_ATTEMPT_REQUEST'])
    need(invref['sha256']==os.environ['EVCS_BATCH_INVOCATION_SHA256'],'Recovery invocation hash differs')
    inv=read(invref)
    need(parent['admission_reference']==ref and parent['queue_reference']==value['queue_reference']==inv['request_reference'] and
         inv['item_request_reference']==value['worker_request_reference'] and inv['plan_id']==value['plan_id'], 'Wrong recovery invocation/queue')
    need(parent['parent_job']['job_id']==inv['parent_job']['job_id'] and
         parent['parent_job']['manifest_reference']==inv['parent_job']['manifest_reference'],'Wrong recovery parent')
    need(actual_argv()==child_argv(value,ref,parent_ref),'Child was not started by explicit recovery argv')
    need(os.getppid()==parent['pid'],'Recovery child parent process differs')
    identity=batch.process_identity(os.getppid())
    need(identity is not None and all(identity[k]==parent['process_identity'][k] for k in ('pid','token','sid','pgid')),'Recovery live parent birth identity differs')
    if 'external_ledger' in value:
        ledger=load_external_ledger(value);_,_,_,pending,pending_ref=ledger._chain(value['external_ledger']['anchor_reference'])
        need(pending is not None and pending_ref==parent.get('external_parent_reservation_reference') and pending['admission_reference']==ref, 'Child lacks exact pending parent reservation')
    restore=install_validator(value)
    memory_restore=None;memory_record=None
    if 'native_memory_recovery' in value:
        memory=load_memory_recovery(value)
        memory_restore,memory_record=memory.install(value['native_memory_recovery']['policy_reference'],value)
    from evcs_experiments import saa_planning_worker as worker
    original_new=worker._new
    receipt=new(Path(inv['attempt_directory'])/'validation_recovery_execution.json',dict(kind=RECEIPT,role='child',
        admission_reference=ref,parent_execution_reference=parent_ref,invocation_reference=invref,
        **({'native_memory_recovery_execution':memory_record} if memory_record is not None else {}),
        parent_job=inv['parent_job'],declared_worker_argv=inv['argv'],actual_argv=actual_argv(),
        adapter_reference=value['adapter_reference'],validator_reference=value['validator_reference'],
        original_module_references=value['original_module_references'],pid=os.getpid(),ppid=os.getppid(),
        process_identity=batch.process_identity(os.getpid()),observed_parent_identity=parent['process_identity'],
        **({'external_parent_reservation_reference':parent['external_parent_reservation_reference']} if 'external_ledger' in value else {})))
    def first_write(path,data):
        if Path(path).resolve()==Path(inv['result_path']).resolve():
            need('validation_recovery_execution_reference' not in data,'Worker result provenance already assigned')
            data=dict(data,validation_recovery_execution_reference=receipt,
                      artifacts=[*data['artifacts'],receipt])
        elif Path(path).name in ('candidate_solver_record.json','worker_checkpoint.json'):
            data=dict(data,validation_recovery_execution_reference=receipt)
        return original_new(path,data)
    worker._new=first_write
    try:
        worker.run_attempt(invref,_development_fixture=_development_fixture)
        return validate_execution_result(batch.file_reference(inv['result_path']),invref)
    finally:
        worker._new=original_new;restore()
        if memory_restore is not None:memory_restore()


def run_parent(ref):
    """Dispatch through original queue; expensive admission runs inside its clock."""
    entered=time.monotonic();value=read(ref)
    original_checked=batch.checked_request;original_result=batch._result;original_subprocess=batch.subprocess
    holder={}
    allocator_restore=None;allocation_record=None
    reservation=None
    if 'external_ledger' in value:
        ledger=load_external_ledger(value);hook=value['external_ledger']
        expected=[value['original_python'],value['adapter_reference']['path'],'parent','--admission',ref['path'],'--expected-sha256',ref['sha256']]
        need(actual_argv()==expected and os.environ['EVCS_JOB_ID']==value['parent_job_id'] and
             Path(os.environ['EVCS_JOB_DIRECTORY']).resolve()==Path(value['parent_job_directory']).resolve(),'External ledger actual parent differs')
        _,_,last,pending,_=ledger._chain(hook['anchor_reference']) if (Path(read(hook['anchor_reference'])['ledger_root'])/'reservation-0001.json').exists() else (None,None,None,None,None)
        need(pending is None,'Unsettled recovery parent blocks restart')
        need(last==hook['expected_last_settlement_reference'],'External ledger head differs from reviewed admission')
        if last is not None:batch.verify_reference(hook['expected_last_settlement_reference'])
        opening=read(hook['anchor_reference'])['initial_queue_snapshot_reference'] if last is None else read(last)['closing_queue_snapshot_reference']
        reservation=ledger.reserve_recovery_parent(hook['anchor_reference'],
          parent_job_request_reference=batch.file_reference(hook['parent_job_request_path']),
          parent_manifest_reference=batch.file_reference(Path(value['parent_job_directory'])/'manifest.json'),
          admission_reference=ref,opening_queue_snapshot_reference=opening)
    def checked(path):
        if not holder:
            admitted,queue=admission(ref)
            need(Path(path).resolve()==Path(admitted['queue_reference']['path']).resolve(),'Foreign recovery queue')
            expected=[admitted['original_python'],admitted['adapter_reference']['path'],'parent','--admission',ref['path'],'--expected-sha256',ref['sha256']]
            need(actual_argv()==expected,'Parent actual argv differs')
            parent=batch._parent(queue)
            need(parent is not None and read(parent['manifest_reference'])['argv']==expected,'Actual job manifest argv differs')
            need(parent is not None and parent['job_id']==admitted['parent_job_id'] and
                 parent['directory']==admitted['parent_job_directory'],'Foreign recovery jobctl parent')
            state=batch.read_json(Path(queue['queue_root'])/'state.json')
            need(state['request_reference']==admitted['queue_reference'] and state['attempt_receipts']==admitted['baseline_receipt_references'] and
                 not state.get('active') and not state.get('halted'),'Original queue history changed')
            carry=batch._carryovers(queue);_,spent=batch._history(state,admitted['queue_reference'],Path(queue['queue_root']),carry)
            need(spent.get(admitted['plan_id'],0.)==admitted['opening_cumulative_seconds'],'Recovery budget opening differs')
            need(len(admitted['legacy_result_references'])==(1 if admitted['domain']=='formal' else 0),'Exactly one original legacy result exemption required')
            baseline=[]
            for r in state['attempt_receipts']:
                row=read(r);baseline.append(row['result_reference'])
                # Exactly admitted original results may lack the new provenance.
            need(baseline==admitted['baseline_result_references'],'Baseline result exemptions differ')
            restore=install_validator(admitted)
            parent_ref=new(admitted['parent_execution_path'],dict(kind=RECEIPT,role='parent',admission_reference=ref,
                queue_reference=admitted['queue_reference'],parent_job=parent,actual_argv=actual_argv(),
                **({'queue_allocation_execution':allocation_record} if allocation_record is not None else {}),
                pid=os.getpid(),process_identity=batch.process_identity(os.getpid()),adapter_entered_monotonic=entered,opening_cumulative_seconds=admitted['opening_cumulative_seconds'],
                **({'external_parent_reservation_reference':reservation} if reservation is not None else {})))
            holder.update(value=admitted,parent_ref=parent_ref,restore=restore)
        return original_checked(path)
    def dispatch(argv,*args,**kwargs):
        need(holder and argv==read(holder['value']['queue_reference'])['items'][0]['argv'],'Unexpected original worker dispatch')
        batch.verify_reference(batch.file_reference(kwargs['env']['EVCS_BATCH_ATTEMPT_REQUEST']))
        return original_subprocess.Popen(child_argv(holder['value'],ref,holder['parent_ref']),*args,**kwargs)
    def result(r,i):
        answer=original_result(r,i)
        if holder:
            validate_execution_result(r,i,baseline_result_references=holder['value']['legacy_result_references'])
        return answer
    batch.checked_request=checked;batch._result=result
    batch.subprocess=SimpleNamespace(**{**vars(original_subprocess),'Popen':dispatch})
    try:
        if value.get('queue_allocator_reference') is not None:
            admitted,_=admission(ref)
            allocator=load_queue_allocator(admitted)
            allocator_restore,allocation_record=allocator.bind_original_queue(batch,admitted['original_module_references']['campaign_batch'])
        output=batch.run_batch(value['queue_reference']['path'],value['queue_root'])
        elapsed=time.monotonic()-entered
        charged=output['cumulative_plan_seconds'][value['plan_id']]-value['opening_cumulative_seconds']
        coverage=new(value['parent_coverage_path'],dict(kind='SAA_RECOVERY_PARENT_COST_COVERAGE_V1',
            parent_execution_reference=holder['parent_ref'],elapsed_to_seal_seconds=elapsed,
            queue_charged_increment_seconds=charged,covered_through_preseal=elapsed<=charged,
            terminal_parent_settlement_required=True,automatic_resume_authorized=False))
        # No final budget claim: jobctl's terminal wall is unavailable while running.
        # An independent outer settlement must charge max(0,parent_wall-queue_charge).
        return dict(queue_result=output,parent_coverage_reference=coverage,terminal_parent_settlement_required=True,
                    **({'external_parent_reservation_reference':reservation} if reservation is not None else {}))
    finally:
        batch.checked_request=original_checked;batch._result=original_result;batch.subprocess=original_subprocess
        if holder:holder['restore']()
        if allocator_restore is not None:allocator_restore()


def settle_and_validate_terminal(ref,reservation_reference,terminal_job_result_reference,
                                 closing_queue_snapshot_reference,*,output_path):
    """Terminal accounting plus exact original candidate/final gate; never re-solves."""
    value,_=admission(ref);ledger=load_external_ledger(value)
    need(ledger.read(reservation_reference)['admission_reference']==ref,'Foreign terminal reservation')
    settlement=ledger.settle_recovery_parent(reservation_reference,terminal_job_result_reference,closing_queue_snapshot_reference)
    card=ledger.read(settlement);joins=[]
    for receipt_ref in card['new_queue_receipt_references']:
        receipt=read(receipt_ref);result=read(receipt['result_reference'])
        if result['status']!='COMPLETE':continue
        audit=read(result['numerical_audit_reference']);candidate=audit['candidate_record_reference']
        finals=[read(r) for r in read(closing_queue_snapshot_reference).get('finalization_receipts',[]) if read(r)['receipt_reference']==receipt_ref]
        need(len(finals)==1,'Complete recovery lacks original final seal')
        joins.append(ledger.verify_recovery_candidate_budget(value['external_ledger']['anchor_reference'],settlement,candidate,
          final_solver_record_reference=finals[0]['references']['solver_record'],execution_result_validator=validate_execution_result))
    return new(output_path,dict(kind='SAA_RECOVERY_TERMINAL_ACCEPTANCE_BINDING_V1',admission_reference=ref,
      settlement_reference=settlement,candidate_final_budget_bindings=joins,
      original_numerical_and_final_gates_retained=True,scientific_acceptance=False,
      status='SETTLED_CANDIDATE_FINAL_BUDGET_BOUND' if joins else 'SETTLED_NO_COMPLETE_CANDIDATE'))


def main(*,_development_fixture=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('role',choices=('parent','child'))
    p.add_argument('--admission',required=True);p.add_argument('--expected-sha256',required=True)
    p.add_argument('--parent-execution');p.add_argument('--parent-sha256');a=p.parse_args()
    ref=batch.file_reference(a.admission);need(ref['sha256']==a.expected_sha256,'Admission hash differs')
    if a.role=='parent':answer=run_parent(ref)
    else:
        parent=batch.file_reference(a.parent_execution);need(parent['sha256']==a.parent_sha256,'Parent execution hash differs')
        answer=run_child(ref,parent,_development_fixture=_development_fixture)
    print(json.dumps(answer,allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())
