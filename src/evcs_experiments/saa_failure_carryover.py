"""Single-use settlement of the diagnosed first SAA pre-assembly failure.

No numerical checkpoint, model or incumbent is migrated. Only the independently
settled whole-parent charge is inherited by one source-bound successor queue.
"""
import json
from decimal import Decimal
from pathlib import Path
from .failure_carryover import _need, _read, _verify, _release, _claim_path, reference
from .random_streams import canonical_hash
from .runtime.jobctl import observation, identity_alive

KIND='SAA_PRE_ASSEMBLY_FAILED_PARENT_CARRYOVER_V1'
ALLOWED_CHANGED_SOURCE={'saa_failure_carryover.py','campaign_batch.py','saa_planning_worker.py'}
ERROR='Template index/reader source is not reviewed'


def successor_request_identity(request):
    return canonical_hash({k:v for k,v in request.items() if k!='failure_carryover_reference'})


def successor_queue_identity(queue):
    value=json.loads(json.dumps(queue))
    for item in value['items']:
        item.pop('failure_carryover_reference',None)
        item['request_reference']={'path':item['request_reference']['path']}
    return canonical_hash(value)


def carryover_review_identity(record):
    value=json.loads(json.dumps(record));value['references'].pop('independent_review',None)
    return canonical_hash(value)


def _source_binding(request,release_ref,sources):
    root=Path(release_ref['path']).parent/'src/evcs_experiments'
    expected=[reference(root/name) for name in sorted(sources)]
    _need(request['planning_source_references']==expected,'SAA request source closure differs')
    _verify(request['helper_reference'])


def _rebound(old_ref,new_ref,index_ref,new_release_ref,diagnosis_ref):
    old=_read(old_ref);new=_read(new_ref)
    _need(old['status']==new['status']=='PASS_INDEPENDENT_TEMPLATE_ASSOCIATION_REVIEW'
          and old['passed'] is new['passed'] is True,'Association is not independently accepted')
    _need({k:v for k,v in old.items() if k not in ('implementation_sources','source_rebinding')}==
          {k:v for k,v in new.items() if k not in ('implementation_sources','source_rebinding')},
          'Association index/proof/axes/scope changed')
    _need(old['index_reference']==index_ref,'Association index differs')
    _verify(index_ref)
    reb=new['source_rebinding']
    _need(reb['previous_review_reference']==old_ref and reb['independent_diagnosis_reference']==diagnosis_ref
          and reb['old_index_and_axes_preserved'] is True and reb['old_numerical_evidence_unchanged'] is True
          and reb['new_numeric_work_performed'] is False,'Source rebound diagnosis/scope differs')
    pairs=reb['byte_identity_pairs'];names=('recourse_template_index.py','recourse_store.py')
    _need(len(pairs)==2,'Exactly two reader/index source pairs required')
    current=[]
    for name in names:
        matches=[p for p in pairs if Path(p['current']['path']).name==name]
        _need(len(matches)==1,'Reader/index pair missing or duplicate');p=matches[0]
        _verify(p['previous']);_verify(p['current'])
        _need(p['previous'] in old['implementation_sources'] and p['exact_bytes_equal'] is True
              and p['previous']['sha256']==p['current']['sha256'] and p['previous']['size_bytes']==p['current']['size_bytes']
              and p['current']['path']==str(Path(new_release_ref['path']).parent/'src/evcs_experiments'/name),
              'Reader/index rebound is not exact current source bytes')
        current.append(p['current'])
    _need(new['implementation_sources']==old['implementation_sources']+current,'Rebound implementation inventory differs')


def verify_saa_failure_carryover(ref,worker_request,queue_root,*,require_claim=True):
    record=_read(ref);refs=record['references']
    _need(record.get('schema_version')==1 and record.get('kind')==KIND,'Unknown SAA carryover schema')
    data={k:_read(v) for k,v in refs.items()}
    review=data['independent_review']
    _need(review['status']=='PASS_SAA_FAILURE_CARRYOVER_REVIEW' and
          review['candidate_payload_sha256']==carryover_review_identity(record),'Independent carryover review is not bound to candidate')
    successor_path=Path(record['successor_queue_request_path'])
    _need(successor_path.is_absolute(),'Absolute successor queue request required')
    successor=json.loads(successor_path.read_text())
    _need(successor_queue_identity(successor)==record['successor_queue_identity_sha256'] and
          successor['queue_root']==record['successor_queue_root'] and len(successor['items'])==1 and
          successor['items'][0]['plan_id']==record['plan_id'] and successor['items'][0]['plan_budget_seconds']==7200 and
          successor['items'][0]['failure_carryover_reference']==ref and
          _read(successor['items'][0]['request_reference'])==worker_request, 'Successor queue source/path/plan binding differs')
    old=data['worker_request'];q=data['queue_request'];job=data['job_result'];jm=data['job_manifest']
    state=data['batch_state'];run=data['batch_result'];inv=data['invocation'];failure=data['failure'];wf=data['worker_failure']
    settlement=data['settlement'];diagnosis=data['independent_diagnosis']
    _need(record['successor_queue_root']==str(Path(queue_root).resolve()) and
          record['successor_output_root']==worker_request['output_root'] and
          record['successor_request_identity_sha256']==successor_request_identity(worker_request),'SAA successor identity differs')
    _need(worker_request['release_reference']==record['new_release_reference'],'Successor release reference differs')
    _need(record['plan_id']==old['plan_id']==worker_request['plan_id'] and
          old['method_id']==worker_request['method_id']=='SAA' and old['domain']=='formal' and old['spec_id']=='S05'
          and old['model_id']=='IEEE33-v54-Q1' and old['model_input_mode']=='assemble','Not a formal SAA assembly failure')
    exceptions={'output_root','planning_source_references','helper_reference','template_association_review_reference',
                'failure_carryover_reference','release_reference'}
    _need({k:v for k,v in old.items() if k not in exceptions}==
          {k:v for k,v in worker_request.items() if k not in exceptions},'SAA scientific input/config/budget changed')
    _need('failure_carryover_reference' not in old and 'model_manifest_reference' not in old
          and old['budget']['cumulative_wall_seconds']==7200,'Only first unassembled original-budget attempt is admitted')
    for key in ('protocol_reference','config_reference','bank_acceptance_reference','summary_reference','audit_reference'):_verify(old[key])
    for template in old['template_references']:_verify(template)
    old_sources,old_engine=_release(record['old_release_reference'])
    new_sources,new_engine=_release(record['new_release_reference'])
    changed={n for n in set(old_sources)|set(new_sources) if old_sources.get(n)!=new_sources.get(n)}
    _need(set(old_sources)<=set(new_sources) and changed<=ALLOWED_CHANGED_SOURCE and old_engine==new_engine,
          'SAA carryover changes unapproved mathematical/source modules')
    _need(record['source_changes']=={n:dict(old=old_sources.get(n),new=new_sources.get(n)) for n in sorted(changed)},'SAA source diff differs')
    _source_binding(old,record['old_release_reference'],old_sources)
    _source_binding(worker_request,record['new_release_reference'],new_sources)
    _need(old['helper_reference']['sha256']==worker_request['helper_reference']['sha256'] and
          old['helper_reference']['size_bytes']==worker_request['helper_reference']['size_bytes'],'Numerical helper changed')
    _rebound(old['template_association_review_reference'],worker_request['template_association_review_reference'],
             old['template_index_reference'],record['new_release_reference'],refs['independent_diagnosis'])
    _need(len(q['items'])==1 and q['items'][0]['request_reference']==refs['worker_request'] and
          q['items'][0]['plan_id']==old['plan_id'] and q['items'][0]['plan_budget_seconds']==7200,
          'Predecessor must be one original fixed SAA plan')
    _need(q['queue_root']==str(Path(refs['batch_state']['path']).parent) and
          state['request_reference']==refs['queue_request']==run['request_reference']==inv['request_reference'],
          'Predecessor queue identity differs')
    _need(state['attempt_receipts']==[] and state['halted']=='VALIDITY_OR_WORKER_FAILURE_REQUIRES_REVIEW'
          and state['active']['invocation_reference']==refs['invocation'] and state['runs']==[refs['batch_result']],
          'Predecessor is not a single unresolved failed attempt')
    _need(inv['item_request_reference']==refs['worker_request'] and inv['plan_id']==old['plan_id'] and
          inv['cumulative_plan_seconds']==0 and inv['opening_carryover_seconds']==0 and
          inv['failure_carryover_reference'] is None and inv['previous_checkpoint'] is None and inv['previous_result_reference'] is None,
          'Prior SAA cost/checkpoint cannot be reset')
    attempt=Path(inv['attempt_directory']);oldroot=Path(old['output_root'])
    _need(list(Path(q['queue_root']).glob('*__attempt_*'))==[attempt],'Unexpected predecessor attempts')
    _need(not any(oldroot.rglob('*')) and
          not any((attempt/n).exists() for n in ('worker_result.json','worker_checkpoint.json','receipt.json','finalization.json')),
          'Old model/phases/results cannot be carried as pre-assembly failure')
    _need(inv['parent_job']['manifest_reference']==refs['job_manifest'] and
          inv['parent_job']['job_id']==job['job_id']==jm['job_id'] and
          '--request' in jm['argv'] and jm['argv'][jm['argv'].index('--request')+1]==refs['queue_request']['path'],
          'Parent job/request binding differs')
    _need(job.get('complete') is True and job['state']=='FAILED' and job['returncode']==2 and
          run['status']=='FAILED_VALIDITY_OR_WORKER' and run['unsettled_plan_seconds'] is True,
          'Predecessor is not terminal failed')
    _need(not observation(job)['blocks_new_launch'] and not identity_alive(state['active']['process']),
          'Predecessor processes remain alive')
    _need(wf['status']=='FAILED_SAA_WORKER_REQUIRES_DIAGNOSIS' and wf['error_type']=='ValueError' and wf['error']==ERROR
          and failure['error']=='Worker exited nonzero: 1','Unreviewed SAA failure reason')
    _need(diagnosis['status']=='DIAGNOSED_PRE_ASSEMBLY_SOURCE_ASSOCIATION_FAILURE' and diagnosis['passed_diagnosis'] is True
          and diagnosis['job_result_reference']==refs['job_result'] and diagnosis['worker_failure_reference']==refs['worker_failure']
          and diagnosis['queue_state_reference']==refs['batch_state'] and diagnosis['assembly_completed'] is False
          and diagnosis['master_calls']==diagnosis['fixed_q_calls']==0,'Independent SAA diagnosis differs')
    mapping={'parent_result_reference':'job_result','parent_manifest_reference':'job_manifest','worker_failure_reference':'worker_failure',
             'queue_failure_reference':'failure','request_reference':'worker_request','queue_request_reference':'queue_request',
             'independent_diagnosis_reference':'independent_diagnosis'}
    _need(all(settlement[k]==refs[v] for k,v in mapping.items()) and
          settlement['source_manifest_reference']==record['old_release_reference'],'Settlement evidence differs')
    amount=Decimal(str(job['elapsed_seconds']))
    _need(amount.is_finite() and 0<amount<7200 and record['accounted_seconds']==float(amount)
          and settlement['kind']=='SAA_PRE_ASSEMBLY_FAILED_PARENT_ACCOUNTING_V1'
          and settlement['status']=='SETTLED_FAILED_PARENT_NOT_RESUME_ADMISSION' and settlement['accounting_PASS'] is True
          and settlement['plan_id']==record['plan_id'] and settlement['opening_cumulative_seconds_decimal']=='0'
          and Decimal(settlement['terminal_parent_seconds_decimal'])==amount==Decimal(settlement['cumulative_accounted_seconds_decimal'])
          and Decimal(settlement['remaining_seconds_decimal'])==Decimal(7200)-amount
          and settlement['cumulative_cap_seconds']==7200 and settlement['parent_charged_once'] is True
          and settlement['nested_seconds_added'] is False,'Settled SAA parent cost reset or double counted')
    _need(float(amount)>=run['elapsed_seconds']>=failure['elapsed_seconds']>=wf['cumulative_accounted_seconds']>=0
          and run['cumulative_plan_seconds']=={old['plan_id']:failure['elapsed_seconds']},'Nested SAA clocks differ')
    claim=dict(carryover_reference=ref,successor_queue_root=record['successor_queue_root'],successor_output_root=record['successor_output_root'])
    if require_claim:_need(json.loads(_claim_path(record).read_text())==claim,'SAA predecessor claimed by another successor')
    return dict(reference=ref,accounted_seconds=float(amount),plan_id=record['plan_id'],successor_queue_root=record['successor_queue_root'])


def claim_saa_failure_carryover(ref,worker_request,queue_root):
    result=verify_saa_failure_carryover(ref,worker_request,queue_root,require_claim=False)
    record=_read(ref);path=_claim_path(record);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(dict(carryover_reference=ref,successor_queue_root=record['successor_queue_root'],
                       successor_output_root=record['successor_output_root']),stream,indent=2);stream.write('\n')
    return result
