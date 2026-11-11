"""Case0-only pre-OOS plan lock and bounded resumable item orchestration.

A separately issued planning acceptance must bind every raw planning/input file.
This runner never issues that numerical acceptance and never edits Case1 state.
"""
from .case0_profiles import context_options, validate_config, LEGACY

import fcntl
import json
import math
import os
import uuid
from pathlib import Path
import time

import numpy as np
from scipy.sparse import save_npz

from .campaign_batch import file_reference,verify_reference
from .case0_campaign import METHODS,campaign_root,read,active_config
from .case0_evaluation import Case0EvaluationContext,case0_items,evaluate_case0_item,_write
from .case0_inputs import Case0Context
from .first_stage import FirstStagePlan,assemble_first_stage
from .evaluation_metrics import plan_identity
from . import recourse
from .recourse import assemble_recourse
from .recourse_store import MATRICES,ARRAYS,physical_fingerprint
from .runtime.jobctl import atomic_json,utcnow


def _need(ok,msg):
    if not ok:raise ValueError(msg)


def _read(ref):
    verify_reference(ref);return read(ref['path'])


def planning_evidence(root):
    """Resolve completed evidence from the authoritative attempt map, never a guessed retry."""
    root=Path(root).resolve();state=read(root/'state.json');config=active_config(root)
    original_ref=file_reference(root/'config.lock.json')
    active_ref=state.get('active_config_reference',original_ref)
    configs={original_ref['sha256']:original_ref,active_ref['sha256']:active_ref}
    evidence={}
    for method in METHODS:
        key=f"size{state['size_index']}/{method}";base=root/'planning'/key
        directory=Path(state.get('configuration_attempt_roots',{}).get(key,str(base)))
        _need(directory in (base,base/'refinement-1',base/'continuation-1',base/'memory-repair-1',base/'compact-metadata-1'),'Foreign Case0 completed attempt root')
        if directory in (base/'continuation-1',base/'memory-repair-1',base/'compact-metadata-1'):
            if directory==base/'compact-metadata-1':
                from .case0_memory_repair import validate_representation as validate_transition
            elif directory==base/'memory-repair-1':
                from .case0_memory_repair import validate_transition
            else:
                from .case0_interruption import validate_transition
            _,transition=validate_transition(root,state)
            _need(key=='size0/EA_DRO' and str(directory)==transition['new_attempt_directory']
                  and (directory/'inputs').is_symlink() and
                  (directory/'inputs').resolve()==Path(transition['original_input_directory']),
                  'Interrupted evidence does not match original input/attempt transition')
        elif directory!=base:
            transition=_read(state['numerical_refinement'])
            _need(key=='size1/OR_DRO' and str(directory)==transition['new_attempt_directory'],
                  'Refined evidence does not match numerical transition')
        result_ref=file_reference(directory/'solve/result.json');row=state['completed'][key]
        _need(row['path']==result_ref['path'] and row['sha256']==result_ref['sha256'],
              'Completed Case0 result changed or wrong attempt')
        start_ref=file_reference(directory/'start.json');start=_read(start_ref)
        _need(start['method']==method and start['config_sha256'] in configs,'Unknown method producer configuration')
        config_ref=configs[start['config_sha256']];producer=_read(config_ref)
        result=_read(result_ref)
        if producer.get('solver_zero_presolve') is True:
            _need(result.get('zero_presolve') is True,
                  'Memory repair result omitted configured exact-zero preprocessing')
        else:
            _need('zero_presolve' not in result,
                  'Planning result claimed unconfigured exact-zero preprocessing')
        if producer.get('solver_compact_recourses') is True:
            _need(result.get('descriptive_metadata_representation')=='lossless_categorical_rows_v1'
                  and type(result.get('descriptive_metadata_code_bytes')) is int
                  and result['descriptive_metadata_code_bytes']>=0,
                  'Planning result omitted configured compact descriptive metadata')
        else:
            _need('descriptive_metadata_representation' not in result and
                  'descriptive_metadata_code_bytes' not in result,
                  'Planning result claimed unconfigured compact descriptive metadata')
        accounting_ref=file_reference(directory/'accounting.json');accounting=_read(accounting_ref)
        spent=accounting.get('cumulative_configuration_seconds',accounting['configuration_wall_seconds'])
        _need(type(spent) in (int,float) and math.isfinite(spent) and 0<=spent<=config['configuration_seconds'],
              'Original configuration accounting exceeds budget')
        _need(key not in state.get('configuration_spent_seconds',{}) or
              state['configuration_spent_seconds'][key]==spent,'Completed configuration ledger differs')
        evidence[method]=dict(result_reference=result_ref,input_identity_reference=file_reference(directory/'inputs/identity.json'),
            producer_config_reference=config_ref,producer_source_sha256=producer['source_sha256'],
            configuration_start_reference=start_ref,configuration_accounting_reference=accounting_ref)
    return config,state,evidence


def _verify_producer_sources(refs, expected):
    """Per-method immutable producer root; current evaluator is a distinct source."""
    _need(isinstance(refs,list) and len(refs)==len(expected),'Complete method producer source inventory required')
    roots=set();seen=set()
    for ref in refs:
        verify_reference(ref);path=Path(ref['path'])
        matches=[name for name,sha in expected.items() if ref['sha256']==sha and
                 path.as_posix().endswith('/'+name)]
        _need(len(matches)==1 and matches[0] not in seen,'Producer source/config identity differs')
        name=matches[0];seen.add(name);roots.add(str(path)[:-len(name)])
    _need(len(roots)==1 and seen==set(expected),'Producer files span different source releases')


def validate_case0_planning_acceptance(root,acceptance_reference,*,_attempt_reference=None):
    """Validate exact accepted artifacts/identities; not a replacement for audit."""
    if _read(acceptance_reference).get('kind') == 'CASE0_USER_ACCEPTED_COMPARISON_ADMISSION_V1':
        from .case0_accepted_evaluation import validate_accepted_comparison
        return validate_accepted_comparison(root,acceptance_reference,_attempt_reference=_attempt_reference)
    root=Path(root).resolve();config,state,method_evidence=planning_evidence(root)
    _need(not state.get('audit_reservation_reference'),'Independent audit parent accounting remains unsettled')
    card=_read(acceptance_reference)
    _need(card['kind']=='CASE0_PLANNING_ACCEPTANCE_V1' and card['passed'] is True,'Independent Case0 planning acceptance required')
    _need(card['config_reference']==state.get('active_config_reference',file_reference(root/'config.lock.json')) and
          card['original_config_reference']==file_reference(root/'config.lock.json') and card['size_index']==state['size_index'],
        'Accepted Case0 configuration/size differs')
    _need(state['size_index'] in (0,1) and ((state['active_configuration'] is None and state.get('pending_evaluation_attempt') is None) or
          (_attempt_reference is not None and state.get('pending_evaluation_attempt')==_attempt_reference and
           state['active_configuration']=='CASE0_EVALUATION:PREPARATION')),'Unsettled planning operation')
    _need(state['status'] in ('PLANNING_REVIEW_REQUIRED','CASE0_EVALUATION_IN_PROGRESS','CASE0_EVALUATION_COMPLETE_PENDING_REVIEW'),
        'Case0 is not ready to lock/evaluate')
    audit=_read(card['numerical_audit_reference'])
    _need(audit['kind']=='CASE0_INDEPENDENT_PLANNING_NUMERICAL_AUDIT_V1' and audit['passed'] is True,
          'Independent original-array numerical audit required')
    for key in ('config_reference','original_config_reference','size_index','result_references','input_identity_references','artifact_references','method_evidence'):
        _need(audit[key]==card[key],'Numerical audit/acceptance scope differs: '+key)
    audit_sources=audit['auditor_source_references'];_need(audit_sources,'Auditor source binding required')
    for ref in audit_sources:verify_reference(ref)
    review=_read(audit['independent_review_reference'])
    _need(review['kind']=='CASE0_NUMERICAL_AUDITOR_REVIEW_V1' and review['passed'] is True and
          review['auditor_source_references']==audit_sources,'Numerical auditor lacks exact independent source review')
    _need(audit['original_master_coefficients_checked'] is True and audit['full_q_kkt_physical_checked'] is True
          and audit['risk_bounds_checked'] is True,'Numerical audit scope incomplete')
    _need(card['method_evidence']==method_evidence,'Method source/config/attempt evidence differs')
    evaluation_sources=card['evaluation_source_references']
    _need(evaluation_sources==[file_reference(p) for p in sorted(Path(__file__).parent.rglob('*.py'))],
          'Current evaluation source inventory differs')
    evaluation_review=_read(card['evaluation_review_reference'])
    _need(evaluation_review['kind']=='CASE0_EVALUATOR_SOURCE_REVIEW_V1' and evaluation_review['passed'] is True
          and evaluation_review['evaluation_source_references']==evaluation_sources,'Evaluator source not independently reviewed')
    expected=set();results={};identities={}
    for method in METHODS:
        directory=Path(method_evidence[method]['result_reference']['path']).parent.parent
        input_directory=Path(method_evidence[method]['input_identity_reference']['path']).parent
        manifest_ref=audit['producer_source_manifest_references'][method];manifest=_read(manifest_ref)
        producer_root=Path(manifest_ref['path']).parent
        inventory={row['path']:row['sha256'] for row in manifest['files']}
        mapping=method_evidence[method]['producer_source_sha256']
        _need(all(inventory.get(name)==sha for name,sha in mapping.items()),'Producer manifest/config differs')
        _verify_producer_sources([file_reference(producer_root/name) for name in mapping],mapping)
        result_ref=card['result_references'][method];identity_ref=card['input_identity_references'][method]
        _need(result_ref==method_evidence[method]['result_reference'] and identity_ref==method_evidence[method]['input_identity_reference']
            and Path(result_ref['path'])==directory/'solve/result.json' and Path(identity_ref['path'])==input_directory/'identity.json'
            and (directory/'inputs').resolve()==input_directory,
            'Acceptance uses foreign method evidence')
        result=_read(result_ref);identity=_read(identity_ref)
        row=state['completed'][f"size{state['size_index']}/{method}"]
        _need(row['path']==result_ref['path'] and row['sha256']==result_ref['sha256'],'Accepted result differs from original planning state')
        _need(result['kind']=='CASE0_FULL_FORMULATION_NUMERICAL_RESULT_V1' and result['method_id']==method
            and result['numerical_valid'] is True and result['full_support_independently_evaluated'] is True
            and result['gap_passed'] is True and 0<=result['gap']<=config['gap_target'], 'Plan requires repair/fallback before OOS')
        _need(result['backend']==config['backend'] and 0<=result['configuration_wall_seconds']<=config['configuration_seconds'],
            'Accepted planning backend/budget differs')
        plan=FirstStagePlan(**result['plan']);_need(plan_identity(plan)==result['plan_sha256'],'Accepted plan hash differs')
        _need(plan.task_sha256==identity['task_sha256'] and identity['lead_h']==(0 if method=='OR_DRO' else 1),
            'Plan/input information contract differs')
        _read(identity['task_file']);results[method]=result;identities[method]=identity
        expected.update(ref['path'] for key,ref in method_evidence[method].items() if key.endswith('_reference'))
        expected.update(str(p.resolve()) for parent in (directory/'solve',input_directory) for p in parent.rglob('*') if p.is_file())
    refs=card['artifact_references'];_need(len(refs)==len(expected) and {r['path'] for r in refs}==expected,
        'Acceptance must bind complete original planning/input artifact inventory')
    for ref in refs:verify_reference(ref)
    _need(len({tuple(identities[m]['vehicle_ids']) for m in METHODS})==1,'Method fleet selection differs')
    _need(identities['EA_DRO']['task_sha256']==identities['SAA']['task_sha256']==identities['ARO']['task_sha256'],
        'Baseline methods changed shared H')
    return config,state,results,identities


def prepare_case0_evaluation_lock(*,acceptance_reference,output_path=None,_attempt_reference=None):
    """Reopen frozen inputs and seal IDs only; no OOS sample is generated here."""
    root=campaign_root();config,state,results,identities=validate_case0_planning_acceptance(root,acceptance_reference,_attempt_reference=_attempt_reference)
    size=config['sizes'][state['size_index']];contexts={}
    for method in METHODS:
        directory=Path(_read(acceptance_reference)['input_identity_references'][method]['path']).parent
        context=Case0Context(directory,**size,lead_hours=0 if method=='OR_DRO' else 1,**context_options(config))
        _need(context.identity==identities[method] and results[method]['scenario_ids']==list(context.planning_ids),
              'Reopened Case0 physical inputs/support differ')
        contexts[method]=context
    plans={m:FirstStagePlan(**r['plan']) for m,r in results.items()};plans['DSP']=plans['EA_DRO']
    ea=plans['EA_DRO'];plans['ZERO']=FirstStagePlan(np.zeros_like(ea.x),np.zeros_like(ea.r),ea.task_sha256,ea.config_hash)
    ids={regime:[contexts['EA_DRO'].sample_id(i,split='oos',regime=regime) for i in range(64)] for regime in ('R1','R2','R3')}
    lock=dict(kind='CASE0_PRE_OOS_LOCK_V1',acceptance_reference=acceptance_reference,
        config_reference=_read(acceptance_reference)['config_reference'],original_config_reference=file_reference(root/'config.lock.json'),size_index=state['size_index'],
        method_plan_sha256={m:plan_identity(p) for m,p in plans.items()},sample_ids_by_regime=ids,
        item_count=768,input_identity_references=_read(acceptance_reference)['input_identity_references'],
        formal_OOS=False,Case1_acceptance=False)
    if config.get('profile',LEGACY)!=LEGACY:
        lock.update(profile=config['profile'],target_event=config['target_event'])
    output=Path(output_path) if output_path else root/'evaluation'/'lock.json';output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        _need(read(output)==lock,'Existing Case0 OOS lock cannot change');ref=file_reference(output)
    else:ref=_write(output,lock)
    return ref,contexts,plans


def settle_case0_evaluation_attempt(*,root,terminal_result_reference):
    """Charge actual terminal parent wall time once; never call this child CPU time."""
    from .runtime.jobctl import observation,TERMINAL
    root=Path(root).resolve();result=_read(terminal_result_reference)
    job=root/'jobs'/result['job_id']
    _need(Path(terminal_result_reference['path'])==job/'result.json','Foreign Case0 terminal result')
    status=read(job/'status.json');observed=observation(status)
    _need(result['state'] in TERMINAL and result['complete'] is True and status['state'] in TERMINAL
          and not observed['blocks_new_launch'] and not observed['supervisor_alive']
          and not observed['command_alive'] and not observed['live_group_members'],'Parent job not terminal/dead')
    with (root/'evaluation/.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=read(root/'state.json');ref=state.get('pending_evaluation_attempt')
        if ref is None:
            settled=state.get('last_evaluation_settlement')
            _need(settled is not None,'No evaluation reservation to settle')
            receipt=_read(settled);_need(receipt['terminal_result_reference']==terminal_result_reference,
                                        'Different terminal accounting already settled')
            return settled
        reservation=_read(ref);directory=Path(ref['path']).parent
        _need(reservation['parent_manifest_reference']==file_reference(job/'manifest.json'),
              'Reservation parent identity differs')
        parent_seconds=result['elapsed_seconds'];opening=reservation['opening_compute_seconds']
        _need(type(parent_seconds) in (float,int) and math.isfinite(parent_seconds) and parent_seconds>=0,
              'Invalid terminal parent elapsed time')
        _need(opening<=state['compute_seconds']<=opening+parent_seconds+1e-6,
              'Provisional/global accounting differs from original parent')
        provisional=state['compute_seconds']-opening;target=opening+parent_seconds
        path=directory/'settlement.json'
        if path.exists():
            receipt=read(path)
            _need(receipt['reservation_reference']==ref and receipt['terminal_result_reference']==terminal_result_reference
                  and receipt['cumulative_compute_seconds']==target,'Conflicting evaluation settlement')
        else:
            receipt=dict(kind='CASE0_EVALUATION_PARENT_SETTLEMENT_V1',reservation_reference=ref,
                terminal_result_reference=terminal_result_reference,parent_manifest_reference=reservation['parent_manifest_reference'],
                parent_wall_seconds=parent_seconds,provisional_seconds_already_charged=provisional,
                additional_seconds=parent_seconds-provisional,cumulative_compute_seconds=target,
                accounting_scope='Actual whole parent wall time including isolation; not child CPU or exact preparation time')
            _write(path,receipt)
        settlement_ref=file_reference(path)
        interrupted=not (directory/'provisional_accounting.json').exists()
        state.update(compute_seconds=target,pending_evaluation_attempt=None,last_evaluation_settlement=settlement_ref)
        if interrupted:
            state['status']='CASE0_EVALUATION_INTERRUPTED_REQUIRES_DIAGNOSIS'
            state['next_action']='Parent wall time settled; diagnose incomplete preparation/item before any resume.'
        atomic_json(root/'state.json',state)
        return settlement_ref


def run_case0_evaluation(*,acceptance_reference,seconds):
    """One bounded Case0 job: exact768 items, peritem evidence, actual total charge."""
    from .case0_isolation import verify_current_cgroup
    verify_current_cgroup()
    _need(type(seconds) in (int,float) and 0<seconds<=7200,'Bounded Case0 job seconds required')
    root=campaign_root();began=time.monotonic();evaluation=root/'evaluation';evaluation.mkdir(exist_ok=True)
    with (evaluation/'.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=read(root/'state.json');_need(state['active_configuration'] is None and state.get('pending_evaluation_attempt') is None and not state.get('audit_reservation_reference'),'Unsettled Case0 operation')
        progress_path=evaluation/'state.json'
        progress=read(progress_path) if progress_path.exists() else dict(completed={},active_item=None,status='PREPARED')
        _need(progress['active_item'] is None and progress['status']!='FAILED','Interrupted item/accounting needs diagnosis')
        config=active_config(root);prior=state['compute_seconds']
        end=began+min(float(seconds)-10,config['total_compute_seconds']-prior-10)
        _need(end>began,'Case0 total budget exhausted')
        # Reserve before source admission/input reopening; a killed finally cannot erase cost.
        job_directory=Path(os.environ.get('EVCS_JOB_DIRECTORY','')).resolve()
        _need(job_directory.parent==root/'jobs' and job_directory.name==os.environ.get('EVCS_JOB_ID'),
              'Evaluation requires its Case0 jobctl parent')
        parent_ref=file_reference(job_directory/'manifest.json')
        parent=_read(parent_ref)
        _need(parent['job_id']==job_directory.name,'Wrong evaluation parent manifest')
        _need(all(type(parent[k]) in (int,float) and math.isfinite(parent[k]) and parent[k]>=0
                  for k in ('timeout_seconds','grace_seconds')) and
              parent['timeout_seconds']+parent['grace_seconds']<=config['total_compute_seconds']-prior,
              'Evaluation parent timeout/grace exceeds remaining total budget')
        attempt=evaluation/'attempts'/uuid.uuid4().hex;attempt.mkdir(parents=True)
        reservation=_write(attempt/'reservation.json',dict(kind='CASE0_EVALUATION_ATTEMPT_V1',
            parent_manifest_reference=parent_ref,acceptance_reference=acceptance_reference,
            opening_compute_seconds=prior,started_monotonic=began,started_at_utc=utcnow(),
            deadline_monotonic=end,parent_wall_settlement_required=True))
        state['pending_evaluation_attempt']=reservation
        state['active_configuration']='CASE0_EVALUATION:PREPARATION'
        atomic_json(root/'state.json',state)
        progress['active_preparation']=reservation;atomic_json(progress_path,progress)
        try:
            lock_ref,contexts,plans=prepare_case0_evaluation_lock(acceptance_reference=acceptance_reference,_attempt_reference=reservation)
            lock=_read(lock_ref)
            _need(time.monotonic()<end,'Case0 preparation exhausted job/total deadline')
            progress['active_preparation']=None;state['active_configuration']=None
            if 'lock_reference' in progress:_need(progress['lock_reference']==lock_ref,'Evaluation lock changed')
            progress['lock_reference']=lock_ref
            sources=[file_reference(p) for p in sorted(Path(__file__).parent.rglob('*.py'))]
            if _read(acceptance_reference).get('kind') == 'CASE0_USER_ACCEPTED_COMPARISON_ADMISSION_V1':
                from .case0_accepted_evaluation import evaluation_sources
                sources=evaluation_sources()
            def verify_lock(ref):
                _need(ref==lock_ref,'Foreign Case0 lock');return lock
            def factory(method,index,regime):
                source='OR_DRO' if method=='OR_DRO' else 'EA_DRO'
                ctx=contexts[source]
                return ctx.case,ctx.tasks,ctx.scenario(index,split='oos',regime=regime,departure_h=0 if method=='DSP' else None)
            template_root=evaluation/'templates';template_root.mkdir(exist_ok=True)
            def provider(case,tasks,scene):
                from .recourse_store import _record,bind_template,load_template
                first=assemble_first_stage(case,tasks);key=physical_fingerprint(first,case,tasks,scene)
                folder=template_root/key;binding=template_root/(key+'.binding.json')
                if folder.exists():
                    ref=file_reference(folder/'record.json');row,_=_record(ref)
                    template=load_template(file_reference(binding),first,case,tasks,scene,
                        expected_template_sha256=row['template_sha256'],expected_scenario_index=0,
                        expected_scenario_id=scene.scenario_id)
                    return template,ref
                template=assemble_recourse(case,tasks,scene);folder.mkdir()
                for name in MATRICES:save_npz(folder/(name+'.npz'),getattr(template,name))
                np.savez_compressed(folder/'arrays.npz',**{k:getattr(template,k) for k in ARRAYS})
                _write(folder/'identity.json',template.as_dict())
                ref=_write(folder/'record.json',dict(kind='recourse_template',template_sha256=template.template_sha256,
                    scenario_index=0,scenario_id=scene.scenario_id,
                    artifacts=[file_reference(p) for p in sorted(folder.iterdir())]))
                bind_template(ref,first,case,tasks,scene,source_references={'case0_config':lock['config_reference']},
                    compiler_reference=file_reference(recourse.__file__),output=binding)
                return template,ref
            context=Case0EvaluationContext(lock_reference=lock_ref,lock_validator=verify_lock,plans=plans,
                physical_factory=factory,template_provider=provider,source_references=sources)
            state['evaluation_opened']=True;state['status']='CASE0_EVALUATION_IN_PROGRESS'
            for item in case0_items():
                item_id=item['item_id'];folder=evaluation/'items'/item_id
                if item_id in progress['completed']:
                    ref=progress['completed'][item_id];_need(file_reference(folder/'record.json')==ref,'Completed item pointer changed')
                    evaluate_case0_item(context,item,output_directory=folder,deadline=end,lp_limit_seconds=60)
                    continue
                if end-time.monotonic()<=61:break
                progress['active_item']=item_id;state['active_configuration']='CASE0_EVALUATION:'+item_id
                atomic_json(progress_path,progress);atomic_json(root/'state.json',state)
                result=evaluate_case0_item(context,item,output_directory=folder,deadline=min(end,time.monotonic()+60),lp_limit_seconds=60)
                progress['completed'][item_id]=result['record_reference'];progress['active_item']=None
                state['active_configuration']=None;state['compute_seconds']=prior+time.monotonic()-began
                atomic_json(progress_path,progress);atomic_json(root/'state.json',state)
            complete=len(progress['completed'])==768
            progress['status']='COMPLETE_PENDING_INDEPENDENT_EVALUATION_AUDIT' if complete else 'TIME_LIMIT'
            state['status']='CASE0_EVALUATION_COMPLETE_PENDING_REVIEW' if complete else 'CASE0_EVALUATION_IN_PROGRESS'
            return progress
        except Exception:
            progress['status']='FAILED';state['status']='CASE0_EVALUATION_FAILED_REQUIRES_DIAGNOSIS';raise
        finally:
            state['compute_seconds']=prior+time.monotonic()-began
            _write(attempt/'provisional_accounting.json',dict(reservation_reference=reservation,
                provisional_seconds=state['compute_seconds']-prior,cumulative_compute_seconds=state['compute_seconds'],
                parent_wall_settlement_required=True))
            state['next_action']='Settle the terminal parent wall ledger before further Case0 work; preserve OOS lock.'
            atomic_json(progress_path,progress);atomic_json(root/'state.json',state)
