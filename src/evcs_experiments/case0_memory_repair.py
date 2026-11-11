"""Single reviewed exact-zero memory recovery; original science and budgets persist."""
from pathlib import Path
import argparse
import json
import math
import time

from .case0_interruption import need, read, _root, _dead_parent, linger_enabled
from .campaign_batch import file_reference, verify_reference
from .case0_profiles import EXPANDED, validate_config
from .runtime.jobctl import atomic_json

KEY='size0/EA_DRO'
MODULE='src/evcs_experiments/case0_memory_repair.py'
CHANGED_SOURCES={MODULE,*('src/evcs_experiments/'+n for n in (
    'case0_campaign.py','case0_evaluation_runner.py','case0_interruption.py',
    'case0_planning.py','case0_planning_audit.py'))}
NUMERICAL={'case0_planning.py':'284f977d9b04b1fff59ceea0287d04f533c6f4e62711356e100cdc8e5f11a38f',
           'case0_planning_audit.py':'683bbdc4ceca62122e0b287b8fd8ee38ab8e3e6871ecc5668ebdff2c331f2b11'}
REFS={'original_config','config','state','original_source_manifest','parent_manifest','parent_result','settlement','prepared','review'}
DIAGNOSTIC_SECONDS=7.349020892754197


def _finite(*values):
    need(all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in values),'Invalid memory recovery budget measurement')


def _same(a,b,label):
    _finite(a,b);need(abs(a-b)<=1e-8,label)


def _sources(old,new):
    need(set(new)==set(old)|{MODULE},'Memory repair source inventory differs')
    need(all(new[k]==v for k,v in old.items() if k not in CHANGED_SOURCES),
         'Memory repair changed mathematical/input/solver source')
    for name,digest in NUMERICAL.items():
        need(new.get('src/evcs_experiments/'+name)==digest,'Memory repair numerical component differs')


def _review(request):
    need(set(request)=={'kind','references','original_release_root','new_source_sha256'} and
         request['kind']=='CASE0_MEMORY_REPAIR_REQUEST_V1' and set(request['references'])==REFS,
         'Wrong memory repair request schema')
    refs=request['references'];review=read(refs['review'])
    need(review['kind']=='CASE0_MEMORY_REPAIR_REVIEW_V1' and review['passed'] is True and
         review['references']=={k:v for k,v in refs.items() if k!='review'} and
         review['new_source_sha256']==request['new_source_sha256'],'Exact memory repair independent review required')
    return refs


def _prepared(reference,new_sources):
    value=read(reference)
    need(value['kind']=='CASE0_MEMORY_REPAIR_PREPARED_COMPONENT_V1' and value['component_pass'] is True and
         all(value[k] is False for k in ('controller_wiring_complete','deployed','launched','production_memory_pass')),
         'Prepared component is not the original unlaunched numerical evidence')
    for ref in value['files'].values():verify_reference(ref)
    card=read(value['files']['component_review.json'])
    need(card['kind']=='CASE0_EXACT_ZERO_COMPONENT_INDEPENDENT_REVIEW_V1' and card['passed'] is True,
         'Exact-zero numerical component lacks independent review')
    for name,digest in NUMERICAL.items():
        ref=value['files'][name];proof=card['source_references'][name]
        need(ref['sha256']==proof['sha256']==digest==new_sources['src/evcs_experiments/'+name] and
             ref['size_bytes']==proof['size_bytes'],'Prepared and deployed numerical bytes differ')
    rows=[json.loads(line) for line in Path(value['files']['zero_reduction_diagnostic.jsonl']['path']).read_text().splitlines()]
    need(len(rows)==3 and [r['stage'] for r in rows]==['loaded','reduced','audited'] and
         rows[-1]['no_optimizer'] is True and rows[-1]['audit']['status']=='PASS_EXACT_ZERO_PROOF' and
         rows[-1]['elapsed_seconds']==value['diagnostic_measured_seconds_pending_preparation_accounting']==DIAGNOSTIC_SECONDS,
         'Original diagnostic expense/proof differs')
    return DIAGNOSTIC_SECONDS


def _settled(root,refs,state):
    from .case0_interruption import validate_transition as original_transition
    config,prior=original_transition(root,state)
    need(refs['original_config']==file_reference(root/'config.lock.json') and
         refs['config']==state['active_config_reference']==prior['active_config_reference'],
         'Original/R2 configuration binding differs')
    need(read(refs['config'])==config and validate_config(config,root)==EXPANDED and
         'solver_zero_presolve' not in config,'Memory repair can only follow original R2 configuration')
    s=read(refs['settlement']);need(s['kind']=='CASE0_OOM_PARENT_SETTLEMENT_V1' and
         s['scientific_pass'] is False and s['OOS_opened'] is False and s['automatic_retry'] is False and
         s['configuration_key']==KEY and s['parent_reference']==refs['parent_result'] and
         s['parent_manifest_reference']==refs['parent_manifest'] and
         s['prior_interruption_transition_reference']==state['operational_interruption'] and
         state['planning_parent_settlement_reference']==state['memory_failure_settlement_reference']==refs['settlement'],
         'Exact settled OOM parent/old transition required')
    for key,ref in s.items():
        if key.endswith('_reference'):read(ref)
    parent=_dead_parent(root,refs['parent_result'],refs['parent_manifest'])
    need(parent['state']=='FAILED' and parent['returncode']==1,'Original failed guard parent required')
    elapsed=parent['elapsed_seconds'];_finite(elapsed);need(elapsed>0,'Empty failed parent expense')
    _same(s['parent_elapsed_seconds'],elapsed,'Parent elapsed differs')
    before=read(s['state_before_reference']);launch=read(s['launch_admission_reference'])
    need(before['active_job_id']==parent['job_id'] and before['active_configuration']==KEY and
         before['status']=='PLANNING' and before['completed']=={} and
         before['planning_launch_admission_reference']==s['launch_admission_reference'] and
         before['operational_interruption']==state['operational_interruption'] and
         before['active_config_reference']==refs['config'],'Failed state/parent/source identity differs')
    for actual,expected,label in (
        (s['opening_compute_seconds'],launch['opening_compute_seconds'],'Opening cost differs'),
        (s['previously_accounted_compute_seconds'],before['compute_seconds'],'Prior global cost differs'),
        (s['already_charged_parent_seconds'],s['previously_accounted_compute_seconds']-s['opening_compute_seconds'],'Already charged cost differs'),
        (s['added_compute_seconds'],elapsed-s['already_charged_parent_seconds'],'Parent incremental cost differs'),
        (s['closing_compute_seconds'],s['previously_accounted_compute_seconds']+s['added_compute_seconds'],'Parent closing cost differs'),
        (s['prior_configuration_seconds'],prior['prior_configuration_seconds'],'Prior configuration cost differs'),
        (s['prior_master_seconds'],prior['prior_master_seconds'],'Prior master cost differs'),
        (s['closing_configuration_seconds'],s['prior_configuration_seconds']+elapsed,'Configuration expense reset'),
        (s['closing_master_seconds'],s['prior_master_seconds']+elapsed,'Master expense reset'),
        (s['remaining_configuration_seconds'],1800-s['closing_configuration_seconds'],'Configuration remainder differs'),
        (s['remaining_master_seconds'],1200-s['closing_master_seconds'],'Master remainder differs')):_same(actual,expected,label)
    need(state['compute_seconds']==s['closing_compute_seconds'] and
         state['configuration_spent_seconds']=={KEY:s['closing_configuration_seconds']} and
         state['configuration_master_spent_seconds']=={KEY:s['closing_master_seconds']},'Settled cumulative budgets changed')
    terminal=read(s['terminal_review_reference']);resource=read(s['resource_observation_reference'])
    need(terminal['kind']=='CASE0_EXPANDED_RECOVERY_TERMINAL_INDEPENDENT_REVIEW_V1' and terminal['passed'] is True and
         resource['unit_properties']['Result']=='oom-kill' and str(resource['unit_properties']['ExecMainStatus'])=='9',
         'Independent owned-unit memory termination evidence required')
    for ref in terminal['references'].values():verify_reference(ref)
    need(terminal['references']['parent_result']==refs['parent_result'] and
         terminal['references']['parent_manifest']==refs['parent_manifest'] and
         terminal['references']['resource_observation']==s['resource_observation_reference'] and
         terminal['references']['guard_failure']==s['guard_failure_reference'], 'Terminal review evidence differs')
    old=root/'planning'/KEY;failed=old/'continuation-1'
    need((failed/'inputs').is_symlink() and (failed/'inputs').resolve()==old/'inputs' and
         file_reference(old/'inputs/identity.json')==s['original_input_identity_reference'], 'Original input path/identity differs')
    for folder,key in ((old,'original_attempt_manifest_reference'),(failed,'failed_attempt_manifest_reference')):
        inventory=read(s[key]);need(inventory['root']==str(folder) and inventory['files'],'Failed attempt inventory root differs')
        # The original inventory predates continuation-1. Never rewrite either attempt.
        actual={str(p.resolve()) for p in folder.rglob('*') if p.is_file() and
                not (folder==old and p.is_relative_to(failed))}
        need(actual=={r['path'] for r in inventory['files']} and len(actual)==len(inventory['files']),
             'Original/failed attempt inventory incomplete')
        for ref in inventory['files']:verify_reference(ref)
        need(not any((folder/name).exists() for name in ('accounting.json','failure.json','solve/result.json')),
             'Conflicting completed attempt evidence')
    return config,s


def validate_transition(root,state):
    """Revalidate immutable source, review and ledger after live state advances."""
    root=_root(root);revision=root/'revisions/memory-repair-1';ref=state['memory_repair'];t=read(ref)
    need(Path(ref['path'])==revision/'transition.json' and t['kind']=='CASE0_MEMORY_REPAIR_TRANSITION_V1',
         'Foreign memory repair transition')
    request=read(t['request_reference']);refs=_review(request);snapshot=read(t['settled_state_snapshot_reference'])
    need(t['settled_state_snapshot_reference']['sha256']==refs['state']['sha256'] and
         t['settled_state_snapshot_reference']['size_bytes']==refs['state']['size_bytes'],'Settled memory snapshot differs')
    from .case0_interruption import validate_transition as original_transition
    historical_config,_=original_transition(root,snapshot)
    original=read(refs['config']);revised=read(t['active_config_reference']);s=read(refs['settlement'])
    need(historical_config==original and snapshot['active_config_reference']==refs['config'],
         'Historical R2 configuration ancestry changed')
    need(t['active_config_reference']==state['active_config_reference'] and
         Path(t['active_config_reference']['path'])==revision/'config.json' and revised['solver_zero_presolve'] is True and
         {k:v for k,v in revised.items() if k not in ('source_sha256','solver_zero_presolve')}==
         {k:v for k,v in original.items() if k!='source_sha256'},'Memory repair scientific/config identity changed')
    _sources(original['source_sha256'],revised['source_sha256']);validate_config(revised,root)
    need(revised['source_sha256']==request['new_source_sha256'] and t['settlement_reference']==refs['settlement'] and
         t['prepared_reference']==refs['prepared'] and snapshot['memory_failure_settlement_reference']==refs['settlement'] and
         t['prior_interruption_transition_reference']==snapshot['operational_interruption']==state['operational_interruption'],
         'Memory repair source/settlement ancestry changed')
    diagnostic=_prepared(refs['prepared'],request['new_source_sha256']);admission=t['admission_seconds'];_finite(admission)
    need(t['diagnostic_seconds']==diagnostic and t['prior_master_seconds']==s['closing_master_seconds'] and
         t['prior_configuration_seconds']==s['closing_configuration_seconds']+admission and
         t['opening_compute_seconds']==s['closing_compute_seconds'] and
         t['cumulative_compute_seconds']==s['closing_compute_seconds']+diagnostic+admission and
         state['memory_diagnostic_accounting_reference']==ref,'Memory repair expense binding changed')
    need(state.get('pending_diagnostic_accounting_reference') is None and
         type(state.get('compute_seconds')) in (int,float) and math.isfinite(state['compute_seconds']) and
         state['compute_seconds']>=t['cumulative_compute_seconds'] and
         state['configuration_spent_seconds'].get(KEY,0)>=t['prior_configuration_seconds'] and
         state['configuration_master_spent_seconds'].get(KEY,0)>=t['prior_master_seconds'],
         'Live memory recovery cumulative budget reset or diagnostic still pending')
    need(t['new_attempt_directory']==str(root/'planning'/KEY/'memory-repair-1') and
         t['original_input_directory']==str(root/'planning'/KEY/'inputs'),'Memory attempt/input path changed')
    return revised,t


def prepare_memory_repair(request_path,expected_sha256):
    began=time.monotonic()
    from . import case0_campaign as c
    root=_root(c.campaign_root());request_ref=file_reference(request_path)
    need(request_ref['sha256']==expected_sha256,'Memory repair request changed')
    request=read(request_ref);need(request.get('kind')=='CASE0_MEMORY_REPAIR_REQUEST_V1','Wrong memory repair request kind')
    revision=root/'revisions/memory-repair-1';revision.mkdir(parents=True,exist_ok=False)
    atomic_json(revision/'reservation.json',dict(request_reference=request_ref,admission_started_monotonic=began,
        incomplete_admission_requires_manual_accounting=True,automatic_retry_allowed=False))
    refs=_review(request);state=read(refs['state'])
    need(refs['state']==file_reference(root/'state.json') and state['status']=='MEMORY_REPAIR_REVIEW_REQUIRED' and
         state['size_index']==0 and state['completed']=={} and state['evaluation_opened'] is False and
         state['active_configuration'] is None and state.get('active_job_id') is None and
         not any(state.get(k) for k in ('planning_parent_pending_settlement','pending_evaluation_attempt',
             'audit_reservation_reference','memory_repair','memory_diagnostic_accounting_reference')),
         'Single settled memory recovery not available')
    need(refs['prepared']==state.get('memory_repair_prepared_reference')==
         state.get('pending_diagnostic_accounting_reference'),'Pending diagnostic authority differs')
    config,s=_settled(root,refs,state);_sources(config['source_sha256'],request['new_source_sha256'])
    need(request['new_source_sha256']==c.sources(),'Current memory source differs from reviewed request')
    diagnostic=_prepared(refs['prepared'],request['new_source_sha256'])
    source_root=Path(request['original_release_root']).resolve();manifest=read(refs['original_source_manifest'])
    need(Path(refs['original_source_manifest']['path'])==source_root/'RELEASE_MANIFEST.json' and
         state['source_release_reference']==refs['original_source_manifest'] and state['release_root']==str(source_root),
         'Original R2 release identity differs')
    inventory={r['path']:r['sha256'] for r in manifest['files']}
    need(len(inventory)==len(manifest['files']),'Duplicate original release inventory')
    for name,digest in config['source_sha256'].items():
        path=source_root/name
        need(not Path(name).is_absolute() and path.resolve().is_relative_to(source_root) and
             inventory.get(name)==digest==c.digest(path),'Original R2 source bytes changed')
    need(c.digest(c.get_paths().data_root/'MANIFEST_SHA256.csv')==config['data_manifest_sha256'],'Frozen data manifest changed')
    linger_enabled();(revision/'state_before.json').write_bytes((root/'state.json').read_bytes())
    revised=dict(config,source_sha256=request['new_source_sha256'],solver_zero_presolve=True)
    atomic_json(revision/'config.json',revised);admission=time.monotonic()-began;_finite(admission)
    cfg=s['closing_configuration_seconds']+admission;master=s['closing_master_seconds']
    compute=s['closing_compute_seconds']+diagnostic+admission
    need(cfg<1800 and master<1200 and compute<config['total_compute_seconds'],'Memory admission exhausts original cumulative budget')
    t=dict(kind='CASE0_MEMORY_REPAIR_TRANSITION_V1',request_reference=request_ref,
        original_config_reference=refs['original_config'],prior_config_reference=refs['config'],
        active_config_reference=file_reference(revision/'config.json'),settled_state_snapshot_reference=file_reference(revision/'state_before.json'),
        settlement_reference=refs['settlement'],prepared_reference=refs['prepared'],
        prior_interruption_transition_reference=state['operational_interruption'],admission_seconds=admission,
        diagnostic_seconds=diagnostic,diagnostic_accounting_scope='Previously uncharged saved-array diagnostic; global only, not native or configuration solve time.',
        prior_configuration_seconds=cfg,prior_master_seconds=master,opening_compute_seconds=s['closing_compute_seconds'],
        cumulative_compute_seconds=compute,new_attempt_directory=str(root/'planning'/KEY/'memory-repair-1'),
        original_input_directory=str(root/'planning'/KEY/'inputs'),scientific_protocol_changed=False,formal_OOS=False,
        Case1_mutated=False,automatic_second_retry=False,
        admission_clock_scope='Function entry through source/input checks and config persistence; final receipt/state persistence is closing operational overhead.')
    atomic_json(revision/'transition.json',t);transition_ref=file_reference(revision/'transition.json')
    state.update(active_config_reference=t['active_config_reference'],memory_repair=transition_ref,
        memory_diagnostic_accounting_reference=transition_ref,pending_diagnostic_accounting_reference=None,
        status='MEMORY_REPAIR_PREPARED',
        configuration_attempt_roots={KEY:t['new_attempt_directory']},configuration_spent_seconds={KEY:cfg},
        configuration_master_spent_seconds={KEY:master},compute_seconds=compute,
        next_action='Run one reviewed exact-zero memory recovery at unchanged Zeta48 inputs and remaining cumulative budgets; no repeat claim or OOS.')
    c.save_state(root,state);c.decision(root,'CASE0_MEMORY_REPAIR_PREPARED',transition_reference=transition_ref)
    return state


# A second representation can be admitted without changing any historical API.
REPRESENTATION='compact-metadata-1'
REPRESENTATION_REFS={'config','state','source_manifest','parent_manifest','parent_result','settlement','component','review'}
COMPACT_MODULE='src/evcs_experiments/compact_recourses.py'
REPRESENTATION_SOURCES={MODULE,COMPACT_MODULE,*('src/evcs_experiments/'+n for n in
    ('case0_campaign.py','case0_evaluation_runner.py','case0_interruption.py','case0_planning.py'))}


def _representation_review(request):
    continuation=request.get('kind')=='CASE0_REPRESENTATION_CONTINUATION_REQUEST_V1'
    expected_refs=REPRESENTATION_REFS|({'failed_admission_settlement'} if continuation else set())
    need(set(request)=={'kind','revision','references','new_source_sha256'} and
         request['kind'] in ('CASE0_REPRESENTATION_REQUEST_V1','CASE0_REPRESENTATION_CONTINUATION_REQUEST_V1') and
         request['revision']==REPRESENTATION and set(request['references'])==expected_refs,
         'Wrong representation request schema')
    refs=request['references'];review=read(refs['review'])
    need(review['kind']==('CASE0_REPRESENTATION_CONTINUATION_REVIEW_V1' if continuation else 'CASE0_REPRESENTATION_REVIEW_V1') and review['passed'] is True and
         review['revision']==REPRESENTATION and review['references']=={k:v for k,v in refs.items() if k!='review'} and
         review['new_source_sha256']==request['new_source_sha256'],'Exact representation independent review required')
    component=read(refs['component'])
    need(component['kind']=='CASE0_COMPACT_RECOURSE_COMPONENT_REVIEW_V1' and component['passed'] is True and
         component['exact_model_unchanged'] is True and
         set(component['source_sha256'])=={COMPACT_MODULE,'src/evcs_experiments/case0_planning.py'} and
         all(request['new_source_sha256'].get(k)==v for k,v in component['source_sha256'].items()),
         'Representation component/source binding differs')
    return refs


def _representation_sources(old,new):
    need(set(new)==set(old)|{COMPACT_MODULE} and COMPACT_MODULE not in old,
         'Representation source inventory differs')
    need(all(new[k]==v for k,v in old.items() if k not in REPRESENTATION_SOURCES),
         'Representation changed mathematical/input/solver source')
    need(old.get('src/evcs_experiments/case0_planning_audit.py')==NUMERICAL['case0_planning_audit.py'],
         'Original numerical checker differs')


def _representation_ancestry(root,refs,state):
    config,prior=validate_transition(root,state)
    need(refs['config']==state['active_config_reference']==prior['active_config_reference'] and
         read(refs['config'])==config and validate_config(config,root)==EXPANDED and
         config.get('solver_zero_presolve') is True and 'solver_compact_recourses' not in config,
         'Original R3 configuration ancestry differs')
    s=read(refs['settlement'])
    need(s['kind']=='CASE0_REPRESENTATION_PARENT_SETTLEMENT_V1' and s['scientific_pass'] is False and
         s['OOS_opened'] is False and s['automatic_retry'] is False and s['configuration_key']==KEY and
         s['parent_reference']==refs['parent_result'] and s['parent_manifest_reference']==refs['parent_manifest'] and
         s['representation_transition_reference']==state['memory_repair'] and
         s['prior_parent_settlement_reference']==prior['settlement_reference'] and
         state['planning_parent_settlement_reference']==state['representation_failure_settlement_reference']==refs['settlement'],
         'Representation settled parent/source ancestry differs')
    for key,ref in s.items():
        if key.endswith('_reference'):verify_reference(ref)
    parent=_dead_parent(root,refs['parent_result'],refs['parent_manifest'])
    need(parent['state']=='FAILED' and parent['returncode']==1,'Failed guarded parent required')
    elapsed=parent['elapsed_seconds'];_finite(elapsed);need(elapsed>0,'Empty failed parent expense')
    before=read(s['state_before_reference']);launch=read(s['launch_admission_reference'])
    need(before['active_job_id']==parent['job_id'] and before['active_configuration']==KEY and
         before['status']=='PLANNING' and before['completed']=={} and
         before['planning_launch_admission_reference']==s['launch_admission_reference'] and
         before['memory_repair']==state['memory_repair'] and before['active_config_reference']==refs['config'] and
         launch['transition_reference']==state['memory_repair'],'Failed state/launch identity differs')
    for actual,expected,label in (
        (s['parent_elapsed_seconds'],elapsed,'Parent elapsed differs'),
        (s['opening_compute_seconds'],launch['opening_compute_seconds'],'Opening cost differs'),
        (s['previously_accounted_compute_seconds'],before['compute_seconds'],'Prior global cost differs'),
        (s['already_charged_parent_seconds'],s['previously_accounted_compute_seconds']-s['opening_compute_seconds'],'Already charged cost differs'),
        (s['added_compute_seconds'],elapsed-s['already_charged_parent_seconds'],'Parent incremental cost differs'),
        (s['closing_compute_seconds'],s['previously_accounted_compute_seconds']+s['added_compute_seconds'],'Parent closing cost differs'),
        (s['prior_configuration_seconds'],prior['prior_configuration_seconds'],'Prior configuration cost differs'),
        (s['prior_master_seconds'],prior['prior_master_seconds'],'Prior master cost differs'),
        (s['closing_configuration_seconds'],s['prior_configuration_seconds']+elapsed,'Configuration expense reset'),
        (s['closing_master_seconds'],s['prior_master_seconds']+elapsed,'Master expense reset'),
        (s['remaining_configuration_seconds'],1800-s['closing_configuration_seconds'],'Configuration remainder differs'),
        (s['remaining_master_seconds'],1200-s['closing_master_seconds'],'Master remainder differs')):_same(actual,expected,label)
    need(state['compute_seconds']==s['closing_compute_seconds'] and
         state['configuration_spent_seconds']=={KEY:s['closing_configuration_seconds']} and
         state['configuration_master_spent_seconds']=={KEY:s['closing_master_seconds']},'Settled cumulative budgets changed')
    terminal=read(s['terminal_review_reference'])
    need(terminal['kind']=='CASE0_R3_TERMINAL_INDEPENDENT_REVIEW_V1' and terminal['passed'] is True and
         terminal['references']['parent_result']==refs['parent_result'] and
         terminal['references']['parent_manifest']==refs['parent_manifest'] and
         terminal['references']['launch_admission']==s['launch_admission_reference'] and
         terminal['references']['resources']==s['resource_observation_reference'] and
         terminal['references']['guard_failure']==s['guard_failure_reference'] and
         terminal['references']['master_log']==s['master_log_reference'] and
         all(terminal['observation_reference'][k]==s['terminal_observation_reference'][k]
             for k in ('sha256','size_bytes')),'Independent R3 terminal evidence differs')
    # This historical card retains its producer's local path. Its designated
    # server mirror is already hashed above; no other reference loses path binding.
    need(set(terminal['observation_reference'])=={'path','sha256','size_bytes'} and
         Path(terminal['observation_reference']['path']).is_absolute(), 'Malformed historical observation provenance')
    for ref in terminal['references'].values():verify_reference(ref)
    base=root/'planning'/KEY
    need(file_reference(base/'inputs/identity.json')==s['original_input_identity_reference'],'Original input identity changed')
    for name,key in (('', 'original_attempt_manifest_reference'),('continuation-1','prior_failed_attempt_manifest_reference'),
                     ('memory-repair-1','failed_attempt_manifest_reference')):
        folder=base/name;inventory=read(s[key])
        need(inventory['root']==str(folder) and inventory['files'],'Original attempt inventory root differs')
        if name:
            need((folder/'inputs').is_symlink() and (folder/'inputs').resolve()==base/'inputs',
                 'Original input symlink changed')
        actual={str(p.resolve()) for p in folder.rglob('*') if p.is_file() and
                (name or not any(p.is_relative_to(base/n) for n in ('continuation-1','memory-repair-1',REPRESENTATION)))}
        need(actual=={r['path'] for r in inventory['files']} and len(actual)==len(inventory['files']),
             'Original/failed attempt inventory incomplete')
        for ref in inventory['files']:verify_reference(ref)
        need(not any((folder/n).exists() for n in ('accounting.json','failure.json','solve/result.json')),
             'Conflicting completed attempt evidence')
    return config,s


def validate_representation(root,state):
    """Validate the immutable representation transition after live state advances."""
    root=_root(root);revision=root/'revisions'/REPRESENTATION;ref=state['representation_revision'];t=read(ref)
    need(Path(ref['path'])==revision/'transition.json' and t['kind']=='CASE0_REPRESENTATION_TRANSITION_V1' and
         t['revision']==REPRESENTATION,'Foreign representation transition')
    request=read(t['request_reference']);refs=_representation_review(request)
    snapshot=read(t['settled_state_snapshot_reference'])
    need(t['settled_state_snapshot_reference']['sha256']==refs['state']['sha256'] and
         t['settled_state_snapshot_reference']['size_bytes']==refs['state']['size_bytes'],'Settled representation snapshot differs')
    historical,prior=validate_transition(root,snapshot)
    original=read(refs['config']);revised=read(t['active_config_reference']);s=read(refs['settlement'])
    need(historical==original and snapshot['active_config_reference']==refs['config'] and
         t['active_config_reference']==state['active_config_reference'] and
         Path(t['active_config_reference']['path'])==revision/'config.json' and
         revised.get('solver_compact_recourses') is True and
         {k:v for k,v in revised.items() if k not in ('source_sha256','solver_compact_recourses')}==
         {k:v for k,v in original.items() if k!='source_sha256'},'Representation scientific/config identity changed')
    _representation_sources(original['source_sha256'],revised['source_sha256']);validate_config(revised,root)
    need(revised['source_sha256']==request['new_source_sha256'] and
         t['settlement_reference']==refs['settlement']==snapshot['representation_failure_settlement_reference'] and
         t['prior_transition_reference']==snapshot['memory_repair']==state['memory_repair'] and
         t['component_reference']==refs['component'],'Representation ancestry changed')
    if request['kind']=='CASE0_REPRESENTATION_CONTINUATION_REQUEST_V1':
        _,s=_failed_representation_closing(root,refs,snapshot,request)
        need(t.get('failed_admission_settlement_reference')==refs['failed_admission_settlement'],
             'Continuation settlement transition binding differs')
        reservation=read(t['continuation_reservation_reference'])
        need(Path(t['continuation_reservation_reference']['path'])==revision/'continuation-1/reservation.json' and
             reservation['request_reference']==t['request_reference'] and
             reservation['original_reservation_reference']==s['reservation_reference'], 'Continuation exclusive claim differs')
    admission=t['admission_seconds'];_finite(admission)
    need(t['prior_master_seconds']==s['closing_master_seconds'] and
         t['prior_configuration_seconds']==s['closing_configuration_seconds']+admission and
         t['opening_compute_seconds']==s['closing_compute_seconds'] and
         t['cumulative_compute_seconds']==s['closing_compute_seconds']+admission,
         'Representation expense binding changed')
    _finite(state['compute_seconds'],state['configuration_spent_seconds'].get(KEY,0),state['configuration_master_spent_seconds'].get(KEY,0))
    need(state['compute_seconds']>=t['cumulative_compute_seconds'] and
         state['configuration_spent_seconds'].get(KEY,0)>=t['prior_configuration_seconds'] and
         state['configuration_master_spent_seconds'].get(KEY,0)>=t['prior_master_seconds'] and
         state.get('pending_diagnostic_accounting_reference') is None,'Representation live cumulative budget reset')
    need(t['new_attempt_directory']==str(root/'planning'/KEY/REPRESENTATION) and
         t['original_input_directory']==str(root/'planning'/KEY/'inputs'),'Representation attempt/input path changed')
    return revised,t


def prepare_representation(request_path,expected_sha256):
    began=time.monotonic()
    from . import case0_campaign as c
    root=_root(c.campaign_root());request_ref=file_reference(request_path)
    need(request_ref['sha256']==expected_sha256,'Representation request changed')
    request=read(request_ref)
    need(request.get('kind')=='CASE0_REPRESENTATION_REQUEST_V1' and request.get('revision')==REPRESENTATION,
         'Wrong representation request kind/revision')
    revision=root/'revisions'/REPRESENTATION;revision.mkdir(parents=True,exist_ok=False)
    atomic_json(revision/'reservation.json',dict(request_reference=request_ref,admission_started_monotonic=began,
        incomplete_admission_requires_manual_accounting=True,automatic_retry_allowed=False))
    refs=_representation_review(request);state=read(refs['state'])
    need(refs['state']==file_reference(root/'state.json') and state['status']=='REPRESENTATION_FAILURE_REVIEW_REQUIRED' and
         state['size_index']==0 and state['completed']=={} and state['evaluation_opened'] is False and
         state['active_configuration'] is None and state.get('active_job_id') is None and
         not any(state.get(k) for k in ('planning_parent_pending_settlement','pending_evaluation_attempt',
             'audit_reservation_reference','representation_revision','pending_diagnostic_accounting_reference')),
         'Single settled representation admission not available')
    config,s=_representation_ancestry(root,refs,state)
    _representation_sources(config['source_sha256'],request['new_source_sha256'])
    need(request['new_source_sha256']==c.sources(),'Current representation source differs')
    _check_representation_source(root,refs,state,config,request)
    need(not (root/'planning'/KEY/REPRESENTATION).exists(),'Representation attempt already exists')
    return _commit_representation(root,revision,request_ref,request,refs,state,config,s,began)




def _check_representation_source(root,refs,state,config,request):
    from . import case0_campaign as c
    source_root=Path(refs['source_manifest']['path']).parent.resolve();manifest=read(refs['source_manifest'])
    need(Path(refs['source_manifest']['path'])==source_root/'RELEASE_MANIFEST.json' and
         state['source_release_reference']==refs['source_manifest'] and state['release_root']==str(source_root),
         'Original R3 release identity differs')
    inventory={r['path']:r['sha256'] for r in manifest['files']}
    need(len(inventory)==len(manifest['files']),'Duplicate original source inventory')
    for name,digest in config['source_sha256'].items():
        p=source_root/name
        need(not Path(name).is_absolute() and p.resolve().is_relative_to(source_root) and
             inventory.get(name)==digest==c.digest(p),'Original R3 source bytes changed')
    need(c.digest(c.get_paths().data_root/'MANIFEST_SHA256.csv')==config['data_manifest_sha256'],'Frozen data manifest changed')


def _commit_representation(root,revision,request_ref,request,refs,state,config,s,began):
    from . import case0_campaign as c
    linger_enabled();(revision/'state_before.json').write_bytes((root/'state.json').read_bytes())
    revised=dict(config,source_sha256=request['new_source_sha256'],solver_compact_recourses=True)
    atomic_json(revision/'config.json',revised);admission=time.monotonic()-began;_finite(admission)
    cfg=s['closing_configuration_seconds']+admission;master=s['closing_master_seconds'];compute=s['closing_compute_seconds']+admission
    need(cfg<1800 and master<1200 and compute<config['total_compute_seconds'],'Representation admission exhausts original budget')
    t=dict(kind='CASE0_REPRESENTATION_TRANSITION_V1',revision=REPRESENTATION,request_reference=request_ref,
        prior_config_reference=refs['config'],active_config_reference=file_reference(revision/'config.json'),
        settled_state_snapshot_reference=file_reference(revision/'state_before.json'),settlement_reference=refs['settlement'],
        component_reference=refs['component'],prior_transition_reference=state['memory_repair'],admission_seconds=admission,
        prior_configuration_seconds=cfg,prior_master_seconds=master,opening_compute_seconds=s['closing_compute_seconds'],
        cumulative_compute_seconds=compute,new_attempt_directory=str(root/'planning'/KEY/REPRESENTATION),
        original_input_directory=str(root/'planning'/KEY/'inputs'),scientific_protocol_changed=False,formal_OOS=False,
        Case1_mutated=False,automatic_second_retry=False,
        admission_clock_scope='Entry through source/input checks and config persistence; final receipt/state persistence is closing operational overhead.')
    if request['kind']=='CASE0_REPRESENTATION_CONTINUATION_REQUEST_V1':
        t.update(failed_admission_settlement_reference=refs['failed_admission_settlement'],
                 continuation_reservation_reference=file_reference(revision/'continuation-1/reservation.json'))
    atomic_json(revision/'transition.json',t);transition_ref=file_reference(revision/'transition.json')
    state.update(active_config_reference=t['active_config_reference'],representation_revision=transition_ref,
        status='REPRESENTATION_PREPARED',configuration_attempt_roots={KEY:t['new_attempt_directory']},
        configuration_spent_seconds={KEY:cfg},configuration_master_spent_seconds={KEY:master},compute_seconds=compute,
        next_action='Run one reviewed lossless compact-metadata representation at unchanged Zeta48 science/resources and remaining cumulative budgets; no automatic repeat or OOS.')
    c.save_state(root,state);c.decision(root,'CASE0_REPRESENTATION_PREPARED',transition_reference=transition_ref)
    return state


def _failed_representation_closing(root,refs,state,request):
    """Check the one settled pre-solver failure without reopening its mutable state."""
    fail=read(refs['failed_admission_settlement']);revision=root/'revisions'/REPRESENTATION
    need(fail['kind']=='CASE0_FAILED_REPRESENTATION_ADMISSION_SETTLEMENT_V1' and
         fail['solver_started'] is False and fail['scientific_pass'] is False and
         state['failed_representation_admission_settlement_reference']==refs['failed_admission_settlement'] and
         state['failed_representation_reservation_reference']==fail['reservation_reference'] and
         Path(fail['reservation_reference']['path'])==revision/'reservation.json',
         'Failed representation admission authority differs')
    reservation=read(fail['reservation_reference']);original=read(reservation['request_reference'])
    need(original['kind']=='CASE0_REPRESENTATION_REQUEST_V1' and
         reservation['automatic_retry_allowed'] is False and reservation['incomplete_admission_requires_manual_accounting'] is True,
         'Only the original failed first admission can continue')
    oldrefs=_representation_review(original)
    before=read(fail['state_before_reference'])
    need(all(fail['state_before_reference'][k]==oldrefs['state'][k] for k in ('sha256','size_bytes')) and
         all(refs[k]==oldrefs[k] for k in ('config','source_manifest','parent_manifest','parent_result','settlement','component')),
         'Failed admission original request/evidence differs')
    need(set(request['new_source_sha256'])==set(original['new_source_sha256']) and
         all(request['new_source_sha256'][k]==v for k,v in original['new_source_sha256'].items() if k!=MODULE),
         'Continuation changed more than the reviewed admission adapter')
    need(before['active_config_reference']==state['active_config_reference']==refs['config'] and
         before['memory_repair']==state['memory_repair'] and
         fail['source_release_reference']==before['source_release_reference']==state['source_release_reference']==refs['source_manifest'] and
         before['release_root']==state['release_root'] and
         before['planning_parent_settlement_reference']==state['planning_parent_settlement_reference']==refs['settlement'],
         'Failed admission original state/source ancestry differs')
    wall=read(fail['failed_prepare_reference']);preflight=read(fail['preflight_reference'])
    seconds=wall['actual_seconds'];development=preflight['development_verification_seconds']
    _finite(seconds,development);need(seconds>0 and wall['requires_accounting_before_any_resume'] is True and
        preflight['passed'] is True,'Failed admission fee evidence differs')
    for actual,expected,label in (
        (fail['opening_compute_seconds'],before['compute_seconds'],'Failed admission opening reset'),
        (fail['added_global_seconds'],seconds+development,'Failed admission global fee differs'),
        (fail['closing_compute_seconds'],before['compute_seconds']+seconds+development,'Failed admission closing reset'),
        (fail['added_configuration_seconds'],seconds,'Failed admission configuration fee differs'),
        (fail['closing_configuration_seconds'],before['configuration_spent_seconds'][KEY]+seconds,'Failed admission configuration reset'),
        (fail['closing_master_seconds'],before['configuration_master_spent_seconds'][KEY],'Failed admission master changed')):_same(actual,expected,label)
    need(state['compute_seconds']==fail['closing_compute_seconds'] and
         state['configuration_spent_seconds']=={KEY:fail['closing_configuration_seconds']} and
         state['configuration_master_spent_seconds']=={KEY:fail['closing_master_seconds']},
         'Failed admission closing authority budget differs')
    return before,fail


def _check_representation_continuation(request_path,expected_sha256):
    from . import case0_campaign as c
    root=_root(c.campaign_root());request_ref=file_reference(request_path)
    need(request_ref['sha256']==expected_sha256,'Representation continuation request changed')
    request=read(request_ref);need(request.get('kind')=='CASE0_REPRESENTATION_CONTINUATION_REQUEST_V1',
        'Wrong representation continuation kind')
    refs=_representation_review(request);state=read(refs['state']);revision=root/'revisions'/REPRESENTATION
    need(refs['state']==file_reference(root/'state.json') and state['status']=='REPRESENTATION_FAILURE_REVIEW_REQUIRED' and
         state['size_index']==0 and state['completed']=={} and state['evaluation_opened'] is False and
         state['active_configuration'] is None and state.get('active_job_id') is None and
         not any(state.get(k) for k in ('planning_parent_pending_settlement','pending_evaluation_attempt',
             'audit_reservation_reference','representation_revision','pending_diagnostic_accounting_reference')),
         'Settled failed representation continuation not available')
    need(revision.is_dir() and {p.name for p in revision.iterdir()}=={'reservation.json'},
         'Original admission reservation is missing, already continued, or has committed output')
    before,closing=_failed_representation_closing(root,refs,state,request)
    config,_=_representation_ancestry(root,refs,before)
    _representation_sources(config['source_sha256'],request['new_source_sha256'])
    need(request['new_source_sha256']==c.sources(),'Current continuation source differs')
    _check_representation_source(root,refs,state,config,request)
    need(not (root/'planning'/KEY/REPRESENTATION).exists(),'Representation attempt already exists')
    need(closing['closing_configuration_seconds']<1800 and closing['closing_master_seconds']<1200 and
         closing['closing_compute_seconds']<config['total_compute_seconds'],'Continuation original budget exhausted')
    linger_enabled()
    return root,revision,request_ref,request,refs,state,config,closing


def check_representation_continuation(request_path,expected_sha256):
    """All real-path admission checks, no reservation/config/state/solver writes."""
    root,revision,request_ref,request,refs,state,config,closing=_check_representation_continuation(request_path,expected_sha256)
    return dict(kind='CASE0_REPRESENTATION_CONTINUATION_PREFLIGHT_V1',passed=True,request_reference=request_ref,
        configuration_reference=refs['config'],failed_admission_settlement_reference=refs['failed_admission_settlement'],
        new_source_sha256=request['new_source_sha256'],opening_compute_seconds=closing['closing_compute_seconds'],
        configuration_spent_seconds=closing['closing_configuration_seconds'],master_spent_seconds=closing['closing_master_seconds'],
        original_input_directory=str(root/'planning'/KEY/'inputs'),new_attempt_directory=str(root/'planning'/KEY/REPRESENTATION),
        no_writes=True,no_solver=True)


def prepare_representation_continuation(request_path,expected_sha256):
    began=time.monotonic()
    root,revision,request_ref,request,refs,state,config,closing=_check_representation_continuation(request_path,expected_sha256)
    claim=revision/'continuation-1';claim.mkdir(exist_ok=False)
    atomic_json(claim/'reservation.json',dict(kind='CASE0_REPRESENTATION_CONTINUATION_RESERVATION_V1',
        request_reference=request_ref,original_reservation_reference=closing['reservation_reference'],
        failed_admission_settlement_reference=refs['failed_admission_settlement'],admission_started_monotonic=began,
        incomplete_admission_requires_manual_accounting=True,automatic_retry_allowed=False))
    need(file_reference(root/'state.json')==refs['state'],'Continuation authority changed during admission')
    return _commit_representation(root,revision,request_ref,request,refs,state,config,closing,began)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',required=True);p.add_argument('--expected-sha256',required=True)
    a=p.parse_args();print(json.dumps(prepare_memory_repair(a.request,a.expected_sha256)))


if __name__=='__main__':main()
