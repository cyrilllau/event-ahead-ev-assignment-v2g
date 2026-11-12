"""One reviewed numerical refinement, preserving original Case0 attempts/budget."""
from pathlib import Path
import math
import time

from .campaign_batch import file_reference,verify_reference


def prepare_numerical_refinement(request_path,expected_sha256):
    began=time.monotonic()
    from . import case0_campaign as c
    from .runtime.jobctl import observation,TERMINAL
    root=c.campaign_root();request_ref=file_reference(request_path)
    if request_ref['sha256']!=expected_sha256:raise ValueError('Refinement request changed')
    request=c.read(request_path);refs=request['references']
    def need(ok,message):
        if not ok:raise ValueError(message)
    need(request['kind']=='CASE0_NUMERICAL_REFINEMENT_REQUEST_V1','Wrong numerical refinement request')
    for ref in refs.values():verify_reference(ref)
    need(refs['config']==file_reference(root/'config.lock.json') and refs['state']==file_reference(root/'state.json'),
         'Refinement original configuration/state changed')
    config=c.read(refs['config']['path']);state=c.read(refs['state']['path']);review=c.read(refs['review']['path'])
    need(review['kind']=='CASE0_NUMERICAL_REFINEMENT_REVIEW_V1' and review['passed'] is True
         and review['references']=={k:v for k,v in refs.items() if k!='review'}
         and review['new_source_sha256']==request['new_source_sha256']==c.sources(),
         'Exact independently reviewed source/evidence transition required')
    need(config['backend']=='gurobi' and config['configuration_seconds']==1800
         and c.digest(c.get_paths().data_root/'MANIFEST_SHA256.csv')==config['data_manifest_sha256'],
         'Original numerical protocol/data differs')
    need(state['size_index']==1 and state['status']=='FAILURE_REQUIRES_REVIEW'
         and state['active_configuration'] is None and state['evaluation_opened'] is False
         and 'numerical_refinement' not in state and 'active_config_reference' not in state,
         'Numerical refinement is not available for this state')
    need(set(state['completed']) >= {'size1/EA_DRO'} and all('size1/'+m not in state['completed'] for m in ('OR_DRO','SAA','ARO')),
         'Expected completed EA and failed OR only at fallback size')
    ea=state['completed']['size1/EA_DRO'];earef=file_reference(ea['path'])
    need(earef['sha256']==ea['sha256'] and earef==refs['completed_ea_result'],'Original EA result changed')
    earow=c.read(ea['path']);need(earow['numerical_valid'] is True and earow['gap_passed'] is True,'Original EA is not a valid completed plan')
    old=root/'planning/size1/OR_DRO'
    for name,relative in (('accounting','accounting.json'),('failure','failure.json'),('master','solve/master.json')):
        need(Path(refs[name]['path']).resolve()==(old/relative).resolve(),'Foreign failed OR evidence')
    accounting=c.read(refs['accounting']['path']);master=c.read(refs['master']['path'])
    spent=accounting['configuration_wall_seconds']
    need(type(spent) in (int,float) and math.isfinite(spent) and 0<spent<1800,'Invalid original OR cumulative time')
    need(accounting['cumulative_case0_compute_seconds']==state['compute_seconds'],'Unsettled original attempt accounting')
    need(master['solver_status']==4 and master['status']=='SOLVER_ERROR' and master['raw_solver']['native_interface']['raw_model_status']==2,
         'Repair requires original native OPTIMAL evidence')
    need(master['raw_solver']['native_interface']['original_certificate']['max_row_violation']>1e-8
         and master['raw_solver']['native_interface']['original_certificate']['max_integrality_violation']<=1e-8,
         'Not the reviewed original feasibility precision failure')
    parent=c.read(refs['last_parent_result']['path'])
    need(parent['complete'] is True and parent['state'] in TERMINAL and
         Path(refs['last_parent_result']['path']).resolve()==(root/'jobs'/parent['job_id']/'result.json').resolve(),
         'Exact terminal Case0 parent result required')
    need('active_job_id' not in state or state['active_job_id']==parent['job_id'],'Last parent identity differs from recorded Case0 job')
    def dead_jobs():
        for directory in (root/'jobs').iterdir():
            if not directory.is_dir() or not (directory/'manifest.json').exists():continue
            status=c.read(directory/'status.json');observed=observation(status)
            need(status['state'] in TERMINAL and not observed['blocks_new_launch']
                 and not observed['supervisor_alive'] and not observed['command_alive']
                 and not observed['live_group_members'],'Case0 job still active/unsettled')
    dead_jobs()
    revision=root/'revisions/numerical-refinement-1';revision.mkdir(parents=True,exist_ok=False)
    c.atomic_json(revision/'reservation.json',dict(request_reference=request_ref,admission_started_monotonic=began,
        incomplete_admission_requires_manual_accounting=True,automatic_retry_allowed=False))
    oldsource=Path(request['original_release_root']).resolve()
    need(Path(refs['original_source_manifest']['path']).resolve()==oldsource/'RELEASE_MANIFEST.json',
         'Original release/manifest location differs')
    manifest=c.read(refs['original_source_manifest']['path'])
    inventory={row['path']:row['sha256'] for row in manifest['files']}
    need(all(inventory.get(name)==sha for name,sha in config['source_sha256'].items()),
         'Original release manifest/config source map differs')
    for name,sha in config['source_sha256'].items():
        path=(oldsource/name).resolve()
        need(path.is_relative_to(oldsource) and c.digest(path)==sha,'Original source release changed')
    native_spent=master['wall_seconds']
    need(type(native_spent) in (int,float) and math.isfinite(native_spent) and 0<native_spent<1200,
         'Invalid original master cumulative time')
    dead_jobs()
    revised=dict(config,source_sha256=c.sources(),solver_feasibility_tolerance=1e-9)
    c.atomic_json(revision/'config.json',revised)
    config_ref=file_reference(revision/'config.json')
    admission_seconds=time.monotonic()-began
    effective_spent=spent+admission_seconds
    need(effective_spent<1800 and state['compute_seconds']+admission_seconds<config['total_compute_seconds'],
         'Numerical admission exhausted remaining budget')
    transition=dict(kind='CASE0_NUMERICAL_REFINEMENT_TRANSITION_V1',request_reference=request_ref,
        original_config_reference=refs['config'],active_config_reference=config_ref,
        original_failed_configuration_seconds=spent,admission_seconds=admission_seconds,
        admission_clock_scope="function entry through verification and revised config persistence",
        prior_configuration_seconds=effective_spent,remaining_configuration_seconds=1800-effective_spent,
        prior_master_seconds=native_spent,remaining_master_seconds=1200-native_spent,
        opening_compute_seconds=state['compute_seconds'],cumulative_compute_seconds=state['compute_seconds']+admission_seconds,completed_ea_reference=earef,
        old_attempt_directory=str(old),new_attempt_directory=str(old/'refinement-1'),
        scientific_threshold=1e-8,solver_feasibility_tolerance=1e-9,scientific_protocol_changed=False,
        formal_OOS=False,Case1_mutated=False)
    c.atomic_json(revision/'transition.json',transition)
    state.update(compute_seconds=state['compute_seconds']+admission_seconds,active_config_reference=config_ref,numerical_refinement=file_reference(revision/'transition.json'),
        configuration_spent_seconds={**state.get('configuration_spent_seconds',{}),'size1/OR_DRO':effective_spent},
        configuration_master_spent_seconds={'size1/OR_DRO':native_spent},
        configuration_attempt_roots={'size1/OR_DRO':str(old/'refinement-1')},status='REFINEMENT_PREPARED',
        next_action='Run the single reviewed OR precision refinement using remaining original budget, then SAA/ARO; preserve completed EA and all old attempts.')
    c.save_state(root,state)
    c.decision(root,'NUMERICAL_PRECISION_REFINEMENT_PREPARED',transition_reference=state['numerical_refinement'],
               compute_seconds_preserved=state['compute_seconds'])
    return state
