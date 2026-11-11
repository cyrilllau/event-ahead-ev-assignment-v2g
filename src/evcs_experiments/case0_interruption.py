"""One reviewed expanded-Case0 operational continuation; no model changes."""
from pathlib import Path
import argparse
import json
import math
import os
import subprocess
import sys
import time

from .campaign_batch import file_reference, verify_reference
from .case0_profiles import EXPANDED, CAMPAIGNS, validate_config
from .runtime.jobctl import atomic_json, observation, TERMINAL, utcnow

KEY = 'size0/EA_DRO'
CHANGED_SOURCES = {'src/evcs_experiments/case0_campaign.py',
                   'src/evcs_experiments/case0_interruption.py',
                   'src/evcs_experiments/case0_evaluation_runner.py',
                   'src/evcs_experiments/case0_planning_audit.py',
                   'src/evcs_experiments/case0_isolation.py'}


def need(ok, message):
    if not ok: raise ValueError(message)


def read(ref):
    verify_reference(ref)
    return json.loads(Path(ref['path']).read_text())


def linger_enabled():
    value = subprocess.run(['loginctl', 'show-user', str(os.getuid()), '-p', 'Linger', '--value'],
                           capture_output=True, text=True, check=True, timeout=5).stdout.strip()
    need(value == 'yes', 'Case0 background execution requires Linger=yes')
    return True


def _root(root):
    root = Path(root).resolve()
    need(root.name == CAMPAIGNS[EXPANDED], 'Only expanded Case0 can continue an interruption')
    return root


def _sources(old, new):
    need(set(new) == set(old) | {'src/evcs_experiments/case0_interruption.py'},
         'Recovery source inventory changed outside the operational files')
    need(all(new[k] == v for k, v in old.items() if k not in CHANGED_SOURCES),
         'Recovery changed a mathematical, input, solver or other source')


def _dead_parent(root, parent_ref, manifest_ref):
    parent = read(parent_ref); manifest = read(manifest_ref)
    job = root / 'jobs' / parent['job_id']
    need(Path(parent_ref['path']) == job / 'result.json' and Path(manifest_ref['path']) == job / 'manifest.json'
         and manifest['job_id'] == parent['job_id'] and parent.get('complete') is True
         and parent.get('state') in TERMINAL, 'Exact terminal original parent required')
    status = json.loads((job / 'status.json').read_text()); obs = observation(status)
    need(status['state'] == parent['state'] and not obs['blocks_new_launch']
         and not obs['supervisor_alive'] and not obs['command_alive'] and not obs['live_group_members'],
         'Original parent still has live or unreviewed work')
    return parent


def validate_transition(root, state):
    """Validate immutable transition even after the live state has advanced."""
    root = _root(root)
    ref = state['operational_interruption']; transition = read(ref)
    revision = root / 'revisions/interruption-1'
    need(Path(ref['path']) == revision / 'transition.json' and
         transition['kind'] == 'CASE0_INTERRUPTION_TRANSITION_V1', 'Foreign interruption transition')
    request = read(transition['request_reference']); refs = request['references']; review = read(refs['review'])
    need(review['kind'] == 'CASE0_INTERRUPTION_REVIEW_V1' and review['passed'] is True
         and review['references'] == {k:v for k,v in refs.items() if k != 'review'}
         and review['new_source_sha256'] == request['new_source_sha256'], 'Independent recovery review changed')
    snapshot = transition['settled_state_snapshot_reference']; read(snapshot)
    need(snapshot['sha256'] == refs['state']['sha256'] and snapshot['size_bytes'] == refs['state']['size_bytes'],
         'Original settled state snapshot differs')
    original = read(refs['config']); revised = read(transition['active_config_reference'])
    need(refs['config'] == file_reference(root / 'config.lock.json') and
         transition['active_config_reference'] == state['active_config_reference'] and
         Path(state['active_config_reference']['path']) == revision / 'config.json', 'Recovery configuration identity differs')
    need({k:v for k,v in revised.items() if k != 'source_sha256'} ==
         {k:v for k,v in original.items() if k != 'source_sha256'}, 'Recovery scientific configuration changed')
    _sources(original['source_sha256'], revised['source_sha256'])
    need(revised['source_sha256'] == request['new_source_sha256'], 'Reviewed recovery source differs')
    validate_config(revised, root)
    settlement = read(refs['settlement'])
    need(transition['settlement_reference'] == refs['settlement'] and
         transition['prior_master_seconds'] == settlement['prior_master_seconds'] and
         transition['prior_configuration_seconds'] == settlement['prior_configuration_seconds'] + transition['admission_seconds']
         and transition['cumulative_compute_seconds'] == settlement['closing_compute_seconds'] + transition['admission_seconds'],
         'Recovery carryover arithmetic changed')
    need(transition['new_attempt_directory'] == str(root / 'planning' / KEY / 'continuation-1') and
         transition['original_input_directory'] == str(root / 'planning' / KEY / 'inputs'), 'Recovery attempt/input path changed')
    return revised, transition


def prepare_interruption(request_path, expected_sha256):
    began = time.monotonic()
    from . import case0_campaign as c
    root = _root(c.campaign_root()); request_ref = file_reference(request_path)
    need(request_ref['sha256'] == expected_sha256, 'Recovery request changed')
    request = read(request_ref); refs = request['references']
    need(request['kind'] == 'CASE0_INTERRUPTION_REQUEST_V1', 'Wrong recovery request kind')
    revision = root/'revisions/interruption-1'; revision.mkdir(parents=True, exist_ok=False)
    atomic_json(revision/'reservation.json', dict(request_reference=request_ref, admission_started_monotonic=began,
        incomplete_admission_requires_manual_accounting=True, automatic_retry_allowed=False))
    need(set(refs) == {'config','state','original_source_manifest','parent_manifest','parent_result','settlement','review'},
         'Recovery reference inventory differs')
    values = {k:read(v) for k,v in refs.items()}
    config, state, settlement, review = (values[k] for k in ('config','state','settlement','review'))
    need(refs['config'] == file_reference(root / 'config.lock.json') and refs['state'] == file_reference(root / 'state.json'),
         'Original recovery configuration/state changed')
    need(review['kind'] == 'CASE0_INTERRUPTION_REVIEW_V1' and review['passed'] is True and
         review['references'] == {k:v for k,v in refs.items() if k != 'review'} and
         review['new_source_sha256'] == request['new_source_sha256'] == c.sources(), 'Exact independent recovery review required')
    need(validate_config(config, root) == EXPANDED and not state['evaluation_opened'] and state['size_index'] == 0
         and state['status'] == 'INTERRUPTION_REVIEW_REQUIRED' and state['active_configuration'] is None
         and state.get('active_job_id') is None and state['completed'] == {}
         and not state.get('planning_parent_pending_settlement') and not state.get('pending_evaluation_attempt')
         and not state.get('audit_reservation_reference') and 'active_config_reference' not in state
         and 'operational_interruption' not in state and 'numerical_refinement' not in state, 'Single interruption is not available')
    need(state['planning_parent_settlement_reference'] == refs['settlement'] and
         settlement['kind'] == 'CASE0_INTERRUPTED_PARENT_SETTLEMENT_V1' and settlement['scientific_pass'] is False
         and settlement['OOS_opened'] is False and settlement['configuration_key'] == KEY,
         'Original interrupted parent has not been exactly settled')
    parent = _dead_parent(root, refs['parent_result'], refs['parent_manifest'])
    for name in ('launch_admission_reference','state_before_reference','original_attempt_manifest_reference','observation_reference'):
        read(settlement[name])
    before_state=read(settlement['state_before_reference'])
    need(before_state['active_job_id']==parent['job_id'] and before_state['active_configuration']==KEY
         and before_state['status']=='PLANNING' and before_state['completed']=={}
         and before_state['planning_launch_admission_reference']==settlement['launch_admission_reference'],
         'Interrupted state/launch/parent identity differs')
    parent_seconds = parent['elapsed_seconds']
    need(type(parent_seconds) in (int,float) and math.isfinite(parent_seconds) and 0 < parent_seconds < 1200
         and settlement['parent_reference'] == refs['parent_result']
         and settlement['parent_elapsed_seconds'] == settlement['prior_configuration_seconds'] == settlement['prior_master_seconds'] == parent_seconds,
         'Conservative nested budget must use the exact original parent wall')
    opening, before, already, added, closing = (settlement[k] for k in ('opening_compute_seconds',
        'previously_accounted_compute_seconds','already_charged_parent_seconds','added_compute_seconds','closing_compute_seconds'))
    need(all(type(v) in (int,float) and math.isfinite(v) for v in (opening,before,already,added,closing))
         and 0 <= already <= parent_seconds and abs(already-(before-opening)) <= 1e-8
         and abs(added-(parent_seconds-already)) <= 1e-8 and abs(closing-(before+added)) <= 1e-8
         and abs(closing-(opening+parent_seconds)) <= 1e-8 and state['compute_seconds'] == closing,
         'Original parent global accounting differs')
    need(state['configuration_spent_seconds'] == {KEY:parent_seconds} and
         state['configuration_master_spent_seconds'] == {KEY:parent_seconds}, 'Nested cumulative budgets changed')
    old = root / 'planning' / KEY
    need(settlement['original_attempt_directory'] == str(old) and not (old/'accounting.json').exists()
         and not (old/'failure.json').exists() and not (old/'solve/result.json').exists(), 'Original interruption has conflicting final evidence')
    inventory = read(settlement['original_attempt_manifest_reference'])
    need(inventory['root'] == str(old) and inventory['files'], 'Original attempt inventory differs')
    files = {str(p.resolve()) for p in old.rglob('*') if p.is_file()}
    need(files == {r['path'] for r in inventory['files']} and len(files) == len(inventory['files']), 'Original attempt inventory is incomplete')
    for ref in inventory['files']: verify_reference(ref)
    need((old/'inputs/identity.json').is_file(), 'Original input identity required')
    source_root = Path(request['original_release_root']).resolve()
    need(Path(refs['original_source_manifest']['path']) == source_root/'RELEASE_MANIFEST.json', 'Original release root differs')
    source_inventory = {r['path']:r['sha256'] for r in values['original_source_manifest']['files']}
    _sources(config['source_sha256'], request['new_source_sha256'])
    for name, sha in config['source_sha256'].items():
        path = (source_root/name).resolve()
        need(path.is_relative_to(source_root) and source_inventory.get(name) == sha and c.digest(path) == sha,
             'Original release source changed')
    need(c.digest(c.get_paths().data_root/'MANIFEST_SHA256.csv') == config['data_manifest_sha256'], 'Frozen input manifest changed')
    linger_enabled()
    (revision/'state_before.json').write_bytes((root/'state.json').read_bytes())
    revised = dict(config, source_sha256=request['new_source_sha256']); atomic_json(revision/'config.json', revised)
    admission = time.monotonic()-began
    need(parent_seconds+admission < 1800 and closing+admission < config['total_compute_seconds'], 'Recovery admission exhausted original budget')
    transition = dict(kind='CASE0_INTERRUPTION_TRANSITION_V1', request_reference=request_ref,
        original_config_reference=refs['config'], active_config_reference=file_reference(revision/'config.json'),
        settled_state_snapshot_reference=file_reference(revision/'state_before.json'), settlement_reference=refs['settlement'],
        admission_seconds=admission, prior_configuration_seconds=parent_seconds+admission,
        admission_clock_scope='Function entry through source/input verification and revised config persistence; final receipt/state persistence is operational closing overhead.',
        prior_master_seconds=parent_seconds, opening_compute_seconds=closing, cumulative_compute_seconds=closing+admission,
        new_attempt_directory=str(old/'continuation-1'), original_input_directory=str(old/'inputs'),
        scientific_protocol_changed=False, formal_OOS=False, Case1_mutated=False, automatic_second_retry=False)
    atomic_json(revision/'transition.json', transition)
    state.update(active_config_reference=transition['active_config_reference'], operational_interruption=file_reference(revision/'transition.json'),
        configuration_attempt_roots={KEY:str(old/'continuation-1')}, configuration_spent_seconds={KEY:parent_seconds+admission},
        compute_seconds=closing+admission, status='INTERRUPTION_PREPARED',
        next_action='Run the single reviewed operational continuation with original H, source-bound inputs and remaining budgets; no second blind retry or OOS.')
    c.save_state(root,state); c.decision(root,'OPERATIONAL_INTERRUPTION_PREPARED',transition_reference=state['operational_interruption'])
    return state


def resource_sample(unit, previous=None):
    """Read only the explicit owned unit; retain last visible counters after collection."""
    from .case0_isolation import unit_name
    unit = unit_name(unit); need(unit.startswith('case0-expanded-'), 'Recovery telemetry requires an expanded Case0 unit')
    result = dict(previous or {}); result['observed_at_utc'] = utcnow(); result['unit'] = unit
    probe = subprocess.run(['systemctl','--user','show',unit,'--property=ControlGroup,Result,ExecMainCode,ExecMainStatus,ActiveState,SubState',
                            '--no-pager'], capture_output=True,text=True,timeout=3,check=False)
    props = dict(line.split('=',1) for line in probe.stdout.splitlines() if '=' in line)
    result['unit_properties'] = props; result['systemctl_returncode'] = probe.returncode
    group = props.get('ControlGroup','')
    if group:
        path = Path(group)
        need(path.is_absolute() and '..' not in path.parts and path.name == unit, 'Foreign telemetry cgroup')
        root = Path('/sys/fs/cgroup')/group.lstrip('/')
        result['control_group'] = group
        for name in ('memory.current','memory.peak','memory.events','cpu.stat'):
            try: raw = (root/name).read_text().strip()
            except FileNotFoundError: continue
            if name in ('memory.current','memory.peak'):
                value = int(raw); result[name] = value
                result['observed_memory_max_bytes'] = max(value,result.get('observed_memory_max_bytes',0))
            else: result[name] = {k:int(v) for k,v in (line.split() for line in raw.splitlines())}
    return result


def guarded_plan(root, operation, unit, *, seconds=7180, cpu=0):
    from . import case0_campaign as c
    from .case0_isolation import unit_name
    root = _root(root); operation = Path(operation).resolve(); unit = unit_name(unit)
    need(root == c.campaign_root().resolve() and operation.is_relative_to(root/'operations') and
         unit.startswith('case0-expanded-') and type(seconds) is int and 0 < seconds <= 7180
         and type(cpu) is int and cpu >= 0, 'Invalid bounded recovery operation')
    state = c.read(root/'state.json')
    if state.get('representation_revision'):
        from .case0_memory_repair import validate_representation
        config, transition = validate_representation(root,state)
        expected_status, revision, transition_key = 'REPRESENTATION_PREPARED', 'compact-metadata-1', 'representation_revision'
        claim_kind = 'CASE0_REPRESENTATION_GUARD_CLAIM_V1'
    elif state.get('memory_repair'):
        from .case0_memory_repair import validate_transition as validate_memory_transition
        config, transition = validate_memory_transition(root,state)
        expected_status, revision, transition_key = 'MEMORY_REPAIR_PREPARED', 'memory-repair-1', 'memory_repair'
        claim_kind = 'CASE0_MEMORY_REPAIR_GUARD_CLAIM_V1'
    else:
        config, transition = validate_transition(root,state)
        expected_status, revision, transition_key = 'INTERRUPTION_PREPARED', 'interruption-1', 'operational_interruption'
        claim_kind = 'CASE0_INTERRUPTION_GUARD_CLAIM_V1'
    c.verify_sources(config)
    need(state['status'] == expected_status and state['active_configuration'] is None
         and not state['evaluation_opened'], 'Recovery is not admitted for a new guarded batch')
    linger_enabled()
    jobid = os.environ.get('EVCS_JOB_ID'); jobdir = Path(os.environ.get('EVCS_JOB_DIRECTORY','')).resolve()
    need(jobid and jobdir == root/'jobs'/jobid, 'Guard must run inside the exact Case0 jobctl parent')
    manifest_ref = file_reference(jobdir/'manifest.json'); manifest = read(manifest_ref)
    need(manifest['job_id'] == jobid, 'Guard parent identity changed')
    argv=manifest['argv']
    need(len(argv)==14 and Path(argv[0]).resolve()==Path(sys.executable).resolve() and
         argv[1:4]==['-m','evcs_experiments.case0_interruption','guarded-plan'], 'Guard parent invocation differs')
    pairs=dict(zip(argv[4::2],argv[5::2]))
    need(set(pairs)=={'--root','--operation','--unit','--seconds','--cpu'} and
         pairs['--root']==str(root) and pairs['--operation']==str(operation) and unit_name(pairs['--unit'])==unit
         and pairs['--seconds']==str(seconds) and pairs['--cpu']==str(cpu), 'Guard argument inventory differs')
    need(manifest['outroot']==str(root/'jobs') and manifest['threads']==1
         and seconds+10 <= manifest['timeout_seconds'] <= 7200 and 0<manifest['grace_seconds']<=5,
         'Guard parent time/resource envelope differs')
    operation.mkdir(parents=True,exist_ok=True)
    claim_path=root/'revisions'/revision/'guard_claim.json'
    with claim_path.open('x') as stream:
        json.dump(dict(kind=claim_kind,parent_manifest_reference=manifest_ref,
            transition_reference=state[transition_key],unit=unit,at_utc=utcnow(),automatic_retry_allowed=False),stream)
    atomic_json(operation/'guard_claim_reference.json',file_reference(claim_path))
    command=[sys.executable,'-m','evcs_experiments.case0_isolation','--unit',unit,'--seconds',str(seconds+10),'--cpu',str(cpu),
             '--retain-failed-unit','--',sys.executable,'-m','evcs_experiments.case0_campaign','plan','--seconds',str(seconds)]
    telemetry = {}; child = subprocess.Popen(command)
    while True:
        sample_started=time.monotonic()
        try: telemetry = resource_sample(unit,telemetry)
        except (OSError,subprocess.SubprocessError,ValueError) as error:
            telemetry['last_observation_error'] = str(error)
        atomic_json(operation/'resource_observation.json',telemetry)
        try: code=child.wait(timeout=max(.001,5-(time.monotonic()-sample_started))); break
        except subprocess.TimeoutExpired: pass
    try: telemetry = resource_sample(unit,telemetry)
    except (OSError,subprocess.SubprocessError,ValueError) as error: telemetry['last_observation_error']=str(error)
    atomic_json(operation/'resource_observation.json',telemetry)
    final=c.read(root/'state.json')
    valid=(code==0 and final.get('active_configuration') is None and final.get('status') in
           ('PLANNING_REVIEW_REQUIRED','PLANNING','BUDGET_EXHAUSTED') and not final.get('evaluation_opened'))
    if final.get('status')=='PLANNING_REVIEW_REQUIRED':
        valid=valid and set(final.get('completed',{}))=={f'size0/{m}' for m in c.METHODS}
    elif final.get('status')=='PLANNING':
        valid=valid and len(final.get('completed',{}))>len(state.get('completed',{}))
    output=dict(kind='CASE0_INTERRUPTION_GUARD_RESULT_V1',passed=bool(valid),child_returncode=code,
        final_state_reference=file_reference(root/'state.json'),active_configuration=final.get('active_configuration'),
        campaign_status=final.get('status'),unit=unit,scientific_acceptance=False)
    if transition_key=='memory_repair':
        output.update(kind='CASE0_MEMORY_REPAIR_GUARD_RESULT_V1',transition_reference=state[transition_key])
    elif transition_key=='representation_revision':
        output.update(kind='CASE0_REPRESENTATION_GUARD_RESULT_V1',transition_reference=state[transition_key])
    atomic_json(operation/('guard_result.json' if valid else 'guard_failure.json'),output)
    return 0 if valid else 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare');p.add_argument('--request',required=True);p.add_argument('--expected-sha256',required=True)
    p=sub.add_parser('guarded-plan');p.add_argument('--root',required=True);p.add_argument('--operation',required=True)
    p.add_argument('--unit',required=True);p.add_argument('--seconds',type=int,default=7180);p.add_argument('--cpu',type=int,default=0)
    args=parser.parse_args(argv)
    if args.action=='prepare': print(json.dumps(prepare_interruption(args.request,args.expected_sha256),indent=2)); return 0
    return guarded_plan(args.root,args.operation,args.unit,seconds=args.seconds,cpu=args.cpu)


if __name__=='__main__': raise SystemExit(main())
