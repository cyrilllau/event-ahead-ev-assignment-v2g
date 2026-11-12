"""Checkpointed formal planning/calibration bank; no formal OOS is generated."""
import argparse
import json
import os
import shutil
from pathlib import Path
import time
import numpy as np
from .config import get_paths
from .evidence import sha256
from .evaluation_lock import file_reference
from .formal_scenarios import PlanningContext
from .joint_scenarios import array_hash
from .moment_calibration import bootstrap_moments,calibrate_moments
from .random_streams import canonical_hash
from .runtime.jobctl import atomic_json
from .validation.validate_s01_replay import network_arcs
from .validation.s02_toys import _serial


def read_ref(ref):
    path=Path(ref['path'])
    if sha256(path)!=ref['sha256']:raise ValueError('Bound source changed: '+str(path))
    return json.loads(path.read_text())


def write_new(path,value):
    with Path(path).open('x') as f:json.dump(_serial(value),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def accepted_protocol_sources(request):
    s04=read_ref(request['s04_acceptance']);s01=read_ref(request['s01_acceptance'])
    if s04.get('spec_id')!='S04' or s04.get('status')!='PASS' or s01.get('status')!='PASS':raise ValueError('Accepted S04/S01 prerequisite required')
    protocol_ref=request['formal_protocol_lock'];protocol=read_ref(protocol_ref)['payload']
    if s04['formal_protocol_lock']['sha256']!=protocol_ref['sha256']:raise ValueError('Wrong accepted formal protocol')
    if protocol['domain']!='formal':raise ValueError('Expected formal protocol')
    for ref in protocol['references']['physical_inputs']:
        if sha256(ref['path'])!=ref['sha256']:raise ValueError('Protocol frozen physical bytes changed')
    manifest=read_ref(dict(path=s01['deliverables_manifest'],sha256=s01['deliverables_manifest_sha256']))
    contract=read_ref(manifest['generator_contract.json'])
    refs={row['target_event']:row['frozen_tasks'] for row in contract['frozen_task_sources']}
    if request['target_ids']!=protocol['target_ids'] or set(refs)!=set(protocol['target_ids']):raise ValueError('Formal baseline bank must retain all eight targets')
    return protocol,refs


def accepted_task_sources(request):
    protocol,refs=accepted_protocol_sources(request)
    if request.get('lead_h')!=1 or request.get('regime')!='R1' or request.get('domain')!='formal':raise ValueError('This bank exporter only handles formal lead1/R1')
    if request.get('split_sizes')!={'planning':20,'calibration':10}:raise ValueError('Formal baseline split sizes changed')
    for ref in refs.values():read_ref(ref)
    return protocol,refs


def _generate_bank(request,*,wall_seconds,outputroot,resume=False,adapter=None):
    """Retain every completed atom and split; never repeat a certified saved atom.

    Return operational completion separately from independent bank acceptance.
    First-stage forecast tasks retain their original accepted bytes. New physical
    samples and the calibration bootstrap use the true formal random namespace.
    """
    if not 0<wall_seconds<=7190:raise ValueError('Invalid bank wall budget')
    started=time.monotonic();out=Path(outputroot).resolve();identity=canonical_hash(request)
    if not resume:out.mkdir(parents=True,exist_ok=False)
    checkpoint=out/'checkpoint.json'
    if resume:
        previous=json.loads(checkpoint.read_text())
        if previous['request_sha256']!=identity:raise ValueError('Bank resume request changed')
        completed=previous['completed_atoms'];prior_seconds=float(previous['cumulative_seconds']);overview=previous['completed_targets']
        for target_row in overview:
            for key in ('planning_index','calibration_index','moments','bootstrap'):
                saved=target_row[key]
                if sha256(saved['path'])!=saved['sha256']:raise ValueError('Completed target evidence changed: '+key)
    else:completed={};prior_seconds=0.;overview=[]
    generated_task_sources={} if adapter is None or not resume else dict(previous.get('task_sources',{}))
    def checkpoint_now(status):
        record=dict(status=status,request_sha256=identity,completed_atoms=completed,completed_targets=overview,cumulative_seconds=prior_seconds+time.monotonic()-started,formal_OOS=False)
        if adapter is not None:record['task_sources']=generated_task_sources
        atomic_json(checkpoint,record);return record
    checkpoint_now('RUNNING')
    protocol,taskrefs=(accepted_task_sources(request) if adapter is None else adapter.sources(request))
    if adapter is not None:taskrefs.update(generated_task_sources)
    paths=get_paths()
    context=(PlanningContext if adapter is None else adapter.context_type)(paths.data_root,network_arcs(paths.data_root))
    if list(context.events)!=request['target_ids']:raise ValueError('Raw case target order differs')
    frozen={ref['path'].split('/data/cases/ieee33-v54-q1/')[-1]:ref['sha256'] for ref in protocol['references']['physical_inputs']}
    for name,digest in context.input_hashes.items():
        if sha256(paths.data_root/name)!=digest or frozen.get(name)!=digest:raise ValueError('Loaded physical input differs from pre-OOS protocol: '+name)
    if not (out/'request.json').exists():write_new(out/'request.json',request)
    for target in request['target_ids']:
        taskdir=out/target;taskdir.mkdir(exist_ok=True)
        if adapter is None:
            bank=read_ref(taskrefs[target]);prepared=context.prepare_target(target,bank,lead_h=1.)
        else:
            remaining=wall_seconds-(time.monotonic()-started)
            if remaining<5:return checkpoint_now('TIME_LIMIT')
            prepared=adapter.prepare(context,target,taskdir,remaining,taskrefs)
            if prepared is None:return checkpoint_now('TIME_LIMIT')
            bank=prepared.bank
            generated_task_sources.update(taskrefs)
        if bank['task_sha256']!=prepared.tasks.task_sha256:raise ValueError('Frozen forecast task identity changed')
        if not (taskdir/'task_source.json').exists():
            (write_new if adapter is None else atomic_json)(taskdir/'task_source.json',taskrefs[target])
        split_features={};split_indices={}
        for split,count in request['split_sizes'].items():
            rows=[];features=[]
            for variant in range(count):
                for base in request['target_ids']:
                    sid=f'{target}/formal/{split}/{base}/v{variant+1:03d}/R1';key=sid
                    folder=taskdir/split/f'{variant:03d}_{base}'
                    if key not in completed and folder.exists():
                        # A crash may occur after manifest commit but before the
                        # global checkpoint. Adopt only a completely hash-verified atom.
                        if (folder/'manifest.json').is_file():
                            orphan=json.loads((folder/'manifest.json').read_text())
                            if orphan.get('scenario_id')!=sid:raise ValueError('Uncheckpointed atom ID changed')
                            for file_ref in orphan['files'].values():
                                if sha256(file_ref['path'])!=file_ref['sha256']:raise ValueError('Uncheckpointed atom bytes changed')
                            completed[key]=file_reference(folder/'manifest.json')
                        else:
                            quarantine=out/'incomplete_writes';quarantine.mkdir(exist_ok=True)
                            os.rename(folder,quarantine/(target+'_'+split+'_'+folder.name+'_'+str(time.time_ns())))
                    if key in completed:
                        ref=completed[key];manifest=read_ref(ref)
                        for name,file_ref in manifest['files'].items():
                            if sha256(file_ref['path'])!=file_ref['sha256']:raise ValueError('Previously completed atom changed: '+sid)
                        ledger=read_ref(manifest['files']['ledger'])
                        if ledger['scenario_id']!=sid:raise ValueError('Resume atom ID changed')
                        with np.load(manifest['files']['arrays']['path'],allow_pickle=False) as z:feature=z['features'].copy()
                    else:
                        if wall_seconds-(time.monotonic()-started)<5:
                            return checkpoint_now('TIME_LIMIT')
                        sample=context.generate(prepared,base,variant,split,domain='formal',regime='R1')
                        if sample.ledger['development'] is not False or sample.ledger['scenario_id']!=sid or sample.ledger['residual_donor_event']==target:raise ValueError('Formal stream/target/LOEO identity mismatch')
                        # Fail-fast shape, finiteness, conservation and semantic
                        # hashes are retained; independent full-bank audit follows.
                        for name,value in sample.arrays.items():
                            if name in ('task_arrival_h','road_travel_time_min'):
                                # +inf records a failed arrival or impassable
                                # arc; NaN and -inf are never physical outcomes.
                                if np.isnan(value).any() or np.isneginf(value).any():raise ValueError('Invalid nonfinite physical field: '+name)
                            elif not np.isfinite(value).all():raise ValueError('Nonfinite physical field: '+name)
                        for name,value in sample.arrays.items():
                            if sample.ledger['array_hashes'].get(name)!=array_hash(value):raise ValueError('Generated field hash differs: '+name)
                        payload={k:v for k,v in dict(sample.ledger).items() if k!='scenario_sha256'}
                        if canonical_hash(payload)!=sample.scenario_sha256:raise ValueError('Scenario semantic provenance changed')
                        folder.mkdir(parents=True,exist_ok=False)
                        np.savez_compressed(folder/'arrays.npz',**sample.arrays)
                        write_new(folder/'ledger.json',dict(sample.ledger));write_new(folder/'diagnostics.json',dict(sample.diagnostics))
                        manifest=dict(scenario_id=sid,scenario_sha256=sample.scenario_sha256,files={name:file_reference(folder/file) for name,file in [('arrays','arrays.npz'),('ledger','ledger.json'),('diagnostics','diagnostics.json')]})
                        write_new(folder/'manifest.json',manifest);completed[key]=file_reference(folder/'manifest.json');ledger=dict(sample.ledger);feature=np.asarray(sample.arrays['features']);checkpoint_now('RUNNING')
                    if ledger['task_sha256']!=prepared.tasks.task_sha256 or ledger['assignment_state_sha256']!=prepared.target.state_sha256:raise ValueError('Scenario changed target state or frozen H')
                    rows.append(dict(scenario_id=sid,scenario_sha256=ledger['scenario_sha256'],manifest=completed[key]));features.append(feature)
            split_features[split]=np.stack(features);split_indices[split]=rows
            indexpath=taskdir/(split+'.index.json')
            value=dict(target_event=target,split=split,development=False,regime='R1',lead_h=(1. if adapter is None else prepared.target.lead_h),task_source=taskrefs[target],task_sha256=prepared.tasks.task_sha256,task_ids=list(prepared.tasks.task_ids),assignment_state_sha256=prepared.target.state_sha256,times_h=prepared.target.replay_times_h.tolist(),scenarios=rows)
            if indexpath.exists():
                if json.loads(indexpath.read_text())!=value:raise ValueError('Completed split index changed')
            else:write_new(indexpath,value)
        momentpath=taskdir/'moments.json';bootstrappath=taskdir/'bootstrap.npz'
        target_committed=any(row['target_event']==target for row in overview)
        if not target_committed and (momentpath.exists() or bootstrappath.exists()):
            # These two files are committed together by completed_targets in
            # the checkpoint. A crash before that commit can leave one missing,
            # a truncated NPZ, or a complete but uncommitted pair. Preserve all
            # existing bytes and rebuild only this inexpensive deterministic
            # calibration from the already verified saved scenario features.
            recovery=out/'incomplete_writes'/('moments_'+target+'_'+str(time.time_ns()))
            recovery.mkdir(parents=True,exist_ok=False);preserved=[]
            for partial in (momentpath,bootstrappath):
                if partial.exists():
                    destination=recovery/partial.name;os.rename(partial,destination)
                    preserved.append(file_reference(destination))
            write_new(recovery/'record.json',dict(target_event=target,
                reason='uncheckpointed moment/bootstrap pair preserved before deterministic reconstruction',
                preserved_files=preserved,scenario_atoms_regenerated=False,formal_OOS=False))
        if not momentpath.exists():
            boot=bootstrap_moments(split_features['calibration'],development=False,replications=2000,alpha=.05)
            upper=(1.,1.,1.,0. if prepared.feature_tasks.empty_H else 1.,1.09)
            moment=calibrate_moments(split_features['planning'],boot,physical_upper=upper)
            if moment.status!='PASS_MOMENT_CALIBRATION':raise ValueError('Formal moment calibration failed: '+target)
            write_new(momentpath,dict(**moment.as_dict(),bootstrap=boot.as_dict(),formal_OOS=False))
            np.savez_compressed(bootstrappath,indices=boot.indices,means=boot.bootstrap_means,center=boot.center,halfwidth=boot.halfwidth)
        else:
            moment=json.loads(momentpath.read_text())
            if moment['status']!='PASS_MOMENT_CALIBRATION':raise ValueError('Stored moment status invalid')
        if not any(row['target_event']==target for row in overview):
            overview.append(dict(target_event=target,H=len(prepared.tasks.task_ids),planning_count=160,calibration_count=80,planning_index=file_reference(taskdir/'planning.index.json'),calibration_index=file_reference(taskdir/'calibration.index.json'),moments=file_reference(momentpath),bootstrap=file_reference(taskdir/'bootstrap.npz')))
        checkpoint_now('RUNNING')
    checkpoint_now('COMPLETE')
    summary=dict(status=('COMPLETE_FORMAL_BASELINE_INPUTS_PENDING_INDEPENDENT_AUDIT' if adapter is None else adapter.complete_status),formal_OOS=False,formal_plans_solved=False,targets=overview,atom_count=len(completed),task_sources=taskrefs,source_manifest=request['source_manifest'],formal_protocol_lock=request['formal_protocol_lock'],cumulative_seconds=prior_seconds+time.monotonic()-started,independent_scientific_audit_required=True)
    if not (out/'summary.json').exists():write_new(out/'summary.json',summary)
    return summary


def generate_bank(request,*,wall_seconds,outputroot,resume=False):
    return _generate_bank(request,wall_seconds=wall_seconds,outputroot=outputroot,resume=resume)


def _worker_main(generate, complete_status):
    worker_started=time.monotonic()
    invocation=Path(os.environ['EVCS_BATCH_ATTEMPT_REQUEST']);digest=sha256(invocation)
    if digest!=os.environ['EVCS_BATCH_INVOCATION_SHA256']:raise ValueError('Worker invocation changed')
    call=json.loads(invocation.read_text());request=read_ref(call['item_request_reference']);out=Path(request['outputroot'])
    if call['domain']!='formal':raise ValueError('Formal bank worker needs formal batch namespace')
    source_manifest=read_ref(request['source_manifest']);source_root=Path(request['source_manifest']['path']).parent
    for row in source_manifest['files']:
        if sha256(source_root/row['path'])!=row['sha256']:raise ValueError('Active worker source changed: '+row['path'])
    if call['previous_checkpoint'] is not None:
        read_ref(call['previous_checkpoint'])
        if sha256(out/'checkpoint.json')!=call['previous_checkpoint']['sha256']:raise ValueError('Active bank checkpoint differs from previous immutable snapshot')
    deadline=call['worker_deadline_monotonic']
    reserve=request.get('bank_export_reserve_seconds',2.)
    if (type(deadline) not in (int,float) or not np.isfinite(deadline) or
        type(reserve) not in (int,float) or not np.isfinite(reserve) or reserve<2):
        raise ValueError('Invalid shared bank deadline/export reserve')
    source_seconds=time.monotonic()-worker_started
    grant=min(7190.,deadline-time.monotonic()-reserve)
    if grant<=0:raise TimeoutError('Source verification consumed bank work grant')
    result=generate(request,wall_seconds=grant,outputroot=out,resume=call['previous_checkpoint'] is not None)
    def within_deadline():
        if time.monotonic()>=deadline:raise TimeoutError('Bank evidence export exceeded shared deadline')
    within_deadline()
    export_started=time.monotonic()
    snapshot=Path(call['attempt_directory'])/'bank_checkpoint_snapshot.json'
    with snapshot.open('xb') as f:f.write((out/'checkpoint.json').read_bytes())
    cp=file_reference(snapshot);artifacts=[cp]
    checkpoint=json.loads(snapshot.read_text())
    for atomref in checkpoint['completed_atoms'].values():
        within_deadline()
        atom=read_ref(atomref);artifacts.append(atomref);artifacts.extend(atom['files'].values())
    for row in checkpoint['completed_targets']:
        artifacts.extend(row[k] for k in ('planning_index','calibration_index','moments','bootstrap'))
    complete=result['status']==complete_status
    if complete:artifacts.append(file_reference(out/'summary.json'))
    response=dict(status='COMPLETE' if complete else 'TIME_LIMIT',invocation_sha256=digest,artifacts=artifacts,scientific_status='PENDING_INDEPENDENT_BANK_AUDIT',formal_OOS=False)
    if not complete:response.update(resumable=True,checkpoint=cp)
    within_deadline()
    response.update(source_verification_seconds=source_seconds,generation_grant_seconds=grant,
        export_reserve_seconds=reserve,export_seconds=time.monotonic()-export_started,
        worker_elapsed_seconds=time.monotonic()-worker_started)
    write_new(Path(call['result_path']),response)
    within_deadline()
    return 0

def worker_main():
    return _worker_main(generate_bank,'COMPLETE_FORMAL_BASELINE_INPUTS_PENDING_INDEPENDENT_AUDIT')


if __name__=='__main__':raise SystemExit(worker_main())
