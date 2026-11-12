"""Read every accepted planning/calibration atom through the installed adapter.

No weather generation, LP/MILP, OOS, bank regeneration or acceptance signing.
An existing output directory is never reused. Failure retains the exact prefix.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

from evcs_experiments.config import get_paths
from evcs_experiments.evidence import sha256
from evcs_experiments.formal_physical_inputs import FormalPhysicalBank
from evcs_experiments.physical_inputs import load_physical_case,load_frozen_tasks
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.runtime.jobctl import atomic_json


def need(value,message):
    if not bool(value):raise ValueError(message)


def reference(path):
    path=Path(path).resolve();return dict(path=str(path),sha256=sha256(path),size_bytes=path.stat().st_size)


def write_new(path,value):
    """Install a complete JSON file atomically, refusing any existing name."""
    path=Path(path);temporary=path.with_name('.'+path.name+'.tmp')
    with temporary.open('x') as f:json.dump(value,f,allow_nan=False,indent=2);f.write('\n')
    try:os.link(temporary,path)
    finally:temporary.unlink()
    return reference(path)


def loaded_source_references(root,source_paths):
    references=[]
    for name,module in tuple(sys.modules.items()):
        if name=='evcs_experiments' or name.startswith('evcs_experiments.'):
            filename=getattr(module,'__file__',None)
            if filename is not None:
                path=Path(filename).resolve()
                need(path.is_relative_to(root) and path in source_paths,'Mixed or unbound imported source: '+name)
                references.append(reference(path))
    need(bool(references),'No installed experiment source closure found')
    return references


def validate(summary,expected_summary_sha256,acceptance,expected_acceptance_sha256,
             source_root,expected_source_manifest_sha256,output,wall_seconds=590):
    started=time.monotonic()
    need(type(wall_seconds) in (int,float) and math.isfinite(wall_seconds) and 0<wall_seconds<=7190,'Invalid finite wall budget')
    out=Path(output).resolve();paths=get_paths()
    protected=(Path(summary).resolve().parent,Path(acceptance).resolve().parent,Path(source_root).resolve(),paths.data_root.resolve(),paths.frozen_source_root.resolve())
    need(not any(out.is_relative_to(root) for root in protected),'Output must be outside all bank/source/frozen input roots')
    out.mkdir(parents=True,exist_ok=False)
    inputs={};targets=[];checks=[];current=dict(target_id=None,split=None,scenario_index=None,scenario_id=None)
    def remaining():
        if time.monotonic()-started>=wall_seconds:raise TimeoutError('Adapter validation wall budget exhausted')
    def verify(ref):
        need(isinstance(ref,dict) and set(ref) in ({'path','sha256','size_bytes'},{'path','sha256','bytes'}),'Unknown strong reference schema')
        need(Path(ref['path']).is_absolute(),'Absolute reference required')
        actual=reference(ref['path']);need(actual['sha256']==ref['sha256'] and actual['size_bytes']==ref.get('size_bytes',ref.get('bytes')),'Bound bytes changed: '+ref['path'])
        previous=inputs.setdefault(actual['path'],actual);need(previous==actual,'Conflicting original reference')
        return actual
    def read(ref):return json.loads(Path(verify(ref)['path']).read_text())
    def checkpoint(status):
        atomic_json(out/'checkpoint.json',dict(status=status,completed_scenarios=len(checks),completed_targets=targets,
            current=current,wall_seconds=time.monotonic()-started,formal_OOS=False,scientific_acceptance=False))
    def source_manifest(ref):
        value=read(ref);root=Path(ref['path']).resolve().parent
        need(isinstance(value['files'],list) and bool(value['files']),'Empty release manifest')
        for member in value['files']:
            remaining();p=(root/member['path']).resolve();need(p.is_relative_to(root),'Release member escapes root')
            r=reference(p);need(r['sha256']==member['sha256'],'Release source changed: '+str(p));verify(r)
        return value
    try:
        sr=reference(summary);ar=reference(acceptance)
        need(sr['sha256']==expected_summary_sha256 and ar['sha256']==expected_acceptance_sha256,'Externally pinned summary/acceptance changed')
        bank=read(sr);card=read(ar)
        need(card['status']=='PASS_COMPONENT' and card['component_id']=='FORMAL_BASELINE_INPUTS' and
            card['spec_id']=='S05' and card['increment_id']=='S05-I01','Accepted full-bank component required')
        for key in ('summary_reference','audit_reference','source_manifest_reference','protocol_reference',
                    'independent_review_reference','job_result_reference','artifact_manifest_reference'):verify(card[key])
        need(card['summary_reference']==sr and card['scenario_count']==1920 and card['target_count']==8 and
            card['formal_OOS'] is False and card['formal_plans_solved'] is False and card['full_S05_pass'] is False,'Wrong input acceptance scope')
        audit=read(card['audit_reference']);protocol=read(card['protocol_reference'])['payload']
        need(audit['status']=='PASS_INDEPENDENT_FORMAL_BANK_AUDIT' and audit['source_summary_sha256']==sr['sha256']
            and Path(audit['source_output']).resolve()==Path(summary).resolve().parent and audit['scenario_count']==1920,
            'Audit does not bind complete original bank')
        for key in ('s01_acceptance','s04_acceptance'):verify(audit[key])
        need(bank['status']=='COMPLETE_FORMAL_BASELINE_INPUTS_PENDING_INDEPENDENT_AUDIT' and bank['atom_count']==1920
            and bank['formal_OOS'] is False and bank['formal_plans_solved'] is False,'Original bank scope differs')
        need(bank['source_manifest']==card['source_manifest_reference'] and bank['formal_protocol_lock']==card['protocol_reference']
            and audit['source_manifest_sha256']==card['source_manifest_reference']['sha256'] and audit['formal_protocol_lock']==card['protocol_reference'],
            'Original source/protocol bindings differ')
        source_manifest(card['source_manifest_reference'])
        paths=get_paths();root=Path(source_root).resolve()
        need(root==paths.project_root.resolve(),'Candidate source root differs from installed project')
        release=reference(root/'RELEASE_MANIFEST.json');need(release['sha256']==expected_source_manifest_sha256,'Candidate release manifest changed')
        manifest=source_manifest(release)
        source_paths={(root/r['path']).resolve() for r in manifest['files']}
        need(Path(__file__).resolve() in source_paths,'Validation entry is not source-bound')
        adapter_path=Path(sys.modules[FormalPhysicalBank.__module__].__file__).resolve()
        need(adapter_path==root/'src/evcs_experiments/formal_physical_inputs.py' and adapter_path in source_paths,'Installed adapter origin differs')
        events=protocol['target_ids']
        need(protocol['domain']=='formal' and len(events)==8 and len(set(events))==8 and
            [r['target_event'] for r in bank['targets']]==events and set(bank['task_sources'])==set(events),'Eight-target identity/order differs')
        write_new(out/'execution.json',dict(summary_reference=sr,acceptance_reference=ar,original_audit_reference=card['audit_reference'],
            original_source_manifest_reference=card['source_manifest_reference'],protocol_reference=card['protocol_reference'],
            candidate_source_manifest_reference=release,adapter_reference=reference(adapter_path),entry_reference=reference(__file__),
            paths=paths.as_dict(),wall_budget_seconds=wall_seconds,requested_scenarios=1920,
            weather_generated=False,solver_calls=0,formal_OOS=False,scientific_acceptance=False))
        case=load_physical_case(paths.data_root)
        for row in bank['targets']:
            remaining();target=row['target_event'];target_started=time.monotonic();target_checks=[]
            current.update(target_id=target,split=None,scenario_index=None,scenario_id=None)
            task_ref=verify(bank['task_sources'][target]);rawtasks=read(task_ref)
            need(len(rawtasks['vehicle_summaries'])==80 and rawtasks['task_count']==row['H'],'Full fleet/H differs')
            tasks=load_frozen_tasks(task_ref['path'],case,task_ref['sha256'])
            need(len(tasks.task_ids)==row['H'],'Loaded full H differs')
            for key in ('moments','bootstrap'):verify(row[key])
            for split,size in (('planning',160),('calibration',80)):
                remaining();current.update(split=split,scenario_index=None,scenario_id=None)
                need(row[split+'_count']==size,'Declared split support count differs')
                ir=verify(row[split+'_index']);index=read(ir)
                need(index['split']==split and index['target_event']==target and index['task_source']==bank['task_sources'][target],
                    'Index split/target/frozen task reference differs')
                adapter=FormalPhysicalBank(ir['path'],case,tasks,expected_index_sha256=ir['sha256'],case_root=paths.data_root)
                need(len(adapter.scenario_ids)==size and tuple(x['scenario_id'] for x in index['scenarios'])==adapter.scenario_ids,
                    'Complete ordered adapter support differs')
                for i,item in enumerate(index['scenarios']):
                    current.update(scenario_index=i,scenario_id=item['scenario_id']);remaining();tick=time.monotonic()
                    atom=read(item['manifest'])
                    for ref in atom['files'].values():verify(ref)
                    scene=adapter.load(i,expected_scenario_sha256=item['scenario_sha256'])
                    need(scene.scenario_id==item['scenario_id'] and scene.scenario_sha256==item['scenario_sha256']
                        and scene.target_event==target and scene.split==split and scene.development is False and scene.task_ids==tasks.task_ids,
                        'Adapter returned different domain/support/task identities')
                    hashes=dict(scene.metadata['scenario_provenance']['array_hashes'])
                    need(scene.metadata['verified_array_count']==26 and len(hashes)==26,'Adapter did not verify every original array')
                    check=dict(target_id=target,split=split,scenario_index=i,scenario_id=scene.scenario_id,
                        scenario_sha256=scene.scenario_sha256,task_sha256=scene.task_sha256,manifest_reference=verify(item['manifest']),
                        verified_array_count=26,array_hashes=hashes,elapsed_seconds=time.monotonic()-tick)
                    checks.append(check);target_checks.append(check)
                    checkpoint('RUNNING');remaining()
            need(len(target_checks)==240,'Target physical read coverage incomplete')
            tr=write_new(out/(target+'.json'),dict(target_id=target,H=row['H'],scenario_count=240,
                planning_count=160,calibration_count=80,scenarios=target_checks,elapsed_seconds=time.monotonic()-target_started))
            targets.append(tr);checkpoint('RUNNING')
        need(len(checks)==1920 and len(targets)==8 and len({c['scenario_id'] for c in checks})==1920,'Full unique bank coverage incomplete')
        remaining()
        loaded_refs=loaded_source_references(root,source_paths)
        for ref in loaded_refs:verify(ref)
        write_new(out/'loaded_sources.json',loaded_refs)
        for ref in list(inputs.values()):verify(ref);remaining()
        status='PASS_FULL_FORMAL_ADAPTER_READ_VALIDATION'
        checkpoint(status)
        result=dict(status=status,scenario_count=len(checks),target_count=len(targets),targets=targets,
            checked_array_count=26*len(checks),wall_seconds=time.monotonic()-started,weather_generated=False,solver_calls=0,
            formal_OOS=False,scientific_acceptance=False,full_S05_pass=False,independent_review_required=True)
        write_new(out/'input_references.json',list(inputs.values()));write_new(out/'result.json',result)
        write_new(out/'artifact_manifest.json',[reference(p) for p in sorted(out.iterdir()) if p.is_file()])
        remaining()
        return result
    except Exception as exc:
        status='TIME_LIMIT_INCOMPLETE_ADAPTER_VALIDATION' if isinstance(exc,TimeoutError) else 'FAIL_FORMAL_ADAPTER_VALIDATION'
        checkpoint(status)
        write_new(out/'completed_prefix.json',checks)
        if not (out/'input_references.json').exists():write_new(out/'input_references.json',list(inputs.values()))
        for name in ('result.json','artifact_manifest.json'):
            if (out/name).exists():(out/name).rename(out/('unaccepted_'+name))
        result=dict(status=status,error_type=type(exc).__name__,error=str(exc),current=current,
            scenario_count=len(checks),target_count=len(targets),wall_seconds=time.monotonic()-started,
            formal_OOS=False,weather_generated=False,solver_calls=0,scientific_acceptance=False,full_S05_pass=False)
        write_new(out/'result.json',result)
        write_new(out/'artifact_manifest.json',[reference(p) for p in sorted(out.iterdir()) if p.is_file()])
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('summary','acceptance','source-root','output'):p.add_argument('--'+key,type=Path,required=True)
    for key in ('expected-summary-sha256','expected-acceptance-sha256','expected-source-manifest-sha256'):p.add_argument('--'+key,required=True)
    p.add_argument('--wall-seconds',type=float,default=590)
    a=p.parse_args()
    result=validate(**vars(a));print(json.dumps(result,allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())
