"""Bounded original-template binding preparation; no solve or acceptance grant."""
import argparse
import json
from pathlib import Path
import time

from evcs_experiments.config import get_paths
from evcs_experiments.evaluation_lock import file_reference, _verify_reference
from evcs_experiments.first_stage import assemble_first_stage
from evcs_experiments.formal_physical_inputs import FormalPhysicalBank
from evcs_experiments.physical_inputs import load_physical_case, load_frozen_tasks
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.recourse_store import bind_template, load_template
from evcs_experiments.recourse_template_index import prepare_template_index
from evcs_experiments.runtime.jobctl import atomic_json
import evcs_experiments.recourse as compiler
import evcs_experiments.recourse_store as reader


def need(value,message):
    if not bool(value):raise ValueError(message)


def read(reference):
    _verify_reference(reference)
    return json.loads(Path(reference['path']).read_text())


def normal(reference):
    ref=dict(reference)
    if 'bytes' in ref:ref['size_bytes']=ref.pop('bytes')
    _verify_reference(ref)
    return ref


def run(request_reference, output, wall_seconds):
    started=time.monotonic();need(10<=wall_seconds<=570,'Bounded preparation needs10..570 seconds')
    request=read(request_reference);refs=request['references']
    required={'raw_request','raw_snapshot','bank_audit','numeric_audit','reader_acceptance','source_manifest'}
    need(set(refs)==required,'Unknown original preparation sources')
    docs={key:read(value) for key,value in refs.items()};raw=docs['raw_request'];snapshot=docs['raw_snapshot']
    need(raw['domain']=='formal' and raw['formal_OOS'] is False,'Only original formal planning input supported')
    ids=raw['problem_identity']['scenario_ids'];need(len(ids)==160 and len(snapshot['templates'])==160 and snapshot['request']==refs['raw_request'],'Full original160 snapshot required')
    need(docs['numeric_audit']['status']=='PASS_INDEPENDENT_CANDIDATE_160_RECOURSE_AUDIT'
         and docs['numeric_audit']['passed'] is True and docs['numeric_audit']['support_ids']==ids
         and docs['numeric_audit']['original_problem_sha256']==canonical_hash(raw['problem_identity']), 'Wrong original numerical proof')
    binding=raw['input_binding']
    need(docs['bank_audit']['status']=='PASS_INDEPENDENT_FORMAL_BANK_AUDIT'
         and docs['bank_audit']['source_summary_sha256']==binding['summary']['sha256'],'Original bank audit differs')
    card=docs['reader_acceptance'];need(card['status']=='PASS_COMPONENT' and card['component_id']=='NUMERIC_TEMPLATE_READER_DEVELOPMENT_AND_ONE_REAL_ATOM','Reader component not accepted')
    _verify_reference(card['source']);need(card['source']['sha256']==file_reference(reader.__file__)['sha256'],'Accepted reader implementation differs')
    compiler_ref=file_reference(compiler.__file__)
    need(compiler_ref['sha256']==binding['planning_source']['recourse.py'],'Original compiler changed')
    source_root=Path(refs['source_manifest']['path']).parent
    for relative in ('src/evcs_experiments/recourse.py','src/evcs_experiments/recourse_store.py',
                     'src/evcs_experiments/recourse_template_index.py','scripts/verification/prepare_formal_template_index.py'):
        rows=[r for r in docs['source_manifest']['files'] if r['path']==relative]
        need(len(rows)==1 and file_reference(source_root/relative)['sha256']==rows[0]['sha256'],'Preparation source changed')
    need(Path(__file__).resolve()==source_root/'scripts/verification/prepare_formal_template_index.py','Driver outside bound installed release')
    task_ref=normal(binding['task_source']);index_ref=normal(binding['target']['planning_index']);index=read(index_ref)
    case=load_physical_case(get_paths().data_root);tasks=load_frozen_tasks(task_ref['path'],case,task_ref['sha256'])
    bank=FormalPhysicalBank(index_ref['path'],case,tasks,expected_index_sha256=index_ref['sha256'],case_root=get_paths().data_root)
    first=assemble_first_stage(case,tasks)
    need(list(bank.scenario_ids)==ids and first.matrix_sha256==raw['problem_identity']['first_stage_sha256'],'Original physical axes differ')
    out=Path(output).resolve();need(not out.is_relative_to(Path(refs['raw_request']['path']).parent),'Output cannot overwrite original raw input')
    out.mkdir(parents=True,exist_ok=False);(out/'bindings').mkdir()
    progress=dict(status='PREPARING_TEMPLATE_INDEX',request_reference=request_reference,binding_references=[],
                  formal_OOS=False,solver_calls=0,independent_association_review_required=True)
    atomic_json(out/'progress.json',progress)
    for i in range(160):
        if time.monotonic()-started>=wall_seconds-10:break
        scene=bank.load(i,expected_scenario_sha256=bank.scenario_sha256s[i])
        sources=dict(original_request=refs['raw_request'],task_source=task_ref,planning_index=index_ref,
                     scenario_manifest=normal(index['scenarios'][i]['manifest']))
        reference=bind_template(snapshot['templates'][i],first,case,tasks,scene,source_references=sources,
                                compiler_reference=compiler_ref,output=out/'bindings'/f'{i:03d}.json')
        progress['binding_references'].append(reference)
        progress['elapsed_seconds']=time.monotonic()-started
        atomic_json(out/'progress.json',progress)
    complete=len(progress['binding_references'])==160
    index_reference=None
    if complete:
        index_reference=prepare_template_index(original_request_reference=refs['raw_request'],
            original_snapshot_reference=refs['raw_snapshot'],binding_references=progress['binding_references'],
            proof_references={key:refs[key] for key in ('bank_audit','numeric_audit','reader_acceptance')},output=out/'index.json')
    result=dict(status='PREPARED_FULL160_INDEX_PENDING_ASSOCIATION_REVIEW' if complete else 'PREPARATION_TIME_LIMIT',
        complete=complete,prepared_count=len(progress['binding_references']),index_reference=index_reference,
        request_reference=request_reference,progress_reference=file_reference(out/'progress.json'),
        elapsed_seconds=time.monotonic()-started,compiler_reference=compiler_ref,solver_calls=0,
        full_support_association_accepted=False,production_routing_changed=False,formal_OOS=False,
        formal_plan_budget_changed=False,preprocessing_allocation_policy='Actual parent time retained; future cross-plan allocation must be declared before use.')
    with (out/'result.json').open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',required=True)
    parser.add_argument('--expected-request-sha256',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--wall-seconds',type=float,default=570)
    args=parser.parse_args();reference=file_reference(args.request)
    need(reference['sha256']==args.expected_request_sha256,'Preparation request changed')
    print(json.dumps(run(reference,args.output,args.wall_seconds),indent=2))


if __name__=='__main__':main()
