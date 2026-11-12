"""Full-H 80-EV/160-support resource and initial-RMP development probe.

Three deterministic full-H atoms measure loading/assembly/RSS. Phase-I uses
all160 atoms. The actual unfixed initial RMP is solved, but this preparatory
probe does not claim a full-support UB, completed C&CG, or full S03 acceptance.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

from evcs_experiments.config import get_paths
from evcs_experiments.evidence import sha256, source_provenance
from evcs_experiments.physical_inputs import load_physical_case, load_frozen_tasks, load_scenario_physical_inputs
from evcs_experiments.first_stage import assemble_first_stage
from evcs_experiments.first_stage_check import check_first_stage
from evcs_experiments.recourse import assemble_recourse
from evcs_experiments.probability import ProbabilitySet
from evcs_experiments.probability_phase1 import solve_phase1
from evcs_experiments.extensive import assemble_extensive, solve_extensive
from evcs_experiments.validation.validate_s03_probability import _physical_acceptance, target_inputs
from evcs_experiments.validation.validate_s03_extensive import prerequisites
from evcs_experiments.validation.validate_s03_ccg import _save_model
from evcs_experiments.validation.s02_toys import _json_file, _save_solution

I03_SHA='96f2f597b8e75f388988643e9446cf5d145771b1e03d1777f9530842466b9c69'
TARGET='michael_2018'


def read_ref(ref):
    p=Path(ref['path'])
    if sha256(p)!=ref['sha256']:raise ValueError('Source hash changed: '+str(p))
    return json.loads(p.read_text())


def rss_bytes():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):return int(line.split()[1])*1024
    raise RuntimeError('Linux RSS unavailable')


def array_bytes(template):
    total=0
    for key in ('A_ub','A_eq','Bx_ub','Bx_eq','Br_ub','Br_eq'):
        a=getattr(template,key);total+=a.data.nbytes+a.indices.nbytes+a.indptr.nbytes
    for key in ('b_ub','b_eq','c','lower_bounds','upper_bounds','component_objectives'):
        total+=getattr(template,key).nbytes
    return total


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-root',type=Path,default=get_paths().data_root)
    parser.add_argument('--s03-i03-acceptance',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--wall-seconds',type=float,default=600.)
    parser.add_argument('--rmp-seconds',type=float,default=180.)
    args=parser.parse_args();started=time.monotonic();paths=get_paths();out=args.output.resolve()
    if not 0<args.wall_seconds<=600 or not 0<args.rmp_seconds<=args.wall_seconds:raise ValueError('Invalid development budget')
    out.mkdir(parents=True,exist_ok=False)
    def progress(phase,**values):
        _json_file(out/'progress.json',{'phase':phase,'elapsed_seconds':time.monotonic()-started,
            'rss_bytes':rss_bytes(),'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,**values})
    progress('prerequisites')
    card=_physical_acceptance(args.s03_i03_acceptance,I03_SHA)
    if card['increment']!='S03-I03' or card['status']!='PASS_INCREMENT' or card['full_S03_pass']:raise ValueError('I03 scope mismatch')
    for name,digest in card['reviews'].items():
        if sha256(args.s03_i03_acceptance.parent/name)!=digest:raise ValueError('I03 review changed')
    ref=card['source_refs']['s03_i01_acceptance'];_,inherited=prerequisites(ref['path'],ref['sha256'])
    s01=read_ref(card['source_refs']['s01_acceptance']);manifest=json.loads(Path(s01['deliverables_manifest']).read_text())
    if sha256(s01['deliverables_manifest'])!=s01['deliverables_manifest_sha256']:raise ValueError('S01 manifest changed')
    contract=read_ref(manifest['generator_contract.json']);index=read_ref(manifest['task_scenario_replay_index.json'])
    taskref=next(v['frozen_tasks'] for v in contract['frozen_task_sources'] if v['target_event']==TARGET)
    rawtasks=read_ref(taskref)
    if len(rawtasks['vehicle_summaries'])!=80:raise ValueError('Expected frozen base fleet80')
    bank=next(v for v in index['archives'] if v['target_event']==TARGET and v['split']=='planning')
    if bank['scenario_count']!=160 or not bank['development']:raise ValueError('Expected full160 development support')
    ids,phi,lo,hi=target_inputs(inherited,TARGET,1.)
    if list(ids)!=bank['scenario_ids']:raise ValueError('Feature/support axis mismatch')
    case=load_physical_case(args.case_root);tasks=load_frozen_tasks(taskref['path'],case,taskref['sha256'])
    if len(tasks.task_ids)!=bank['task_count']:raise ValueError('Full H changed')
    first=assemble_first_stage(case,tasks);a=ProbabilitySet(ids,tuple(inherited['feature_order']),phi,lo,hi)
    phase=solve_phase1(a,time_limit_seconds=min(30.,args.wall_seconds-(time.monotonic()-started)))
    _json_file(out/'phase1.json',phase.as_dict())
    if not phase.feasible or not phase.basis_verified:raise ValueError('Full-support feasible initialization failed')
    _json_file(out/'execution.json',{'paths':paths.as_dict(),'python':sys.version,'argv':sys.argv,
        'source_manifest':{'path':paths.project_root/'RELEASE_MANIFEST.json','sha256':sha256(paths.project_root/'RELEASE_MANIFEST.json')},
        'config_path':os.environ.get('EVCS_CONFIG'),'runtime_policy_sha256':sha256(paths.spec_root/'runtime_policy.json'),
        'I03_acceptance':{'path':args.s03_i03_acceptance,'sha256':I03_SHA},'task_source':taskref,'bank':bank,
        'target':TARGET,'rho':1.,'base_fleet':80,'vehicles_with_tasks':len(set(tasks.vehicle_ids)),
        'H':len(tasks.task_ids),'support':len(ids),'full_H_and_routes_unchanged':True,'OOS':False})
    templates={};samples=[];base_rss=rss_bytes()
    def get_template(i):
        if i in templates:return templates[i]
        began=time.monotonic();before=rss_bytes()
        scene=load_scenario_physical_inputs(bank['file']['path'],bank['scenario_ledger']['path'],i,case,tasks,
            expected_npz_sha256=bank['file']['sha256'],expected_ledger_sha256=bank['scenario_ledger']['sha256'])
        if scene.scenario_id!=ids[i]:raise ValueError('Loaded scenario ID changed')
        loaded=time.monotonic();template=assemble_recourse(case,tasks,scene)
        record={'scenario_index':i,'scenario_id':ids[i],'template_sha256':template.template_sha256,
            'H':len(template.task_ids),'columns':len(template.c),'inequality_rows':len(template.b_ub),'equality_rows':len(template.b_eq),
            'load_seconds':loaded-began,'assembly_seconds':time.monotonic()-loaded,'rss_before':before,
            'rss_after':rss_bytes(),'array_bytes':array_bytes(template)}
        templates[i]=template;samples.append(record);progress('template_assembled',samples=samples)
        return template
    for i in (0,80,159):get_template(i)
    per_template=max(max(s['array_bytes'],max(0,s['rss_after']-s['rss_before'])) for s in samples)
    estimated_bank=base_rss+160*per_template
    memory={'base_rss_bytes':base_rss,'sample_indices':[0,80,159],'samples':samples.copy(),
        'conservative_per_template_bytes':per_template,'estimated_160_template_resident_bytes':estimated_bank,
        'estimate_is_not_measured_full_bank':True,'includes_python_metadata_via_RSS':True,
        'full_bank_above_32GiB':estimated_bank>32*1024**3,'full_bank_loaded':False}
    _json_file(out/'memory_estimate.json',memory)
    # Only the initial master copies are needed for this bounded timing probe.
    initial=list(phase.initial_support_indices)
    predicted_initial=rss_bytes()+len(initial)*per_template*4
    if predicted_initial>32*1024**3:
        report={'status':'MEMORY_PREFLIGHT_ONLY','reason':'Estimated initial master plus solver workspace exceeds original32GiB budget',
                'initial_RMP_executed':False,'lower_bound_usd':None,'upper_bound_usd':None}
    else:
        progress('initial_RMP_templates',initial_indices=initial)
        copies=tuple(get_template(i) for i in initial)
        restricted=ProbabilitySet(tuple(ids[i] for i in initial),a.feature_ids,a.features[initial],a.lower,a.upper)
        began=time.monotonic()
        model=assemble_extensive(first,copies,risk_model='DRO',scenario_ids=restricted.support_ids,probability_set=restricted)
        assembly_seconds=time.monotonic()-began
        remaining=args.wall_seconds-(time.monotonic()-started)
        if remaining<=0:raise TimeoutError('Development budget exhausted before initial master solve')
        progress('initial_RMP_solve',columns=len(model.c),rows=model.A.shape[0],nnz=model.A.nnz)
        solved=solve_extensive(model,time_limit_seconds=min(args.rmp_seconds,remaining),mip_relative_gap=.001)
        d=out/'initial_rmp';d.mkdir();_save_model(d,model,solved)
        if solved.solution is not None:
            check=check_first_stage(case,tasks,solved.solution.plan);_json_file(d/'first_stage_check.json',check)
            for i,y in zip(initial,solved.solution.recourse_solutions):_save_solution(d/f'witness_{i}.npz',y)
            if not check['passed']:raise ValueError('Initial master first-stage physical failure')
        report={'status':'INITIAL_RMP_PROFILE','initial_RMP_executed':True,'initial_support_indices':initial,
            'master_assembly_seconds':assembly_seconds,'master_solve_seconds':solved.wall_seconds,
            'solver_status':solved.status,'raw_solver_status':solved.solver_status,'lower_bound_usd':solved.lower_bound_usd,
            'restricted_incumbent_objective_usd':solved.objective_usd,'full_support_upper_bound_usd':None,
            'RMP_incumbent_is_not_full_support_UB':True,'MIP_gap':solved.mip_gap}
    report.update(target=TARGET,base_fleet=80,H=len(tasks.task_ids),full_support=160,
        total_wall_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        memory_estimate=memory,full_S03_pass=False,full_CCG_completed=False,full_support_recourse_evaluation_executed=False,
        next_action='Use measured memory/load/assembly/RMP timings to implement bounded lazy templates if needed, then full-support C&CG evaluation; retain cumulative compute.')
    _json_file(out/'profile.json',report);_json_file(out/'model_code_manifest.json',source_provenance())
    progress('complete',status=report['status'])
    _json_file(out/'artifact_sha256.json',{str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print(json.dumps({k:report[k] for k in ('status','H','full_support','full_S03_pass','total_wall_seconds')}));return 0


if __name__=='__main__':raise SystemExit(main())
