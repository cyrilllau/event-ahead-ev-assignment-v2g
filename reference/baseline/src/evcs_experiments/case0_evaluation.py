"""Case0 item scheduling and resumable raw fixed-plan evaluation component.

The caller provides an independently reviewed Case0 pre-OOS lock validator and
physical/template factories. This component does not confer lock admission or
independent full-result acceptance. It never accesses Case1 campaign state.
"""
from collections import OrderedDict
import json
from pathlib import Path
import time

import numpy as np

from .case0_profiles import lock_target

from .campaign_batch import file_reference,verify_reference
from .evaluation_metrics import metrics_from_solution,plan_identity
from .first_stage import assemble_first_stage
from .recourse import solve_fixed_plan
from .recourse_store import physical_fingerprint

METHODS={'R1':('EA_DRO','DSP','OR_DRO','SAA','ARO','ZERO'),
         'R2':('EA_DRO','SAA','ARO'),'R3':('EA_DRO','SAA','ARO')}


def case0_items(samples_per_regime=64):
    if type(samples_per_regime) is not int or samples_per_regime<1:raise ValueError('Invalid sample count')
    return [dict(item_id=f'{regime}.{i:03d}.{method}',regime=regime,sample_index=i,method_id=method)
            for regime,methods in METHODS.items() for i in range(samples_per_regime) for method in methods]


def _read(ref):
    verify_reference(ref);return json.loads(Path(ref['path']).read_text())


def _serial(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if hasattr(v,'items'):return {k:_serial(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [_serial(x) for x in v]
    return v


def _write(p,v):
    with Path(p).open('x') as f:json.dump(_serial(v),f,indent=2,allow_nan=False);f.write('\n')
    return file_reference(p)


class Case0EvaluationContext:
    """One bounded process validates the lock once; shares exact physical templates.

    lock_validator(lock_reference) must actually validate all Case0 plans,
    numerical/scale/source/budget locks. A mere status field is not a validator.
    The validator returns method_plan_sha256 and sample_ids_by_regime; both are
    checked against supplied plans and every realized sample.
    physical_factory(method,index,regime) returns (case,tasks,scene).
    template_provider(case,tasks,scene) returns (template, original_record_ref).
    Source admission for those callable implementations remains with the caller.
    """
    def __init__(self,*,lock_reference,lock_validator,plans,physical_factory,template_provider,
                 source_references,samples_per_regime=64,cache_capacity=1):
        if type(cache_capacity) is not int or cache_capacity<1:raise ValueError('Invalid cache capacity')
        verify_reference(lock_reference)
        for ref in source_references:verify_reference(ref)
        if file_reference(__file__) not in source_references:raise ValueError('Case0 evaluator source not bound')
        verified=lock_validator(lock_reference)
        if set(plans)!=set(METHODS['R1']):raise ValueError('All six execution plans required')
        hashes={m:plan_identity(p) for m,p in plans.items()}
        if verified['method_plan_sha256']!=hashes:raise ValueError('Plans differ from validated Case0 lock')
        self.target_id=lock_target(verified)
        self.sample_ids=verified['sample_ids_by_regime']
        if set(self.sample_ids)!=set(METHODS) or any(len(v)!=samples_per_regime or len(set(v))!=len(v) for v in self.sample_ids.values()):
            raise ValueError('Locked sample support differs')
        if hashes['DSP']!=hashes['EA_DRO']:raise ValueError('DSP changed EA plan')
        if np.any(plans['ZERO'].x) or np.any(plans['ZERO'].r):raise ValueError('ZERO procurement must be zero')
        self.lock_reference=lock_reference;self.plans=dict(plans);self.plan_hashes=hashes
        self.factory=physical_factory;self.provider=template_provider;self.sources=source_references
        self.items={r['item_id']:r for r in case0_items(samples_per_regime)}
        self.cache=OrderedDict();self.capacity=cache_capacity;self.cache_hits=0;self.cache_misses=0

    def template(self,case,tasks,scene):
        first=assemble_first_stage(case,tasks);key=physical_fingerprint(first,case,tasks,scene)
        if key in self.cache:
            self.cache_hits+=1;self.cache.move_to_end(key);return self.cache[key],key
        template,ref=self.provider(case,tasks,scene);verify_reference(ref)
        if (template.first_stage.matrix_sha256!=first.matrix_sha256 or template.task_sha256!=tasks.task_sha256
            or template.config_hash!=tasks.config_hash or template.scenario_sha256!=scene.scenario_sha256):
            raise ValueError('Provider changed original physical template')
        self.cache[key]=(template,ref);self.cache_misses+=1
        while len(self.cache)>self.capacity:self.cache.popitem(last=False)
        return (template,ref),key


def evaluate_case0_item(context,item,*,output_directory,deadline,lp_limit_seconds=60):
    """Save one LP/metrics record or verify and reuse an identical completed item.

    OOS sample authorization is external to development primitives and must be
    supplied by the Case0 lock validator. Only pending-independent-audit evidence
    is returned here; raw primal/dual and exact template reference are retained.
    """
    started=time.monotonic()
    if not np.isfinite(deadline) or deadline<=started or not 0<lp_limit_seconds<=60:raise ValueError('Invalid item deadline')
    if context.items.get(item.get('item_id'))!=item:raise ValueError('Undeclared evaluation item')
    method=item['method_id'];plan=context.plans[method]
    if plan_identity(plan)!=context.plan_hashes[method]:raise ValueError('Fixed plan changed')
    root=Path(output_directory).resolve();record_path=root/'record.json'
    binding=dict(lock_reference=context.lock_reference,item=item,plan_sha256=context.plan_hashes[method],source_references=context.sources)
    if root.exists():
        if (root/'failure.json').exists():raise ValueError('Partial/failed item requires diagnosis')
        if not record_path.is_file():raise ValueError('Partial item requires diagnosis; no automatic overwrite')
        ref=file_reference(record_path);record=_read(ref)
        if record['binding']!=binding or record['status']!='COMPLETE_CASE0_ITEM_PENDING_INDEPENDENT_AUDIT':raise ValueError('Changed item or failed prior evidence')
        for artifact in record['artifacts']:verify_reference(artifact)
        verify_reference(record['template_reference'])
        if time.monotonic()>=deadline:raise TimeoutError('Case0 reuse verification deadline exceeded')
        return dict(record_reference=ref,reused=True,solver_calls=0)
    root.mkdir(parents=True,exist_ok=False)
    try:
        case,tasks,scene=context.factory(method,item['sample_index'],item['regime'])
        if (scene.development is not True or scene.split!='oos' or scene.target_event!=context.target_id or scene.regime!=item['regime']
            or scene.scenario_id!=context.sample_ids[item['regime']][item['sample_index']]):
            raise ValueError('Wrong Case0 domain/target/locked sample')
        if scene.metadata.get('departure_h')!=(0 if method in ('DSP','OR_DRO') else -1):raise ValueError('Wrong fixed execution departure')
        if tasks.task_sha256!=plan.task_sha256 or tasks.config_hash!=plan.config_hash:raise ValueError('Factory changed fixed H/config')
        if case.model_revision!='v54-q1' or case.capability_active_coordinate!='accredited_reserve_r_av':raise ValueError('Case0 requires original Q1 physics')
        (template,template_ref),key=context.template(case,tasks,scene)
        remaining=deadline-time.monotonic()-1
        if remaining<=0:raise TimeoutError('Case0 template preparation consumed item deadline')
        solved=solve_fixed_plan(template,plan,time_limit_seconds=min(lp_limit_seconds,remaining))
        artifacts=[_write(root/'solver.json',solved.as_dict()),_write(root/'plan.json',dict(x=plan.x,r=plan.r,task_sha256=plan.task_sha256,config_hash=plan.config_hash))]
        if solved.solution is not None:
            np.save(root/'primal.npy',template.pack(solved.solution),allow_pickle=False);artifacts.append(file_reference(root/'primal.npy'))
        if not solved.optimal or not solved.certificate.get('optimal_certificate_valid'):raise ValueError('Uncertified Case0 LP '+solved.status)
        dual={k:np.asarray(solved.certificate['dual_'+k+'_marginals']) for k in ('inequality','equality','lower','upper')}
        bu,be=template.fixed_rhs(plan);np.savez_compressed(root/'dual_rhs.npz',**dual,b_ub=bu,b_eq=be);artifacts.append(file_reference(root/'dual_rhs.npz'))
        metrics=metrics_from_solution(case,tasks,scene,plan,solved.solution)
        if metrics['plan_sha256_after']!=context.plan_hashes[method]:raise ValueError('Evaluation changed plan')
        artifacts.append(_write(root/'metrics.json',metrics))
        if time.monotonic()>=deadline:raise TimeoutError('Case0 item evidence deadline exceeded')
        record=dict(status='COMPLETE_CASE0_ITEM_PENDING_INDEPENDENT_AUDIT',binding=binding,
            scenario_id=scene.scenario_id,scenario_sha256=scene.scenario_sha256,physical_fingerprint=key,
            template_sha256=template.template_sha256,template_reference=template_ref,artifacts=artifacts,
            elapsed_seconds=time.monotonic()-started,formal_OOS=False,scientific_acceptance=False)
        ref=_write(record_path,record)
        if time.monotonic()>=deadline:raise TimeoutError('Case0 final item seal deadline exceeded')
        return dict(record_reference=ref,reused=False,solver_calls=1)
    except Exception as exc:
        _write(root/'failure.json',dict(error_type=type(exc).__name__,error=str(exc),partial_outputs_preserved=True,formal_OOS=False))
        raise
