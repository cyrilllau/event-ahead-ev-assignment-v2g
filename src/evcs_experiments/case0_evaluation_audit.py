"""Independent read-only Case0 OOS witness audit; never optimize a fixed plan.

Producer artifacts remain immutable. Reassembly verifies coefficients, and a
small physical-fingerprint cache avoids repeating shared scenario templates.
"""
from collections import OrderedDict
from .case0_profiles import CAMPAIGNS, context_options, validate_config, lock_target

import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.sparse import load_npz
from .campaign_batch import file_reference,verify_reference,read_json
from .case0_evaluation import case0_items
from .case0_planning_audit import _kkt
from .first_stage import FirstStagePlan,assemble_first_stage
from .first_stage_check import check_first_stage
from .recourse import assemble_recourse
from .recourse_check import check_recourse
from .recourse_store import MATRICES,ARRAYS,_record,_numeric,_header,_identity_check,physical_fingerprint
from .evaluation_metrics import metrics_from_solution,plan_identity


def _need(ok,msg):
    if not bool(ok):raise ValueError(msg)


def _time(deadline):
    _need(type(deadline) in (int,float) and math.isfinite(deadline),'Finite OOS audit deadline required')
    if time.monotonic()>=deadline:raise TimeoutError('Case0 OOS audit deadline')


def _read(ref):
    verify_reference(ref);return read_json(ref['path'])


def _same(a,b,label,policy_sources=None):
    """Exact coefficients and reproducible reporting arithmetic, not looser tolerances."""
    if isinstance(a,dict):
        _need(isinstance(b,dict) and set(a)==set(b),label+' keys differ')
        for k in a:_same(a[k],b[k],label+'.'+k,policy_sources)
    elif isinstance(a,(list,tuple)):
        _need(isinstance(b,(list,tuple)) and len(a)==len(b),label+' axis differs')
        for i,(x,y) in enumerate(zip(a,b)):_same(x,y,label+f'[{i}]',policy_sources)
    elif isinstance(a,(float,int)) and not isinstance(a,bool):
        _need(type(b) in (float,int) and math.isfinite(float(a)) and math.isfinite(float(b)) and a==b,label+' differs')
    else:
        if label in ('Original reported metrics.physical_check.tolerances.source',
                     'Original reported metrics.physical_check.first_stage.tolerances.source') and a!=b:
            _need(policy_sources is not None and (a,b)==(policy_sources['producer_policy_reference']['path'],
                  policy_sources['auditor_policy_reference']['path']),'Unbound runtime-policy source correspondence')
            for key in ('producer_policy_reference','auditor_policy_reference'):verify_reference(policy_sources[key])
        else:_need(type(a)==type(b) and a==b,label+' differs')


def policy_source_correspondence(producer_manifest_reference,auditor_manifest_reference):
    """Only this typed policy-file path may differ; bytes and both inventories must agree."""
    refs=[]
    for manifest_ref in (producer_manifest_reference,auditor_manifest_reference):
        manifest=_read(manifest_ref);root=Path(manifest_ref['path']).parent
        relative='docs/experiments/runtime_policy.json'
        rows=[row for row in manifest['files'] if row['path']==relative]
        _need(len(rows)==1,'Reviewed manifest lacks unique runtime policy')
        ref=file_reference(root/relative)
        _need(ref['sha256']==rows[0]['sha256'] and ref['size_bytes']==rows[0]['size_bytes'],
              'Runtime policy differs from reviewed source manifest')
        refs.append(ref)
    _need(refs[0]['sha256']==refs[1]['sha256'] and refs[0]['size_bytes']==refs[1]['size_bytes'],
          'Original and auditor runtime-policy bytes differ')
    return dict(producer_policy_reference=refs[0],auditor_policy_reference=refs[1],
                producer_manifest_reference=producer_manifest_reference,auditor_manifest_reference=auditor_manifest_reference)


class OriginalTemplateAuditCache:
    def __init__(self,capacity=3):
        _need(type(capacity) is int and capacity>=1,'Positive template cache required')
        self.capacity=capacity;self.cache=OrderedDict();self.assemblies=0;self.hits=0;self.original_references={}

    def original(self,reference,case,tasks,scene,deadline):
        _time(deadline);first=assemble_first_stage(case,tasks)
        key=physical_fingerprint(first,case,tasks,scene)
        if key in self.cache:
            oldref,t=self.cache[key]
            _need(oldref==reference,'Same physical template changed original reference')
            verify_reference(reference);self.hits+=1;self.cache.move_to_end(key)
            return t,key
        row,refs=_record(reference)
        _need(row['scenario_id']==scene.scenario_id and type(row['scenario_index']) is int and row['scenario_index']==0,
              'Original template record scenario label differs')
        matrices,arrays=_numeric(refs)
        for ref in [reference,*refs.values()]:self.original_references[ref['path']]=ref
        meta=_header(Path(refs['identity.json']['path']))
        _identity_check(meta,row,first,case,tasks,scene,matrices,arrays)
        _time(deadline);t=assemble_recourse(case,tasks,scene);self.assemblies+=1
        _need(t.template_sha256==row['template_sha256'],'Rebuilt original template hash differs')
        for name in MATRICES:
            expected=getattr(t,name);actual=matrices[name]
            _need(actual.shape==expected.shape and (actual!=expected).nnz==0,'Original coefficient differs: '+name)
        for name in ARRAYS:_need(np.array_equal(arrays[name],getattr(t,name)),'Original objective/RHS/domain differs: '+name)
        self.cache[key]=(reference,t)
        while len(self.cache)>self.capacity:self.cache.popitem(last=False)
        _time(deadline);return t,key


def audit_case0_evaluation_item(*,record_reference,item,lock_reference,lock,plan,case,tasks,scene,
                                producer_source_references,cache,deadline,policy_sources=None):
    """Actual saved original-array/primal-dual audit. No solver callable is used."""
    _time(deadline);row=_read(record_reference);folder=Path(record_reference['path']).parent
    _need(row['status']=='COMPLETE_CASE0_ITEM_PENDING_INDEPENDENT_AUDIT' and row['formal_OOS'] is False
          and row['scientific_acceptance'] is False,'Wrong original item scope/status')
    binding=dict(lock_reference=lock_reference,item=item,plan_sha256=plan_identity(plan),source_references=producer_source_references)
    _need(row['binding']==binding and lock['method_plan_sha256'][item['method_id']]==plan_identity(plan),'Original item/plan/source binding differs')
    _need(scene.development is True and scene.split=='oos' and scene.target_event==lock_target(lock)
          and scene.regime==item['regime'] and scene.scenario_id==lock['sample_ids_by_regime'][item['regime']][item['sample_index']],
          'Original paired development sample differs')
    _need(scene.metadata.get('departure_h')==(0 if item['method_id'] in ('OR_DRO','DSP') else -1),
          'Original information/departure contract differs')
    _need(plan.task_sha256==tasks.task_sha256==scene.task_sha256 and plan.config_hash==tasks.config_hash==scene.config_hash,
          'Original fixed H/config differs')
    _need(row['scenario_id']==scene.scenario_id and row['scenario_sha256']==scene.scenario_sha256,'Original scenario identity differs')
    _need(case.model_revision=='v54-q1' and case.capability_active_coordinate=='accredited_reserve_r_av','Wrong original physical model')
    expected={'solver.json','plan.json','primal.npy','dual_rhs.npz','metrics.json'}
    refs={Path(r['path']).name:r for r in row['artifacts']}
    _need(len(row['artifacts'])==len(expected) and set(refs)==expected and
          {p.name for p in folder.iterdir()}==expected|{'record.json'},'Original item inventory incomplete/uncommitted')
    for ref in refs.values():
        _need(Path(ref['path']).parent==folder,'Foreign item artifact');verify_reference(ref)
    savedplan=FirstStagePlan(**_read(refs['plan.json']))
    _need(plan_identity(savedplan)==plan_identity(plan),'Saved fixed plan changed')
    firstcheck=check_first_stage(case,tasks,plan);_need(firstcheck['passed'],'Original first-stage feasibility failed')
    t,key=cache.original(row['template_reference'],case,tasks,scene,deadline)
    _need(key==row['physical_fingerprint'] and t.template_sha256==row['template_sha256'],'Original physical template binding differs')
    primal=np.load(refs['primal.npy']['path'],allow_pickle=False);solver=_read(refs['solver.json'])
    with np.load(refs['dual_rhs.npz']['path'],allow_pickle=False) as raw:
        _need(set(raw.files)=={'inequality','equality','lower','upper','b_ub','b_eq'},'Original dual/RHS inventory differs')
        for name in ('inequality','equality','lower','upper'):
            _need(np.array_equal(raw[name],np.asarray(solver['certificate']['dual_'+name+'_marginals'])),
                  'Original raw/solver dual mismatch: '+name)
        bu,be=t.fixed_rhs(plan)
        _need(np.array_equal(raw['b_ub'],bu) and np.array_equal(raw['b_eq'],be),'Original fixed-plan RHS differs')
    kkt=_kkt(t,plan,primal,solver)
    physical=check_recourse(case,tasks,scene,plan,t.unpack(primal),check_objective_bound=True)
    _need(physical['passed'],'Original-unit OOS physics failed')
    recomputed=metrics_from_solution(case,tasks,scene,plan,t.unpack(primal))
    _same(_read(refs['metrics.json']),recomputed,'Original reported metrics',policy_sources)
    _need(abs(sum(recomputed['costs'][name] for name in ('Fshed_usd','Fshort_usd','FV2G_usd','Fdeg_usd'))-kkt['qop_usd'])
          <=1e-4+1e-7*max(abs(kkt['qop_usd']),1.),'Independent operating cost differs')
    for ref in refs.values():verify_reference(ref)
    verify_reference(record_reference);_time(deadline)
    return dict(item=item,record_reference=record_reference,scenario_id=scene.scenario_id,
        scenario_sha256=scene.scenario_sha256,plan_sha256=plan_identity(plan),template_sha256=t.template_sha256,
        physical_fingerprint=key,passed=True,kkt=kkt,metrics=recomputed['metrics'],costs=recomputed['costs'],
        timeseries_kw=recomputed['timeseries_kw'],energy_kwh=recomputed['energy_kwh'],na_reasons=recomputed['na_reasons'])


def validate_case0_oos_axis(root,lock,progress):
    """Exact declared768 item identities; no sample generation or matrix work."""
    root=Path(root).resolve();items=case0_items();expected={item['item_id'] for item in items}
    _need(lock['item_count']==768 and set(lock['method_plan_sha256'])=={'EA_DRO','DSP','OR_DRO','SAA','ARO','ZERO'},
          'Six fixed execution plans required')
    _need(set(lock['sample_ids_by_regime'])=={'R1','R2','R3'} and all(
        len(ids)==64 and len(set(ids))==64 for ids in lock['sample_ids_by_regime'].values()),'Paired64 sample axes differ')
    _need(set(progress['completed'])==expected and {p.name for p in (root/'evaluation/items').iterdir()}==expected,
          'OOS768 coverage incomplete/extra')
    for item in items:
        ref=progress['completed'][item['item_id']]
        _need(Path(ref['path'])==root/'evaluation/items'/item['item_id']/'record.json','Foreign OOS completed item')
    return items


def verify_evaluation_parent_settlement(reference,state):
    settlement_ref=reference;settlement=_read(settlement_ref)
    _need(settlement_ref==state.get('last_evaluation_settlement') and
          settlement['kind']=='CASE0_EVALUATION_PARENT_SETTLEMENT_V1' and
          type(settlement['cumulative_compute_seconds']) in (int,float) and
          math.isfinite(settlement['cumulative_compute_seconds']) and
          0<=settlement['cumulative_compute_seconds']<=state['compute_seconds'],
          'Original evaluation parent settlement differs or exceeds current ledger')
    for name in ('terminal_result_reference','parent_manifest_reference','reservation_reference'):
        verify_reference(settlement[name])
    return settlement_ref


def audit_case0_evaluation(request,deadline):
    """Read exact locked 768-item group, output new independent audit only."""
    from .case0_inputs import Case0Context
    from .case0_evaluation_runner import planning_evidence
    _time(deadline);root=Path(request['root']).resolve()
    _need(root.name in CAMPAIGNS.values(),'Independent Case0 root required')
    lockref=request['lock_reference'];lock=_read(lockref);progress=_read(request['evaluation_state_reference'])
    _need(Path(lockref['path'])==root/'evaluation/lock.json','Foreign Case0 evaluation lock')
    _need(lock['kind']=='CASE0_PRE_OOS_LOCK_V1' and lock['item_count']==768 and lock['formal_OOS'] is False
          and lock['Case1_acceptance'] is False,'Wrong Case0 lock scope')
    _need(progress['status']=='COMPLETE_PENDING_INDEPENDENT_EVALUATION_AUDIT' and progress['active_item'] is None
          and progress.get('active_preparation') is None and progress['lock_reference']==lockref,'Evaluation not fully completed/locked')
    items=validate_case0_oos_axis(root,lock,progress)
    acceptance_ref=request['planning_acceptance_reference'];_need(lock['acceptance_reference']==acceptance_ref,'Planning acceptance differs from pre-OOS lock')
    acceptance=_read(acceptance_ref);planning=_read(acceptance['numerical_audit_reference'])
    _need(acceptance['kind']=='CASE0_PLANNING_ACCEPTANCE_V1' and acceptance['passed'] is True and
          planning['kind']=='CASE0_INDEPENDENT_PLANNING_NUMERICAL_AUDIT_V1' and planning['passed'] is True,
          'Independent pre-OOS planning acceptance missing')
    for key in ('config_reference','original_config_reference','size_index','result_references','input_identity_references','artifact_references','method_evidence'):
        _need(acceptance[key]==planning[key],'Original planning acceptance audit scope differs')
    for key in ('original_master_coefficients_checked','full_q_kkt_physical_checked','risk_bounds_checked'):
        _need(planning[key] is True,'Original planning audit scope incomplete')
    for ref in planning['auditor_source_references']:_time(deadline);verify_reference(ref)
    oldreview=_read(planning['independent_review_reference'])
    _need(oldreview['passed'] is True and oldreview['kind']=='CASE0_NUMERICAL_AUDITOR_REVIEW_V1' and
          oldreview['auditor_source_references']==planning['auditor_source_references'],'Original planning auditor source review differs')
    for ref in acceptance['artifact_references']:_time(deadline);verify_reference(ref)
    config,state,evidence=planning_evidence(root)
    _need(lock_target(lock)==config['target_event'],'OOS lock target differs from scientific configuration')
    _need(evidence==acceptance['method_evidence'] and state['size_index']==lock['size_index']==acceptance['size_index'],
          'Original completed planning evidence changed')
    _need(lock['config_reference']==acceptance['config_reference'] and lock['original_config_reference']==acceptance['original_config_reference'],
          'Original lock configuration changed')
    _need(state['active_configuration'] is None and state.get('pending_evaluation_attempt') is None,
          'Evaluation parent budget not settled')
    settlement_ref=verify_evaluation_parent_settlement(request['evaluation_parent_settlement_reference'],state)
    producer=request['producer_source_manifest_reference'];manifest=_read(producer);producerroot=Path(producer['path']).parent
    producer_sources=acceptance['evaluation_source_references']
    inventory={r['path']:r['sha256'] for r in manifest['files']}
    for ref in producer_sources:
        _time(deadline);verify_reference(ref);p=Path(ref['path'])
        _need(p.is_relative_to(producerroot) and inventory.get(str(p.relative_to(producerroot)))==ref['sha256'],
              'Original evaluation source/manifest differs')
    evalreview=_read(acceptance['evaluation_review_reference'])
    _need(evalreview['kind']=='CASE0_EVALUATOR_SOURCE_REVIEW_V1' and evalreview['passed'] is True
          and evalreview['evaluation_source_references']==producer_sources,'Original evaluator was not source reviewed')
    sources=[file_reference(p) for p in sorted(Path(__file__).parent.rglob('*.py'))]
    review=_read(request['independent_review_reference'])
    auditor_manifest_ref=request['auditor_source_manifest_reference'];auditor_manifest=_read(auditor_manifest_ref)
    auditor_root=Path(auditor_manifest_ref['path']).parent
    auditor_inventory={r['path']:r['sha256'] for r in auditor_manifest['files']}
    _need(all(Path(ref['path']).is_relative_to(auditor_root) and
              auditor_inventory.get(str(Path(ref['path']).relative_to(auditor_root)))==ref['sha256'] for ref in sources),
          'Current auditor package differs from source manifest')
    policy_sources=policy_source_correspondence(producer,auditor_manifest_ref)
    _need(review['kind']=='CASE0_EVALUATION_AUDITOR_REVIEW_V1' and review['passed'] is True
          and review['auditor_source_references']==sources,'Current OOS auditor exact source not reviewed')
    for name in ('case0_inputs.py','case0_planning_audit.py','first_stage.py','first_stage_check.py','recourse.py','recourse_check.py',
                 'recourse_store.py','evaluation_metrics.py','random_streams.py'):
        _need(inventory.get('src/evcs_experiments/'+name)==file_reference(Path(__file__).parent/name)['sha256'],
              'Original numerical/reporting primitive changed: '+name)
    contexts={};plans={}
    for method in ('EA_DRO','OR_DRO','SAA','ARO'):
        _time(deadline);entry=evidence[method];identity=_read(entry['input_identity_reference'])
        _need(entry['input_identity_reference']==lock['input_identity_references'][method],'Locked input identity differs')
        directory=Path(entry['input_identity_reference']['path']).parent
        _need((directory/'tasks.json').is_file(),'Original tasks missing; audit cannot generate inputs')
        ctx=Case0Context(directory,case_root=request.get('case_root'),**config['sizes'][state['size_index']],
                         lead_hours=0 if method=='OR_DRO' else 1,**context_options(config))
        _need(ctx.identity==identity,'Reopened original input changed')
        contexts[method]=ctx;plans[method]=FirstStagePlan(**_read(entry['result_reference'])['plan'])
    plans['DSP']=plans['EA_DRO'];ea=plans['EA_DRO']
    plans['ZERO']=FirstStagePlan(np.zeros_like(ea.x),np.zeros_like(ea.r),ea.task_sha256,ea.config_hash)
    _need({m:plan_identity(p) for m,p in plans.items()}==lock['method_plan_sha256'],'Locked six fixed plans changed')
    _need(set(lock['sample_ids_by_regime'])=={'R1','R2','R3'},'Wrong regime lock')
    for regime in ('R1','R2','R3'):
        _need(lock['sample_ids_by_regime'][regime]==[contexts['EA_DRO'].sample_id(i,split='oos',regime=regime) for i in range(64)],
              'Original paired sample lock changed')
    cache=OriginalTemplateAuditCache();reports=[]
    for item in items:
        _time(deadline);method=item['method_id'];ctx=contexts['OR_DRO' if method=='OR_DRO' else 'EA_DRO']
        scene=ctx.scenario(item['sample_index'],split='oos',regime=item['regime'],departure_h=0 if method=='DSP' else None)
        ref=progress['completed'][item['item_id']]
        _need(Path(ref['path'])==root/'evaluation/items'/item['item_id']/'record.json','Foreign OOS completed item')
        reports.append(audit_case0_evaluation_item(record_reference=ref,item=item,lock_reference=lockref,lock=lock,
            plan=plans[method],case=ctx.case,tasks=ctx.tasks,scene=scene,producer_source_references=producer_sources,
            cache=cache,deadline=deadline,policy_sources=policy_sources))
    # Immutable snapshots and all row refs stay stable throughout; no producer file rewritten.
    for ref in (lockref,request['evaluation_state_reference'],acceptance_ref):verify_reference(ref)
    for report in reports:_time(deadline);verify_reference(report['record_reference'])
    for ref in cache.original_references.values():_time(deadline);verify_reference(ref)
    card=dict(kind='CASE0_INDEPENDENT_EVALUATION_NUMERICAL_AUDIT_V1',passed=True,item_count=768,
        lock_reference=lockref,evaluation_state_reference=request['evaluation_state_reference'],
        planning_acceptance_reference=acceptance_ref,evaluation_parent_settlement_reference=settlement_ref,producer_source_manifest_reference=producer,
        auditor_source_references=sources,auditor_source_manifest_reference=auditor_manifest_ref,
        policy_source_correspondence=policy_sources,independent_review_reference=request['independent_review_reference'],
        original_coefficients_checked=True,fixed_plan_rhs_checked=True,full_primal_dual_kkt_checked=True,
        physical_metrics_checked=True,paired_axes_checked=True,solver_calls=0,
        template_assemblies=cache.assemblies,template_cache_hits=cache.hits,items=reports,
        formal_OOS=False,Case1_acceptance=False,scope='Single-target Case0 development OOS only; no population superiority claim')
    _time(deadline)
    if request.get('output_path'):
        output=Path(request['output_path']).resolve()
        _need(not output.exists() and not output.is_relative_to(root/'evaluation') and not output.is_relative_to(root/'planning'),
              'Independent audit output must be new and outside original evidence')
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('x') as stream:json.dump(card,stream,indent=2,allow_nan=False);stream.write('\n')
    _time(deadline);return card
