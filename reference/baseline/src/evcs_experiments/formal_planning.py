"""Audited baseline EA/ARO problem preparation and lossless C&CG evidence.

No solver or worker starts here. Production callers must separately require the
bank component acceptance, hold the campaign queue lock, and count preparation,
export, failed attempts and previous calls in the cumulative planning budget.
"""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass, fields
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse import save_npz

from .ccg import CCGProblem
from .config import get_paths
from .evaluation_lock import file_reference
from .evidence import sha256
from .first_stage import assemble_first_stage
from .first_stage_check import check_first_stage
from .formal_physical_inputs import FormalPhysicalBank
from .joint_scenarios import array_hash
from .lazy_recourses import LazyRecourses
from .physical_inputs import load_physical_case, load_frozen_tasks
from .probability import ProbabilitySet
from .probability_check import check_probability_solution
from .random_streams import canonical_hash
from .recourse import assemble_recourse
from .recourse_check import check_recourse
from .runtime.jobctl import atomic_json
from .scenario_features import FEATURE_NAMES


def _need(value, message):
    if not bool(value): raise ValueError(message)


def _serial(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Mapping): return {k:_serial(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [_serial(v) for v in value]
    return value


def _write(path, value):
    with Path(path).open('x') as f: json.dump(_serial(value),f,indent=2,allow_nan=False);f.write('\n')
    return file_reference(path)


def _verify(ref):
    path=Path(ref['path'])
    _need(path.is_absolute() and sha256(path)==ref['sha256'], 'Bound planning evidence changed: '+str(path))
    if 'size_bytes' in ref: _need(path.stat().st_size==ref['size_bytes'],'Bound file size changed')
    if 'bytes' in ref: _need(path.stat().st_size==ref['bytes'],'Bound file size changed')
    return path


def _read(ref): return json.loads(_verify(ref).read_text())


def _seal(folder, metadata):
    refs=[file_reference(p) for p in sorted(folder.iterdir()) if p.is_file()]
    return _write(folder/'record.json',dict(**metadata,artifacts=refs))


def _verified_record(path):
    path=Path(path)
    _need(path.is_absolute() and path.resolve()==path and path.is_file(),
          'Raw record must use its canonical absolute file path')
    value=json.loads(path.read_text())
    refs=value['artifacts']
    _need(all(Path(ref['path']).is_absolute() and Path(ref['path']).resolve()==Path(ref['path'])
              and Path(ref['path']).parent==path.parent and Path(ref['path']).is_file() for ref in refs)
          and len({ref['path'] for ref in refs})==len(refs),
          'Raw artifacts must be unique canonical files in their record directory')
    _need({Path(ref['path']).name for ref in refs}=={p.name for p in Path(path).parent.iterdir() if p.name!='record.json'},
        'Raw evidence directory is incomplete or contains unrecorded files')
    for ref in refs:_verify(ref)
    if value.get('kind') == 'recourse_template_reference_v1':
        from .recourse_template_index import resolve_template_reference
        _need(len(refs)==1 and Path(refs[0]['path']).name=='template_source.json',
              'Invalid template reference inventory')
        source=_read(refs[0]);original=resolve_template_reference(source)
        _need(all(value[key]==original[key] for key in ('scenario_index','scenario_id','template_sha256')),
              'Template reference metadata differs from original')
        return original
    return value


def _save_solution(path, solution):
    np.savez_compressed(path,**{field.name:getattr(solution,field.name) for field in fields(solution)})


def _plan(plan):return dict(x=plan.x,r=plan.r,task_sha256=plan.task_sha256,config_hash=plan.config_hash)


class RawPlanningEvidence:
    """Append-only original matrices, witnesses and checks for every callback.

    ``domain='development'`` is reserved for explicit synthetic unit fixtures.
    This exporter cannot confer acceptance or unlock formal OOS. Restarting
    requires the same problem identity and verifies every saved artifact first.
    """
    def __init__(self, root, case, tasks, problem, scene_loader, *, domain, input_binding, resume=False, resume_snapshot_reference=None, template_provider=None):
        _need(domain in ('formal','development'), 'Explicit evidence domain required')
        self.root=Path(root).resolve();self.case=case;self.tasks=tasks;self.problem=problem;self.scene=scene_loader
        self.ids=problem.probability_set.support_ids;self.domain=domain;self.q={};self.events=[]
        self.template_provider=template_provider
        if template_provider is not None:
            _need(input_binding.get('template_provider')==template_provider.identity,
                  'Evidence input must bind the reviewed template provider')
        request=dict(schema_version=1,domain=domain,problem_identity=problem.identity,input_binding=input_binding,
            exporter_sha256=sha256(__file__),formal_OOS=False,independent_scientific_audit_required=True)
        if resume:
            _need(resume_snapshot_reference is not None, 'Resume requires externally pinned evidence snapshot')
            snapshot=_read(resume_snapshot_reference)
            _need(snapshot['evidence_root']==str(self.root), 'Evidence snapshot origin changed')
            for ref in [snapshot['request'],snapshot['first_stage'],*snapshot['templates'],*snapshot['events']]:
                _verify(ref)
                if Path(ref['path']).name=='record.json':_verified_record(ref['path'])
            _need(snapshot['request']==file_reference(self.root/'request.json') and
                snapshot['first_stage']==file_reference(self.root/'first_stage'/'record.json'), 'Snapshot fixed source changed')
            _need(snapshot['templates']==[file_reference(p) for p in sorted((self.root/'templates').glob('*/record.json'))],
                'Snapshot template inventory changed')
            _need(snapshot['events']==json.loads((self.root/'history.json').read_text()), 'Current raw history differs from pinned snapshot')
            _need(json.loads((self.root/'request.json').read_text())==_serial(request),'Raw evidence resume identity changed')
            history=json.loads((self.root/'history.json').read_text())
            for ref in history:_verified_record(_verify(ref))
            _need([r['path'] for r in history]==[str(p) for p in sorted((self.root/'events').glob('*/record.json'))], 'Raw callback history differs from journal')
            for i,path in enumerate(sorted((self.root/'events').glob('*/record.json'))):
                _need(path.parent.name.split('_')[0]==f'{i:06d}','Raw callback history was truncated')
                row=_verified_record(path);ref=file_reference(path);self.events.append(ref);self._retain_q({**row,'event_reference':ref})
            _need(len(list((self.root/'events').iterdir()))==len(self.events),'Uncommitted callback requires diagnosis')
            _need(len(list((self.root/'templates').iterdir()))==len(snapshot['templates']),'Uncommitted template requires diagnosis')
            for path in (self.root/'templates').glob('*/record.json'):self._verified_template_record(path)
            _verified_record(self.root/'first_stage'/'record.json')
        else:
            self.root.mkdir(parents=True,exist_ok=False);(self.root/'events').mkdir();(self.root/'templates').mkdir()
            _write(self.root/'request.json',request);_write(self.root/'history.json',[])
            folder=self.root/'first_stage';folder.mkdir();first=problem.first_stage
            save_npz(folder/'matrix.npz',first.A)
            np.savez_compressed(folder/'arrays.npz',**{k:getattr(first,k) for k in
                ('row_lower','row_upper','lower_bounds','upper_bounds','integrality','c')})
            _write(folder/'identity.json',dict(matrix_sha256=first.matrix_sha256,task_ids=first.task_ids,periods=first.periods,
                task_sha256=first.task_sha256,config_hash=first.config_hash,variable_map=first.variable_map,
                constraint_map=first.constraint_map,units=first.units))
            _seal(folder,dict(kind='first_stage'))
        self.request_sha256=sha256(self.root/'request.json')

    def _verified_template_record(self, path):
        row=_verified_record(path)
        local=json.loads(Path(path).read_text())
        if self.template_provider is None:
            _need(local['kind']=='recourse_template', 'Template pointer lacks this run admission')
        else:
            _need(local['kind']=='recourse_template_reference_v1', 'Reviewed run requires admitted template pointers')
            source=_read(local['artifacts'][0])
            expected=self.template_provider.identity
            _need(all(source.get(key)==value for key,value in expected.items()),
                  'Template pointer admission differs from this run')
        return row

    def snapshot(self, path):
        """Write an immutable complete evidence envelope outside mutable state.

        The worker pins the returned reference in its immutable attempt receipt.
        Future callbacks append new files and leave this snapshot valid.
        """
        _need(sha256(self.root/'request.json')==self.request_sha256,'Evidence request changed')
        templates=[file_reference(p) for p in sorted((self.root/'templates').glob('*/record.json'))]
        _need(len(templates)==len(list((self.root/'templates').iterdir())), 'Uncommitted template requires diagnosis')
        _need(self.events==json.loads((self.root/'history.json').read_text()), 'Callback journal differs')
        _need(len(list((self.root/'events').iterdir()))==len(self.events),'Uncommitted callback requires diagnosis')
        _need([r['path'] for r in self.events]==[str(p) for p in sorted((self.root/'events').glob('*/record.json'))], 'Callback inventory differs')
        first=file_reference(self.root/'first_stage'/'record.json')
        for ref in [first,*self.events]:_verified_record(_verify(ref))
        for ref in templates:self._verified_template_record(_verify(ref))
        return _write(path,dict(schema_version=1,evidence_root=str(self.root),request=file_reference(self.root/'request.json'),
            first_stage=first,templates=templates,events=self.events,
            status='PENDING_INDEPENDENT_NUMERICAL_AUDIT',full_support_verified=False,formal_OOS=False))

    def _retain_q(self,row):
        if row.get('event')=='recourse' and row.get('status')=='RAW_EVIDENCE_RETAINED_PENDING_AUDIT' and row.get('optimal_certificate_valid') is True:
            self.q.setdefault(row['plan_sha256'],{})[row['scenario_index']]=(row['q_usd'],row['event_reference'])

    def template(self,index,template):
        _need(template.scenario_sha256==self.problem.recourses.scenario_sha256s[index], 'Template scenario differs')
        folder=self.root/'templates'/f'{index:03d}'
        if folder.exists():
            row=self._verified_template_record(folder/'record.json')
            _need(row['template_sha256']==template.template_sha256,'Reassembled coefficient identity changed')
            if self.template_provider is not None:
                source=self.template_provider.evidence_reference(index,template)
                _need(_read(file_reference(folder/'template_source.json'))==source,
                      'Existing template reference admission changed')
                return source['original_template_reference']
            return file_reference(folder/'record.json')
        if self.template_provider is not None:
            from .recourse_template_index import REFERENCE_KIND, resolve_template_reference
            source=self.template_provider.evidence_reference(index,template)
            resolve_template_reference(source)
            folder.mkdir()
            _write(folder/'template_source.json',source)
            _seal(folder,dict(kind=REFERENCE_KIND,scenario_index=index,scenario_id=self.ids[index],
                              template_sha256=template.template_sha256))
            return source['original_template_reference']
        folder.mkdir()
        for name in ('A_ub','A_eq','Bx_ub','Bx_eq','Br_ub','Br_eq'):save_npz(folder/(name+'.npz'),getattr(template,name))
        np.savez_compressed(folder/'arrays.npz',**{k:getattr(template,k) for k in
            ('b_ub','b_eq','c','lower_bounds','upper_bounds','component_objectives')})
        _write(folder/'identity.json',template.as_dict())
        return _seal(folder,dict(kind='recourse_template',scenario_index=index,scenario_id=self.ids[index],
            template_sha256=template.template_sha256))

    def sink_factory(self, call_directory):
        call_directory=Path(call_directory).resolve();call_directory.mkdir(parents=True,exist_ok=True)
        call_refs=[]
        def sink(event,iteration,**data):
            _need(event in ('rmp','recourse','probability','phase1','iteration'),'Unknown C&CG callback; evidence cannot be discarded')
            _need(sha256(self.root/'request.json')==self.request_sha256,'Raw evidence request changed')
            folder=self.root/'events'/f'{len(self.events):06d}_{event}';folder.mkdir()
            meta=dict(event=event,iteration=iteration,driver_call_directory=str(call_directory),domain=self.domain)
            try:
                self._export(folder,meta,event,data)
            except Exception as exc:
                _write(folder/'failure.json',dict(error_type=type(exc).__name__,error=str(exc)))
                ref=_seal(folder,dict(**meta,status='FAILED_EVIDENCE_CHECK'))
                self.events.append(ref);call_refs.append(ref);atomic_json(call_directory/'raw_events.json',call_refs);atomic_json(self.root/'history.json',self.events)
                raise
            ref=_seal(folder,dict(**meta,status='RAW_EVIDENCE_RETAINED_PENDING_AUDIT'))
            self.events.append(ref);call_refs.append(ref);atomic_json(call_directory/'raw_events.json',call_refs);atomic_json(self.root/'history.json',self.events)
            self._retain_q({**meta,'status':'RAW_EVIDENCE_RETAINED_PENDING_AUDIT','event_reference':ref})
        return sink

    def _scene(self,i):
        scene=self.scene(i)
        _need(scene.development is (self.domain=='development') and scene.scenario_id==self.ids[i]
            and scene.scenario_sha256==self.problem.recourses.scenario_sha256s[i], 'Physical scene domain/support differs')
        return scene

    def _export(self,folder,meta,event,data):
        problem=self.problem;a=problem.ambiguity
        if event=='rmp':
            model,result=data['model'],data['result']
            _write(folder/'model.json',model.as_dict());_write(folder/'solver.json',result.as_dict())
            save_npz(folder/'matrix.npz',model.A)
            np.savez_compressed(folder/'model_arrays.npz',**{k:getattr(model,k) for k in
                ('row_lower','row_upper','lower_bounds','upper_bounds','c','integrality','objective_constant_usd')})
            _need(model.first_stage.matrix_sha256==problem.first_stage.matrix_sha256 and model.risk_model==problem.risk_model,
                'RMP problem identity changed')
            refs=[]
            for sid,template in zip(model.scenario_ids,model.recourses):refs.append(self.template(self.ids.index(sid),template))
            _write(folder/'template_references.json',refs)
            if 'native_interface' in result.raw_solver and result.raw_solver.get('mip_start_supplied',True):
                from .extensive import full_shedding_start
                np.savez_compressed(folder/'mip_start.npz',vector=full_shedding_start(model))
            if result.solution is not None:
                solution=result.solution;plan=solution.plan
                np.savez_compressed(folder/'solution.npz',vector=solution.vector,x=plan.x,r=plan.r)
                _write(folder/'solution.json',solution.as_dict())
                check=check_first_stage(self.case,self.tasks,plan);_write(folder/'first_stage_check.json',check)
                _need(check['passed'],'Original first-stage check failed')
                _need(len(solution.recourse_solutions)==len(model.scenario_ids),'RMP witness count changed')
                for sid,y in zip(model.scenario_ids,solution.recourse_solutions):
                    i=self.ids.index(sid);_save_solution(folder/f'witness_{i:03d}.npz',y)
                    check=check_recourse(self.case,self.tasks,self._scene(i),plan,y,check_objective_bound=False)
                    _write(folder/f'physical_{i:03d}.json',check);_need(check['passed'],'Original RMP physical check failed')
        elif event=='recourse':
            i,plan,result=data['index'],data['plan'],data['result'];template=problem.recourses[i]
            _need(type(i) is int and 0<=i<len(self.ids),'Invalid recourse callback index')
            _write(folder/'solver.json',result.as_dict());_write(folder/'plan.json',_plan(plan))
            _write(folder/'template_reference.json',self.template(i,template))
            bub,beq=template.fixed_rhs(plan);np.savez_compressed(folder/'fixed_rhs.npz',b_ub=bub,b_eq=beq)
            plan_hash=canonical_hash(_serial(_plan(plan)))
            meta.update(scenario_index=i,scenario_id=self.ids[i],plan_sha256=plan_hash,
                optimal_certificate_valid=bool(result.optimal and result.certificate.get('optimal_certificate_valid')),
                q_usd=None if result.objective_usd is None else float(result.objective_usd+problem.scenario_cost_constants_usd[i]))
            if result.solution is not None:
                _save_solution(folder/'solution.npz',result.solution)
                check=check_recourse(self.case,self.tasks,self._scene(i),plan,result.solution,check_objective_bound=result.optimal)
                _write(folder/'physical_check.json',check);_need(check['passed'],'Original fixed-plan physical check failed')
                vector=template.pack(result.solution);np.savez_compressed(folder/'primal_vector.npz',vector=vector)
                if result.optimal:
                    cert=result.certificate
                    dual={k:np.asarray(cert['dual_'+k+'_marginals']) for k in ('inequality','equality','lower','upper')}
                    stationarity=template.c-template.A_ub.T@dual['inequality']-template.A_eq.T@dual['equality']-dual['lower']-dual['upper']
                    np.savez_compressed(folder/'dual_kkt.npz',**dual,stationarity=stationarity,
                        inequality_slack=bub-template.A_ub@vector,equality_residual=template.A_eq@vector-beq,
                        inequality_complementarity=dual['inequality']*(bub-template.A_ub@vector),
                        lower_complementarity=dual['lower'][np.isfinite(template.lower_bounds)]*(vector-template.lower_bounds)[np.isfinite(template.lower_bounds)],
                        upper_complementarity=dual['upper'][np.isfinite(template.upper_bounds)]*(template.upper_bounds-vector)[np.isfinite(template.upper_bounds)])
        elif event in ('phase1','probability'):
            result=data['result'];_write(folder/'solver.json',result.as_dict())
            _write(folder/'probability_input.json',dict(input_ambiguity=problem.probability_set.as_dict(),effective_ambiguity=a.as_dict()))
            _need(tuple(result.support_ids)==self.ids and result.ambiguity_sha256==a.ambiguity_sha256,'Probability identity changed')
            if event=='probability':
                plan=data['plan'];plan_hash=canonical_hash(_serial(_plan(plan)));values=self.q.get(plan_hash,{})
                _need(set(values)==set(range(len(self.ids))),'Full-support raw recourse evidence is incomplete')
                q=[];refs=[]
                for i in range(len(self.ids)):
                    value,ref=values[i];row=_verified_record(_verify(ref));_need(row['status']=='RAW_EVIDENCE_RETAINED_PENDING_AUDIT','Invalid retained recourse')
                    q.append(value);refs.append(ref)
                _write(folder/'q_input.json',dict(support_ids=self.ids,q_usd=q,plan=_plan(plan),recourse_evidence=refs))
                check=check_probability_solution(a,q,result,q_support_ids=self.ids)
                _write(folder/'probability_check.json',check.as_dict())
                if result.certified_optimal:_need(check.passed,'Independent probability check failed')
        else:_write(folder/'trace.json',data['record'])


@dataclass(frozen=True)
class PreparedFormalProblem:
    case: object
    tasks: object
    bank: FormalPhysicalBank
    problem: CCGProblem
    evidence: RawPlanningEvidence
    input_binding: dict
    preparation_seconds: float

    @property
    def sink_factory(self):return self.evidence.sink_factory


@dataclass(frozen=True)
class BaselinePhysicalPreparation:
    """Accepted physical inputs before choosing the optimization objective.

    No C&CG object or raw C&CG callback is created for SAA. Template reuse is
    physical only; method identity stays in each fresh optimization binding.
    """
    case: object
    tasks: object
    bank: FormalPhysicalBank
    first_stage: object
    moment_ambiguity: ProbabilitySet
    input_binding: dict
    template_loader: object
    template_provider: object
    preparation_seconds: float


def prepare_baseline_physics(*, summary_reference, audit_reference, target_id, method_id,
                             case_root=None, template_index_reference=None,
                             template_association_review_reference=None):
    """Verify the accepted full160 baseline bank without assembling templates."""
    started=time.monotonic();_need(method_id in ('EA','ARO','SAA'),'Only baseline EA, ARO and SAA physics implemented')
    summary=_read(summary_reference);audit=_read(audit_reference)
    _need(audit['status']=='PASS_INDEPENDENT_FORMAL_BANK_AUDIT' and audit['source_summary_sha256']==summary_reference['sha256']
        and Path(audit['source_output']).resolve()==Path(summary_reference['path']).resolve().parent,
        'Independent audit does not bind this bank summary')
    _need(audit['formal_OOS'] is False and audit['formal_plans_solved'] is False and audit['scenario_count']==1920
        and summary['atom_count']==1920 and summary['formal_OOS'] is False,'Wrong complete planning/calibration bank scope')
    _need(summary['status']=='COMPLETE_FORMAL_BASELINE_INPUTS_PENDING_INDEPENDENT_AUDIT','Unknown bank summary schema/status')
    source=_read(summary['source_manifest']);protocol=_read(summary['formal_protocol_lock'])['payload']
    source_root=Path(summary['source_manifest']['path']).resolve().parent
    _need(isinstance(source['files'],list) and bool(source['files']), 'Empty source manifest')
    for member in source['files']:
        path=(source_root/member['path']).resolve()
        _need(path.is_relative_to(source_root) and sha256(path)==member['sha256'],'Audited release source bytes changed')
    _need(audit['source_manifest_sha256']==summary['source_manifest']['sha256'] and
        audit['formal_protocol_lock']==summary['formal_protocol_lock'],'Audited source/protocol binding differs')
    for key in ('s01_acceptance','s04_acceptance'):_read(audit[key])
    _need(protocol['domain']=='formal' and list(summary['task_sources'])==protocol['target_ids']
        and len(summary['targets'])==8 and [r['target_event'] for r in summary['targets']]==protocol['target_ids'],
        'Full target bank order differs from frozen protocol')
    config=protocol['configuration'];baseline=config['baseline']
    _need(baseline['support_S']==160 and baseline['fleet_E']==80 and baseline['rho']==1 and baseline['lead_h']==1
        and baseline['SOC_arr_min']==.3 and baseline['tau_blk_h']==1,'Unsupported formal baseline settings')
    rows=[row for row in summary['targets'] if row['target_event']==target_id];_need(len(rows)==1,'Unknown target')
    row=rows[0];_need(row['planning_count']==160 and row['calibration_count']==80,'Full support count differs')
    for key in ('planning_index','calibration_index','moments','bootstrap'):_verify(row[key])
    index=_read(row['planning_index']);task_ref=summary['task_sources'][target_id]
    _need(index['task_source']==task_ref and index['split']=='planning','Frozen task source/planning split differs')
    rawtasks=_read(task_ref)
    _need(len(rawtasks['vehicle_summaries'])==80 and rawtasks['task_count']==row['H'], 'Full frozen fleet/H differs')
    case=load_physical_case(case_root);tasks=load_frozen_tasks(_verify(task_ref),case,task_ref['sha256'])
    _need(len(tasks.task_ids)==row['H'] and tasks.config['basic_soc']==.3 and tasks.tau_blk_h==1.,'Full baseline H changed')
    bank=FormalPhysicalBank(row['planning_index']['path'],case,tasks,expected_index_sha256=row['planning_index']['sha256'],case_root=case_root)
    features=[]
    for item in index['scenarios']:
        manifest=_read(item['manifest']);ledger=_read(manifest['files']['ledger']);p=_verify(manifest['files']['arrays'])
        _verify(manifest['files']['diagnostics'])
        _need(manifest['scenario_id']==ledger['scenario_id']==item['scenario_id'] and
            manifest['scenario_sha256']==ledger['scenario_sha256']==item['scenario_sha256'] and
            canonical_hash({k:v for k,v in ledger.items() if k!='scenario_sha256'})==item['scenario_sha256'],
            'Ordered feature atom semantic identity differs')
        with np.load(p,allow_pickle=False) as arrays:feature=arrays['features'].copy()
        _need(array_hash(feature)==ledger['array_hashes']['features'] and feature.shape==(5,),'Feature input hash/shape differs')
        features.append(feature)
    moment=_read(row['moments']);_need(moment['status']=='PASS_MOMENT_CALIBRATION' and moment['formal_OOS'] is False,'Unaccepted moment calibration')
    bounds=moment['bounds_by_rho']['1.0'];_need(bounds['rho']==1. and bounds['feasibility']['valid_witness'],'rho1 lacks feasibility witness')
    first=assemble_first_stage(case,tasks)
    _need((template_index_reference is None)==(template_association_review_reference is None),
          'Both template index and independent association review are required')
    provider=None
    if template_index_reference is not None:
        from .recourse_template_index import TemplateProvider
        provider=TemplateProvider(template_index_reference,template_association_review_reference,
            first_stage=first,case=case,tasks=tasks,scenario_ids=bank.scenario_ids,
            scenario_sha256s=bank.scenario_sha256s,domain='formal')
    binding=dict(summary=summary_reference,audit=audit_reference,target=row,task_source=task_ref,
        formal_protocol_lock=summary['formal_protocol_lock'],bank_source_manifest=summary['source_manifest'],
        planning_source={p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        physical_case_sha256=case.case_sha256,task_sha256=tasks.task_sha256,config_hash=tasks.config_hash,
        assignment_state_sha256=index['assignment_state_sha256'],method_id=method_id,rho=1.,formal_OOS=False)
    if provider is not None:binding['template_provider']=provider.identity
    def load(i):
        for ref in (summary_reference,audit_reference,row['planning_index'],row['moments'],task_ref):_verify(ref)
        scene=bank.load(i,expected_scenario_sha256=bank.scenario_sha256s[i])
        return provider.load(i,scene) if provider is not None else assemble_recourse(case,tasks,scene)
    ambiguity=ProbabilitySet(bank.scenario_ids,FEATURE_NAMES,np.asarray(features),bounds['lower'],bounds['upper'])
    return BaselinePhysicalPreparation(case,tasks,bank,first,ambiguity,binding,load,provider,time.monotonic()-started)


def load_formal_problem(*, summary_reference, audit_reference, target_id, method_id,
                        evidence_root, case_root=None, resume_evidence=False, resume_snapshot_reference=None,
                        template_index_reference=None, template_association_review_reference=None):
    """Prepare the existing baseline EA/ARO C&CG and lossless raw exporter."""
    started=time.monotonic()
    _need(method_id in ('EA','ARO'),'Only baseline EA and ARO C&CG factory implemented')
    prepared=prepare_baseline_physics(summary_reference=summary_reference,audit_reference=audit_reference,
        target_id=target_id,method_id=method_id,case_root=case_root,
        template_index_reference=template_index_reference,
        template_association_review_reference=template_association_review_reference)
    first,bank,binding=prepared.first_stage,prepared.bank,prepared.input_binding
    holder={}
    def load(i):
        template=prepared.template_loader(i)
        holder['evidence'].template(i,template)
        return template
    cache=LazyRecourses(bank.scenario_ids,bank.scenario_sha256s,first.matrix_sha256,binding,load,capacity=2)
    problem=CCGProblem(first,cache,prepared.moment_ambiguity,risk_model='DRO' if method_id=='EA' else 'ARO')
    evidence=RawPlanningEvidence(evidence_root,prepared.case,prepared.tasks,problem,bank.load,
        domain='formal',input_binding=binding,resume=resume_evidence,
        resume_snapshot_reference=resume_snapshot_reference,template_provider=prepared.template_provider)
    holder['evidence']=evidence
    return PreparedFormalProblem(prepared.case,prepared.tasks,bank,problem,evidence,binding,time.monotonic()-started)
