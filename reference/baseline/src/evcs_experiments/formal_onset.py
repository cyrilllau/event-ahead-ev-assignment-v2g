"""Accepted onset-bank OR preparation; no input generation or OOS access.

The mandatory component card pins the independently audited lead0 H, all
planning/calibration atoms and their new moment calibration. The generic worker
separately admits current source and preserves the original cumulative budget.
"""
from pathlib import Path
import time
import numpy as np
from .formal_planning import (PreparedFormalProblem, BaselinePhysicalPreparation,
    RawPlanningEvidence, _need, _read, _verify)
from .ccg import CCGProblem
from .evidence import sha256
from .first_stage import assemble_first_stage
from .onset_physical_inputs import OnsetPhysicalBank
from .physical_inputs import load_physical_case, load_frozen_tasks
from .probability import ProbabilitySet
from .joint_scenarios import array_hash
from .lazy_recourses import LazyRecourses
from .random_streams import canonical_hash
from .recourse import assemble_recourse
from .scenario_features import FEATURE_NAMES


def validate_onset_acceptance(*, onset_acceptance_reference, summary_reference,
                              audit_reference, protocol_reference=None):
    """Verify the pinned component and its audit chain; never grants plan/OOS PASS."""
    acceptance=_read(onset_acceptance_reference)
    _need(acceptance.get('status')=='PASS_FORMAL_ONSET_INPUT_BANK_COMPONENT'
        and acceptance.get('formal_OOS') is False and acceptance.get('formal_plans_solved') is False
        and acceptance.get('whole_S05_PASS') is False,
        'OR requires independent onset input component acceptance')
    _need(acceptance.get('bank_summary')==summary_reference and acceptance.get('numerical_audit')==audit_reference,
        'Onset component acceptance binds a different summary/audit')
    for key in ('auditor_source','development_review','production_review','operational_completion'):_verify(acceptance[key])
    summary=_read(summary_reference);audit=_read(audit_reference)
    _need(audit.get('bank_kind')=='onset_v1' and type(audit.get('lead_h')) in (int,float) and audit['lead_h']==0.
        and audit.get('audit_source_sha256')==acceptance['auditor_source']['sha256'],
        'Onset audit kind/epoch/checker source differs')
    _need(audit['status']=='PASS_INDEPENDENT_FORMAL_BANK_AUDIT' and audit['source_summary_sha256']==summary_reference['sha256']
        and Path(audit['source_output']).resolve()==Path(summary_reference['path']).resolve().parent,
        'Independent audit does not bind this bank summary')
    _need(audit['formal_OOS'] is False and audit['formal_plans_solved'] is False and audit['scenario_count']==1920
        and audit.get('full_S05_pass') is False and summary.get('formal_plans_solved') is False
        and summary['atom_count']==1920 and summary['formal_OOS'] is False,'Wrong complete planning/calibration bank scope')
    if protocol_reference is not None:
        _need(summary['formal_protocol_lock']==protocol_reference,'Onset accepted protocol differs from worker protocol')
    _need(summary['status']=='COMPLETE_FORMAL_ONSET_INPUTS_PENDING_INDEPENDENT_AUDIT','Unknown bank summary schema/status')
    _need(audit['formal_protocol_lock']==summary['formal_protocol_lock']
        and audit['source_manifest_sha256']==summary['source_manifest']['sha256'],
        'Onset audit source/protocol binding differs')
    _verify(summary['formal_protocol_lock']);_verify(summary['source_manifest'])
    return acceptance


def prepare_onset_physics(*, summary_reference, audit_reference, onset_acceptance_reference, target_id, method_id,
                             case_root=None, template_index_reference=None,
                             template_association_review_reference=None):
    """Verify accepted full160 onset inputs without assembling templates."""
    started=time.monotonic();_need(method_id=='OR','Onset factory accepts OR only')
    validate_onset_acceptance(onset_acceptance_reference=onset_acceptance_reference,
        summary_reference=summary_reference,audit_reference=audit_reference)
    summary=_read(summary_reference);audit=_read(audit_reference)
    source=_read(summary['source_manifest']);envelope=_read(summary['formal_protocol_lock']);protocol=envelope['payload']
    _need(envelope.get('kind')=='protocol_lock' and envelope.get('payload_sha256')==canonical_hash(protocol),'Onset formal protocol envelope differs')
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
    _need(index['target_event']==target_id and index['lead_h']==0. and type(index['lead_h']) in (int,float),
        'Onset target/lead0 index differs')
    _need(index['task_source']==task_ref and index['split']=='planning','Frozen task source/planning split differs')
    rawtasks=_read(task_ref)
    _need(len(rawtasks['vehicle_summaries'])==80 and rawtasks['task_count']==row['H'], 'Full frozen fleet/H differs')
    case=load_physical_case(case_root);tasks=load_frozen_tasks(_verify(task_ref),case,task_ref['sha256'])
    _need(len(tasks.task_ids)==row['H'] and tasks.config['basic_soc']==.3 and tasks.tau_blk_h==1.,'Full baseline H changed')
    bank=OnsetPhysicalBank(row['planning_index']['path'],case,tasks,expected_index_sha256=row['planning_index']['sha256'],case_root=case_root)
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
    binding=dict(onset_acceptance=onset_acceptance_reference,information_contract='OR_ONSET_SAME_KNOWN_FORECAST_v1',lead_h=0.,summary=summary_reference,audit=audit_reference,target=row,task_source=task_ref,
        formal_protocol_lock=summary['formal_protocol_lock'],bank_source_manifest=summary['source_manifest'],
        planning_source={p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        physical_case_sha256=case.case_sha256,task_sha256=tasks.task_sha256,config_hash=tasks.config_hash,
        assignment_state_sha256=index['assignment_state_sha256'],method_id=method_id,rho=1.,formal_OOS=False)
    if provider is not None:binding['template_provider']=provider.identity
    def load(i):
        for ref in (onset_acceptance_reference,summary_reference,audit_reference,row['planning_index'],row['moments'],task_ref):_verify(ref)
        scene=bank.load(i,expected_scenario_sha256=bank.scenario_sha256s[i])
        return provider.load(i,scene) if provider is not None else assemble_recourse(case,tasks,scene)
    ambiguity=ProbabilitySet(bank.scenario_ids,FEATURE_NAMES,np.asarray(features),bounds['lower'],bounds['upper'])
    return BaselinePhysicalPreparation(case,tasks,bank,first,ambiguity,binding,load,provider,time.monotonic()-started)


def load_formal_onset_problem(*, summary_reference, audit_reference, onset_acceptance_reference, target_id, method_id,
                        evidence_root, case_root=None, resume_evidence=False, resume_snapshot_reference=None,
                        template_index_reference=None, template_association_review_reference=None):
    """Prepare OR with its own onset state/H/moments and the common DRO engine."""
    started=time.monotonic()
    _need(method_id=='OR','Onset factory accepts OR only')
    prepared=prepare_onset_physics(summary_reference=summary_reference,audit_reference=audit_reference,
        onset_acceptance_reference=onset_acceptance_reference,target_id=target_id,method_id=method_id,case_root=case_root,
        template_index_reference=template_index_reference,
        template_association_review_reference=template_association_review_reference)
    first,bank,binding=prepared.first_stage,prepared.bank,prepared.input_binding
    holder={}
    def load(i):
        template=prepared.template_loader(i)
        holder['evidence'].template(i,template)
        return template
    cache=LazyRecourses(bank.scenario_ids,bank.scenario_sha256s,first.matrix_sha256,binding,load,capacity=2)
    problem=CCGProblem(first,cache,prepared.moment_ambiguity,risk_model='DRO')
    evidence=RawPlanningEvidence(evidence_root,prepared.case,prepared.tasks,problem,bank.load,
        domain='formal',input_binding=binding,resume=resume_evidence,
        resume_snapshot_reference=resume_snapshot_reference,template_provider=prepared.template_provider)
    holder['evidence']=evidence
    return PreparedFormalProblem(prepared.case,prepared.tasks,bank,problem,evidence,binding,time.monotonic()-started)
