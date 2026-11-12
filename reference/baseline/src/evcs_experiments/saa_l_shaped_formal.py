"""Source-bound formal input adapter for the fixed-empirical SAA decomposition.

No extensive model is assembled. All 160 original numeric templates are read in
order with one resident recourse. Input preflight is not solver admission: the
original formal queue prerequisites, Linux parent accounting, candidate audit,
and source-specific dispatch admission remain required outside this module.
"""
from collections.abc import Sequence
import json
from pathlib import Path
import time
import numpy as np
from .campaign_batch import file_reference, verify_reference
from .config import get_paths
from .formal_saa import PreparedFormalSAA, prepare_formal_saa
from .formal_physical_inputs import FormalPhysicalBank
from .lazy_recourses import LazyRecourses
from .planning_budget import validate_budget
from .random_streams import canonical_hash
from .recourse_template_index import TemplateProvider
from .saa_l_shaped import DevelopmentSAAProblem, need
from .saa_l_shaped_provider import DiskAtomStore

FORMAL_ALGORITHM_ID='SAA_FIXED_EMPIRICAL_MULTICUT_L_SHAPED_FORMAL_V1'
PREFLIGHT_KIND='SAA_L_SHAPED_FORMAL_INPUT_PREFLIGHT_V1'
_TOKEN=object()


def _read(reference):
    verify_reference(reference)
    path=Path(reference['path'])
    need(path.is_absolute() and path.resolve()==path,'Canonical source reference required')
    return json.loads(path.read_text())


def _sources():
    return [file_reference(p) for p in sorted(Path(__file__).parent.rglob('*.py'))]


def _worker_input(reference):
    """Validate the immutable input/source contract without impersonating jobctl."""
    request=_read(reference)
    need(request.get('schema_version')==1 and request.get('domain')=='formal'
         and request.get('model_id')=='IEEE33-v54-Q1' and request.get('spec_id')=='S05'
         and request.get('method_id')=='SAA','Original formal SAA worker request required')
    need(request.get('planning_source_references')==_sources(),'Formal input implementation source closure differs')
    for r in request['planning_source_references']:verify_reference(r)
    budget=validate_budget(request['budget'])
    need(budget==dict(cumulative_wall_seconds=7200,single_master_seconds=600,
         final_certificate_reserve_seconds=1790,shutdown_reserve_seconds=10,
         master_threads=4,recourse_workers=1,blas_threads=1,normal_gap_relative=.005,rmp_mip_gap=.001),
         'Original formal planning budget differs')
    config=_read(request['config_reference']);protocol=_read(request['protocol_reference'])['payload']
    need(protocol['domain']=='formal' and protocol['configuration']==config
         and config['planning_budget']==budget and request['target_id'] in protocol['target_ids'],
         'Frozen formal input config/protocol differs')
    card=_read(request['bank_acceptance_reference'])
    need(card['status']=='PASS_COMPONENT' and card['component_id']=='FORMAL_BASELINE_INPUTS'
         and card['spec_id']=='S05' and card['increment_id']=='S05-I01'
         and card['scenario_count']==1920 and card['target_count']==8
         and card['formal_OOS'] is False and card['formal_plans_solved'] is False
         and card['full_S05_pass'] is False,'Original accepted baseline input scope differs')
    for k in ('summary_reference','audit_reference','source_manifest_reference','protocol_reference',
              'independent_review_reference','job_result_reference','artifact_manifest_reference'):
        verify_reference(card[k])
    for k in ('summary_reference','audit_reference','protocol_reference'):
        need(card[k]==request[k],'Original accepted bank does not bind requested source')
    summary=_read(request['summary_reference']);audit=_read(request['audit_reference'])
    need(card['source_manifest_reference']==summary['source_manifest']
         and summary['formal_protocol_lock']==request['protocol_reference']
         and audit['source_summary_sha256']==request['summary_reference']['sha256']
         and audit['source_manifest_sha256']==card['source_manifest_reference']['sha256']
         and audit['formal_protocol_lock']==request['protocol_reference']
         and audit['status']=='PASS_INDEPENDENT_FORMAL_BANK_AUDIT',
         'Formal bank audit/source/protocol binding differs')
    need(request.get('template_index_reference') is not None
         and request.get('template_association_review_reference') is not None,
         'Formal decomposition requires reviewed original numeric templates; no reassembly fallback')
    return request


def _physical_identity(problem):
    index=problem.provider.verify()
    return dict(case_sha256=problem.case.case_sha256,task_sha256=problem.tasks.task_sha256,
        config_hash=problem.tasks.config_hash,first_stage_sha256=problem.first_stage.matrix_sha256,
        scenario_ids=list(problem.support_ids),scenario_sha256s=list(problem.recourses.scenario_sha256s),
        template_sha256s=[r['template_sha256'] for r in index['rows']],
        constants_usd=problem.constants_usd.tolist(),empirical_probabilities=[1/160]*160)


def validate_formal_input_reference(reference, *, expected_request=None, expected_identity=None):
    """Read a caller-retained preflight pin; never infer a pin from mutable bytes."""
    body=_read(reference)
    need(body['kind']==PREFLIGHT_KIND and body['domain']=='formal'
         and body['algorithm_id']==FORMAL_ALGORITHM_ID and body['formal_admission'] is False
         and body['scientific_acceptance'] is False and body['formal_OOS'] is False
         and body['solver_calls']==0 and body['original_model_unchanged'] is True,
         'Formal input preflight is not a solver or production acceptance')
    request=_worker_input(body['worker_request_reference'])
    need(expected_request is None or expected_request==body['worker_request_reference'],
         'Formal input uses a different worker request')
    need(body['source_references']==request['planning_source_references']==_sources(),
         'Formal preflight source closure differs')
    identity=body['physical_identity'];rows=body['rows'];ids=identity['scenario_ids']
    need(len(ids)==160 and len(set(ids))==160 and len(rows)==160
         and len(identity['scenario_sha256s'])==len(identity['template_sha256s'])==160
         and identity['constants_usd']==[0.]*160 and identity['empirical_probabilities']==[1/160]*160,
         'Complete unchanged160 empirical physical support required')
    from .recourse_template_index import ReviewedTemplateIndex
    index=ReviewedTemplateIndex(request['template_index_reference'],request['template_association_review_reference'])
    source=index.verify()
    need(body['provider_identity']==index.identity and source['domain']=='formal'
         and source['scenario_ids']==ids and source['scenario_sha256s']==identity['scenario_sha256s']
         and source['first_stage_sha256']==identity['first_stage_sha256'],
         'Formal preflight differs from reviewed original template association')
    for i,row in enumerate(rows):
        need(row['scenario_index']==i and row['scenario_id']==ids[i]
             and row['scenario_sha256']==identity['scenario_sha256s'][i]
             and row['template_sha256']==identity['template_sha256s'][i]==source['rows'][i]['template_sha256']
             and row['first_stage_sha256']==identity['first_stage_sha256']
             and row['numeric_loaded'] is True and row['nonnegative_cost_floor_verified'] is True
             and type(row['recourse_columns']) is int and row['recourse_columns']>0
             and type(row['coefficient_nnz']) is int and row['coefficient_nnz']>=0,
             'Missing, reordered or incompletely loaded formal numeric template')
    binding=body['input_binding']
    need(binding['summary']==request['summary_reference'] and binding['audit']==request['audit_reference']
         and binding['formal_protocol_lock']==request['protocol_reference']
         and binding['target']['target_event']==request['target_id']
         and binding['physical_case_sha256']==identity['case_sha256']
         and binding['task_sha256']==identity['task_sha256'] and binding['config_hash']==identity['config_hash']
         and binding['method_id']=='SAA' and binding['rho']==1. and binding['formal_OOS'] is False
         and binding['algorithm_id']==FORMAL_ALGORITHM_ID
         and binding['objective_spec']==dict(risk_model='SAA',law='uniform_empirical',
             support_ids=ids,weights=[1/160]*160,moment_ambiguity_used_for_optimization=False),
         'Formal preflight physical binding differs')
    need(type(body['elapsed_seconds']) in (int,float) and np.isfinite(body['elapsed_seconds'])
         and body['elapsed_seconds']>=0 and body['numeric_template_count']==160
         and body['resident_template_capacity']==1,'Invalid complete preflight accounting/scope')
    if expected_identity is not None:
        expected=dict(domain='formal',algorithm_id=FORMAL_ALGORITHM_ID,formal_input_reference=reference,
            first_stage_sha256=identity['first_stage_sha256'],support_ids=ids,
            scenario_sha256s=identity['scenario_sha256s'],template_sha256s=identity['template_sha256s'],
            constants_usd=identity['constants_usd'],empirical_probabilities=identity['empirical_probabilities'])
        need(all(expected_identity.get(k)==v for k,v in expected.items()),'Formal problem/preflight identity differs')
    return body


class _Scenes(Sequence):
    def __init__(self,bank):self.bank=bank
    def __len__(self):return len(self.bank.scenario_ids)
    def __getitem__(self,i):
        if not 0<=i<len(self):raise IndexError(i)
        return self.bank.load(i,expected_scenario_sha256=self.bank.scenario_sha256s[i])


class FormalStoredSAAProblem(DevelopmentSAAProblem):
    """Constructed only by the original-input factory, with no eager 160-bank copy."""
    def __init__(self,prepared,request_reference,evidence_root,*,_token):
        need(_token is _TOKEN and type(prepared) is PreparedFormalSAA,
             'Use original formal SAA input factory')
        physical=prepared.physical;provider=physical.template_provider
        need(isinstance(physical.bank,FormalPhysicalBank) and isinstance(provider,TemplateProvider),
             'Original formal physical bank and reviewed numeric provider required')
        need(physical.tasks.config.get('synthetic_validation_only') is not True,
             'Synthetic tasks cannot enter formal domain')
        ids=tuple(physical.bank.scenario_ids);hashes=tuple(physical.bank.scenario_sha256s)
        need(len(ids)==160 and len(set(ids))==160 and provider.index['domain']=='formal',
             'Formal input requires complete160 support')
        need(np.array_equal(prepared.empirical_probabilities,np.full(160,1/160)),
             'Formal decomposition must retain original uniform empirical law')
        for name,value in dict(case=physical.case,tasks=physical.tasks,domain='formal',
             first_stage=physical.first_stage,support_ids=ids,provider=provider,
             constants_usd=np.frombuffer(np.zeros(160).tobytes(),dtype=float),
             input_binding=dict(prepared.input_binding,algorithm_id=FORMAL_ALGORITHM_ID,
                 original_input_factory_algorithm_id=prepared.input_binding['algorithm_id']),worker_request_reference=request_reference,
             scenarios=_Scenes(physical.bank),evidence_root=str(evidence_root),capacity=1,
             _factory_token=_TOKEN,formal_input_reference=None).items():object.__setattr__(self,name,value)
        def loader(i):
            scene=self.scene_loader(i)
            need(scene.development is False and scene.physical_case_sha256==self.case.case_sha256,
                 'Formal scenario domain/case differs')
            template=provider.load(i,scene)
            need(np.all(template.c>=0) and np.all(template.lower_bounds[template.c>0]>=0),
                 'Original operating cost floor invalid')
            return template
        object.__setattr__(self,'recourses',LazyRecourses(ids,hashes,self.first_stage.matrix_sha256,
            dict(domain='formal',provider=provider.identity,worker_request_reference=request_reference,
                 algorithm_id=FORMAL_ALGORITHM_ID),loader,capacity=1))

    def scene_loader(self,i):return self.scenarios[i]

    @property
    def identity(self):
        physical=_physical_identity(self)
        return dict(domain='formal',algorithm_id=FORMAL_ALGORITHM_ID,
            first_stage_sha256=physical['first_stage_sha256'],support_ids=physical['scenario_ids'],
            scenario_sha256s=physical['scenario_sha256s'],template_sha256s=physical['template_sha256s'],
            constants_usd=physical['constants_usd'],empirical_probabilities=physical['empirical_probabilities'],
            provider_identity=self.provider.identity,evidence_root=self.evidence_root,cache_capacity=1,
            formal_input_reference=self.formal_input_reference)


def load_formal_l_shaped_problem(worker_request_reference, *, evidence_root, preflight_path, deadline):
    """Load all original160 templates once, with a shared caller deadline.

    The preflight receipt is reusable input evidence. It contains no Q solve,
    master bound, failed-parent cost settlement or production permission.
    """
    began=time.monotonic()
    need(type(deadline) in (int,float) and np.isfinite(deadline) and deadline>began,
         'Finite shared formal input deadline required')
    root=Path(evidence_root);output=Path(preflight_path);paths=get_paths()
    for p in (root,output):
        need(p.is_absolute() and p.resolve()==p,'Explicit canonical formal output path required')
        need(not any(p.is_relative_to(Path(x).resolve()) for x in
             (paths.data_root,paths.frozen_source_root,paths.project_root/'src')),
             'Formal input evidence overlaps frozen/source files')
    need(not output.exists(),'Cannot overwrite formal input preflight')
    request=_worker_input(worker_request_reference)
    prepared=prepare_formal_saa(summary_reference=request['summary_reference'],audit_reference=request['audit_reference'],
        target_id=request['target_id'],template_index_reference=request['template_index_reference'],
        template_association_review_reference=request['template_association_review_reference'])
    problem=FormalStoredSAAProblem(prepared,worker_request_reference,root,_token=_TOKEN)
    original=problem.provider.index
    for p in (root,output):
        need(not p.is_relative_to(Path(original['original_request_reference']['path']).parent)
             and p!=Path(original['original_snapshot_reference']['path'])
             and not any(p.is_relative_to(Path(row['original_template_reference']['path']).parent)
                         for row in original['rows']),
             'Formal evidence overlaps original frozen raw/template inputs')
    rows=[]
    for i,sid in enumerate(problem.support_ids):
        if time.monotonic()>=deadline:raise TimeoutError('Formal full160 numeric input shared deadline')
        template=problem.recourses[i]
        rows.append(dict(scenario_index=i,scenario_id=sid,scenario_sha256=template.scenario_sha256,
            template_sha256=template.template_sha256,first_stage_sha256=template.first_stage.matrix_sha256,
            numeric_loaded=True,nonnegative_cost_floor_verified=True,recourse_columns=len(template.c),
            coefficient_nnz=sum(getattr(template,k).nnz for k in ('A_ub','A_eq','Bx_ub','Bx_eq','Br_ub','Br_eq'))))
        del template
    if time.monotonic()>=deadline:raise TimeoutError('Formal full160 numeric input shared deadline')
    body=dict(kind=PREFLIGHT_KIND,domain='formal',algorithm_id=FORMAL_ALGORITHM_ID,
        worker_request_reference=worker_request_reference,source_references=request['planning_source_references'],
        provider_identity=problem.provider.identity,physical_identity=_physical_identity(problem),
        input_binding=problem.input_binding,rows=rows,numeric_template_count=len(rows),resident_template_capacity=1,
        elapsed_seconds=time.monotonic()-began,original_model_unchanged=True,solver_calls=0,
        formal_admission=False,scientific_acceptance=False,formal_OOS=False)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(body,stream,indent=2,allow_nan=False);stream.write('\n')
    object.__setattr__(problem,'formal_input_reference',file_reference(output))
    validate_formal_problem(problem)
    object.__setattr__(problem,'atom_store',DiskAtomStore(root,problem))
    return problem


def validate_formal_problem(problem):
    need(type(problem) is FormalStoredSAAProblem and getattr(problem,'_factory_token',None) is _TOKEN,
         'Only original-input typed formal SAA problem is allowed')
    need(problem.domain=='formal' and problem.formal_input_reference is not None,
         'Formal input factory has not completed full160 numeric validation')
    body=validate_formal_input_reference(problem.formal_input_reference,
        expected_request=problem.worker_request_reference,expected_identity=problem.identity)
    need(body['physical_identity']==_physical_identity(problem) and body['input_binding']==problem.input_binding,
         'Formal physical object differs from pinned numeric input')
    return body


def restore_formal_l_shaped_problem(formal_input_reference, *, evidence_root, deadline):
    """Rebuild input objects from the same pinned complete preflight on resume.

    Numeric templates are rehashed and checked on every later actual load. The
    original complete scan is inherited by its trusted reference; it is not
    repeated and cannot be replaced with an unpinned new scan on restart.
    """
    need(type(deadline) in (int,float) and np.isfinite(deadline) and deadline>time.monotonic(),
         'Finite shared formal restore deadline required')
    body=validate_formal_input_reference(formal_input_reference)
    request=_worker_input(body['worker_request_reference'])
    prepared=prepare_formal_saa(summary_reference=request['summary_reference'],audit_reference=request['audit_reference'],
        target_id=request['target_id'],template_index_reference=request['template_index_reference'],
        template_association_review_reference=request['template_association_review_reference'])
    problem=FormalStoredSAAProblem(prepared,body['worker_request_reference'],Path(evidence_root),_token=_TOKEN)
    object.__setattr__(problem,'formal_input_reference',formal_input_reference)
    validate_formal_problem(problem)
    if time.monotonic()>=deadline:raise TimeoutError('Formal input restore shared deadline')
    object.__setattr__(problem,'atom_store',DiskAtomStore(Path(evidence_root),problem))
    return problem
