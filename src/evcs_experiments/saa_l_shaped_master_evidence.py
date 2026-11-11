"""Development small-master evidence and independent original-row audit.

Native HiGHS/Gurobi global bounds remain trusted solver output bound to the actual
caller and coefficients. This does not replay a branch tree, certify full-Q,
produce an extensive manifest, or admit formal execution.
"""
from pathlib import Path
import json,time
import numpy as np
import scipy
from scipy.sparse import csr_matrix
from .campaign_batch import file_reference,verify_reference
from .config import get_paths
from .dual_cut_bound import build_dual_cut
from .random_streams import canonical_hash
from .relaxation_bound import coefficient_sha256

PREPARATION='SAA_L_SHAPED_MASTER_PREPARATION_DEVELOPMENT_V1'
EXECUTION='SAA_L_SHAPED_MASTER_EXECUTION_DEVELOPMENT_V1'
ARRAYS=('c','row_lower','row_upper','lower_bounds','upper_bounds','integrality')

def need(ok,message):
    if not bool(ok):raise ValueError(message)

def _read(ref):
    path=Path(ref['path']);need(path.is_absolute() and path.resolve()==path,'Aliased evidence path')
    verify_reference(ref);return json.loads(path.read_text())

def _write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    return file_reference(path)

def _sources():
    return {n:file_reference(Path(__file__).with_name(n)) for n in (
        'saa_l_shaped_master_evidence.py','saa_l_shaped.py','first_stage.py','recourse.py',
        'dual_cut_bound.py','relaxation_bound.py','random_streams.py','campaign_batch.py',
        'saa_l_shaped_native.py','native_gurobi.py')}

def _scope(problem):
    if problem.domain=='formal':
        from .saa_l_shaped_formal import validate_formal_problem
        validate_formal_problem(problem)
    else:
        need(problem.domain=='development' and problem.tasks.config.get('synthetic_validation_only') is True,
             'Only typed formal or synthetic development small-master evidence is admitted')

def _hash(m):return coefficient_sha256(dict(m,objective_constant_usd=0.))

def _finite(value):return None if value is None or not np.isfinite(value) else float(value)


def native_bound_contribution(execution):
    """Read an externally pinned event; this alone is not its matrix audit.

    A genuine native time limit may return no finite global bound. Such an
    event stays in the history and contributes no bound. Every other admitted
    event must carry a finite bound; a missing value is never a zero bound.
    """
    nr=execution['native_result'];record=execution['master_record']
    code=nr['status']
    need(type(code) is int and code in (0,1) and nr['success'] is (code==0)
         and type(record['solver_status']) is int and record['solver_status']==code
         and record['message']==nr['message'],
         'Native status differs or cannot admit master event')
    if 'native_adapter_reference' in execution:
        adapter=_read(execution['native_adapter_reference'])
        original=adapter['native_record']
        need(execution['bound_source']=='actual_gurobi_result.mip_dual_bound',
             'Actual Gurobi bound source differs')
        if original['status']=='MODEL_LOADING_TIME_LIMIT':
            normalized=adapter['normalized_result']
            need(type(original['raw_model_status']) is int and original['raw_model_status']==1
                 and code==1 and original['effective_run_time_limit_seconds']==0
                 and original['solver_run_seconds']==0 and original['valid_incumbent'] is False
                 and normalized['limit_status']=='TIME_LIMIT'
                 and normalized['status']==1 and normalized['success'] is False
                 and all(nr[k] is None and normalized[k] is None
                         for k in ('x','fun','mip_dual_bound','mip_gap','mip_node_count')),
                 'Typed Gurobi model-loading time limit differs')
        else:
            need(type(original['raw_model_status']) is int and original['raw_model_status'] in (2,9)
                 and original['status']==('OPTIMAL' if original['raw_model_status']==2 else 'TIME_LIMIT')
                 and code==(0 if original['raw_model_status']==2 else 1)
                 and not original['certificate_failures']
                 and original['solver_bound_without_constant_usd']==nr['mip_dual_bound'],
                 'Actual Gurobi status/bound provenance differs')
    else:
        need(execution['bound_source']=='actual_scipy_milp_result.mip_dual_bound',
             'Actual native global bound source differs')
    lb=nr['mip_dual_bound']
    if lb is None:
        need(code==1 and record['lower_bound_usd'] is None,
             'Only native TIME_LIMIT may retain an event without a finite bound')
        return None
    need(type(lb) in (int,float) and np.isfinite(lb)
         and type(record['lower_bound_usd']) in (int,float) and record['lower_bound_usd']==lb,
         'Finite actual native global bound differs')
    return float(lb)

class DevelopmentMasterEvidence:
    """Called before/after the original single native call, inside its budget."""
    def __init__(self,output_root):
        self.root=Path(output_root)
        need(self.root.is_absolute() and self.root.resolve()==self.root,'Canonical absolute master root required')
        p=get_paths()
        for protected in (p.data_root,p.frozen_source_root,p.project_root/'src'):
            need(not self.root.is_relative_to(Path(protected).resolve()),'Master evidence overlaps frozen/source content')
        self.root.mkdir(parents=True,exist_ok=True)
        self.sequence=len(list(self.root.glob('master_*')))
        self.records=[]
    @property
    def identity(self):return dict(kind=PREPARATION,root=str(self.root),sources=_sources())
    def _solver_metadata(self):
        return dict(solver_backend='scipy.optimize.milp/HiGHS',scipy_version=scipy.__version__)
    def prepare(self,*,problem,cuts,model,requested_limit,mip_gap):
        _scope(problem)
        need(problem.domain!='formal' or isinstance(self,NativeDevelopmentMasterEvidence),'Formal master requires Gurobi')
        if getattr(problem,'provider',None) is not None:
            index=problem.provider.index
            need(not self.root.is_relative_to(Path(index['original_request_reference']['path']).parent),
                 'Master evidence overlaps original raw input evidence')
            need(self.root!=Path(index['original_snapshot_reference']['path']),'Master evidence overlaps original snapshot')
            for row in index['rows']:
                need(not self.root.is_relative_to(Path(row['original_template_reference']['path']).parent),
                     'Master evidence overlaps original numeric template')
        self.sequence+=1;folder=self.root/f'master_{self.sequence:06d}';folder.mkdir()
        a=csr_matrix(model['matrix']);path=folder/'submitted_matrix.npz'
        with path.open('xb') as f:
            np.savez_compressed(f,data=a.data,indices=a.indices,indptr=a.indptr,shape=np.asarray(a.shape),
                                **{k:np.asarray(model[k]) for k in ARRAYS})
        prep=dict(kind=PREPARATION,problem_identity=problem.identity,source_references=_sources(),
            coefficient_sha256=_hash(model),matrix_reference=file_reference(path),cuts=cuts,
            requested_limit_seconds=requested_limit,mip_relative_gap=mip_gap,
            first_stage_columns=model['first_stage_columns'],theta_columns=model['theta_columns'],
            recourse_columns=model['recourse_columns'],cut_rows=model['cut_rows'],
            **self._solver_metadata(),
            formal_admission=False,extensive_model_manifest=False)
        return _write(folder/'preparation.json',prep)
    def finish(self,*,token,model,raw,record,effective_options,native_elapsed_seconds):
        prep=_read(token);need(_hash(model)==prep['coefficient_sha256'],'Actual submitted native coefficients changed')
        native=dict(status=int(raw.status),message=str(raw.message),success=bool(raw.success),
            x=None if raw.x is None else np.asarray(raw.x).tolist(),fun=_finite(raw.fun),
            mip_dual_bound=_finite(getattr(raw,'mip_dual_bound',None)),
            mip_gap=_finite(getattr(raw,'mip_gap',None)),mip_node_count=_finite(getattr(raw,'mip_node_count',None)))
        body=dict(kind=EXECUTION,preparation_reference=token,source_references=_sources(),
            submitted_coefficient_sha256=_hash(model),native_result=native,master_record=record,
            effective_options=effective_options,native_elapsed_seconds=native_elapsed_seconds,
            bound_source='actual_scipy_milp_result.mip_dual_bound',formal_admission=False,
            native_branch_and_bound_proof_replayed=False,scientific_acceptance=False)
        ref=_write(Path(token['path']).parent/'execution.json',body);self.records.append(ref);return ref


class NativeDevelopmentMasterEvidence(DevelopmentMasterEvidence):
    """Explicit Gurobi caller for the same development matrix and full audit.

    A native backend here does not admit a formal domain, worker or parent fee.
    """
    def __init__(self,output_root,*,threads=1):
        need(type(threads) is int and 1<=threads<=4,'Native master threads must be1..4')
        super().__init__(output_root);self.threads=threads;self._native_runs={}
    @property
    def identity(self):
        return dict(super().identity,solver_backend='gurobipy',gurobi_version=[12,0,3],threads=self.threads)
    def _solver_metadata(self):
        return dict(solver_backend='gurobipy',gurobi_version=[12,0,3],threads=self.threads)
    def solve_native(self,*,token,model,limit,mip_gap):
        from .saa_l_shaped_native import solve_native_small_master
        prep=_read(token)
        need(prep['solver_backend']=='gurobipy' and _hash(model)==prep['coefficient_sha256'],
             'Native request differs from pre-call matrix')
        result=solve_native_small_master(model,time_limit_seconds=limit,mip_relative_gap=mip_gap,
            threads=self.threads,output_root=Path(token['path']).parent/'native')
        self._native_runs[token['path']]=result
        return result
    def finish(self,*,token,model,raw,record,effective_options,native_elapsed_seconds):
        prep=_read(token);result=self._native_runs.pop(token['path'])
        need(result.raw is raw and _hash(model)==prep['coefficient_sha256']
             and effective_options==result.effective_options and native_elapsed_seconds==result.native_elapsed_seconds,
             'Caller changed actual native result/options')
        native=dict(status=int(raw.status),message=str(raw.message),success=bool(raw.success),
            x=None if raw.x is None else np.asarray(raw.x).tolist(),fun=_finite(raw.fun),
            mip_dual_bound=_finite(getattr(raw,'mip_dual_bound',None)),
            mip_gap=_finite(getattr(raw,'mip_gap',None)),mip_node_count=_finite(getattr(raw,'mip_node_count',None)))
        body=dict(kind=EXECUTION,preparation_reference=token,source_references=_sources(),
            submitted_coefficient_sha256=_hash(model),native_result=native,master_record=record,
            effective_options=effective_options,native_elapsed_seconds=native_elapsed_seconds,
            native_adapter_reference=result.evidence_reference,
            bound_source='actual_gurobi_result.mip_dual_bound',formal_admission=False,
            native_branch_and_bound_proof_replayed=False,scientific_acceptance=False)
        ref=_write(Path(token['path']).parent/'execution.json',body);self.records.append(ref);return ref


def _verify_native_adapter(prep,execution):
    from .saa_l_shaped_native import verify_native_small_master_evidence
    need(prep['gurobi_version']==[12,0,3] and type(prep['threads']) is int and 1<=prep['threads']<=4,
         'Native backend/version/threads differs')
    ref=execution['native_adapter_reference']
    need(Path(ref['path']).parent==Path(execution['preparation_reference']['path']).parent/'native',
         'Foreign native execution root')
    adapter=verify_native_small_master_evidence(ref,expected_coefficient_sha256=prep['coefficient_sha256'])
    need(adapter['effective_options']==execution['effective_options']
         and adapter['native_elapsed_seconds']==execution['native_elapsed_seconds'],
         'Native execution options/elapsed differ')
    normalized=adapter['normalized_result'];actual=execution['native_result']
    need(all(normalized[k]==v for k,v in actual.items()),'Native normalized result differs')


def _load_model(ref):
    verify_reference(ref);path=Path(ref['path']);need(path.resolve()==path,'Aliased master coefficients')
    with np.load(path,allow_pickle=False) as z:
        need(set(z.files)==set(ARRAYS)|{'data','indices','indptr','shape'},'Wrong submitted coefficient inventory')
        values={k:z[k].copy() for k in z.files}
    need(all(a.dtype.kind in 'fiu' for a in values.values()),'Invalid master array type')
    shape=values.pop('shape');need(shape.shape==(2,) and np.all(shape>0),'Wrong master shape')
    a=csr_matrix((values.pop('data'),values.pop('indices'),values.pop('indptr')),shape=tuple(shape))
    a.check_format(full_check=True);return dict(matrix=a,**values)


def _original_cut(problem,i,handle):
    atom=problem.atom_store.load(handle,i) if handle.get('kind')=='SAA_L_SHAPED_DISK_ATOM_REFERENCE_V1' else handle
    t=problem.recourses[i];anchor=np.asarray(atom['anchor'],dtype=float);plan=problem.plan(anchor)
    need(problem.first_stage.evaluate(plan).feasible,'Original cut anchor is infeasible')
    z=np.asarray(atom['primal'],dtype=float);need(z.shape==t.c.shape and np.isfinite(z).all(),'Original Q primal differs')
    bu,be=t.fixed_rhs(plan)
    residual=max(float(np.max(t.A_ub@z-bu,initial=0)),float(np.max(abs(t.A_eq@z-be),initial=0)),
        float(np.max(t.lower_bounds-z,initial=0)),float(np.max(z-t.upper_bounds,initial=0)))
    q=float(t.c@z+problem.constants_usd[i]);need(residual<=1e-7 and abs(q-atom['q_usd'])<=1e-8,'Original cut Q witness differs')
    raw={k:getattr(t,k) for k in ('A_ub','A_eq','Bx_ub','Bx_eq','Br_ub','Br_eq','b_ub','b_eq','c','lower_bounds','upper_bounds')}
    raw.update(scenario_id=problem.support_ids[i],template_sha256=t.template_sha256)
    cut,g=build_dual_cut(raw,atom['dual'],scenario_constant_usd=float(problem.constants_usd[i]),
        anchor_first_stage=anchor,anchor_q_usd=atom["q_usd"])
    need(cut['passed'] and cut==atom['cut'] and np.array_equal(g,atom['gradient']),'Original-matrix dual plane differs')
    return cut,g


def audit_master_evidence(problem,trusted_record_reference,*,deadline):
    """No solver/builder calls. Caller must supply its independently retained pin.

    Rehashing an attacker-rewritten execution record is not native provenance.
    This audit checks the admitted native bound and its exact coefficient/source
    linkage; the solver's branch-and-bound proof itself is not reconstructed.
    """
    began=time.monotonic();_scope(problem)
    def alive():
        if not np.isfinite(deadline) or time.monotonic()>=deadline:raise TimeoutError('Master audit deadline')
    alive();execution=_read(trusted_record_reference);pref=execution['preparation_reference'];prep=_read(pref)
    folder=Path(pref['path']).parent
    need(Path(trusted_record_reference['path'])==folder/'execution.json'
         and Path(prep['matrix_reference']['path'])==folder/'submitted_matrix.npz','Master evidence paths differ')
    need(execution['kind']==EXECUTION and prep['kind']==PREPARATION and prep['problem_identity']==problem.identity,
         'Master problem/algorithm/support identity differs')
    sources=_sources()
    need(prep['source_references']==execution['source_references']==sources,'Actual master implementation source differs')
    for ref in sources.values():verify_reference(ref)
    need(prep['formal_admission'] is False and prep['extensive_model_manifest'] is False
         and execution['formal_admission'] is False and execution['scientific_acceptance'] is False
         and execution['native_branch_and_bound_proof_replayed'] is False,'Development evidence scope differs')
    need(problem.domain!='formal' or prep['solver_backend']=='gurobipy','Formal master evidence must be Gurobi')
    if prep['solver_backend']=='gurobipy':
        _verify_native_adapter(prep,execution)
    else:
        need(prep['solver_backend']=='scipy.optimize.milp/HiGHS' and prep['scipy_version']==scipy.__version__
             and 'native_adapter_reference' not in execution,'Native backend/version differs')
    m=_load_model(prep['matrix_reference']);coefficient=_hash(m)
    need(coefficient==prep['coefficient_sha256']==execution['submitted_coefficient_sha256'],'Pre-call coefficient hash differs')
    f=problem.first_stage;A=m['matrix'];n=len(problem.support_ids);d=len(f.c);r=f.A.shape[0];cuts=prep['cuts']
    need(len(cuts)==n,'Cut support axis differs');count=sum(map(len,cuts))
    need(A.shape==(r+count,d+n) and prep['first_stage_columns']==d and prep['theta_columns']==n
         and prep['recourse_columns']==0 and prep['cut_rows']==count,'Small-master dimensions differ')
    same=lambda a,b:np.array_equal(np.asarray(a),np.asarray(b))
    need((A[:r,:d]!=f.A).nnz==0 and A[:r,d:].nnz==0,'Original first-stage coefficients changed')
    for key in ('row_lower','row_upper'):need(same(m[key][:r],getattr(f,key)),'Original first-stage rows changed')
    for key in ('lower_bounds','upper_bounds','integrality'):need(same(m[key][:d],getattr(f,key)),'Original first-stage domain changed')
    need(same(m['c'],np.r_[f.c,np.full(n,1/n)]),'Fixed empirical objective differs')
    need(same(m['lower_bounds'][d:],problem.constants_usd) and np.isposinf(m['upper_bounds'][d:]).all()
         and same(m['integrality'][d:],np.zeros(n)),'Original theta domain/constants differ')
    j=r;plane_hashes=[]
    for i,group in enumerate(cuts):
        alive();t=problem.recourses[i]
        need(np.all(t.c>=0) and np.all(t.lower_bounds[t.c>0]>=0),'Theta floor lacks original nonnegative-cost contract')
        del t
        for atom in group:
            alive();cut,g=_original_cut(problem,i,atom)
            need(same(A[j,:d].toarray().ravel(),-g),'Retained original cut gradient changed')
            tail=np.zeros(n);tail[i]=1
            need(same(A[j,d:].toarray().ravel(),tail) and m['row_lower'][j]==cut['intercept_usd']
                 and np.isposinf(m['row_upper'][j]),'Retained original cut theta/intercept changed')
            plane_hashes.append(cut['cut_sha256']);j+=1
            del cut,g
    nr=execution['native_result'];record=execution['master_record'];opts=execution['effective_options']
    if prep['solver_backend']=='gurobipy':
        need(set(opts)=={'time_limit','mip_rel_gap','threads','primal_feasibility_tolerance','dual_feasibility_tolerance'}
             and opts['threads']==prep['threads'] and opts['primal_feasibility_tolerance']==opts['dual_feasibility_tolerance']==1e-8,
             'Native Gurobi resource/tolerance options differ')
    else:
        need(set(opts)=={'time_limit','mip_rel_gap','threads','mip_feasibility_tolerance'}
             and opts['threads']==1 and opts['mip_feasibility_tolerance']==1e-8,
             'Native HiGHS resource/tolerance options differ')
    need(0<opts['time_limit']<=prep['requested_limit_seconds']<=600
         and opts['mip_rel_gap']==prep['mip_relative_gap'] and 0<=opts['mip_rel_gap']<=.001,
         'Native time/gap options differ')
    need(type(execution['native_elapsed_seconds']) in (int,float) and np.isfinite(execution['native_elapsed_seconds'])
         and execution['native_elapsed_seconds']>=0,'Invalid native elapsed time')
    lb=native_bound_contribution(execution)
    need(record['first_stage_columns']==d and record['theta_columns']==n and record['recourse_columns']==0
         and record['cut_rows']==count and record['matrix_shape']==list(A.shape),'Native master axis differs')
    violation=None;objective=None
    if nr['x'] is None:
        need(nr['status']==1 and record['primal'] is None and record['incumbent_objective_usd'] is None
             and nr['fun'] is None,'Missing native incumbent metadata differs')
    else:
        x=np.asarray(nr['x'],dtype=float)
        need(x.shape==(d+n,) and np.isfinite(x).all() and same(x,record['primal']),'Retained native primal differs')
        act=A@x;integer=m['integrality']==1
        violation=max(float(np.max(m['row_lower']-act,initial=0)),float(np.max(act-m['row_upper'],initial=0)),
            float(np.max(m['lower_bounds']-x,initial=0)),float(np.max(x-m['upper_bounds'],initial=0)),
            float(np.max(abs(x[integer]-np.rint(x[integer])),initial=0)))
        need(violation<=1e-8,'Native incumbent violates original small-master rows')
        objective=float(m['c']@x);tol=1e-4+1e-7*max(1.,abs(objective))
        need(nr['fun'] is not None and abs(objective-nr['fun'])<=tol
             and abs(objective-record['incumbent_objective_usd'])<=tol and (lb is None or lb<=objective+tol),
             'Native objective/LB differs from original matrix')
    if prep['solver_backend']=='gurobipy':
        raw_native=_read(execution['native_adapter_reference'])['native_record']
        native_provenance=dict(backend='gurobipy',raw_model_status=raw_native['raw_model_status'],
            status_name=raw_native['status'],normalized_status=nr['status'])
    else:
        native_provenance=dict(backend='scipy.optimize.milp/HiGHS',raw_model_status=nr['status'],
            status_name='OPTIMAL' if nr['status']==0 else 'TIME_LIMIT',normalized_status=nr['status'])
    alive()
    return dict(kind=('INDEPENDENT_SAA_SMALL_MASTER_AUDIT_FORMAL_V1' if problem.domain=='formal' else 'INDEPENDENT_SAA_SMALL_MASTER_AUDIT_DEVELOPMENT_V1'),passed=True,
        execution_reference=trusted_record_reference,preparation_reference=pref,coefficient_sha256=coefficient,
        problem_identity_sha256=canonical_hash(problem.identity),source_references=sources,
        first_stage_columns=d,theta_columns=n,recourse_columns=0,cut_rows=count,cut_sha256s=plane_hashes,
        native_lower_bound_usd=lb,incumbent_objective_usd=objective,max_original_unit_violation=violation,
        contributes_lower_bound=lb is not None,
        bound_disposition='FINITE_NATIVE_GLOBAL_BOUND' if lb is not None else 'TIME_LIMIT_NO_FINITE_NATIVE_BOUND',
        native_status=nr['status'],native_provenance=native_provenance,
        native_branch_and_bound_proof_replayed=False,native_solver_output_trusted=True,
        fixed_plan_full_Q_audited=False,solver_calls=0,formal_admission=False,scientific_acceptance=False,
        elapsed_seconds=time.monotonic()-began)
