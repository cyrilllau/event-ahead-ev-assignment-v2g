"""Development fixed-uniform SAA multicut L-shaped reference kernel.

The master contains the original first stage and one theta per support atom;
no recourse copy or probability optimization is assembled in that master.
This is not the paper's C&CG algorithm, a formal worker, or an OOS entry.
"""
from dataclasses import dataclass
import hashlib,json,time,warnings
from pathlib import Path
import numpy as np
from scipy.optimize import Bounds,LinearConstraint,milp
from scipy.sparse import csr_matrix,hstack,vstack
from .first_stage import FirstStagePlan,assemble_first_stage
from .recourse import assemble_recourse,solve_fixed_plan
from .dual_cut_bound import build_dual_cut
from .random_streams import canonical_hash
from .runtime.jobctl import atomic_json

ALGORITHM_ID='SAA_FIXED_EMPIRICAL_MULTICUT_L_SHAPED_DEVELOPMENT_V1'

def need(ok,message):
    if not bool(ok):raise ValueError(message)

def source_identity():
    names=('saa_l_shaped.py','first_stage.py','recourse.py','dual_cut_bound.py','random_streams.py')
    return {name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}

@dataclass(frozen=True)
class DevelopmentSAAProblem:
    case: object
    tasks: object
    scenarios: tuple
    constants_usd: object
    domain: str='development'

    def __post_init__(self):
        need(self.domain=='development','Formal and OOS decomposition are unsupported')
        scenes=tuple(self.scenarios)
        need(scenes and all(s.development is True for s in scenes) and self.tasks.config.get('synthetic_validation_only') is True,
             'Only explicit synthetic development physical fixtures are admitted')
        ids=tuple(s.scenario_id for s in scenes)
        need(len(set(ids))==len(ids),'Duplicate support')
        constants=np.asarray(self.constants_usd,dtype=float)
        need(constants.shape==(len(ids),) and np.isfinite(constants).all(),'Invalid scenario constants')
        first=assemble_first_stage(self.case,self.tasks)
        recourses=tuple(assemble_recourse(self.case,self.tasks,s) for s in scenes)
        need(all(r.first_stage.matrix_sha256==first.matrix_sha256 for r in recourses),'Changed first-stage domain')
        need(all(np.all(r.c>=0) and np.all(r.lower_bounds[r.c>0]>=0) for r in recourses),
             'Initial theta lower bounds require original nonnegative operating costs and variables')
        object.__setattr__(self,'scenarios',scenes);object.__setattr__(self,'first_stage',first)
        object.__setattr__(self,'recourses',recourses);object.__setattr__(self,'support_ids',ids)
        object.__setattr__(self,'constants_usd',np.frombuffer(constants.tobytes(),dtype=float))

    @property
    def identity(self):
        return dict(domain=self.domain,algorithm_id=ALGORITHM_ID,first_stage_sha256=self.first_stage.matrix_sha256,
            support_ids=list(self.support_ids),scenario_sha256s=[r.scenario_sha256 for r in self.recourses],
            template_sha256s=[r.template_sha256 for r in self.recourses],constants_usd=self.constants_usd.tolist(),
            empirical_probabilities=[1/len(self.support_ids)]*len(self.support_ids))

    def plan(self,vector):
        h=len(self.first_stage.task_ids);t=len(self.first_stage.periods);v=np.asarray(vector)
        return FirstStagePlan(v[:h],v[h:h+h*t].reshape(h,t),self.first_stage.task_sha256,self.first_stage.config_hash)


def _template(problem,i):
    t=problem.recourses[i]
    value={k:getattr(t,k) for k in ('A_ub','A_eq','Bx_ub','Bx_eq','Br_ub','Br_eq','b_ub','b_eq','c','lower_bounds','upper_bounds')}
    value.update(scenario_id=problem.support_ids[i],template_sha256=t.template_sha256)
    return value


def _validate_atom(problem,i,atom):
    if getattr(problem,'atom_store',None) is not None and atom.get('kind')=='SAA_L_SHAPED_DISK_ATOM_REFERENCE_V1':
        atom=problem.atom_store.load(atom,i)
    t=problem.recourses[i];anchor=np.asarray(atom['anchor']);plan=problem.plan(anchor)
    need(problem.first_stage.evaluate(plan).feasible,'Stored cut anchor is infeasible')
    z=np.asarray(atom['primal'],dtype=float);need(z.shape==t.c.shape and np.isfinite(z).all(),'Stored Q primal differs')
    bu,be=t.fixed_rhs(plan)
    residual=max(float(np.max(t.A_ub@z-bu,initial=0)),float(np.max(abs(t.A_eq@z-be),initial=0)),
                 float(np.max(t.lower_bounds-z,initial=0)),float(np.max(z-t.upper_bounds,initial=0)))
    q=float(t.c@z+problem.constants_usd[i]);need(residual<=1e-7 and abs(q-atom['q_usd'])<=1e-8,'Stored full-Q witness changed')
    cut,g=build_dual_cut(_template(problem,i),atom['dual'],scenario_constant_usd=float(problem.constants_usd[i]),
        anchor_first_stage=anchor,anchor_q_usd=atom["q_usd"])
    need(cut['passed'] and cut==atom['cut'] and np.array_equal(g,atom['gradient']),'Stored dual cut failed original coefficients')
    return cut,g


def _anchor_matches(atom,anchor):
    return atom.get('anchor_sha256')==canonical_hash(anchor) if atom.get('kind')=='SAA_L_SHAPED_DISK_ATOM_REFERENCE_V1' else atom['anchor']==anchor


def _plane_sha(atom):
    return atom['plane_sha256'] if atom.get('kind')=='SAA_L_SHAPED_DISK_ATOM_REFERENCE_V1' else canonical_hash(dict(gradient=atom['gradient'],intercept=atom['cut']['intercept_usd']))


def build_master(problem,cuts):
    """Original first-stage domain plus theta >= each retained affine minorant."""
    f=problem.first_stage;n=len(problem.support_ids);d=len(f.c)
    need(len(cuts)==n,'Cut support axis differs')
    rows=[];lower=[]
    for i,atoms in enumerate(cuts):
        for atom in atoms:
            cut,g=_validate_atom(problem,i,atom)
            row=np.r_[-g,np.zeros(n)];row[d+i]=1
            rows.append(csr_matrix(row.reshape(1,-1)));lower.append(cut['intercept_usd'])
    A=hstack([f.A,csr_matrix((f.A.shape[0],n))],format='csr')
    if rows:A=vstack([A,*rows],format='csr')
    return dict(matrix=A,row_lower=np.r_[f.row_lower,lower],row_upper=np.r_[f.row_upper,np.full(len(rows),np.inf)],
        lower_bounds=np.r_[f.lower_bounds,problem.constants_usd],upper_bounds=np.r_[f.upper_bounds,np.full(n,np.inf)],
        integrality=np.r_[f.integrality,np.zeros(n,dtype=int)],c=np.r_[f.c,np.full(n,1/n)],
        first_stage_columns=d,theta_columns=n,recourse_columns=0,cut_rows=len(rows))


def _master(problem,cuts,limit,gap,master_sink=None):
    began=time.monotonic();m=build_master(problem,cuts)
    token=None if master_sink is None else master_sink.prepare(problem=problem,cuts=cuts,model=m,requested_limit=limit,mip_gap=gap)
    grant=limit-(time.monotonic()-began)
    if grant<=0:raise TimeoutError('Master preparation exhausted remaining budget')
    from .saa_l_shaped_master_evidence import NativeDevelopmentMasterEvidence
    if isinstance(master_sink,NativeDevelopmentMasterEvidence):
        native=master_sink.solve_native(token=token,model=m,limit=grant,mip_gap=gap)
        raw=native.raw;effective_options=native.effective_options
        native_elapsed=native.native_elapsed_seconds
    else:
        effective_options=dict(time_limit=grant,mip_rel_gap=gap,threads=1,mip_feasibility_tolerance=1e-8)
        native_started=time.monotonic()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',category=RuntimeWarning,message='Unrecognized options detected.*')
            raw=milp(m['c'],integrality=m['integrality'],bounds=Bounds(m['lower_bounds'],m['upper_bounds']),
                constraints=LinearConstraint(m['matrix'],m['row_lower'],m['row_upper']),
                options=effective_options)
        native_elapsed=time.monotonic()-native_started
    finite=lambda x:None if x is None or not np.isfinite(x) else float(x)
    lb=finite(getattr(raw,'mip_dual_bound',None)) if raw.status in (0,1) else None
    record=dict(solver_status=int(raw.status),message=str(raw.message),lower_bound_usd=lb,
        incumbent_objective_usd=finite(raw.fun),primal=None,first_stage_columns=m['first_stage_columns'],
        theta_columns=m['theta_columns'],recourse_columns=0,cut_rows=m['cut_rows'],matrix_shape=list(m['matrix'].shape))
    if raw.x is not None:
        v=np.asarray(raw.x);need(v.shape==m['c'].shape and np.isfinite(v).all(),'Invalid master incumbent')
        activity=m['matrix']@v;ii=m['integrality']==1
        error=max(float(np.max(m['row_lower']-activity,initial=0)),float(np.max(activity-m['row_upper'],initial=0)),
            float(np.max(m['lower_bounds']-v,initial=0)),float(np.max(v-m['upper_bounds'],initial=0)),
            float(np.max(abs(v[ii]-np.rint(v[ii])),initial=0)))
        need(error<=1e-8,'Master original-unit feasibility failed')
        objective=float(m['c']@v);need(abs(objective-raw.fun)<=1e-4+1e-7*max(1.,abs(objective)),'Master objective mismatch')
        need(lb is None or lb<=objective+1e-4+1e-7*max(1.,abs(objective)),'Contradictory native master bound')
        record.update(primal=v.tolist(),max_violation=error,incumbent_objective_usd=objective)
    if master_sink is not None:
        record['evidence_reference']=master_sink.finish(token=token,model=m,raw=raw,record=dict(record),
            effective_options=effective_options,native_elapsed_seconds=native_elapsed)
    return record


def run_saa_l_shaped(problem,*,checkpoint_path=None,resume=False,expected_checkpoint_sha256=None,budget_seconds=30.,wall_seconds=30.,
                     master_time_limit_seconds=600.,recourse_time_limit_seconds=60.,mip_relative_gap=.001,
                     optimality_relative_gap=.005,max_iterations=1000,evidence_sink=None,master_sink=None):
    """One bounded development session; cumulative budget cannot grow on resume.

    Completed original LP primal/duals are checkpointed. Resume validates them
    against unchanged source and physical matrices, including a partial sweep.
    The resume file hash must come from an already trusted immutable reference;
    recomputing it from an edited checkpoint does not establish provenance.
    An explicit native evidence sink may select Gurobi; production wall sealing
    and formal source admission remain outside this development kernel.
    """
    started=time.monotonic();need(isinstance(problem,DevelopmentSAAProblem),'Expected typed SAA physical problem')
    if problem.domain=='formal':
        from .saa_l_shaped_formal import validate_formal_problem
        from .saa_l_shaped_master_evidence import NativeDevelopmentMasterEvidence
        from .saa_l_shaped_audit_bridge import FormalFixedQEvidence
        validate_formal_problem(problem)
        need(isinstance(master_sink,NativeDevelopmentMasterEvidence),'Formal small master requires original native Gurobi evidence')
        need(isinstance(evidence_sink,FormalFixedQEvidence) and evidence_sink.problem is problem,
             'Formal kernel requires bound original LP event sink')
    else:need(problem.domain=='development','Unsupported SAA physical domain')
    for value in (budget_seconds,wall_seconds,master_time_limit_seconds,recourse_time_limit_seconds):
        need(type(value) in (int,float) and np.isfinite(value) and value>0,'Invalid time budget')
    need(master_time_limit_seconds<=600,'Single master exceeds original 600-second ceiling')
    need(type(max_iterations) is int and max_iterations>0,'Invalid iteration count')
    need(0<=mip_relative_gap<=.001 and 0<=optimality_relative_gap<=.005,'Invalid convergence tolerance')
    path=None if checkpoint_path is None else Path(checkpoint_path)
    cfg=dict(budget_seconds=budget_seconds,master_time_limit_seconds=master_time_limit_seconds,
        recourse_time_limit_seconds=recourse_time_limit_seconds,mip_relative_gap=mip_relative_gap,
        optimality_relative_gap=optimality_relative_gap)
    if master_sink is not None:cfg['master_evidence_identity']=master_sink.identity
    if resume:
        need(path is not None,'Resume requires checkpoint')
        need(isinstance(expected_checkpoint_sha256,str) and len(expected_checkpoint_sha256)==64
             and all(v in '0123456789abcdef' for v in expected_checkpoint_sha256),'Resume requires explicit trusted checkpoint hash')
        raw=path.read_bytes();need(hashlib.sha256(raw).hexdigest()==expected_checkpoint_sha256,'Trusted checkpoint bytes changed')
        envelope=json.loads(raw);s=envelope['state']
        need(envelope['sha256']==canonical_hash(s),'Checkpoint checksum changed')
        need(s['identity']==problem.identity and s['source_identity']==source_identity() and s['config']==cfg,
             'Checkpoint problem/source/budget/config changed')
        need(s['status'] in ('TIME_LIMIT','ITERATION_LIMIT','CONVERGED'),'Failed checkpoint needs diagnosis')
        if getattr(problem,'atom_store',None) is not None and 'last_incomplete_recourse' in s:
            problem.atom_store.verify_incomplete(s['last_incomplete_recourse'])
        elapsed=s['elapsed_seconds'];sessions=s['sessions']
        need(type(elapsed) in (int,float) and np.isfinite(elapsed) and elapsed>=0 and sessions,
             'Invalid cumulative elapsed time')
        need(all(type(v['elapsed_seconds']) in (int,float) and np.isfinite(v['elapsed_seconds']) and v['elapsed_seconds']>=0
                 and type(v['wall_seconds']) in (int,float) and np.isfinite(v['wall_seconds']) and v['wall_seconds']>0 for v in sessions)
             and abs(sum(v['elapsed_seconds'] for v in sessions)-elapsed)<=1e-9,'Session accounting mismatch')
        need(s['status']!='CONVERGED' or elapsed<=budget_seconds,'Converged checkpoint exceeds cumulative budget')
        need(len(s['cuts'])==len(problem.support_ids),'Checkpoint cut support count changed')
        masters=s['master_history'];bounds=[]
        for master in masters:
            lb=master['lower_bound_usd']
            need(master['recourse_columns']==0 and master['first_stage_columns']==len(problem.first_stage.c)
                 and master['theta_columns']==len(problem.support_ids),'Stored master axis changed')
            if lb is not None:
                need(master['solver_status'] in (0,1) and np.isfinite(lb),'Invalid recorded native lower bound');bounds.append(lb)
        need(s['lower_bound_usd']==(max(bounds) if bounds else None),'Lower-bound summary differs from native history')
        need(s['iterations']==len(s['trace']) and all(row['master'] in masters for row in s['trace']),
             'Iteration/native history mismatch')
        for i,atoms in enumerate(s['cuts']):
            for atom in atoms:_validate_atom(problem,i,atom)
        if s['pending'] is not None:
            pending=s['pending'];need(len(pending['atoms'])<=len(problem.support_ids) and pending['master'] in masters
                and pending['anchor']==pending['master']['primal'][:len(problem.first_stage.c)],'Pending sweep/master differs')
            for i,atom in enumerate(s['pending']['atoms']):
                need(_anchor_matches(atom,s['pending']['anchor']),'Pending Q belongs to a different plan');_validate_atom(problem,i,atom)
        if s['best'] is not None:
            best=s['best'];need(len(best['atoms'])==len(problem.support_ids),'Incomplete stored upper-bound sweep')
            for i,atom in enumerate(best['atoms']):
                need(_anchor_matches(atom,best['anchor']),'Best Q belongs to a different plan');_validate_atom(problem,i,atom)
            plan=problem.plan(best['anchor']);need(best['plan_sha256']==canonical_hash(dict(x=plan.x.tolist(),r=plan.r.tolist())),'Best plan identity changed')
            q=[a['q_usd'] for a in best['atoms']];fpre=float(problem.first_stage.c@np.asarray(best['anchor']))
            need(q==best['q_usd'] and fpre==best['Fpre_usd'] and float(fpre+np.mean(q))==s['upper_bound_usd'],'Stored upper bound changed')
        else:need(s['upper_bound_usd'] is None,'Upper bound lacks a complete physical witness')
        if s['status']=='CONVERGED':
            need(s['lower_bound_usd'] is not None and s['upper_bound_usd'] is not None
                 and max(0.,s['upper_bound_usd']-s['lower_bound_usd'])/max(1.,abs(s['upper_bound_usd']))<=optimality_relative_gap,'Converged checkpoint gap differs')
            return s
    else:
        need(expected_checkpoint_sha256 is None,'Initial run cannot consume a resume hash')
        need(path is None or not path.exists(),'Refusing to overwrite checkpoint')
        s=dict(identity=problem.identity,source_identity=source_identity(),config=cfg,status='INITIALIZED',
            elapsed_seconds=0.,sessions=[],master_history=[],cuts=[[] for _ in problem.support_ids],trace=[],pending=None,
            lower_bound_usd=None,upper_bound_usd=None,best=None,iterations=0,algorithm_id=problem.identity['algorithm_id'],
            risk_certificate_kind='fixed_empirical',scientific_acceptance=False,formal_admission=False,formal_OOS=False)
    s.pop('resource_stop_reason',None)  # Prior immutable checkpoint retains its original stop reason.
    previous=s['elapsed_seconds'];s['sessions'].append(dict(elapsed_seconds=0.,wall_seconds=wall_seconds))
    def left(limit=np.inf):return max(0.,min(limit,wall_seconds-(time.monotonic()-started),budget_seconds-previous-(time.monotonic()-started)))
    def save(status):
        elapsed=time.monotonic()-started;s['elapsed_seconds']=previous+elapsed;s['sessions'][-1]['elapsed_seconds']=elapsed;s['status']=status
        if path is not None:atomic_json(path,dict(state=s,sha256=canonical_hash(s)))
        return json.loads(json.dumps(s,allow_nan=False))
    for _ in range(max_iterations):
        if left()<=0:return save('TIME_LIMIT')
        if s['pending'] is None:
            if problem.domain=='formal' and left()<master_time_limit_seconds:
                s['resource_stop_reason']='INSUFFICIENT_REMAINING_FOR_FULL_FORMAL_MASTER_GRANT'
                return save('TIME_LIMIT')
            try:
                master=(_master(problem,s['cuts'],left(master_time_limit_seconds),mip_relative_gap) if master_sink is None
                    else _master(problem,s['cuts'],left(master_time_limit_seconds),mip_relative_gap,master_sink))
            except TimeoutError:return save('TIME_LIMIT')
            s['master_history'].append(master)
            lb=master['lower_bound_usd']
            if lb is not None:s['lower_bound_usd']=lb if s['lower_bound_usd'] is None else max(lb,s['lower_bound_usd'])
            if master['primal'] is None:return save('TIME_LIMIT' if master['solver_status']==1 else 'MASTER_FAILED')
            need(master['solver_status'] in (0,1),'Unexpected native master failure')
            vector=master['primal'][:len(problem.first_stage.c)]
            s['pending']=dict(master=master,anchor=vector,atoms=[])
            save('RUNNING')
        pending=s['pending'];plan=problem.plan(pending['anchor'])
        for i in range(len(pending['atoms']),len(problem.support_ids)):
            if left()<=0:return save('TIME_LIMIT')
            template=problem.recourses[i]
            result=solve_fixed_plan(template,plan,time_limit_seconds=left(recourse_time_limit_seconds))
            if evidence_sink is not None:evidence_sink(iteration=s['iterations']+1,scenario_index=i,template=template,plan=plan,result=result)
            if not result.optimal or not result.certificate['optimal_certificate_valid']:
                s['last_incomplete_recourse']=(problem.atom_store.retain_incomplete(i,plan,result)
                    if getattr(problem,'atom_store',None) is not None else result.as_dict())
                return save('TIME_LIMIT' if result.solver_status==1 else 'RECOURSE_FAILED')
            dual={k:list(result.certificate['dual_'+k+'_marginals']) for k in ('inequality','equality','lower','upper')}
            q=float(result.objective_usd+problem.constants_usd[i]);cut,g=build_dual_cut(_template(problem,i),dual,
                scenario_constant_usd=float(problem.constants_usd[i]),anchor_first_stage=pending['anchor'],anchor_q_usd=q)
            atom=dict(anchor=pending['anchor'],q_usd=q,primal=problem.recourses[i].pack(result.solution).tolist(),
                dual=dual,cut=cut,gradient=g.tolist());_validate_atom(problem,i,atom)
            if getattr(problem,'atom_store',None) is not None:atom=problem.atom_store.retain(i,atom)
            pending['atoms'].append(atom);save('RUNNING')
            # Release current scenario objects before the next bounded-cache load.
            del template,result,dual,g,cut,atom
        q=[a['q_usd'] for a in pending['atoms']];fpre=float(problem.first_stage.c@np.asarray(pending["anchor"]))
        ub=float(fpre+np.mean(q));plan_hash=canonical_hash(dict(x=plan.x.tolist(),r=plan.r.tolist()))
        if s['upper_bound_usd'] is None or ub<s['upper_bound_usd']:
            s['upper_bound_usd']=ub;s['best']=dict(anchor=pending['anchor'],q_usd=q,Fpre_usd=fpre,plan_sha256=plan_hash,atoms=pending['atoms'])
        lb=s['lower_bound_usd'];best=s['upper_bound_usd'];tol=1e-4+1e-7*max(1.,abs(best),abs(lb or 0.))
        need(lb is None or lb<=best+tol,'Bound contradiction')
        gap=None if lb is None else max(0.,best-lb)/max(1.,abs(best))
        newcuts=0
        for i,atom in enumerate(pending['atoms']):
            # Same numerical affine plane is useful only once, regardless of anchor.
            if not any(_plane_sha(a)==_plane_sha(atom) for a in s['cuts'][i]):
                s['cuts'][i].append(atom);newcuts+=1
        s['iterations']+=1;s['trace'].append(dict(iteration=s['iterations'],plan_sha256=plan_hash,
            repeated_plan=any(r['plan_sha256']==plan_hash for r in s['trace']),master=pending['master'],q_usd=q,
            candidate_upper_bound_usd=ub,lower_bound_usd=lb,upper_bound_usd=best,relative_gap=gap,added_cuts=newcuts))
        s['pending']=None
        if left()<=0:return save('TIME_LIMIT')
        if gap is not None and gap<=optimality_relative_gap:return save('CONVERGED')
        if pending['master']['solver_status']==1:return save('TIME_LIMIT')
        if newcuts==0:return save('STALLED_NO_NEW_CUT')
        save('RUNNING')
    return save('ITERATION_LIMIT')
