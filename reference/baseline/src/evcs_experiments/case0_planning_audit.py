"""Read-only original-array and physical audit of saved Case0 planning.

No MILP, recourse, probability optimization or Case1 admission is performed.
Producer versions remain explicit; the independent checker rederives each
extensive block and rechecks saved primal/dual witnesses against physical inputs.
"""
from .case0_profiles import context_options, validate_config

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.sparse import load_npz,hstack,csr_matrix

from .campaign_batch import file_reference,verify_reference,read_json
from .first_stage import FirstStagePlan,assemble_first_stage
from .first_stage_check import check_first_stage
from .recourse import assemble_recourse
from .recourse_check import check_recourse
from .probability_check import check_probability_solution
from .random_streams import canonical_hash
from .config import get_paths

METHODS=('EA_DRO','OR_DRO','SAA','ARO')
ZERO_REDUCTION_AUDITOR_SHA256='cc0f4695194e0978685d4a539273ebc0f5b3bc94ea7908831da9777318fdc2d0'


def _need(ok,msg):
    if not bool(ok):raise ValueError(msg)


def _time(deadline):
    _need(type(deadline) in (float,int) and math.isfinite(deadline),'Finite audit deadline required')
    if time.monotonic()>=deadline:raise TimeoutError('Case0 independent planning audit deadline')


def _read(ref):
    verify_reference(ref);return read_json(ref['path'])


def _close(a,b,label):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    _need(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),label+' shape/nonfinite')
    _need(np.all(abs(a-b)<=1e-4+1e-7*np.maximum(1.,np.maximum(abs(a),abs(b)))),label+' differs')


def _equal(a,b,label):
    _need(np.array_equal(a,b),label+' differs from original coefficients')


def _max(a):return float(np.max(a,initial=0.))


def audit_case0_zero_reduction(matrix,arrays,proof,vector,native=None):
    """Replay the existing independent exact proof, then check its zero lift.

    The checker remains the unchanged, separately hashed audit script. No
    presolver, optimizer, coefficient tolerance or new mathematical reduction
    is introduced here. The caller also checks every ORIGINAL physical row.
    """
    path=get_paths().project_root/'scripts/audits/audit_zero_reduction.py'
    source=file_reference(path)
    _need(source['sha256']==ZERO_REDUCTION_AUDITOR_SHA256,'Independent zero-reduction checker source differs')
    _need(type(proof) is dict and type(proof.get('original_columns')) is int and
          proof['original_columns']==matrix.shape[1] and proof.get('original_constraint_check_required') is True,
          'Zero-reduction original column/check scope differs')
    spec=importlib.util.spec_from_file_location('case0_existing_independent_zero_auditor',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    checked=module.audit_zero_reduction(matrix,arrays['row_lower'],arrays['row_upper'],
        arrays['lower_bounds'],arrays['upper_bounds'],proof)
    v=np.asarray(vector,float);cols=np.asarray(proof['retained_columns'],int)
    zeros=np.asarray(proof['zero_columns'],int)
    _need(v.shape==arrays['c'].shape and np.isfinite(v).all(),'Zero-lift original vector shape/nonfinite')
    _need(np.array_equal(v[zeros],np.zeros(len(zeros))),'Eliminated original columns are not exactly zero')
    lifted=np.zeros(len(v));lifted[cols]=v[cols]
    _need(np.array_equal(v,lifted),'Zero-reduction original vector lift differs')
    _close(arrays['c']@v,arrays['c'][cols]@v[cols],'Zero-lift original objective')
    if native is not None:
        rows=np.asarray(proof['retained_rows'],int)
        lo=arrays['row_lower'][rows];hi=arrays['row_upper'][rows]
        equal=np.isfinite(lo)&np.isfinite(hi)&(lo==hi)
        mapping={'eq':np.flatnonzero(equal),'le':np.flatnonzero(np.isfinite(hi)&~equal),
                 'ge':np.flatnonzero(np.isfinite(lo)&~equal)}
        rowmap={k:dict(count=len(v),source_rows_sha256=hashlib.sha256(v.astype('<i8').tobytes()).hexdigest())
                for k,v in mapping.items()}
        rowmap.update(original_rows=len(rows),translated_rows=sum(len(v) for v in mapping.values()),
            free_rows=int(np.count_nonzero(np.isneginf(lo)&np.isposinf(hi))),
            ranged_rows=int(np.count_nonzero(np.isfinite(lo)&np.isfinite(hi)&(lo<hi))))
        _need(native['original_matrix_shape']==proof['reduced_shape'] and
              native['original_matrix_nnz']==proof['reduced_nnz'] and native['row_map_summary']==rowmap and
              native['model_columns']==len(cols) and native['model_rows']==rowmap['translated_rows'],
              'Native reduced matrix/row translation differs from original proof')
    return dict(checked,passed=True,checker_source_reference=source,
        original_zero_lift_checked=True,original_objective_mapping_checked=True,
        original_objective_constant_usd=float(arrays['objective_constant_usd']),
        eliminated_rows=matrix.shape[0]-checked['retained_rows'],
        reduced_nnz=proof['reduced_nnz'],coefficient_threshold=None)


def _physical_master_arrays(A,arrays,header,first,templates,ids,P,constants,deadline):
    """Independent block inspection; never call assemble_extensive."""
    risk=header['risk_model'];nf=len(first.c);h=len(first.task_ids);ns=len(ids)
    _need(risk in ('DRO','SAA','ARO'),'Unknown risk model')
    slices={'x':[0,h],'r':[h,nf],'first_stage':[0,nf],'recourses':{}}
    cursor=nf
    for sid,t in zip(ids,templates):slices['recourses'][sid]=[cursor,cursor+len(t.c)];cursor+=len(t.c)
    riskstart=cursor
    if risk=='DRO':
        _need(P is not None and tuple(P.support_ids)==tuple(ids),'DRO ambiguity axis differs')
        for key,length in (('theta0',1),('alpha',len(P.feature_ids)),('beta',len(P.feature_ids))):slices[key]=[cursor,cursor+length];cursor+=length
    elif risk=='ARO':slices['risk_threshold']=[cursor,cursor+1];cursor+=1
    n=cursor;_need(header['variable_slices']==slices and A.shape[1]==n,'Original extensive variable map differs')
    row=0;blocks=[]
    def block(expected,lo,hi,scope,kind,digest):
        nonlocal row
        _time(deadline);stop=row+expected.shape[0]
        _need(A[row:stop].shape==expected.shape and (A[row:stop]!=expected).nnz==0,'Original extensive coefficient block differs: '+kind)
        _equal(arrays['row_lower'][row:stop],lo,'row lower');_equal(arrays['row_upper'][row:stop],hi,'row upper')
        blocks.append(dict(scope=scope,kind=kind,row_start=row,row_stop=stop,source_sha256=digest));row=stop
    block(hstack([first.A,csr_matrix((first.A.shape[0],n-nf))],format='csr'),first.row_lower,first.row_upper,'first_stage','first_stage',first.matrix_sha256)
    for sid,t in zip(ids,templates):
        a,b=slices['recourses'][sid]
        for sense in ('ub','eq'):
            matrix=getattr(t,'A_'+sense);rhs=getattr(t,'b_'+sense)
            expected=hstack([-getattr(t,'Bx_'+sense),-getattr(t,'Br_'+sense),csr_matrix((len(rhs),a-nf)),matrix,csr_matrix((len(rhs),n-b))],format='csr')
            block(expected,np.full(len(rhs),-np.inf) if sense=='ub' else rhs,rhs,sid,'recourse_'+sense,t.template_sha256)
    cost=np.zeros(n);cost[:nf]=first.c
    lower=np.r_[first.lower_bounds,*[t.lower_bounds for t in templates],np.zeros(n-riskstart)]
    upper=np.r_[first.upper_bounds,*[t.upper_bounds for t in templates],np.full(n-riskstart,np.inf)]
    integ=np.r_[first.integrality,np.zeros(n-nf,dtype=int)];pi=None;offset=0.
    if risk=='SAA':
        pi=np.full(ns,1./ns);_equal(header['empirical_probabilities'],pi,'uniform SAA law')
        for sid,t in zip(ids,templates):cost[slice(*slices['recourses'][sid])]= (1./ns)*t.c
        offset=float(pi@constants)
    else:
        _need(header['empirical_probabilities'] is None,'Non-SAA empirical law')
        lower[riskstart]=-np.inf;cost[riskstart]=1.
        if risk=='DRO':cost[slice(*slices['alpha'])]=-P.lower;cost[slice(*slices['beta'])]=P.upper
        for i,(sid,t) in enumerate(zip(ids,templates)):
            a,b=slices['recourses'][sid]
            riskrow=np.r_[-1.,P.features[i],-P.features[i]] if risk=='DRO' else np.array([-1.])
            expected=hstack([csr_matrix((1,a)),csr_matrix(t.c.reshape(1,-1)),csr_matrix((1,riskstart-b)),csr_matrix(riskrow.reshape(1,-1))],format='csr')
            block(expected,np.array([-np.inf]),np.array([-constants[i]]),sid,'risk_epigraph',t.template_sha256)
    _need(row==A.shape[0] and header['block_map']==blocks,'Unexpected extensive rows/block map')
    for k,v in [('c',cost),('lower_bounds',lower),('upper_bounds',upper),('integrality',integ)]:_equal(arrays[k],v,k)
    _equal(arrays['objective_constant_usd'],offset,'constant');_equal(header['objective_constant_usd'],offset,'header constant')
    _need(header['first_stage_sha256']==first.matrix_sha256 and header['recourse_template_sha256s']==[t.template_sha256 for t in templates] and
          header['scenario_ids']==list(ids) and header['scenario_sha256s']==[t.scenario_sha256 for t in templates], 'Physical template identities differ')
    ambiguity=None if P is None else P.ambiguity_sha256
    _need(header['ambiguity_sha256']==ambiguity,'Ambiguity identity differs')
    metadata=dict(risk_model=risk,scenario_ids=ids,scenario_sha256s=[t.scenario_sha256 for t in templates],template_sha256s=[t.template_sha256 for t in templates],first_stage_sha256=first.matrix_sha256,ambiguity_sha256=ambiguity,empirical_probabilities=None if pi is None else pi.tolist(),slices=slices,objective_constant_usd=offset)
    digest=hashlib.sha256(json.dumps(metadata,sort_keys=True,separators=(',',':')).encode())
    for a in (A.data,A.indices,A.indptr,arrays['row_lower'],arrays['row_upper'],lower,upper,cost,integ,constants):
        v=np.asarray(a,dtype='<i8' if np.asarray(a).dtype.kind in 'iu' else '<f8');digest.update(str(v.shape).encode());digest.update(v.tobytes())
    _need(digest.hexdigest()==header['model_sha256'],'Original extensive semantic/array hash differs')
    return slices,offset


def _kkt(t,plan,primal,saved):
    """Independently inspect saved LP vectors, not producer audit booleans."""
    _need(saved['status']=='OPTIMAL' and saved['solver_status']==0 and saved['has_solution'] is True,'Saved fixed Q lacks optimal witness')
    c=saved['certificate'];_need(c['template_sha256']==t.template_sha256,'Q template identity differs')
    y,e,lo,hi=(np.asarray(c[k],float) for k in ('dual_inequality_marginals','dual_equality_marginals','dual_lower_marginals','dual_upper_marginals'))
    bub,beq=t.fixed_rhs(plan);_need(primal.shape==t.c.shape and y.shape==bub.shape and e.shape==beq.shape and lo.shape==hi.shape==primal.shape and all(np.isfinite(v).all() for v in (primal,y,e,lo,hi)), 'Q witness axes/nonfinite')
    pe=max(_max(t.A_ub@primal-bub),_max(abs(t.A_eq@primal-beq)),_max(t.lower_bounds-primal),_max(primal-t.upper_bounds))
    station=_max(abs(t.c-t.A_ub.T@y-t.A_eq.T@e-lo-hi));fl=np.isfinite(t.lower_bounds);fu=np.isfinite(t.upper_bounds)
    signs=max(_max(y),_max(-lo),_max(hi),_max(abs(lo[~fl])),_max(abs(hi[~fu])))
    _need(pe<=1e-7 and max(station,signs)<=1e-8,'Original full-Q primal/dual KKT violated')
    primal_cost=float(t.c@primal);dual=float(bub@y+beq@e+t.lower_bounds[fl]@lo[fl]+t.upper_bounds[fu]@hi[fu])
    _close(primal_cost,dual,'Q strong duality');_close(saved['objective_usd'],primal_cost,'Saved Q objective')
    return dict(primal_error=pe,stationarity_error=station,dual_sign_error=signs,qop_usd=primal_cost,dual_usd=dual)


def audit_case0_method(*,result_reference,case,tasks,scenes,probability_set,deadline):
    """Audit saved raw artifacts against caller's exact physical input context."""
    _time(deadline);result=_read(result_reference);out=Path(result_reference['path']).parent
    method=result['method_id'];_need(method in METHODS and result['kind']=='CASE0_FULL_FORMULATION_NUMERICAL_RESULT_V1','Wrong Case0 result')
    risk='DRO' if method.endswith('DRO') else method;_need(result['risk_model']==risk,'Method/risk mismatch')
    ids=tuple(result['scenario_ids']);scenes=tuple(scenes)
    _need(len(ids)==len(scenes)>0 and len(set(ids))==len(ids),'Incomplete planning support')
    first=assemble_first_stage(case,tasks);templates=[]
    for sid,scene in zip(ids,scenes):
        _time(deadline);_need(scene.development is True and scene.split=='planning' and scene.regime=='R1' and scene.scenario_id==sid,'Case0 physical planning domain/axis differs')
        templates.append(assemble_recourse(case,tasks,scene))
    header=read_json(out/'model.json');A=load_npz(out/'matrix.npz').tocsr()
    with np.load(out/'model_arrays.npz',allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    _need(np.isfinite(A.data).all(),'Nonfinite master coefficient');constants=np.asarray(header['scenario_cost_constants_usd'],float)
    _need(constants.shape==(len(ids),) and np.isfinite(constants).all(),'Invalid scenario constants')
    # Case0 caller never supplies external scenario constants.
    _equal(constants,np.zeros(len(ids)),'Case0 zero additive scenario costs')
    slices,offset=_physical_master_arrays(A,arrays,header,first,templates,ids,probability_set if risk=='DRO' else None,constants,deadline)
    _need(result['model_sha256']==header['model_sha256'] and header['risk_model']==risk,'Result/model identity differs')
    with np.load(out/'plan_and_master_vector.npz',allow_pickle=False) as z:
        vector=z['vector'].copy();plan=FirstStagePlan(z['x'],z['r'],str(z['task_sha256'].item()),str(z['config_hash'].item()))
    _need(canonical_hash(result['plan'])==result['plan_sha256'],'Plan hash differs')
    _equal(plan.x,result['plan']['x'],'plan x');_equal(plan.r,result['plan']['r'],'plan reserve')
    _need(plan.task_sha256==tasks.task_sha256==result['plan']['task_sha256'] and plan.config_hash==result['plan']['config_hash'],'Plan task/config differs')
    _need(vector.shape==arrays['c'].shape and np.isfinite(vector).all(),'Master primal shape/nonfinite')
    _equal(vector[:len(first.c)],np.r_[plan.x,plan.r.ravel()],'Master first-stage vector')
    activity=A@vector
    errors=[_max(arrays['row_lower']-activity),_max(activity-arrays['row_upper']),_max(arrays['lower_bounds']-vector),_max(vector-arrays['upper_bounds']),_max(abs(vector[arrays['integrality']==1]-np.rint(vector[arrays['integrality']==1])))]
    _need(max(errors)<=1e-8,'Original master primal/integrality violated')
    master=read_json(out/'master.json');raw=master['raw_solver'];status=raw['status']
    zero_check=None
    if 'zero_reduction' in raw:
        _need(result.get('zero_presolve') is True,'Unrequested Case0 exact reduction')
        zero_check=audit_case0_zero_reduction(A,arrays,raw['zero_reduction'],vector,
            raw.get('native_interface') if result['backend']=='gurobi' else None)
        _need(result['zero_reduction_proof_sha256']==zero_check['proof_sha256'],'Result zero-reduction proof binding differs')
        source=result['zero_reduction_checker_reference'];verify_reference(source)
        _need(source['sha256']==ZERO_REDUCTION_AUDITOR_SHA256,'Producer zero-reduction checker source differs')
        saved=read_json(out/'zero_reduction_audit.json')
        _need(saved.get('checker_source_reference')==source and
              {k:v for k,v in saved.items() if k!='checker_source_reference'}==
              {k:v for k,v in zero_check.items() if k!='checker_source_reference'},
              'Saved exact reduction audit differs from independently replayed proof')
    else:
        _need('zero_presolve' not in result and 'zero_reduction_proof_sha256' not in result and
              'zero_reduction_checker_reference' not in result and not (out/'zero_reduction_audit.json').exists(),
              'Case0 exact reduction claim lacks native proof')
    _need(type(status) is int and status in (0,1) and master['solver_status']==status==result['native_status_code'],'Unusable native master status')
    _need(master['status']==result['native_status'] and master['solution']['model_sha256']==header['model_sha256'],'Native master identity/status differs')
    _equal(master['solution']['x'],plan.x,'Native plan x');_equal(master['solution']['r'],plan.r,'Native plan reserve')
    opts=raw['options'];_need(opts['threads']==1 and opts['mip_rel_gap']==.005 and 0<opts['time_limit']<=1200.,'Original Case0 master resources differ')
    if result['backend']=='gurobi':
        native=raw['native_interface'];eff=native['effective_options']
        _need(raw['solver']=='gurobipy/Gurobi' and native['gurobi_version']==[12,0,3] and native['certificate_failures']==[] and
              native['valid_incumbent'] is True and native['is_mip'] is True and native['bound_source']=='MIP_ObjBound', 'Original native bound certificate metadata differs')
        _need(native['raw_model_status']==(2 if status==0 else 9) and native['status']==('OPTIMAL' if status==0 else 'TIME_LIMIT'),'Native raw/normalized status differs')
        _need(eff['Threads']==1 and eff['MIPGap']==.005 and eff['FeasibilityTol'] in (1e-8,1e-9) and eff['OptimalityTol']==eff['IntFeasTol']==1e-8,'Native scientific/resource tolerances differ')
        _close(native['solver_bound_without_constant_usd'],raw['raw_lower_bound_without_constant_usd'],'Native bound source')
    else:_need(result['backend']=='highs' and raw['solver'].startswith(('scipy.optimize.milp/HiGHS','SciPy-bundled HiGHS')),'Backend mismatch')
    lower=float(raw['raw_lower_bound_without_constant_usd'])+offset;witness=float(arrays['c']@vector)+offset
    _close(raw['raw_objective_without_constant_usd']+offset,witness,'Native objective offset');_close(master['objective_usd'],witness,'Master objective');_close(master['lower_bound_usd'],lower,'Master lower bound')
    firstcheck=check_first_stage(case,tasks,plan);_need(firstcheck['passed'],'Independent first-stage physical violation')
    q=[];checks=[];costrows=[]
    for i,(sid,t,scene) in enumerate(zip(ids,templates,scenes)):
        _time(deadline);folder=out/f'scenario_{i:03d}'
        with np.load(folder/'primal.npz',allow_pickle=False) as z:y=z['fixed'].copy();w=z['witness'].copy()
        _equal(w,vector[slice(*slices['recourses'][sid])],'Native recourse witness vector')
        fixed=read_json(folder/'fixed_solver.json');kkt=_kkt(t,plan,y,fixed)
        wc=check_recourse(case,tasks,scene,plan,t.unpack(w),check_objective_bound=False)
        physical=check_recourse(case,tasks,scene,plan,t.unpack(y),check_objective_bound=True)
        _need(wc['passed'] and physical['passed'],'Independent full-Q/native-witness physical violation')
        _close(kkt['qop_usd'],physical['costs']['Qop'],'Independent physical Q cost')
        q.append(float(physical['costs']['Qop']));checks.append(kkt);costrows.append(dict(scenario_id=sid,**physical['costs'],scenario_constant_usd=0.))
        _need(q[-1]<=wc['costs']['Qop']+1e-4+1e-7*max(1.,abs(q[-1])),'Fixed Q exceeds extensive witness')
    q=np.asarray(q)
    if risk=='DRO':
        prob=read_json(out/'probability.json');pcheck=check_probability_solution(probability_set,q,prob,q_support_ids=ids)
        _need(pcheck.passed,'Independent original moment-risk witness violated');riskvalue=float(prob['dual_objective_usd'])
    else:riskvalue=float(q.max()) if risk=='ARO' else float(q.mean())
    upper=float(firstcheck['costs']['Fpre_usd']+riskvalue);gap=max(0.,upper-lower)/max(1.,abs(upper))
    for key,value in [('q_usd',q),('Fpre_usd',firstcheck['costs']['Fpre_usd']),('risk_value_usd',riskvalue),('lower_bound_usd',lower),('upper_bound_usd',upper),('gap',gap)]:_close(result[key],value,key)
    _need(lower<=upper+1e-4+1e-7*max(1.,abs(upper)) and upper<=witness+1e-4+1e-7*max(1.,abs(witness)) and gap<=.005,'Independent risk bound/gap order failed')
    _need(len(result['cost_components'])==len(costrows),'Cost component coverage')
    for saved,computed in zip(result['cost_components'],costrows):
        _need(saved['scenario_id']==computed['scenario_id'],'Cost component axis')
        for key in computed:
            if key!='scenario_id':_close(saved[key],computed[key],'Cost '+key)
    _time(deadline)
    report=dict(method_id=method,passed=True,model_sha256=header['model_sha256'],plan_sha256=result['plan_sha256'],scenario_count=len(ids),lower_bound_usd=lower,upper_bound_usd=upper,gap=gap,native_status=master['status'],native_branch_and_bound_proof_replayed=False,master_max_violation=max(errors),full_q_checks=checks,original_master_coefficients_checked=True,full_q_kkt_physical_checked=True,risk_bounds_checked=True)
    if zero_check is not None:report['zero_reduction_audit']=zero_check
    return report


def audit_case0_planning(*,request,deadline):
    """Audit the selected four-method group; only writes the requested audit card.

    The caller owns parent accounting and interruption settlement.  Existing input
    artifacts are mandatory, so reopening cannot generate tasks or calibration.
    """
    from .case0_evaluation_runner import planning_evidence
    from .case0_inputs import Case0Context
    _time(deadline);root=Path(request['root']).resolve()
    config,state,evidence=planning_evidence(root)
    _need(state['status']=='PLANNING_REVIEW_REQUIRED' and state['size_index'] in (0,1),
          'Case0 planning review state required')
    original=file_reference(root/'config.lock.json')
    active=state.get('active_config_reference',original)
    _need(request['config_reference']==active and request['original_config_reference']==original,
          'Audit request configuration differs')
    _need(request['method_evidence']==evidence,'Audit request method evidence differs')
    sources=[file_reference(p) for p in sorted(Path(__file__).parent.rglob('*.py'))]
    review=_read(request['independent_review_reference'])
    _need(review['kind']=='CASE0_NUMERICAL_AUDITOR_REVIEW_V1' and review['passed'] is True and
          review['auditor_source_references']==sources,'Auditor exact source lacks independent review')
    manifests=request['producer_source_manifest_references']
    _need(set(manifests)==set(METHODS),'Complete original producer source manifests required')
    # These inherited numerical/input compilers must retain producer bytes. The
    # orchestration/auditor is deliberately a separately reviewed source version.
    numerical=('case0_inputs.py','first_stage.py','first_stage_check.py','recourse.py',
               'recourse_check.py','probability_check.py','random_streams.py',
               'onset_physical_inputs.py','joint_physical_inputs.py')
    artifact_paths=set();before={};reports={};identities={}
    for method in METHODS:
        _time(deadline);entry=evidence[method];manifest=_read(manifests[method])
        producer=Path(manifests[method]['path']).parent
        inventory={row['path']:row['sha256'] for row in manifest['files']}
        _need(len(inventory)==len(manifest['files']),'Duplicate producer inventory path')
        mapping=entry['producer_source_sha256']
        for name,digest in mapping.items():
            _time(deadline);path=producer/name
            _need(not Path(name).is_absolute() and path.resolve().is_relative_to(producer.resolve()),'Producer source path escapes')
            _need(inventory.get(name)==digest and file_reference(path)['sha256']==digest,'Producer source/config/manifest differs')
        for name in numerical:
            current=Path(__file__).parent/name
            if current.exists():
                key='src/evcs_experiments/'+name
                _need(mapping.get(key)==file_reference(current)['sha256'],'Numerical compiler differs from original producer: '+name)
        directory=Path(entry['result_reference']['path']).parent.parent
        input_directory=Path(entry['input_identity_reference']['path']).parent
        identity=_read(entry['input_identity_reference']);identities[method]=identity
        saved_result=_read(entry['result_reference'])
        _need(saved_result['method_id']==method and saved_result['backend']==config['backend'],'Method/backend producer identity differs')
        _need((directory/'inputs').resolve()==input_directory and (input_directory/'tasks.json').is_file(),'Original task input required')
        if method.endswith('DRO'):
            _need((input_directory/'calibration_manifest.json').is_file(),'Original DRO calibration required; audit cannot generate it')
        paths={str(p.resolve()) for parent in (directory/'solve',input_directory) for p in parent.rglob('*') if p.is_file()}
        paths.update(ref['path'] for key,ref in entry.items() if key.endswith('_reference'))
        artifact_paths.update(paths)
        for path in sorted(paths):_time(deadline);before[path]=file_reference(path)
        ctx=Case0Context(input_directory,case_root=request.get('case_root'),
            lead_hours=0 if method=='OR_DRO' else 1,**context_options(config),**config['sizes'][state['size_index']])
        _need(ctx.identity==identity and identity['formal_OOS'] is False,'Original input identity differs')
        P=ctx.probability_set if method.endswith('DRO') else None
        scenes=[]
        for i in range(len(ctx.planning_ids)):
            _time(deadline);sample=ctx.sample(i);scenes.append(sample.physical)
            if P is not None:_equal(P.features[i],sample.features,'Original planning feature row')
        reports[method]=audit_case0_method(result_reference=entry['result_reference'],case=ctx.case,
            tasks=ctx.tasks,scenes=scenes,probability_set=P,deadline=deadline)
    _need(len({tuple(i['vehicle_ids']) for i in identities.values()})==1,'Method fleet differs')
    _need(identities['EA_DRO']['task_sha256']==identities['SAA']['task_sha256']==identities['ARO']['task_sha256'], 'Baseline H differs')
    for path in sorted(artifact_paths):
        _time(deadline);_need(file_reference(path)==before[path],'Original artifact mutated during audit')
    _need(planning_evidence(root)[2]==evidence,'Completed method evidence changed during audit')
    card=dict(kind='CASE0_INDEPENDENT_PLANNING_NUMERICAL_AUDIT_V1',passed=True,
        config_reference=active,original_config_reference=original,size_index=state['size_index'],
        result_references={m:evidence[m]['result_reference'] for m in METHODS},
        input_identity_references={m:evidence[m]['input_identity_reference'] for m in METHODS},
        artifact_references=[before[p] for p in sorted(artifact_paths)],method_evidence=evidence,
        producer_source_manifest_references=manifests,auditor_source_references=sources,
        independent_review_reference=request['independent_review_reference'],method_reports=reports,
        original_master_coefficients_checked=True,full_q_kkt_physical_checked=True,risk_bounds_checked=True,
        native_branch_and_bound_proof_replayed=False,formal_case1_pass=False,
        original_artifacts_unchanged=True)
    _time(deadline)
    if request.get('output_path'):
        output=Path(request['output_path']).resolve()
        _need(not output.exists() and not output.is_relative_to(root/'planning'),'Audit output must be new and outside original planning')
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(card,indent=2,allow_nan=False)+'\n')
        _time(deadline)
    return card


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',required=True);parser.add_argument('--seconds',required=True,type=float)
    args=parser.parse_args();_need(math.isfinite(args.seconds) and args.seconds>0,'Positive audit duration required')
    deadline=time.monotonic()+args.seconds
    card=audit_case0_planning(request=read_json(args.request),deadline=deadline)
    print(json.dumps(dict(kind=card['kind'],passed=card['passed'],size_index=card['size_index'])))


if __name__=='__main__':main()
