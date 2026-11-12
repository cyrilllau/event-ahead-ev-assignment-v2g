"""Independent original-array and physical audit of a fresh full160 candidate.

This component calls no optimizer and does not assemble a planning problem.
Its reviewed parent owns runtime/source isolation and cumulative accounting.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib,importlib.util,json,math,sys,time
import numpy as np
from scipy.sparse import load_npz
from evcs_experiments.config import get_paths
from evcs_experiments.first_stage_check import check_first_stage
from evcs_experiments.recourse_check import check_recourse
from evcs_experiments.physical_inputs import load_physical_case,load_frozen_tasks
from evcs_experiments.formal_physical_inputs import FormalPhysicalBank
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.runtime.jobctl import atomic_json

PASS='PASS_INDEPENDENT_CANDIDATE_160_RECOURSE_AUDIT'
HELPER_SHA256='041613210cfd024382f270d8c709ccd0de4586c2a86758a7b20c15739854714d'
REQUIRED=('audit_script','helper_script','candidate_result','candidate_job_status','original_checkpoint',
          'original_raw_request','original_history','saved160_acceptance','original_source_manifest')

def need(ok,msg):
    if not bool(ok):raise ValueError(msg)

def first_digest(meta,A,v):
    body={k:meta[k] for k in ('task_sha256','config_hash','task_ids','periods','variable_map','constraint_map','units')}
    h=hashlib.sha256(json.dumps(body,sort_keys=True).encode())
    for x in (A.data,A.indices,A.indptr,v['row_lower'],v['row_upper'],v['lower_bounds'],v['upper_bounds'],v['integrality'],v['c']):
        a=np.asarray(x,dtype='<i8' if np.asarray(x).dtype.kind in 'iu' else '<f8');h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def validate_completion(result,cp,snapshot,identity):
    ids=identity['scenario_ids'];q=np.asarray(cp['q_usd'],dtype=float)
    need(result['status']==cp['status']=='FULL_Q_PENDING_INDEPENDENT_AUDIT_AND_PROBABILITY' and result['completed_count']==cp['completed_count']==cp['next_index']==160,'Incomplete candidate evaluation')
    need(len(ids)==len(set(ids))==160 and cp['support_ids']==ids and q.shape==(160,) and np.isfinite(q).all(),'Full ordered finite q required')
    need(result['full_q_available'] is True and result['probability_evaluated'] is False and result['formal_OOS'] is False and cp['formal_OOS'] is False,'Wrong completion scope')
    need(cp['plan_sha256']==result['plan_sha256'] and result['problem_sha256']==canonical_hash(identity),'Wrong candidate identity')
    need(cp['recourse_event_references']==snapshot['events'] and len(snapshot['events'])==160 and len({r['path'] for r in snapshot['events']})==160 and snapshot['templates']==[],'New raw callback/template coverage differs')
    return ids,q

def audit_scenario(h,*,index,scenario_id,plan_dict,constant,expected_template_sha256,meta,matrices,arrays,saved,case,tasks,scene,expected_iteration=1):
    """Recompute one genuine saved scenario's physics, prices, RHS and KKT."""
    plan=SimpleNamespace(x=np.asarray(plan_dict['x']),r=np.asarray(plan_dict['r']),task_sha256=plan_dict['task_sha256'],config_hash=plan_dict['config_hash'])
    event=saved['event'];solver=saved['solver'];solution=saved['solution'];z=saved['vector']
    need(event['event']=='recourse' and event['status']=='RAW_EVIDENCE_RETAINED_PENDING_AUDIT' and event['optimal_certificate_valid'] is True
         and event['scenario_index']==index and event['scenario_id']==scenario_id and event['plan_sha256']==canonical_hash(plan_dict) and type(expected_iteration) is int and expected_iteration>=1 and event['iteration']==expected_iteration,'Wrong raw scenario/plan/status')
    need(saved['plan']==plan_dict and scene.scenario_id==scenario_id,'Changed physical plan/scenario')
    digest=h.template_digest(meta,matrices,arrays,meta['first_stage_sha256'] if 'first_stage_sha256' in meta else saved['first_stage_sha256'])
    need(digest==expected_template_sha256==meta['template_sha256']==solver['certificate']['template_sha256'],'Original template semantic hash differs')
    need(meta['scenario_sha256']==scene.scenario_sha256 and meta['task_sha256']==tasks.task_sha256 and meta['config_hash']==tasks.config_hash and tuple(meta['task_ids'])==tasks.task_ids,'Template physical identity differs')
    h.same(z,np.concatenate([np.asarray(solution[k]).ravel() for k in h.ORDER]),'Primal vector differs from physical witness')
    physical=check_recourse(case,tasks,scene,plan,SimpleNamespace(**solution),check_objective_bound=True)
    need(physical['passed'],'Independent original-unit physical check failed')
    kkt=h.audit_kkt(matrices,arrays,plan,z,saved['dual'],solver,saved['fixed'])
    v=arrays;vs=meta['variable_slices'];dt=case.dt_h;cost=np.zeros_like(v['component_objectives'])
    cost[0,slice(*vs['shed'])]=np.broadcast_to(case.c_shed_usd_per_kwh[None,:]*dt,solution['shed'].shape).ravel()
    cost[1,slice(*vs['short'])]=case.c_short_usd_per_kwh*dt
    cost[2,slice(*vs['p'])]=np.broadcast_to(tasks.c_v2g_usd_per_kwh[:,None]*dt,solution['p'].shape).ravel()
    cost[3,slice(*vs['p'])]=np.broadcast_to(tasks.c_deg_usd_per_kwh[:,None]*dt/tasks.eta_dis[:,None],solution['p'].shape).ravel()
    h.same(v['component_objectives'],cost,'Original physical cost coefficients/units changed')
    h.same(v['c'],cost.sum(axis=0),'Original objective aggregation changed')
    tol=1e-4+1e-7*abs(kkt['primal_objective_usd'])
    h.close(physical['costs']['Qop'],kkt['primal_objective_usd'],'Independent physical objective differs',tol)
    need(math.isfinite(constant) and event['q_usd']==solver['objective_usd']+constant,'Original scenario constant or q differs')
    h.close(event['q_usd'],kkt['primal_objective_usd']+constant,'Original-array q differs',tol)
    return dict(index=index,scenario_id=scenario_id,q_usd=event['q_usd'],template_sha256=digest,physical_passed=True,**kkt)

def audit_candidate_recourse(request_refs,*,output,deadline):
    """Return a full160 numerical audit, or retain failure/progress and raise.

Required strong references are listed in REQUIRED. The original raw arrays and
new candidate artifacts are rehashed, then numerical/physical checks are rerun.
"""
    need(math.isfinite(deadline) and 0<deadline-time.monotonic()<=7190,'Invalid bounded audit deadline')
    need(all(k in request_refs for k in REQUIRED),'Missing audit input reference')
    hp=Path(request_refs['helper_script']['path']).resolve(strict=True)
    need(hashlib.sha256(hp.read_bytes()).hexdigest()==request_refs['helper_script']['sha256']==HELPER_SHA256,'Changed reviewed helper source')
    spec=importlib.util.spec_from_file_location('_candidate_original_array_auditor',hp);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    a=h.Audit(deadline-time.monotonic());out=Path(output).resolve();paths=get_paths();done=[];current=None
    # No output can replace source, candidate evidence, original evidence or data.
    protected=[paths.data_root.resolve(),paths.frozen_source_root.resolve(),Path(__file__).resolve().parents[2],Path(request_refs['candidate_result']['path']).resolve().parent,Path(request_refs['original_raw_request']['path']).resolve().parent]
    need(not any(out.is_relative_to(p) for p in protected),'Audit output overlaps protected content')
    out.mkdir(parents=True,exist_ok=False)
    try:
        for ref in request_refs.values():a.verify(ref)
        need(a.ref(__file__)==request_refs['audit_script'],'Audit executable not bound')
        result=h.read(request_refs['candidate_result']['path']);job=h.read(request_refs['candidate_job_status']['path'])
        need(job['complete'] is True and job['state']=='SUCCEEDED' and job['returncode']==0 and job['job_id']==result['parent']['job_id'],'Candidate parent not terminal success')
        a.verify(result['parent']['manifest_reference']);a.verify(result['request_reference']);jobrequest=h.read(result['request_reference']['path'])
        need(jobrequest['kind']=='S05_FIXED_CANDIDATE_FULL_SUPPORT_JOB_V1' and jobrequest['resume'] is False,'Wrong candidate source request')
        for name in ('checkpoint_reference','raw_snapshot_reference','candidate_binding_reference','original_candidate_binding'):a.verify(result[name])
        cp=h.read(result['checkpoint_reference']['path']);snapshot=h.read(result['raw_snapshot_reference']['path'])
        old=h.read(request_refs['original_checkpoint']['path']);state=old['state'];identity=state['identity'];oldraw=h.read(request_refs['original_raw_request']['path'])
        card=h.read(request_refs['saved160_acceptance']['path'])
        need(old['sha256']==canonical_hash(state) and state['problem_sha256']==canonical_hash(identity)==card['problem_sha256'],'Original problem semantic identity differs')
        need(card['status']=='PASS_SAVED160_COMPONENT_REVIEW' and card['complete_lp_count']==160,'Original templates not accepted')
        for key in ('original_checkpoint','original_raw_request','original_history','original_source_manifest'):need(request_refs[key] in card['references'].values(),'Unbound original source '+key)
        ids,q=validate_completion(result,cp,snapshot,identity);raw=Path(snapshot['evidence_root']).resolve()
        need(raw==Path(request_refs['candidate_result']['path']).parent/'raw','Wrong new raw origin')
        for name in ('request','first_stage'):a.verify(snapshot[name])
        need(snapshot['request']['path']==str(raw/'request.json') and snapshot['first_stage']['path']==str(raw/'first_stage/record.json'),'Wrong raw snapshot root')
        newraw=h.read(snapshot['request']['path']);need(newraw==oldraw and oldraw['problem_identity']==identity and oldraw['domain']=='formal','Original kernel/exporter/input identity differs')
        a.verify(cp['raw_history_reference']);need(h.read(cp['raw_history_reference']['path'])==snapshot['events'],'Pinned raw journal differs')
        need([r['path'] for r in snapshot['events']]==[str(p) for p in sorted((raw/'events').glob('*/record.json'))] and len(list((raw/'events').iterdir()))==160 and not list((raw/'templates').iterdir()),'Uncommitted/extra raw callback or template')
        a.verify(cp['plan_reference']);plan_dict=h.read(cp['plan_reference']['path']);planhash=canonical_hash(plan_dict)
        need(planhash==cp['plan_sha256'] and cp['candidate_binding_reference']==result['candidate_binding_reference'],'Candidate plan/binding changed')
        binding=h.read(result['candidate_binding_reference']['path']);a.verify(binding['vector_reference']);a.verify(binding['original_binding_reference'])
        need(binding['original_binding_reference']==result['original_candidate_binding'] and binding['plan_sha256']==planhash and binding['problem_sha256']==state['problem_sha256'],'Candidate native binding differs')
        vector=h.npz(binding['vector_reference']['path'])[binding['vector_key']];H=len(plan_dict['x']);T=np.asarray(plan_dict['r']).shape[1]
        need(H==1008 and T==24 and binding['first_stage_slice']==[0,H*(T+1)],'Original candidate H/T changed')
        h.same(vector[:H],plan_dict['x'],'Native task plan changed');h.same(vector[H:H*(T+1)],np.asarray(plan_dict['r']).ravel(),'Native reserve plan changed')
        physical_binding=oldraw['input_binding'];target=physical_binding['target']
        for key in ('summary','audit','task_source','formal_protocol_lock','bank_source_manifest'):a.verify(physical_binding[key])
        a.verify(target['planning_index']);index=h.read(target['planning_index']['path'])
        need([x['scenario_id'] for x in index['scenarios']]==ids and index['development'] is False and index['split']=='planning' and index['lead_h']==1 and index['regime']=='R1','Wrong original physical support')
        # Checkers in R13 must retain original R4 bytes; no planning factory is invoked.
        original_files={r['path']:r for r in h.read(request_refs['original_source_manifest']['path'])['files']}
        for fn in (check_first_stage,check_recourse,load_physical_case,load_frozen_tasks,FormalPhysicalBank):
            module=sys.modules[fn.__module__];p=Path(module.__file__);row=original_files['src/evcs_experiments/'+p.name]
            need(a.ref(p)['sha256']==row['sha256'],'Original numerical/physical checker source changed')
        policy=a.ref(paths.spec_root/'runtime_policy.json');need(policy['sha256']==original_files['docs/experiments/runtime_policy.json']['sha256'],'Original physical tolerances changed');a.verify(policy)
        case=load_physical_case(paths.data_root);tasks=load_frozen_tasks(physical_binding['task_source']['path'],case,physical_binding['task_source']['sha256'])
        a.verify(dict(path=str(paths.data_root/'MANIFEST_SHA256.csv'),sha256=case.manifest_sha256,size_bytes=(paths.data_root/'MANIFEST_SHA256.csv').stat().st_size))
        for name,digest in case.input_hashes.items():
            p=paths.data_root/name;a.verify(dict(path=str(p),sha256=digest,size_bytes=p.stat().st_size))
        need(len(tasks.task_ids)==H and tuple(index['task_ids'])==tasks.task_ids and tasks.task_sha256==plan_dict['task_sha256'] and tasks.config_hash==plan_dict['config_hash'],'Frozen task identity changed')
        plan=SimpleNamespace(x=np.asarray(plan_dict['x']),r=np.asarray(plan_dict['r']),task_sha256=tasks.task_sha256,config_hash=tasks.config_hash)
        firstcheck=check_first_stage(case,tasks,plan);need(firstcheck['passed'],'Independent first-stage physical check failed');a.write(out/'first_stage_physical.json',firstcheck)
        firstrow=a.record(snapshot['first_stage']);fd=Path(snapshot['first_stage']['path']).parent;fm=h.read(fd/'identity.json');fv=h.npz(fd/'arrays.npz');FA=load_npz(fd/'matrix.npz').tocsr()
        need(first_digest(fm,FA,fv)==fm['matrix_sha256']==identity['first_stage_sha256'],'First-stage numerical semantic digest differs')
        firstnum=h.rmp_residuals(FA,fv,np.r_[plan.x,plan.r.ravel()]);Fpre=float(fv['c']@np.r_[plan.x,plan.r.ravel()])
        h.close(Fpre,firstcheck['costs']['Fpre_usd'],'Independent first-stage cost differs',1e-4+1e-7*abs(Fpre));a.write(out/'first_stage_numerical.json',dict(Fpre_usd=Fpre,**firstnum))
        bank=FormalPhysicalBank(target['planning_index']['path'],case,tasks,expected_index_sha256=target['planning_index']['sha256'],case_root=paths.data_root)
        oldhistory=h.read(request_refs['original_history']['path']);need(len(oldhistory)==163,'Original template event axis incomplete');constants=np.asarray(identity['constants_usd']);need(constants.shape==(160,),'Original constants differ')
        for i in range(160):
            current=i;a.remaining();tick=time.monotonic();er=snapshot['events'][i];event=a.record(er);ed=Path(er['path']).parent
            need(event['domain']=='formal','Non-formal candidate event')
            oldevent=h.read(oldhistory[i+2]['path']);a.verify(oldhistory[i+2]);pointer=next(x for x in oldevent['artifacts'] if Path(x['path']).name=='template_reference.json');a.verify(pointer)
            tr=h.read(ed/'template_reference.json');need(tr==h.read(pointer['path']),'Candidate refers to a different original template');template=a.record(tr);td=Path(tr['path']).parent
            need(template['scenario_index']==i and template['scenario_id']==ids[i],'Original template scenario axis differs')
            meta=h.metadata_header(td/'identity.json');v=h.npz(td/'arrays.npz');m={k:load_npz(td/(k+'.npz')).tocsr() for k in ('A_ub','A_eq','Bx_ub','Bx_eq','Br_ub','Br_eq')}
            atomref=index['scenarios'][i]['manifest'];a.verify(atomref);atom=h.read(atomref['path'])
            for r in atom['files'].values():a.verify(r)
            scene=bank.load(i,expected_scenario_sha256=index['scenarios'][i]['scenario_sha256'])
            saved=dict(event=event,plan=h.read(ed/'plan.json'),solver=h.read(ed/'solver.json'),solution=h.npz(ed/'solution.npz'),vector=h.npz(ed/'primal_vector.npz')['vector'],dual=h.npz(ed/'dual_kkt.npz'),fixed=h.npz(ed/'fixed_rhs.npz'),first_stage_sha256=identity['first_stage_sha256'])
            lp=audit_scenario(h,index=i,scenario_id=ids[i],plan_dict=plan_dict,constant=float(constants[i]),expected_template_sha256=state['pending']['recourse_certificates'][i]['template_sha256'],meta=meta,matrices=m,arrays=v,saved=saved,case=case,tasks=tasks,scene=scene)
            need(lp['q_usd']==q[i],'Completed candidate q differs from new raw witness');lp['seconds']=time.monotonic()-tick;done.append(lp);a.write(out/f'lp_{i:03d}.json',lp)
            atomic_json(out/'progress.json',dict(status='INDEPENDENT_AUDIT_IN_PROGRESS',completed_lp_count=len(done),expected_lp_count=160,plan_sha256=planhash,solver_calls=0,scientific_acceptance=False))
        a.remaining();a.write(out/'input_references.json',list(a.refs.values()))
        answer=dict(status=PASS,passed=True,complete_lp_count=160,plan_sha256=planhash,original_problem_sha256=state['problem_sha256'],support_ids=ids,q_usd=q.tolist(),Fpre_usd=Fpre,references=request_refs,
            checkpoint_reference=result['checkpoint_reference'],raw_snapshot_reference=result['raw_snapshot_reference'],original_first_stage_sha256=identity['first_stage_sha256'],first_stage_physical_passed=True,
            maximum_primal_residual=max(x['primal_residual'] for x in done),maximum_dual_stationarity=max(x['stationarity'] for x in done),maximum_duality_gap_usd=max(x['duality_gap_usd'] for x in done),
            primal_tolerance=1e-7,dual_tolerance=1e-8,original_objective_tolerance_unchanged=True,hashed_files=len(a.refs),hashed_bytes=a.bytes,wall_seconds=time.monotonic()-a.started,
            solver_calls=0,factory_calls=0,original_templates_reassembled=False,probability_audited=False,probability_PASS=False,scientific_acceptance=False,formal_OOS=False,normal_ccg_checkpoint_compatible=False)
        a.write(out/'audit.json',answer);manifest=dict(files=[a.ref(p) for p in sorted(out.iterdir()) if p.is_file()]);a.write(out/'artifact_manifest.json',manifest);a.remaining()
        return dict(answer,audit_reference=a.ref(out/'audit.json'),artifact_manifest_reference=a.ref(out/'artifact_manifest.json'))
    except Exception as exc:
        failure=dict(status='TIME_LIMIT_INCOMPLETE_CANDIDATE_AUDIT' if isinstance(exc,TimeoutError) else 'FAIL_CANDIDATE_RECOURSE_AUDIT',passed=False,complete_lp_count=len(done),current_index=current,error_type=type(exc).__name__,error=str(exc),references=request_refs,solver_calls=0,factory_calls=0,scientific_acceptance=False,formal_OOS=False)
        if not (out/'audit.json').exists():a.write(out/'audit.json',failure)
        else:a.write(out/'terminal_failure.json',failure)
        if not (out/'input_references.json').exists():a.write(out/'input_references.json',list(a.refs.values()))
        raise
