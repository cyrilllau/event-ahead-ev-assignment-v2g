"""Supervised fresh evaluation of an accepted candidate with the old kernel.

The wrapper and component are separately source-bound; all imported evcs modules
must come from the unchanged original release. No master or probability solve.
"""
from datetime import datetime,timezone
from decimal import Decimal
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,os,sys,time
import numpy as np
from evcs_experiments.config import get_paths
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.runtime.jobctl import observation
from evcs_experiments.formal_planning import load_formal_problem

KIND='S05_FIXED_CANDIDATE_FULL_SUPPORT_JOB_V1'
OLD_MANIFEST_SHA='9d0e674c6eaafa55c589ed3ddb1b137afbc637ff768f19234c6eee099c74053d'


def need(ok,msg):
    if not bool(ok):raise ValueError(msg)


def read(p):return json.loads(Path(p).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))


def reference(p):
    p=Path(p).resolve(strict=True);h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(4194304):h.update(b)
    return dict(path=str(p),sha256=h.hexdigest(),size_bytes=p.stat().st_size)


def verify(r):
    need(set(r)=={'path','sha256','size_bytes'} and Path(r['path']).is_absolute() and reference(r['path'])==r,'Changed strong reference')
    return Path(r['path'])


def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
    return reference(p)


def source_files(manifest_reference):
    path=verify(manifest_reference);root=path.parent;manifest=read(path);files={}
    need(isinstance(manifest['files'],list) and manifest['files'],'Empty source manifest')
    for row in manifest['files']:
        p=(root/row['path']).resolve();need(p.is_relative_to(root) and str(p) not in files,'Source path escape/duplicate')
        r=dict(path=str(p),sha256=row['sha256'],size_bytes=row['size_bytes']);verify(r);files[str(p)]=r
    return root,files


def check_loaded_source(files):
    for name,module in tuple(sys.modules.items()):
        if name=='evcs_experiments' or name.startswith('evcs_experiments.'):
            p=getattr(module,'__file__',None)
            if p is not None:need(reference(p)==files.get(str(Path(p).resolve())),'Mixed original package source')


def budget_guard(req,settled):
    need(req['kind']==KIND and req['domain']=='formal' and req['formal_OOS'] is False and req['resume'] is False,'Fresh formal evaluation required')
    need(req['plan_id']==settled['plan_id'] and settled['kind']=='S05_EPIGRAPH_MASTER_PARENT_ACCOUNTING_V1'
         and settled['status']=='SETTLED_AUXILIARY_MASTER_PENDING_INDEPENDENT_RESULT_REVIEW','Wrong settled predecessor')
    need(settled['accounting_PASS'] is True and settled['nested_seconds_added'] is False and settled['cumulative_seconds_provisional'] is False
         and settled['external_preprocessing_charged_once'] is True and settled['parent_charged_once'] is True,'Unsettled cumulative cost')
    b=settled['opening_budget'];opening=Decimal(b['previous_cumulative_seconds_decimal'])+Decimal(b['external_preprocessing_seconds_decimal'])
    total=opening+Decimal(settled['terminal_parent_seconds_decimal']);remaining=Decimal(7200)-total
    need(total.is_finite() and total>=0 and str(total)==settled['cumulative_accounted_seconds_decimal']
         and str(remaining)==settled['remaining_seconds_decimal'] and float(total)==settled['cumulative_accounted_seconds']
         and float(remaining)==settled['remaining_seconds'] and settled['cumulative_cap_seconds']==7200,'Incorrect exact cumulative cost')
    need(total==Decimal('3971.56097236330092'),'Unexpected candidate predecessor budget')
    p=req['parent_job'];need(p['timeout_seconds']==3200 and p['grace_seconds']==5 and p['threads']==1
         and req['wall_seconds']==3190 and req['shutdown_reserve_seconds']==10 and req['completion_uses_final_evidence_reserve'] is True,'Wrong completion resources')
    need(type(req['lp_time_limit_seconds']) in (int,float) and math.isfinite(req['lp_time_limit_seconds']) and 0<req['lp_time_limit_seconds']<=120,'Invalid LP cap')
    need(remaining>=Decimal(p['timeout_seconds']+p['grace_seconds']+req['shutdown_reserve_seconds']),'Completion exceeds remaining original budget')
    return dict(cumulative_seconds_decimal=str(total),remaining_seconds_decimal=str(remaining),cumulative_cap_seconds=7200)


def bind_parent(req,rr):
    paths=get_paths();p=req['parent_job'];d=Path(p['directory']).resolve();root=paths.project_root.resolve()
    need(d==paths.campaign_root.resolve()/'jobs'/p['job_id'],'Wrong parent directory')
    need(os.environ.get('EVCS_JOB_ID')==p['job_id'] and os.environ.get('EVCS_JOB_DIRECTORY')==str(d),'Not in requested parent')
    m=read(d/'manifest.json');s=read(d/'status.json')
    need(m['argv']==[sys.executable,'-B',str(Path(__file__).resolve()),'--request',rr['path'],'--expected-request-sha256',rr['sha256']]
         and m['cwd']==str(root) and m['outroot']==str(d.parent) and m['job_id']==p['job_id'] and m['spec_id']=='S05','Wrong supervised command')
    need(m['threads']==p['threads']==1 and m['timeout_seconds']==p['timeout_seconds']==3200 and m['grace_seconds']==p['grace_seconds']==5,'Wrong parent limits')
    need(verify(p['controller_reference'])==Path(m['controller_path']).resolve()==root/'src/evcs_experiments/runtime/jobctl.py','Wrong old job controller')
    need(Path(sys.executable).absolute()==root/'.venv/bin/python','Wrong original interpreter')
    need(s['state']=='RUNNING' and os.getpid()==s['command_pid'] and os.getsid(0)==s['supervisor_sid'] and observation(s)['supervisor_alive'],'Parent not live')
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):need(os.environ.get(k)=='1','Wrong numerical threads')
    age=(datetime.now(timezone.utc)-datetime.fromisoformat(m['created_at_utc'])).total_seconds();need(0<=age<3190,'Expired parent')
    return dict(job_id=p['job_id'],directory=str(d),manifest_reference=reference(d/'manifest.json'),deadline_monotonic=time.monotonic()+3190-age)


def candidate_authorization(refs,*,campaign=None):
    names=('adoption_record','candidate_binding','candidate_result','candidate_native','candidate_review','current_settlement','old_checkpoint','raw_request','saved160_acceptance')
    for k in names:verify(refs[k])
    adoption=read(refs['adoption_record']['path']);binding=read(refs['candidate_binding']['path']);result=read(refs['candidate_result']['path']);review=read(refs['candidate_review']['path'])
    need(adoption['id']=='D091' and adoption['candidate']==refs['candidate_binding'] and adoption['result']==refs['candidate_result']
        and adoption['review']==refs['candidate_review'] and adoption['settlement']==refs['current_settlement'],'Wrong adopted candidate')
    if campaign is not None:
        decisions=[json.loads(v) for v in (Path(campaign)/'decisions.jsonl').read_text().splitlines() if v.strip()]
        need([v for v in decisions if v.get('id')=='D091']==[adoption],'Adoption differs from authoritative decision')
        state=read(Path(campaign)/'state.json')['specs']['S05']
        need(state['pending_candidate_binding']==refs['candidate_binding'] and state['epigraph_settlement']==refs['current_settlement']
             and state['epigraph_result_review']==refs['candidate_review'],'Authoritative candidate changed')
    need(review['status']=='PASS_INDEPENDENT_EPIGRAPH_AUXILIARY_RESULT_REVIEW' and review['passed'] is True,'Candidate independent review missing')
    need(review['references']['result']==refs['candidate_result'] and review['references']['candidate_binding']==refs['candidate_binding']
         and review['references']['native_solver']==refs['candidate_native'],'Review binds other candidate')
    verify(binding['vector_reference']);verify(binding['source_request'])
    need(binding['source_native_result']==refs['candidate_native'] and result['candidate_vector']==binding['vector_reference']
        and result['request_reference']==binding['source_request'],'Native candidate linkage differs')
    need(binding['vector_key']=='vector' and binding['first_stage_slice']==[0,25200] and binding['formal_OOS'] is False and binding['lockable'] is False,'Wrong candidate scope')
    native=read(refs['candidate_native']['path'])
    need(native['valid_incumbent'] is True and native['certificate_failures']==[] and result['candidate_available'] is True
        and native['incumbent_without_constant_usd']==result['candidate_auxiliary_objective_usd'],'Invalid original native incumbent')
    cp=read(refs['old_checkpoint']['path']);old=cp['state'];identity=old['identity'];card=read(refs['saved160_acceptance']['path'])
    need(cp['sha256']==canonical_hash(old) and old['problem_sha256']==canonical_hash(identity),'Old scientific identity changed')
    need(binding['problem_sha256']==result['problem_sha256']==review['original_problem_sha256']==old['problem_sha256']==card['problem_sha256'],'Candidate problem differs')
    need(binding['first_stage_sha256']==result['first_stage_sha256']==identity['first_stage_sha256'],'Candidate first-stage differs')
    need(binding['plan_sha256']==result['candidate_plan_sha256']==review['candidate_plan_sha256'],'Candidate plan differs')
    raw=read(refs['raw_request']['path']);need(raw['problem_identity']==identity and raw['domain']=='formal' and raw['formal_OOS'] is False,'Wrong original formal raw source')
    need(card['status']=='PASS_SAVED160_COMPONENT_REVIEW' and card['complete_lp_count']==160
        and refs['old_checkpoint'] in card['references'].values() and refs['raw_request'] in card['references'].values(),'Unaccepted original160')
    return binding,identity


def reserve(req,rr,parent,root):
    refs=req['references'];key=canonical_hash(refs['current_settlement'])
    path=Path(root)/'candidate_evaluation_attempts'/key;path.mkdir(parents=True,exist_ok=False)
    write(path/'reservation.json',dict(request_reference=rr,parent=parent,predecessor_settlement=refs['current_settlement'],candidate_binding=refs['candidate_binding'],requires_terminal_parent_settlement=True,automatic_relaunch=False))
    return path


def execute(path,expected):
    began=time.monotonic();rr=reference(path);need(rr['sha256']==expected,'Changed request');req=read(path);refs=req['references'];paths=get_paths();parent=bind_parent(req,rr)
    out=Path(req['output_root']).resolve();need(out==paths.campaign_root.resolve()/'candidate_evaluations'/parent['job_id'],'Wrong fresh output')
    for k in ('current_settlement','candidate_binding','adoption_record'):verify(refs[k])
    attempt=reserve(req,rr,parent,paths.campaign_root.resolve());prepared=None;component=None;timing={};adapter=None
    try:
        for r in refs.values():verify(r)
        oldroot,oldfiles=source_files(refs['old_source_manifest'])
        need(refs['old_source_manifest']['sha256']==OLD_MANIFEST_SHA and oldroot==paths.project_root.resolve(),'Wrong original R4 source')
        check_loaded_source(oldfiles);newroot,newfiles=source_files(refs['driver_source_manifest'])
        need(newroot!=oldroot and newfiles.get(str(Path(__file__).resolve()))==refs['wrapper_script'],'Wrapper source unbound')
        component_path=verify(refs['component_script']);need(newfiles.get(str(component_path))==refs['component_script'],'Component source unbound')
        need(verify(refs['runtime_config'])==oldroot/'configs/powerai.toml' and Path(os.environ.get('EVCS_CONFIG','')).resolve()==Path(refs['runtime_config']['path']),'Wrong old configuration')
        settled=read(refs['current_settlement']['path']);budget=budget_guard(req,settled);binding,identity=candidate_authorization(refs,campaign=paths.campaign_root)
        need(req['target_id']=='michael_2018' and req['method_id']=='EA' and req['plan_id']==settled['plan_id'],'Wrong candidate campaign identity')
        for k in ('summary','audit'):
            need(refs[k]==read(refs['raw_request']['path'])['input_binding'][k],'Wrong original factory inputs')
        need(time.monotonic()<parent['deadline_monotonic']-10,'Preparation exhausted budget')
        out.mkdir(parents=True,exist_ok=False)
        timing['source_and_authorization_seconds']=time.monotonic()-began
        spec=importlib.util.spec_from_file_location('_bound_candidate_evaluation_component',component_path);component=importlib.util.module_from_spec(spec);spec.loader.exec_module(component)
        prepared=load_formal_problem(summary_reference=refs['summary'],audit_reference=refs['audit'],target_id=req['target_id'],method_id=req['method_id'],evidence_root=out/'raw',case_root=paths.data_root)
        timing['factory_preparation_seconds']=prepared.preparation_seconds
        need(prepared.problem.identity==identity,'Original factory problem/source identity differs');check_loaded_source(oldfiles)
        adapted=dict(binding,result_reference=binding['source_native_result'],original_binding_reference=refs['candidate_binding'],candidate_result_reference=refs['candidate_result'],candidate_review_reference=refs['candidate_review'])
        br=write(out/'adapter_binding.json',adapted);plan=component.extract_candidate(br,prepared.problem.first_stage)
        need(canonical_hash(component.serial_plan(plan))==binding['plan_sha256'],'Extracted plan differs from accepted candidate')
        t=time.monotonic();adapter=component.ReusedTemplateEvidenceAdapter(prepared.evidence,old_history_reference=refs['old_history'],old_checkpoint_reference=refs['old_checkpoint'],saved160_acceptance_reference=refs['saved160_acceptance'],output=out/'template_reuse').install();timing['adapter_setup_seconds']=time.monotonic()-t
        write(out/'execution.json',dict(kind=KIND,request_reference=rr,parent=parent,budget=budget,source_references=refs,candidate_binding_reference=br,original_problem_sha256=canonical_hash(identity),plan_sha256=binding['plan_sha256'],preparation_timing=timing,probability_calls=0,master_calls=0,formal_OOS=False))
        record=component.evaluate_prepared_candidate(prepared,plan,candidate_binding_reference=br,output=out/'evaluation',deadline=parent['deadline_monotonic'],lp_time_limit_seconds=req['lp_time_limit_seconds'],shutdown_reserve_seconds=req['shutdown_reserve_seconds'])
        timing['evaluation_component_seconds']=record['elapsed_seconds']
        with (out/'final_checkpoint.json').open('xb') as f:f.write((out/'evaluation/checkpoint.json').read_bytes())
        cpref=reference(out/'final_checkpoint.json')
        need(cpref['sha256']==reference(out/'evaluation/checkpoint.json')['sha256'],'Terminal checkpoint changed')
        t=time.monotonic();snapshot=prepared.evidence.snapshot(out/'raw_snapshot.json');timing['snapshot_export_seconds']=time.monotonic()-t;check_loaded_source(oldfiles)
        status=record['status'] if time.monotonic()<parent['deadline_monotonic'] else 'TIME_LIMIT_AFTER_EVIDENCE_EXPORT'
        result=dict(kind=KIND,status=status,request_reference=rr,parent=parent,budget=budget,checkpoint_reference=cpref,raw_snapshot_reference=snapshot,candidate_binding_reference=br,original_candidate_binding=refs['candidate_binding'],plan_sha256=binding['plan_sha256'],problem_sha256=canonical_hash(identity),completed_count=record['completed_count'],full_q_available=record['completed_count']==160,probability_evaluated=False,full_support_verified=False,scientific_acceptance=False,lockable=False,formal_OOS=False,normal_ccg_checkpoint_compatible=False,automatic_resume_supported=False,timing=timing,attempt_elapsed_seconds=time.monotonic()-began,requires_terminal_parent_settlement=True,cumulative_seconds_provisional=True)
        write(out/'result.json',result);write(attempt/'completion.json',dict(request_reference=rr,result_reference=reference(out/'result.json'),requires_terminal_parent_settlement=True))
        print(json.dumps(dict(status=status,completed_count=record['completed_count'],scientific_acceptance=False)))
        return result
    except Exception as exc:
        failure=dict(error=type(exc).__name__,message=str(exc),request_reference=rr,parent=parent,elapsed_seconds=time.monotonic()-began,timing=timing,requires_terminal_parent_settlement=True,scientific_acceptance=False,automatic_resume_supported=False)
        if (out/'evaluation/checkpoint.json').exists():
            with (out/'failed_checkpoint.json').open('xb') as f:f.write((out/'evaluation/checkpoint.json').read_bytes())
            failure['checkpoint_reference']=reference(out/'failed_checkpoint.json')
        if prepared is not None and time.monotonic()<parent['deadline_monotonic']-5:
            try:failure['raw_snapshot_reference']=prepared.evidence.snapshot(out/'failure_raw_snapshot.json')
            except Exception as snapshot_error:failure['snapshot_failure']=str(snapshot_error)
        write(attempt/'failure.json',failure)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',type=Path,required=True);p.add_argument('--expected-request-sha256',required=True);a=p.parse_args();execute(a.request,a.expected_request_sha256)
