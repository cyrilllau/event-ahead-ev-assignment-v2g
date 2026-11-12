"""One supervised, cumulative-budgeted fresh-candidate audit and probability job.

Installed imports come from pinned R13; explicitly imported new scripts come
from one independently reviewed release. No CCG, master, OOS or recourse solve.
"""
from datetime import datetime,timezone
from decimal import Decimal
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,os,sys,time
from evcs_experiments.config import get_paths
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.runtime.jobctl import observation

KIND='S05_CANDIDATE_COMPLETION_JOB_V1'
RUNTIME_SHA='2c52546bf0fa390723b2c245fb29da13565776ba81ac9fae48cc7e56b322fc8e'
OLD_SHA='9d0e674c6eaafa55c589ed3ddb1b137afbc637ff768f19234c6eee099c74053d'
PHYSICS=('recourse_check.py','first_stage_check.py','physical_inputs.py','formal_physical_inputs.py','random_streams.py')


def need(ok,msg):
    if not bool(ok):raise ValueError(msg)


def read(path):return json.loads(Path(path).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))


def reference(path):
    p=Path(path).resolve(strict=True);h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(4194304):h.update(b)
    return dict(path=str(p),sha256=h.hexdigest(),size_bytes=p.stat().st_size)


def verify(r):
    need(set(r)=={'path','sha256','size_bytes'} and Path(r['path']).is_absolute() and reference(r['path'])==r,'Changed strong reference');return Path(r['path'])


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    return reference(path)


def remaining(deadline,reserve=10):
    value=deadline-time.monotonic()-reserve;need(value>0,'Completion budget exhausted');return value


def source_files(ref,deadline):
    p=verify(ref);root=p.parent;files={};rows=read(p)['files'];need(rows,'Empty source manifest')
    for row in rows:
        remaining(deadline);p=(root/row['path']).resolve();need(p.is_relative_to(root) and str(p) not in files,'Source path escape/duplicate')
        r=dict(path=str(p),sha256=row['sha256'],size_bytes=row['size_bytes']);verify(r);files[str(p)]=r
    return root,files


def loaded_sources(files):
    for name,module in tuple(sys.modules.items()):
        if name=='evcs_experiments' or name.startswith('evcs_experiments.'):
            p=getattr(module,'__file__',None)
            if p:need(files.get(str(Path(p).resolve()))==reference(p),'Mixed installed package source')



def loaded_original_equivalence(runtime,files,oldroot,oldfiles):
    """Only corrected probability differs from the original imported kernel."""
    for name,module in tuple(sys.modules.items()):
        if name=='evcs_experiments' or name.startswith('evcs_experiments.'):
            path=getattr(module,'__file__',None)
            if not path:continue
            p=Path(path).resolve();need(p.is_relative_to(runtime),'Foreign installed module')
            row=files[str(p)];rel=p.relative_to(runtime)
            if str(rel)=='src/evcs_experiments/probability.py':
                need(row['sha256']=='39a33c1dab694a65113eeb9424c2eaf7e056d4e3478ec4d0f3c84b454754d68c','Wrong corrected probability')
            else:
                old=oldfiles.get(str(oldroot/rel));need(old is not None and row['sha256']==old['sha256'] and row['size_bytes']==old['size_bytes'],'Imported original helper/kernel changed')


def import_script(name,ref,files):
    p=verify(ref);need(files.get(str(p))==ref,'Unbound explicit script')
    spec=importlib.util.spec_from_file_location(name,p);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module


def budget_guard(req,settled,terminal):
    need(req['kind']==KIND and req['domain']=='formal' and req['resume'] is False and req['formal_OOS'] is False,'Wrong completion scope')
    need(settled['kind']=='S05_CANDIDATE_EVALUATION_PARENT_ACCOUNTING_V1' and settled['status']=='SETTLED_FULL_Q_PENDING_INDEPENDENT_AUDIT_AND_PROBABILITY'
         and settled['plan_id']==req['plan_id'],'Wrong settled predecessor')
    need(settled['accounting_PASS'] is True and settled['parent_charged_once'] is True and settled['nested_seconds_added'] is False
         and settled['cumulative_seconds_provisional'] is False and settled['original_cumulative_cap_seconds']==settled['cumulative_cap_seconds']==7200,'Unsettled original budget')
    opening=Decimal(settled['opening_cumulative_seconds_decimal']);elapsed=Decimal(settled['terminal_parent_seconds_decimal']);total=opening+elapsed;left=Decimal(7200)-total
    need(terminal['complete'] is True and terminal['state']=='SUCCEEDED' and terminal['returncode']==0 and Decimal(str(terminal['elapsed_seconds']))==elapsed,'Wrong actual terminal parent cost')
    need(opening==Decimal('3971.56097236330092') and elapsed==Decimal('1365.467456') and total==Decimal(settled['cumulative_accounted_seconds_decimal'])==Decimal('5337.02842836330092')
         and left==Decimal(settled['remaining_seconds_decimal'])==Decimal('1862.97157163669908'),'Cumulative cost reset or arithmetic differs')
    p=req['parent_job'];need(p['timeout_seconds']==1800 and p['grace_seconds']==5 and p['threads']==1 and req['wall_seconds']==1790 and req['shutdown_reserve_seconds']==10,'Wrong completion resources')
    need(type(req['probability_time_limit_seconds']) in (int,float) and math.isfinite(req['probability_time_limit_seconds']) and 0<req['probability_time_limit_seconds']<=30,'Invalid probability cap')
    need(left>=Decimal(p['timeout_seconds']+p['grace_seconds']+req['shutdown_reserve_seconds']),'Completion exceeds remaining original budget')
    return dict(opening_cumulative_seconds_decimal=str(total),remaining_seconds_decimal=str(left),cumulative_cap_seconds=7200)


def bind_parent(req,rr):
    paths=get_paths();root=paths.project_root.resolve();p=req['parent_job'];d=Path(p['directory']).resolve()
    need(d==paths.campaign_root.resolve()/'jobs'/p['job_id'],'Wrong parent directory')
    need(os.environ.get('EVCS_JOB_ID')==p['job_id'] and os.environ.get('EVCS_JOB_DIRECTORY')==str(d),'Not in requested supervisor')
    m=read(d/'manifest.json');s=read(d/'status.json')
    need(m['argv']==[sys.executable,'-B',str(Path(__file__).resolve()),'--request',rr['path'],'--expected-request-sha256',rr['sha256']]
        and m['cwd']==str(root) and m['outroot']==str(d.parent) and m['job_id']==p['job_id'] and m['spec_id']=='S05','Wrong exact parent command')
    need(m['timeout_seconds']==p['timeout_seconds']==1800 and m['grace_seconds']==p['grace_seconds']==5 and m['threads']==p['threads']==1,'Wrong parent limits')
    need(Path(sys.executable).absolute()==root/'.venv/bin/python' and verify(p['controller_reference'])==root/'src/evcs_experiments/runtime/jobctl.py'==Path(m['controller_path']).resolve(),'Wrong R13 runtime/controller')
    need(s['state']=='RUNNING' and s['command_pid']==os.getpid() and s['supervisor_sid']==os.getsid(0) and observation(s)['supervisor_alive'],'Parent is not live')
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):need(os.environ.get(k)=='1','Wrong numerical threads')
    age=(datetime.now(timezone.utc)-datetime.fromisoformat(m['created_at_utc'])).total_seconds();need(0<=age<1790,'Expired parent')
    return dict(job_id=p['job_id'],directory=str(d),manifest_reference=reference(d/'manifest.json'),deadline_monotonic=time.monotonic()+1790-age)


def reserve(req,rr,parent,campaign):
    sr=req['references']['current_settlement'];verify(sr)
    path=Path(campaign)/'candidate_completion_attempts'/canonical_hash(sr);path.mkdir(parents=True,exist_ok=False)
    write(path/'reservation.json',dict(request_reference=rr,parent=parent,predecessor_settlement=sr,automatic_relaunch=False,requires_terminal_parent_settlement=True));return path


def input_bindings(req,*,campaign):
    refs=req['references'];ar=req['audit_references'];pr=req['probability_references']
    for group in (refs,ar,pr):
        for ref in group.values():verify(ref)
    links=[(ar['audit_script'],refs['auditor_script']),(ar['candidate_result'],refs['evaluation_result']),
        (pr['driver_source'],refs['probability_script']),(pr['evaluation_result'],refs['evaluation_result']),(pr['checkpoint'],refs['final_checkpoint']),
        (pr['full_q'],refs['full_q']),(pr['execution'],refs['evaluation_execution']),
        (pr['original_checkpoint'],ar['original_checkpoint']),(pr['original_history'],ar['original_history']),(pr['saved160_acceptance'],ar['saved160_acceptance'])]
    need(all(a==b for a,b in links),'Cross-component input references differ')
    result=read(verify(refs['evaluation_result']));snapshot=read(verify(refs['raw_snapshot']));cp=read(verify(refs['final_checkpoint']));full=read(verify(refs['full_q']))
    need(result['checkpoint_reference']==refs['final_checkpoint'] and result['raw_snapshot_reference']==refs['raw_snapshot'] and cp==full
         and result['completed_count']==160 and result['full_q_available'] is True and result['formal_OOS'] is False and result['probability_evaluated'] is False,'R14 completion records differ')
    need(snapshot['events']==full['recourse_event_references'] and snapshot['request']==pr['new_raw_request'] and snapshot['full_support_verified'] is False and snapshot['formal_OOS'] is False,'Raw snapshot binding differs')
    need(result['original_candidate_binding']==pr['original_binding'] and result['plan_sha256']==full['plan_sha256'],'Wrong evaluated candidate')
    settled=read(verify(refs['current_settlement']));sr=settled['references']
    need(sr['candidate_result']==refs['evaluation_result'] and sr['terminal_parent_result']==refs['previous_parent_result'] and sr['parent_manifest']==refs['previous_parent_manifest']
         and result['parent']['manifest_reference']==refs['previous_parent_manifest'],'Wrong settled terminal references')
    terminal=read(verify(refs['previous_parent_result']));status=read(verify(ar['candidate_job_status']))
    need(status==terminal and status['job_id']==result['parent']['job_id'],'Wrong terminal status/job')
    obs=observation(status)
    need(not any((obs['supervisor_alive'],obs['command_alive'],obs['live_group_members'],obs['blocks_new_launch'])),'Original parent or descendants still live')
    s=read(Path(campaign)/'state.json')['specs']['S05']
    need(s['current_formal_plan_id']==req['plan_id'] and s['pending_candidate_binding']==pr['original_binding'] and s['candidate_evaluation_settlement']==refs['current_settlement']
         and s['candidate_evaluation_result']==refs['evaluation_result'],'Authoritative candidate/settlement changed')
    need(req['target_id']=='michael_2018' and req['method_id']=='EA','Wrong formal target/method')
    return budget_guard(req,settled,terminal),result


def pipeline(auditor,component,req,out,deadline):
    """No dependency injection in execute; isolated orchestration is unit-testable."""
    remaining(deadline,45)
    audit=auditor.audit_candidate_recourse(req['audit_references'],output=out/'recourse_audit',deadline=deadline-40)
    full=read(verify(req['references']['full_q']))
    need(audit['status']=='PASS_INDEPENDENT_CANDIDATE_160_RECOURSE_AUDIT' and audit['passed'] is True and audit['complete_lp_count']==160
         and audit['plan_sha256']==full['plan_sha256'] and audit['support_ids']==full['support_ids'] and audit['q_usd']==full['q_usd']
         and audit['references']==req['audit_references'],'Incomplete or different independent recourse audit')
    verify(audit['audit_reference']);verify(audit['artifact_manifest_reference'])
    cap=min(float(req['probability_time_limit_seconds']),remaining(deadline))
    result=component.complete_candidate_probability(req['probability_references'],output=out/'probability',time_limit_seconds=cap)
    remaining(deadline)
    need(result['probability_independent_check_passed'] is True and result['upper_bound_usd'] is not None
         and result['plan_sha256']==audit['plan_sha256'] and result['original_problem_sha256']==audit['original_problem_sha256'],'Probability/independent recourse result differs or failed')
    need(abs(float(result['first_stage_cost_usd'])-float(audit['Fpre_usd']))<=1e-7,'Independent original first cost differs')
    return audit,result



def finalize(out,attempt,request_reference,result,deadline):
    """Late export never produces a successful supervised completion."""
    write(out/'result.json',result);remaining(deadline)
    write(attempt/'completion.json',dict(request_reference=request_reference,result_reference=reference(out/'result.json'),requires_terminal_parent_settlement=True))
    remaining(deadline)


def execute(path,expected):
    began=time.monotonic();rr=reference(path);need(rr['sha256']==expected,'Changed request');req=read(path);refs=req['references'];paths=get_paths();parent=bind_parent(req,rr)
    out=Path(req['output_root']).resolve();need(out==paths.campaign_root.resolve()/'candidate_completions'/parent['job_id'],'Wrong completion output')
    attempt=reserve(req,rr,parent,paths.campaign_root);deadline=parent['deadline_monotonic']
    try:
        budget,oldresult=input_bindings(req,campaign=paths.campaign_root)
        runtime,files=source_files(refs['runtime_source_manifest'],deadline)
        need(refs['runtime_source_manifest']['sha256']==RUNTIME_SHA and runtime==paths.project_root.resolve(),'Wrong pinned R13 source')
        loaded_sources(files);newroot,newfiles=source_files(refs['driver_source_manifest'],deadline)
        need(newroot!=runtime and newfiles.get(str(Path(__file__).resolve()))==refs['wrapper_script'],'Wrapper source closure differs')
        oldroot,oldfiles=source_files(req['audit_references']['original_source_manifest'],deadline)
        need(req['audit_references']['original_source_manifest']['sha256']==OLD_SHA,'Wrong original R4 source')
        for name in PHYSICS:
            a=files[str(runtime/'src/evcs_experiments'/name)];b=oldfiles[str(oldroot/'src/evcs_experiments'/name)]
            need(a['sha256']==b['sha256'] and a['size_bytes']==b['size_bytes'],'Original physical checker changed')
        helper=verify(req['audit_references']['helper_script']);need(files.get(str(helper))==req['audit_references']['helper_script'],'Unbound R13 numerical helper')
        need(verify(refs['runtime_config'])==runtime/'configs/powerai.toml' and Path(os.environ.get('EVCS_CONFIG','')).resolve()==Path(refs['runtime_config']['path']),'Wrong runtime configuration')
        auditor=import_script('_bound_candidate_recourse_auditor',refs['auditor_script'],newfiles)
        component=import_script('_bound_candidate_probability_component',refs['probability_script'],newfiles);loaded_sources(files);loaded_original_equivalence(runtime,files,oldroot,oldfiles)
        out.mkdir(parents=True,exist_ok=False)
        write(out/'execution.json',dict(kind=KIND,request_reference=rr,parent=parent,budget=budget,input_result=refs['evaluation_result'],verification_elapsed_seconds=time.monotonic()-began,formal_OOS=False))
        audit,prob=pipeline(auditor,component,req,out,deadline);loaded_sources(files);loaded_original_equivalence(runtime,files,oldroot,oldfiles);remaining(deadline)
        result=dict(kind=KIND,status='FULL_SUPPORT_NUMERICAL_COMPLETION_PENDING_FORMAL_PLAN_LOCK',request_reference=rr,parent=parent,budget=budget,
            recourse_audit=audit['audit_reference'],audit_artifact_manifest=audit['artifact_manifest_reference'],probability_result=reference(out/'probability/result.json'),
            plan_sha256=prob['plan_sha256'],original_problem_sha256=prob['original_problem_sha256'],upper_bound_usd=prob['upper_bound_usd'],lower_bound_usd=None,
            complete_lp_count=160,recourse_independent_check_passed=True,probability_independent_check_passed=True,
            full_support_verified=True,scientific_acceptance=False,lockable=False,formal_OOS=False,normal_ccg_checkpoint_compatible=False,
            master_calls=0,recourse_solver_calls=0,attempt_elapsed_seconds=time.monotonic()-began,requires_terminal_parent_settlement=True,cumulative_seconds_provisional=True)
        for key in ('wrapper_script','auditor_script','probability_script'):verify(refs[key])
        finalize(out,attempt,rr,result,deadline);return result
    except Exception as exc:
        write(attempt/'failure.json',dict(request_reference=rr,parent=parent,error=type(exc).__name__,message=str(exc),elapsed_seconds=time.monotonic()-began,
            requires_terminal_parent_settlement=True,cumulative_seconds_provisional=True,scientific_acceptance=False,automatic_relaunch=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--request',type=Path,required=True);p.add_argument('--expected-request-sha256',required=True);a=p.parse_args();execute(a.request,a.expected_request_sha256)
