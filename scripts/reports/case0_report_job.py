#!/usr/bin/env python3
"""Reserved Case0 statistics subprocess; original audit settlement charges parent wall."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from evcs_experiments.campaign_batch import file_reference,verify_reference
from evcs_experiments.case0_campaign import active_config,campaign_root,read,save_state
from evcs_experiments.case0_isolation import verify_current_cgroup
from evcs_experiments.runtime.jobctl import atomic_json,observation,utcnow,identity_alive,process_identity


def need(ok,message):
    if not ok:raise ValueError(message)


def checked(ref):
    verify_reference(ref);return read(ref['path'])


def _supervisor_remaining(manifest,status):
    """Conservative Linux parent clock: process birth precedes jobctl timer.

    Start ticks are rounded down by the kernel; CLOCK_BOOTTIME also charges
    suspended time. Neither can extend jobctl's monotonic timeout. A reused
    PID/boot or non-Linux identity is rejected, never replaced with wall time.
    """
    seconds=manifest.get('timeout_seconds')
    need(type(seconds) in (int,float) and math.isfinite(seconds) and 0<seconds<=7200,
         'Invalid parent timeout_seconds')
    if status.get('state') not in ('LAUNCHING','RUNNING'):
        raise ValueError('Parent is not admitting active work')
    identity=status.get('supervisor')
    if not identity or not identity_alive(identity):raise ValueError('Parent supervisor is not alive')
    current=process_identity(identity['pid'])
    if (current is None or current['token']!=identity['token'] or
            current['sid']!=status.get('supervisor_sid') or current['pid']!=current['sid']):
        raise ValueError('Parent supervisor identity differs')
    token=current['token'].split(':')
    if len(token)!=3 or token[0]!='linux' or not hasattr(time,'CLOCK_BOOTTIME'):
        raise ValueError('Verified parent clock requires Linux boot/start ticks')
    ticks=int(token[2]);frequency=os.sysconf('SC_CLK_TCK')
    if ticks<0 or frequency<=0:raise ValueError('Invalid supervisor start clock')
    now=time.monotonic();boot=time.clock_gettime(time.CLOCK_BOOTTIME)
    elapsed=boot-ticks/frequency
    if not math.isfinite(elapsed) or elapsed<0:raise ValueError('Invalid parent elapsed clock')
    remaining=manifest['timeout_seconds']-elapsed
    return dict(clock_kind='linux_supervisor_birth_boottime_v1',supervisor=current,
                timeout_seconds=manifest['timeout_seconds'],elapsed_seconds=elapsed,
                observed_monotonic=now,remaining_seconds=remaining,
                conservative_deadline_monotonic=now+remaining)


def run_report_job(request_reference,config_path):
    began=time.monotonic();isolation=verify_current_cgroup()
    request=checked(request_reference);root=campaign_root().resolve()
    need(request['kind']=='CASE0_REPORT_JOB_REQUEST_V1','Wrong reporting request')
    folder=Path(request['operation_root']).resolve();output=Path(request['output_directory']).resolve()
    need(folder.is_relative_to(root/'operations') and folder.is_dir() and not output.exists()
         and output.is_relative_to(folder),'Report output must be new within its operation')
    seconds=request['seconds'];need(type(seconds) in (int,float) and math.isfinite(seconds) and 0<seconds<=7180,'Bounded reporting duration required')
    job_id=os.environ['EVCS_JOB_ID'];job=root/'jobs'/job_id
    need(Path(os.environ['EVCS_JOB_DIRECTORY']).resolve()==job,'Foreign reporting job parent')
    manifest=read(job/'manifest.json');status=read(job/'status.json')
    expected=[sys.executable,str(Path(__file__).resolve()),'--config',str(Path(config_path).resolve()),
              '--request',request_reference['path'],'--sha256',request_reference['sha256']]
    need(manifest['job_id']==job_id and Path(manifest['outroot']).resolve()==root/'jobs'
         and manifest['argv'][-len(expected):]==expected,'Parent manifest/report invocation differs')
    clock=_supervisor_remaining(manifest,status)
    with (root/'.audit.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=read(root/'state.json');config=active_config(root)
        need(state['status']=='CASE0_NUMERICAL_PASS_REPORT_PENDING' and state['evaluation_opened'] is True,
             'Numerical acceptance is required before reporting')
        need(state['active_configuration'] is None and not state.get('pending_evaluation_attempt') and
             not state.get('audit_reservation_reference') and state.get('active_job_id') in (None,job_id),
             'Unsettled Case0 work blocks reporting')
        need(config['total_compute_seconds']==21600 and
             manifest['timeout_seconds']+manifest['grace_seconds']<=21600-state['compute_seconds'],
             'Remaining Case0 budget cannot cover reporting parent')
        for other in (root/'jobs').glob('*/status.json'):
            if other.parent!=job:need(not observation(read(other)).get('blocks_new_launch'),'Another Case0 parent remains active')
        reservation=folder/'reservation.json';need(not reservation.exists(),'Reporting operation already reserved')
        atomic_json(reservation,dict(kind='CASE0_AUDIT_RESERVATION_V1',at_utc=utcnow(),job_id=job_id,
            operation_root=str(folder),opening_compute_seconds=state['compute_seconds'],
            parent_manifest_reference=file_reference(job/'manifest.json'),request_reference=request_reference,
            duration_limit_seconds=seconds,isolation=isolation,work_scope='statistics_only_no_optimization'))
        state['audit_reservation_reference']=file_reference(reservation);state['active_job_id']=job_id
        state['next_action']='Reporting parent running; settle exact terminal parent wall before accepting statistics.'
        save_state(root,state)
        try:
            acceptance=checked(request['evaluation_acceptance_reference']);audit=checked(request['numerical_audit_reference'])
            need(state['evaluation_acceptance']==request['evaluation_acceptance_reference'] and
                 acceptance['kind']=='CASE0_EVALUATION_ACCEPTANCE_V1' and acceptance['passed'] is True and
                 acceptance['item_count']==768 and acceptance['numerical_audit_reference']==request['numerical_audit_reference'] and
                 acceptance['formal_OOS'] is False and acceptance['Case1_acceptance'] is False,'Wrong accepted Case0 evaluation')
            need(audit['kind']=='CASE0_INDEPENDENT_EVALUATION_NUMERICAL_AUDIT_V1' and audit['passed'] is True and
                 audit['item_count']==len(audit['items'])==768 and audit['formal_OOS'] is False and audit['Case1_acceptance'] is False,
                 'Wrong independent Case0 numerical audit')
            for key in ('original_coefficients_checked','fixed_plan_rhs_checked','full_primal_dual_kkt_checked','physical_metrics_checked','paired_axes_checked'):
                need(audit[key] is True,'Incomplete independent numerical audit scope')
            for key in ('lock_reference','planning_acceptance_reference','evaluation_parent_settlement_reference'):
                verify_reference(acceptance[key]);need(acceptance[key]==audit[key],'Accepted audit evidence differs: '+key)
            checked(acceptance['audit_parent_settlement_reference'])
            source=request['summary_script_reference'];verify_reference(source)
            need(Path(source['path']).name=='case0_summary.py','Only fixed statistics script is permitted')
            need(request['wrapper_reference']==file_reference(__file__),'Report wrapper source differs')
            review=checked(request['summary_review_reference'])
            need(review['kind']=='CASE0_SUMMARY_SOURCE_REVIEW_V1' and review['passed'] is True and
                 review['summary_script_reference']==source,'Statistics script lacks exact source review')
            argv=[sys.executable,source['path'],'--audit',request['numerical_audit_reference']['path'],
                  '--sha256',request['numerical_audit_reference']['sha256'],'--output',str(output)]
            remaining=min(began+seconds,clock['conservative_deadline_monotonic']-10)-time.monotonic()
            need(remaining>0,'Reporting parent has no child/seal allowance')
            atomic_json(folder/'child_request.json',dict(argv=argv,request_reference=request_reference,
                parent_manifest_reference=file_reference(job/'manifest.json'),timeout_seconds=remaining,
                reservation_reference=file_reference(reservation)))
            with (folder/'stdout.log').open('x') as stdout,(folder/'stderr.log').open('x') as stderr:
                completed=subprocess.run(argv,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                    cwd=str(folder),timeout=remaining,check=False)
            need(completed.returncode==0,'Statistics child failed: '+str(completed.returncode))
            verify_reference(source);verify_reference(request['numerical_audit_reference'])
            need(output.is_dir(),'Statistics child did not create report output')
            refs=[file_reference(p) for p in sorted(output.rglob('*')) if p.is_file()]
            need(refs,'Statistics output is empty')
            result=dict(kind='CASE0_REPORT_JOB_OUTPUT_V1',request_reference=request_reference,artifacts=refs,
                child_request_reference=file_reference(folder/'child_request.json'),elapsed_seconds=time.monotonic()-began,
                terminal_parent_settlement_required=True,numerical_acceptance=False,solver_calls=0)
            atomic_json(folder/'report_job_output.json',result);return result
        except Exception as error:
            atomic_json(folder/'failure.json',dict(kind='CASE0_REPORT_JOB_FAILURE_V1',request_reference=request_reference,
                error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-began,
                reservation_reference=file_reference(reservation),terminal_parent_settlement_required=True))
            raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--request',required=True);p.add_argument('--sha256',required=True);a=p.parse_args(argv)
    os.environ['EVCS_CONFIG']=str(Path(a.config).resolve());ref=file_reference(a.request)
    need(ref['sha256']==a.sha256,'Reporting request hash differs')
    print(json.dumps(run_report_job(ref,a.config),allow_nan=False))


if __name__=='__main__':main()
