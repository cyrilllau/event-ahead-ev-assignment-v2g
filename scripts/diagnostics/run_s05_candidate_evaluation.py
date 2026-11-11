"""Prepared component: fixed-candidate original recourse evaluation, no CLI launch.

A reviewed parent must bind old package/new script sources, authorize the fresh
candidate, supply a hard deadline, and settle actual parent time. This module
cannot launch/resume jobs, calculate probabilities, or create a formal plan lock.
"""
from pathlib import Path
import json,hashlib,math,time
from time import perf_counter
from contextlib import contextmanager
import numpy as np
from evcs_experiments.first_stage import FirstStagePlan
from evcs_experiments.first_stage_check import check_first_stage
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.recourse import solve_fixed_plan
from evcs_experiments.runtime.jobctl import atomic_json


def need(ok,message):
    if not ok:raise ValueError(message)


def reference(path):
    p=Path(path).resolve(strict=True);h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(4194304):h.update(b)
    return dict(path=str(p),sha256=h.hexdigest(),size_bytes=p.stat().st_size)


def verify(ref):
    need(set(ref)=={'path','sha256','size_bytes'} and reference(ref['path'])==ref,'Changed strong reference')
    return Path(ref['path'])


def read(ref):return json.loads(verify(ref).read_text())


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    return reference(path)


def serial_plan(plan):return dict(x=plan.x.tolist(),r=plan.r.tolist(),task_sha256=plan.task_sha256,config_hash=plan.config_hash)

@contextmanager
def timed_phase(timings,key):
    began=perf_counter()
    try:yield
    finally:timings[key]+=perf_counter()-began


def extract_candidate(binding_reference,first_stage):
    """Read a reviewed binding to actual native result/vector; never round x.

Binding keys: result_reference, vector_reference, vector_key,
first_stage_slice=[0,n], first_stage_sha256. The immutable parent request must
pin this binding and independently verify its relation to the native result.
"""
    b=read(binding_reference);verify(b['result_reference']);p=verify(b['vector_reference'])
    n=first_stage.A.shape[1];h=len(first_stage.task_ids);t=len(first_stage.periods)
    need(b['first_stage_sha256']==first_stage.matrix_sha256 and b['first_stage_slice']==[0,n],'Candidate first-stage identity/slice differs')
    with np.load(p,allow_pickle=False) as z:
        need(b['vector_key'] in z.files,'Missing native candidate vector');v=np.asarray(z[b['vector_key']])
    need(v.ndim==1 and len(v)>=n and np.isfinite(v).all(),'Invalid native candidate vector')
    plan=FirstStagePlan(v[:h],v[h:n].reshape(h,t),first_stage.task_sha256,first_stage.config_hash)
    need(first_stage.evaluate(plan).feasible,'Native candidate violates original integer first-stage domain')
    return plan


class ReusedTemplateEvidenceAdapter:
    """Explicit single-exporter adapter; old template artifacts remain immutable.

Each used original record and all its artifacts are fully hashed once in this
invocation. A freshly assembled original-kernel template must match its accepted
semantic digest before the exporter may reference it. No old q/witness is reused.
"""
    def __init__(self,evidence,*,old_history_reference,old_checkpoint_reference,saved160_acceptance_reference,output):
        self.evidence=evidence;self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False)
        self.refs=dict(old_history=old_history_reference,old_checkpoint=old_checkpoint_reference,saved160_acceptance=saved160_acceptance_reference)
        card=read(saved160_acceptance_reference)
        need(card['status']=='PASS_SAVED160_COMPONENT_REVIEW' and card['complete_lp_count']==160,'Unaccepted original templates')
        need(old_history_reference in card['references'].values() and old_checkpoint_reference in card['references'].values(),'Wrong accepted source')
        cp=read(old_checkpoint_reference);self.state=cp['state'];self.history=read(old_history_reference)
        need(cp['sha256']==canonical_hash(self.state) and self.state['problem_sha256']==canonical_hash(self.state['identity'])==card['problem_sha256'],'Original checkpoint semantic identity differs')
        need(len(self.history)==163 and len(evidence.ids)==160 and list(evidence.ids)==self.state['identity']['scenario_ids'],'Wrong full support')
        need(evidence.problem.first_stage.matrix_sha256==self.state['identity']['first_stage_sha256'],'Wrong first-stage coefficients')
        need(evidence.problem.identity==self.state['identity'],'Prepared model/source/ambiguity identity differs from original kernel')
        self.used={};self.previous=evidence.template;self.installed=False
        self.original_template_verification_seconds=0.
        write(self.output/'binding.json',dict(references=self.refs,original_problem_sha256=self.state['problem_sha256'],reuse_scope='original template coefficients only; new plan RHS/LP/physical/KKT evidence required',formal_OOS=False))

    def install(self):
        need(not self.installed,'Adapter already installed');self.evidence.template=self.template;self.installed=True
        return self

    def template(self,index,template):
        need(type(index) is int and 0<=index<160,'Wrong template index')
        need(self.evidence.problem.identity==self.state['identity'],'Prepared model identity changed during evaluation')
        expected=self.state['pending']['recourse_certificates'][index]['template_sha256']
        need(template.template_sha256==expected and template.scenario_sha256==self.evidence.problem.recourses.scenario_sha256s[index]
             and template.first_stage.matrix_sha256==self.state['identity']['first_stage_sha256'],'Reassembled original template differs')
        if index in self.used:return self.used[index]['template_reference']
        verify_started=perf_counter()
        er=self.history[index+2];event=read(er)
        need(event['event']=='recourse' and event['scenario_index']==index and event['scenario_id']==self.evidence.ids[index],'Wrong original event')
        match=[r for r in event['artifacts'] if Path(r['path']).name=='template_reference.json'];need(len(match)==1,'Missing unique original template reference')
        tr=read(match[0]);row=read(tr)
        need(row['kind']=='recourse_template' and row['scenario_index']==index and row['scenario_id']==self.evidence.ids[index] and row['template_sha256']==expected,'Wrong original template record')
        for r in row['artifacts']:verify(r)
        receipt=dict(index=index,scenario_id=self.evidence.ids[index],template_sha256=expected,event_reference=er,template_pointer_reference=match[0],template_reference=tr,artifact_count=len(row['artifacts']),all_original_artifacts_hashed=True)
        write(self.output/f'{index:03d}.json',receipt);self.used[index]=receipt
        self.original_template_verification_seconds+=perf_counter()-verify_started
        return tr

    def inventory(self):
        return dict(references=self.refs,templates=[self.used[i] for i in sorted(self.used)],new_identity_json_exports=0,old_q_reused=False,formal_OOS=False,
                    original_template_verification_seconds=self.original_template_verification_seconds)


def evaluate_prepared_candidate(prepared,plan,*,candidate_binding_reference,output,deadline,lp_time_limit_seconds=120,shutdown_reserve_seconds=10,_development_hooks=None):
    """Run one fresh full-support pass, checkpoint each new q; no automatic resume.

Prepared must be created by the reviewed old factory and its evidence.template
adapter installed before any lazy recourse access. The caller owns source,
parent/cumulative budget checks. Checkpoints preserve progress but this component
never claims production restart support or a completed probability certificate.
"""
    need(math.isfinite(deadline) and type(lp_time_limit_seconds) in (float,int) and 0<lp_time_limit_seconds<=120 and shutdown_reserve_seconds>=10,'Invalid bounded evaluation clock')
    adapter=getattr(prepared.evidence.template,'__self__',None)
    need(isinstance(adapter,ReusedTemplateEvidenceAdapter) and adapter.installed,'Original template adapter must be explicitly installed')
    need(not prepared.evidence.events,'Fresh candidate evaluation cannot append an unrelated prior raw history')
    binding=read(candidate_binding_reference);need(binding['first_stage_sha256']==prepared.problem.first_stage.matrix_sha256,'Wrong candidate binding')
    extracted=extract_candidate(candidate_binding_reference,prepared.problem.first_stage)
    need(serial_plan(extracted)==serial_plan(plan),'Plan differs from pinned native candidate')
    hooks=_development_hooks
    if hooks is not None:need(prepared.evidence.domain=='development','No formal dependency injection')
    physical=check_first_stage if hooks is None else hooks['check_first_stage'];solve=solve_fixed_plan if hooks is None else hooks['solve']
    check=physical(prepared.case,prepared.tasks,plan);need(check['passed'],'Candidate physical first-stage check failed')
    ids=list(prepared.problem.probability_set.support_ids);need(len(ids)==160,'Full original160 support required')
    out=Path(output);out.mkdir(parents=True,exist_ok=False);write(out/'first_stage_check.json',check);planref=write(out/'plan.json',serial_plan(plan))
    q=[None]*160;events=[];began=time.monotonic();sink=prepared.evidence.sink_factory(out/'raw_callback')
    timings=dict(template_lookup_and_assembly_seconds=0.,solve_call_seconds=0.,raw_export_seconds=0.,checkpoint_write_seconds=0.)
    def checkpoint(status,next_index,error=None):
        value=dict(status=status,plan_reference=planref,plan_sha256=canonical_hash(serial_plan(plan)),candidate_binding_reference=candidate_binding_reference,
            support_ids=ids,q_usd=q,completed_count=len(events),next_index=next_index,recourse_event_references=events,
            raw_history_reference=reference(prepared.evidence.root/'history.json'),template_reuse_inventory=adapter.inventory(),
            elapsed_seconds=time.monotonic()-began,probability_evaluated=False,full_support_verified=False,scientific_acceptance=False,
            formal_OOS=False,normal_ccg_checkpoint_compatible=False,automatic_resume_supported=False,error=error,
            phase_timings_seconds=dict(timings),timing_scope='Completed template verification is nested within lookup/assembly; phase timers include failures. Checkpoint timing excludes the currently serialized write. All phases are nested within actual parent wall time and are not additional budget charges.')
        tick=perf_counter();atomic_json(out/'checkpoint.json',value);timings['checkpoint_write_seconds']+=perf_counter()-tick
        return value
    checkpoint('IN_PROGRESS',0)
    for i in range(160):
        remaining=deadline-time.monotonic()-shutdown_reserve_seconds
        if remaining<=0:return checkpoint('TIME_LIMIT',i)
        try:
            with timed_phase(timings,'template_lookup_and_assembly_seconds'):template=prepared.problem.recourses[i]
            remaining=deadline-time.monotonic()-shutdown_reserve_seconds
            if remaining<=0:return checkpoint('TIME_LIMIT',i)
            with timed_phase(timings,'solve_call_seconds'):result=solve(template,plan,time_limit_seconds=min(float(lp_time_limit_seconds),remaining))
            with timed_phase(timings,'raw_export_seconds'):sink('recourse',1,index=i,plan=plan,result=result)
            need(result.optimal and result.certificate.get('optimal_certificate_valid') is True and result.objective_usd is not None,'New recourse lacks an optimal certificate')
            value=float(result.objective_usd+prepared.problem.scenario_cost_constants_usd[i]);need(math.isfinite(value),'Nonfinite new q')
            eventref=prepared.evidence.events[-1];event=read(eventref)
            need(event['scenario_index']==i and event['plan_sha256']==canonical_hash(serial_plan(plan)) and event['q_usd']==value and event['optimal_certificate_valid'] is True,'New raw export disagrees with solver/plan')
            q[i]=value;events.append(eventref);checkpoint('IN_PROGRESS',i+1)
        except Exception as exc:
            checkpoint('FAILED_RECOURSE',i,dict(type=type(exc).__name__,message=str(exc)));raise
        if time.monotonic()>=deadline-shutdown_reserve_seconds:return checkpoint('TIME_LIMIT',i+1)
    record=checkpoint('FULL_Q_PENDING_INDEPENDENT_AUDIT_AND_PROBABILITY',160)
    write(out/'full_q.json',record);return record


if __name__=='__main__':
    raise SystemExit('Prepared component only: production parent/source/candidate/continuation binding is required; no standalone launch CLI.')
