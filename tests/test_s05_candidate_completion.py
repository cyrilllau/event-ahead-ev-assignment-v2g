"""Supervision, provenance, cumulative budget and staged-completion guards."""
import copy,importlib.util,json,os,tempfile,time,types,unittest
from pathlib import Path
from unittest.mock import patch
from evcs_experiments.config import get_paths
p=Path(os.environ.get('EVCS_TEST_PROJECT_ROOT',get_paths().project_root))/'scripts/diagnostics/run_s05_candidate_completion.py'
s=importlib.util.spec_from_file_location('candidate_completion_wrapper',p);d=importlib.util.module_from_spec(s);s.loader.exec_module(d)


def request():
    return dict(kind=d.KIND,domain='formal',formal_OOS=False,resume=False,plan_id='p',wall_seconds=1790,shutdown_reserve_seconds=10,probability_time_limit_seconds=30,
        parent_job=dict(timeout_seconds=1800,grace_seconds=5,threads=1))


def settlement():
    return dict(kind='S05_CANDIDATE_EVALUATION_PARENT_ACCOUNTING_V1',status='SETTLED_FULL_Q_PENDING_INDEPENDENT_AUDIT_AND_PROBABILITY',plan_id='p',accounting_PASS=True,
        parent_charged_once=True,nested_seconds_added=False,cumulative_seconds_provisional=False,original_cumulative_cap_seconds=7200,cumulative_cap_seconds=7200,
        opening_cumulative_seconds_decimal='3971.56097236330092',terminal_parent_seconds_decimal='1365.467456',cumulative_accounted_seconds_decimal='5337.02842836330092',remaining_seconds_decimal='1862.97157163669908')


def terminal():return dict(complete=True,state='SUCCEEDED',returncode=0,elapsed_seconds=1365.467456)


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve();self.addCleanup(self.tmp.cleanup)
    def save(self,name,value):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);return d.write(p,value)
    def test_actual_decimal_budget_preserved(self):
        b=d.budget_guard(request(),settlement(),terminal());self.assertEqual(b['opening_cumulative_seconds_decimal'],'5337.02842836330092');self.assertEqual(b['remaining_seconds_decimal'],'1862.97157163669908')
    def test_budget_reset_wrong_terminal_cost_or_nested_cost_rejected(self):
        for key,value in [('opening_cumulative_seconds_decimal','0'),('terminal_parent_seconds_decimal','1364.185625012964'),('cumulative_accounted_seconds_decimal','0'),('remaining_seconds_decimal','7200'),('nested_seconds_added',True),('parent_charged_once',False),('accounting_PASS',False),('cumulative_seconds_provisional',True)]:
            s=settlement();s[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):d.budget_guard(request(),s,terminal())
    def test_invalid_scope_caps_and_probability_grant_rejected(self):
        for key,value in [('formal_OOS',True),('domain','development'),('resume',True),('wall_seconds',7200),('shutdown_reserve_seconds',0),('probability_time_limit_seconds',True),('probability_time_limit_seconds',31)]:
            q=request();q[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):d.budget_guard(q,settlement(),terminal())
        for key,value in [('timeout_seconds',1900),('grace_seconds',10),('threads',4)]:
            q=request();q['parent_job'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):d.budget_guard(q,settlement(),terminal())
    def test_exit_zero_does_not_allow_nonterminal_parent(self):
        t=terminal();t['complete']=False
        with self.assertRaises(ValueError):d.budget_guard(request(),settlement(),t)
    def test_exactly_one_successor_per_settlement_even_changed_candidate(self):
        sr=self.save('settlement.json',settlement());q=dict(references=dict(current_settlement=sr),candidate='first');rr=self.save('request.json',q)
        d.reserve(q,rr,dict(job_id='j1'),self.root);q['candidate']='changed'
        with self.assertRaises(FileExistsError):d.reserve(q,rr,dict(job_id='j2'),self.root)
    def test_unsupervised_wrong_request_fails_before_reservation(self):
        r=self.save('request.json',{})
        with patch.object(d,'bind_parent') as parent,patch.object(d,'reserve') as reserve:
            with self.assertRaises(ValueError):d.execute(r['path'],'0'*64)
            parent.assert_not_called();reserve.assert_not_called()
        q=dict(parent_job=dict(job_id='wrong',directory=str(get_paths().campaign_root/'jobs/wrong')))
        with patch.dict(os.environ,{'EVCS_JOB_ID':'other'}),self.assertRaisesRegex(ValueError,'supervisor'):d.bind_parent(q,{})
    def test_original_same_size_source_tamper_escape_and_clock(self):
        source=self.root/'release';source.mkdir();f=source/'m.py';f.write_text('abc');r=d.reference(f)
        mf=source/'RELEASE_MANIFEST.json';mf.write_text(json.dumps(dict(files=[dict(path='m.py',sha256=r['sha256'],size_bytes=3)])))
        d.source_files(d.reference(mf),time.monotonic()+60);f.write_text('abd')
        with self.assertRaises(ValueError):d.source_files(d.reference(mf),time.monotonic()+60)
        with self.assertRaisesRegex(ValueError,'budget'):d.source_files(d.reference(mf),time.monotonic()-1)
        mf.write_text(json.dumps(dict(files=[dict(path='../escape',sha256='x',size_bytes=1)])))
        with self.assertRaises((ValueError,FileNotFoundError)):d.source_files(d.reference(mf),time.monotonic()+60)
    def pipeline_fixture(self):
        q=request();full=dict(plan_sha256='plan',support_ids=['s'+str(i) for i in range(160)],q_usd=[3.]*160)
        q['references']=dict(full_q=self.save('full.json',full));q['audit_references']={'synthetic':self.save('auditinput.json',{})};q['probability_references']={'synthetic':self.save('probinput.json',{})}
        audit=dict(status='PASS_INDEPENDENT_CANDIDATE_160_RECOURSE_AUDIT',passed=True,complete_lp_count=160,plan_sha256='plan',support_ids=full['support_ids'],q_usd=full['q_usd'],original_problem_sha256='problem',Fpre_usd=7.,references=q['audit_references'])
        audit['audit_reference']=self.save('audit.json',audit);audit['artifact_manifest_reference']=self.save('artifacts.json',{})
        result=dict(probability_independent_check_passed=True,upper_bound_usd=10.,plan_sha256='plan',original_problem_sha256='problem',first_stage_cost_usd=7.)
        calls=[]
        def run_audit(refs,*,output,deadline):calls.append(('audit',deadline));return audit
        def run_probability(refs,*,output,time_limit_seconds):calls.append(('probability',time_limit_seconds));return result
        return q,audit,result,calls,types.SimpleNamespace(audit_candidate_recourse=run_audit),types.SimpleNamespace(complete_candidate_probability=run_probability)
    def test_audit_precedes_probability_and_exact_shared_q_is_required(self):
        q,a,r,c,aud,prob=self.pipeline_fixture();deadline=time.monotonic()+100
        result=d.pipeline(aud,prob,q,self.root/'out',deadline)
        self.assertEqual([x[0] for x in c],['audit','probability']);self.assertEqual(c[0][1],deadline-40);self.assertLessEqual(c[1][1],30);self.assertEqual(result[1]['upper_bound_usd'],10.)
    def test_failed_or_other_plan_audit_never_reaches_probability(self):
        for key,value in [('passed',False),('complete_lp_count',159),('plan_sha256','other'),('support_ids',['other']*160),('q_usd',[4.]*160),('references',{})]:
            with tempfile.TemporaryDirectory() as tmp:
                old=self.root;self.root=Path(tmp);q,a,r,c,aud,prob=self.pipeline_fixture();self.root=old;a[key]=value
                with self.assertRaises(ValueError):d.pipeline(aud,prob,q,Path(tmp)/'out',time.monotonic()+100)
                self.assertEqual([x[0] for x in c],['audit'])
    def test_probability_failure_or_exhausted_post_audit_budget_has_no_complete_result(self):
        q,a,r,c,aud,prob=self.pipeline_fixture();r['probability_independent_check_passed']=False
        with self.assertRaises(ValueError):d.pipeline(aud,prob,q,self.root/'out',time.monotonic()+100)
        c.clear()
        with self.assertRaisesRegex(ValueError,'budget'):d.pipeline(aud,prob,q,self.root/'out',time.monotonic()+20)
        self.assertEqual(c,[])
    def test_late_probability_completion_is_rejected(self):
        q,a,r,c,aud,prob=self.pipeline_fixture();clock=[0.]
        def late(*args,**kwargs):clock[0]=95.;return r
        prob.complete_candidate_probability=late
        with patch.object(d.time,'monotonic',side_effect=lambda:clock[0]),self.assertRaisesRegex(ValueError,'budget'):d.pipeline(aud,prob,q,self.root/'out',100.)
    def binding_fixture(self):
        def saved(name,value):return self.save('bindings/'+name,value)
        raw=saved('raw.json',{});original=saved('binding.json',{});oldcp=saved('oldcp.json',{});history=saved('history.json',[]);card=saved('card.json',{})
        mf=saved('parent_manifest.json',{});job=dict(terminal(),job_id='oldj');jr=saved('terminal.json',job);js=saved('status.json',job)
        full=dict(plan_sha256='plan',recourse_event_references=[]);cp=saved('cp.json',full);fq=saved('full.json',full)
        snap=saved('snapshot.json',dict(events=[],request=raw,full_support_verified=False,formal_OOS=False))
        result=saved('result.json',dict(checkpoint_reference=cp,raw_snapshot_reference=snap,completed_count=160,full_q_available=True,formal_OOS=False,probability_evaluated=False,
            original_candidate_binding=original,plan_sha256='plan',parent=dict(manifest_reference=mf,job_id='oldj')))
        settled=settlement();settled['references']=dict(candidate_result=result,terminal_parent_result=jr,parent_manifest=mf);sr=saved('settlement.json',settled)
        audit=saved('audit.py',{});prob=saved('prob.py',{});execution=saved('execution.json',{})
        q=request();q.update(target_id='michael_2018',method_id='EA',references=dict(current_settlement=sr,previous_parent_result=jr,previous_parent_manifest=mf,
            evaluation_result=result,final_checkpoint=cp,full_q=fq,raw_snapshot=snap,evaluation_execution=execution,auditor_script=audit,probability_script=prob),
            audit_references=dict(audit_script=audit,candidate_result=result,original_checkpoint=oldcp,original_history=history,saved160_acceptance=card,candidate_job_status=js),
            probability_references=dict(driver_source=prob,evaluation_result=result,checkpoint=cp,full_q=fq,execution=execution,original_checkpoint=oldcp,original_history=history,
                saved160_acceptance=card,new_raw_request=raw,original_binding=original))
        state=dict(specs=dict(S05=dict(current_formal_plan_id='p',pending_candidate_binding=original,candidate_evaluation_settlement=sr,candidate_evaluation_result=result)))
        self.save('state.json',state);return q,state
    def test_actual_schema_cross_refs_and_authoritative_state(self):
        q,state=self.binding_fixture();obs=dict(supervisor_alive=False,command_alive=False,live_group_members=[],blocks_new_launch=False)
        with patch.object(d,'observation',return_value=obs):
            b,r=d.input_bindings(q,campaign=self.root);self.assertEqual(b['remaining_seconds_decimal'],'1862.97157163669908')
            q['probability_references']['checkpoint']=q['references']['full_q']
            with self.assertRaisesRegex(ValueError,'Cross-component'):d.input_bindings(q,campaign=self.root)
    def test_terminal_descendants_and_stale_authoritative_settlement_rejected(self):
        q,state=self.binding_fixture();obs=dict(supervisor_alive=False,command_alive=False,live_group_members=[123],blocks_new_launch=True)
        with patch.object(d,'observation',return_value=obs),self.assertRaisesRegex(ValueError,'descendants'):d.input_bindings(q,campaign=self.root)
        state['specs']['S05']['candidate_evaluation_settlement']=q['references']['previous_parent_result'];(self.root/'state.json').write_text(json.dumps(state))
        obs.update(live_group_members=[],blocks_new_launch=False)
        with patch.object(d,'observation',return_value=obs),self.assertRaisesRegex(ValueError,'Authoritative'):d.input_bindings(q,campaign=self.root)
    def test_late_result_export_cannot_write_completion_receipt(self):
        out=self.root/'out';out.mkdir();attempt=self.root/'attempt';attempt.mkdir();clock=[0.];original=d.write
        def slow(path,value):
            ref=original(path,value)
            if Path(path).name=='result.json':clock[0]=95.
            return ref
        with patch.object(d,'write',side_effect=slow),patch.object(d.time,'monotonic',side_effect=lambda:clock[0]),self.assertRaisesRegex(ValueError,'budget'):
            d.finalize(out,attempt,{},dict(scientific_acceptance=False),100.)
        self.assertTrue((out/'result.json').exists());self.assertFalse((attempt/'completion.json').exists())
    def test_immutable_receipt_never_overwritten(self):
        r=self.save('r.json',{'old':True})
        with self.assertRaises(FileExistsError):d.write(r['path'],{'new':True})
        self.assertEqual(d.read(r['path']),{'old':True})

if __name__=='__main__':unittest.main()
