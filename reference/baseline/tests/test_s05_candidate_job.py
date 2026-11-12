"""Production wrapper guards; no formal data, LP, license or server required."""
import copy,hashlib,importlib.util,json,os
from pathlib import Path
import tempfile,types,unittest
from unittest.mock import patch
from evcs_experiments.config import get_paths
from evcs_experiments.random_streams import canonical_hash

p=Path(os.environ.get('EVCS_TEST_PROJECT_ROOT',get_paths().project_root))/'scripts/diagnostics/run_s05_candidate_job.py'
spec=importlib.util.spec_from_file_location('candidate_job_wrapper',p);d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)


def settled():
    return dict(kind='S05_EPIGRAPH_MASTER_PARENT_ACCOUNTING_V1',status='SETTLED_AUXILIARY_MASTER_PENDING_INDEPENDENT_RESULT_REVIEW',
        plan_id='p',accounting_PASS=True,nested_seconds_added=False,cumulative_seconds_provisional=False,
        external_preprocessing_charged_once=True,parent_charged_once=True,
        opening_budget=dict(previous_cumulative_seconds_decimal='3948.418466',external_preprocessing_seconds_decimal='20.62929236330092'),
        terminal_parent_seconds_decimal='2.513214',cumulative_accounted_seconds_decimal='3971.56097236330092',remaining_seconds_decimal='3228.43902763669908',
        cumulative_accounted_seconds=3971.560972363301,remaining_seconds=3228.439027636699,cumulative_cap_seconds=7200)


def request():
    return dict(kind=d.KIND,domain='formal',formal_OOS=False,resume=False,plan_id='p',wall_seconds=3190,shutdown_reserve_seconds=10,
        lp_time_limit_seconds=60,completion_uses_final_evidence_reserve=True,parent_job=dict(timeout_seconds=3200,grace_seconds=5,threads=1))


class CandidateBudgetTests(unittest.TestCase):
    def test_real_decimal_budget_and_completion_reserve_use(self):
        r=d.budget_guard(request(),settled());self.assertEqual(r['cumulative_seconds_decimal'],'3971.56097236330092');self.assertEqual(r['remaining_seconds_decimal'],'3228.43902763669908')
    def test_same_budget_no_reset_resume_or_injected_scope(self):
        for key,value in [('resume',True),('formal_OOS',True),('domain','development'),('wall_seconds',7200),('shutdown_reserve_seconds',0),('completion_uses_final_evidence_reserve',False),('lp_time_limit_seconds',float('nan'))]:
            q=request();q[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):d.budget_guard(q,settled())
    def test_cost_flags_or_decimal_tamper_rejected(self):
        for key,value in [('nested_seconds_added',True),('accounting_PASS',False),('cumulative_seconds_provisional',True),('external_preprocessing_charged_once',False),('parent_charged_once',False),('cumulative_accounted_seconds_decimal','0'),('remaining_seconds_decimal','7200'),('plan_id','other')]:
            s=settled();s[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):d.budget_guard(request(),s)
    def test_parent_lp_and_shutdown_caps(self):
        for key,value in [('timeout_seconds',3300),('grace_seconds',10),('threads',4)]:
            q=request();q['parent_job'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):d.budget_guard(q,settled())
        q=request();q['lp_time_limit_seconds']=121
        with self.assertRaises(ValueError):d.budget_guard(q,settled())


class CandidateProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve();self.addCleanup(self.tmp.cleanup)
    def save(self,name,value):
        p=self.root/name;p.write_text(json.dumps(value));return d.reference(p)
    def fixture(self):
        ids=['s'+str(i) for i in range(160)];identity=dict(first_stage_sha256='first',scenario_ids=ids);problem=canonical_hash(identity);state=dict(identity=identity,problem_sha256=problem)
        cp=self.save('cp.json',dict(state=state,sha256=canonical_hash(state)));raw=self.save('raw.json',dict(problem_identity=identity,domain='formal',formal_OOS=False))
        vector=self.root/'v.npz';vector.write_bytes(b'originalvector');vr=d.reference(vector);source=self.save('source_request.json',{})
        nr=self.save('native.json',dict(valid_incumbent=True,certificate_failures=[],incumbent_without_constant_usd=12.))
        binding=dict(vector_reference=vr,source_request=source,source_native_result=nr,vector_key='vector',first_stage_slice=[0,25200],formal_OOS=False,lockable=False,problem_sha256=problem,first_stage_sha256='first',plan_sha256='plan')
        br=self.save('binding.json',binding)
        result=dict(candidate_vector=vr,request_reference=source,candidate_available=True,candidate_auxiliary_objective_usd=12.,problem_sha256=problem,first_stage_sha256='first',candidate_plan_sha256='plan')
        rr=self.save('result.json',result)
        review=dict(status='PASS_INDEPENDENT_EPIGRAPH_AUXILIARY_RESULT_REVIEW',passed=True,references=dict(result=rr,candidate_binding=br,native_solver=nr),original_problem_sha256=problem,candidate_plan_sha256='plan')
        cr=self.save('review.json',review);sr=self.save('settlement.json',{})
        ar=self.save('adoption.json',dict(id='D091',candidate=br,result=rr,review=cr,settlement=sr))
        card=self.save('saved160.json',dict(status='PASS_SAVED160_COMPONENT_REVIEW',complete_lp_count=160,problem_sha256=problem,references=dict(checkpoint=cp,raw_request=raw)))
        return dict(adoption_record=ar,candidate_binding=br,candidate_result=rr,candidate_native=nr,candidate_review=cr,current_settlement=sr,old_checkpoint=cp,raw_request=raw,saved160_acceptance=card),binding
    def test_whole_actual_schema_candidate_chain(self):
        refs,binding=self.fixture();b,i=d.candidate_authorization(refs);self.assertEqual(b,binding);self.assertEqual(len(i['scenario_ids']),160)
    def test_same_size_vector_change_rejected(self):
        refs,b=self.fixture();Path(b['vector_reference']['path']).write_bytes(b'changed!vector')
        with self.assertRaises(ValueError):d.candidate_authorization(refs)
    def test_cross_candidate_result_pointer_rejected(self):
        refs,b=self.fixture();other=self.save('other.json',dict(valid_incumbent=True));refs['candidate_native']=other
        with self.assertRaises(ValueError):d.candidate_authorization(refs)
    def test_authoritative_decision_and_state_required(self):
        refs,b=self.fixture();campaign=self.root/'campaign';campaign.mkdir();adoption=d.read(refs['adoption_record']['path'])
        (campaign/'decisions.jsonl').write_text(json.dumps(adoption)+'\n');s=dict(specs=dict(S05=dict(pending_candidate_binding=refs['candidate_binding'],epigraph_settlement=refs['current_settlement'],epigraph_result_review=refs['candidate_review'])))
        (campaign/'state.json').write_text(json.dumps(s));d.candidate_authorization(refs,campaign=campaign)
        s['specs']['S05']['pending_candidate_binding']=refs['candidate_result'];(campaign/'state.json').write_text(json.dumps(s))
        with self.assertRaises(ValueError):d.candidate_authorization(refs,campaign=campaign)
    def test_same_predecessor_candidate_cannot_be_claimed_twice(self):
        refs,_=self.fixture();q=dict(references=refs);rr=self.save('request.json',q);pr=dict(job_id='j');a=d.reserve(q,rr,pr,self.root)
        self.assertTrue((a/'reservation.json').exists())
        with self.assertRaises(FileExistsError):d.reserve(q,rr,dict(job_id='newjob'),self.root)
        altered=copy.deepcopy(q);altered['references']['candidate_binding']=refs['candidate_result']
        with self.assertRaises(FileExistsError):d.reserve(altered,rr,dict(job_id='differentcandidate'),self.root)
    def test_manifest_escape_duplicate_and_same_size_source_change(self):
        release=self.root/'release';release.mkdir();p=release/'module.py';p.write_text('abc\n');r=d.reference(p);row=dict(path='module.py',sha256=r['sha256'],size_bytes=r['size_bytes'])
        mf=release/'RELEASE_MANIFEST.json'
        for rows in ([dict(row,path='../outside.py')],[row,row]):
            mf.write_text(json.dumps(dict(files=rows)))
            with self.assertRaises((ValueError,FileNotFoundError)):d.source_files(d.reference(mf))
        mf.write_text(json.dumps(dict(files=[row])));d.source_files(d.reference(mf));p.write_text('abd\n')
        with self.assertRaises(ValueError):d.source_files(d.reference(mf))
    def test_wrong_request_rejected_before_parent_and_factory(self):
        r=self.save('request.json',{})
        with patch.object(d,'bind_parent') as parent,patch.object(d,'load_formal_problem') as factory:
            with self.assertRaises(ValueError):d.execute(r['path'],'0'*64)
            parent.assert_not_called();factory.assert_not_called()
    def test_unsupervised_call_cannot_reach_factory(self):
        q=dict(parent_job=dict(job_id='wrong',directory=str(get_paths().campaign_root/'jobs/wrong')))
        with patch.dict(d.os.environ,{'EVCS_JOB_ID':'other'}),self.assertRaises(ValueError):d.bind_parent(q,{})
    def test_existing_receipt_not_overwritten(self):
        p=self.root/'r.json';d.write(p,dict(status='failed'))
        with self.assertRaises(FileExistsError):d.write(p,dict(status='passed'))
        self.assertEqual(d.read(p)['status'],'failed')


if __name__=='__main__':unittest.main()
