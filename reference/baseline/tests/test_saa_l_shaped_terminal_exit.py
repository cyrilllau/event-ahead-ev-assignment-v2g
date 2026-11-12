"""A real native time limit retains the old candidate and every master event.

The positive path makes one ordinary native/full-Q iteration, then invokes the
same native HiGHS solver with an explicitly recorded 1 ns limit. Metadata
mutation negatives are controlled fixtures, never new production authorization.
"""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import time
import unittest
from unittest.mock import patch
import numpy as np
from evcs_experiments import saa_l_shaped as kernel
from evcs_experiments import saa_l_shaped_candidate as candidate
from evcs_experiments import saa_l_shaped_master_evidence as masters
from evcs_experiments import saa_fixed_plan_audit as full_q
from evcs_experiments.campaign_batch import file_reference
import test_saa_l_shaped_audit_bridge as fixtures
import test_saa_l_shaped_candidate as candidate_tests


class LShapedTerminalExitTests(unittest.TestCase):
    build = classmethod(candidate_tests.LShapedCandidateTests.build.__func__)
    metadata = candidate_tests.LShapedCandidateTests.metadata

    @classmethod
    def setUpClass(cls):
        cls.fixture = fixtures.LShapedAuditBridgeTests('test_real_kernel_events_original_audit_same_plan_negative_constants_once')
        cls.fixture.setUp(); cls.addClassCleanup(cls.fixture.doCleanups)
        f=cls.fixture; cls.root=f.f.root.resolve(); cls.problem=f.problem
        cls.sources=list({r['path']:r for r in [file_reference(p) for p in (candidate.__file__,kernel.__file__,masters.__file__,full_q.__file__)]
            +f.args['source_references']+f.f.args['source_references']+list(masters._sources().values())+[f.f.args['helper_reference']]}.values())
        cls.sink=masters.DevelopmentMasterEvidence(cls.root/'masters'); path=cls.root/'kernel.json'
        cls.partial=kernel.run_saa_l_shaped(cls.problem,checkpoint_path=path,evidence_sink=f.sink,
            master_sink=cls.sink,max_iterations=1)
        cls.partial_cp=cls.root/'partial_kernel_copy.json';cls.partial_cp.write_bytes(path.read_bytes())
        cls.partial_pair=cls.build(cls.partial,cls.partial_cp,'partial')
        cls.old_refs=[r['evidence_reference'] for r in cls.partial['master_history']]
        old_pin=file_reference(path); native=kernel.milp; cls.observed=[]
        def tiny_native(c,**kwargs):
            # Update the original options object before the actual call. The
            # evidence sink records exactly these effective submitted options.
            kwargs['options']['time_limit']=1e-9
            raw=native(c,**kwargs)
            cls.observed.append(dict(options=dict(kwargs['options']),status=int(raw.status),
                message=str(raw.message),x_is_none=raw.x is None,
                bound_is_none=getattr(raw,'mip_dual_bound',None) is None))
            return raw
        with patch.object(kernel,'milp',side_effect=tiny_native):
            cls.final=kernel.run_saa_l_shaped(cls.problem,checkpoint_path=path,resume=True,
                expected_checkpoint_sha256=old_pin['sha256'],evidence_sink=f.sink,master_sink=cls.sink)
        cls.final_pair=cls.build(cls.final,path,'terminal_without_new_bound')
        candidate._write(cls.root/'terminal_exit_summary.json',dict(
            kind='REAL_NATIVE_TIME_LIMIT_NO_NEW_BOUND_DEVELOPMENT_EXIT',
            native_calls=cls.observed,old_checkpoint_reference=file_reference(cls.partial_cp),
            old_master_references=cls.old_refs,terminal_checkpoint_reference=file_reference(path),
            candidate_reference=cls.final_pair[0],joint_audit=cls.final_pair[2],
            formal_admission=False,whole_parent_admission=False))

    def test_real_native_time_limit_preserves_negative_old_bound_and_full_q(self):
        result=self.metadata(); joint=self.final_pair[2]
        self.assertGreaterEqual(len(self.observed),1)
        self.assertEqual(self.observed[-1]['status'],1)
        self.assertTrue(self.observed[-1]['x_is_none'])
        self.assertTrue(self.observed[-1]['bound_is_none'])
        self.assertTrue(all(r['status']==0 for r in self.observed[:-1]))
        self.assertEqual(self.final['status'],'TIME_LIMIT')
        self.assertLessEqual(result['upper_bound_usd'],self.partial['upper_bound_usd'])
        finite=[r['lower_bound_usd'] for r in self.final['master_history'][:-1] if r['lower_bound_usd'] is not None]
        self.assertEqual(result['lower_bound_usd'],max(finite))
        self.assertLess(result['lower_bound_usd'],0)
        self.assertEqual(result['upper_bound_usd'],self.final['trace'][-1]['upper_bound_usd'])
        self.assertEqual(result['native_master_status'],'TIME_LIMIT')
        self.assertEqual(result['raw_engine_status'],'TIME_LIMIT')
        self.assertFalse(joint['convergence_claim']);self.assertFalse(joint['gap_passed'])
        self.assertEqual(len(joint['master_audits']),len(self.old_refs)+len(self.observed))
        self.assertEqual(self.final_pair[1]['master_execution_references'][:len(self.old_refs)],self.old_refs)
        last=joint['master_audits'][-1]
        self.assertTrue(last['passed']);self.assertIsNone(last['native_lower_bound_usd'])
        self.assertFalse(last['contributes_lower_bound'])
        self.assertEqual(last['bound_disposition'],'TIME_LIMIT_NO_FINITE_NATIVE_BOUND')
        native=joint['master_native_provenance'][-1]
        self.assertEqual(native['backend'],'scipy.optimize.milp/HiGHS')
        self.assertEqual(native['raw_model_status'],1)
        self.assertEqual(native['normalized_status'],1)
        self.assertEqual(native['status_name'],'TIME_LIMIT')
        self.assertTrue(joint['full_q_audit']['full_support_recourse_verified'])
        self.assertFalse(last['fixed_plan_full_Q_audited'])
        self.assertEqual(joint['solver_calls'],0);self.assertFalse(joint['formal_admission'])

    def test_no_bound_status_failure_and_zero_substitution_rejected(self):
        original=candidate._read(self.final_pair[1]['master_execution_references'][-1])
        for mode in ('optimal','infeasible','unbounded','numerical','success','zero','nonfinite','source'):
            changed=deepcopy(original)
            if mode in ('optimal','infeasible','unbounded','numerical'):
                code={'optimal':0,'infeasible':2,'unbounded':3,'numerical':4}[mode]
                changed['native_result'].update(status=code,success=code==0)
                changed['master_record']['solver_status']=code
            elif mode=='success':changed['native_result']['success']=True
            elif mode=='zero':changed['master_record']['lower_bound_usd']=0
            elif mode=='nonfinite':
                changed['native_result']['mip_dual_bound']=float('nan')
                changed['master_record']['lower_bound_usd']=float('nan')
            else:changed['bound_source']='incumbent_objective'
            with self.subTest(mode=mode),self.assertRaises(ValueError):masters.native_bound_contribution(changed)

    def test_dropping_last_event_or_relabeling_its_bound_contribution_rejected(self):
        evidence=self.final_pair[1]
        omitted={**evidence,'master_execution_references':evidence['master_execution_references'][:-1]}
        with self.assertRaisesRegex(ValueError,'Every old and new'):
            candidate.write_development_candidate(self.problem,plan_id='omitted',**omitted,output_path=self.root/'never.json')
        for mode in ('contributes','bound','status','raw_backend','raw_code'):
            joint=deepcopy(self.final_pair[2]); last=joint['master_audits'][-1]
            if mode=='contributes':last['contributes_lower_bound']=True
            elif mode=='bound':last['native_lower_bound_usd']=0
            elif mode=='status':last['native_status']=0
            elif mode=='raw_backend':last['native_provenance']['backend']='gurobipy'
            else:last['native_provenance']['raw_model_status']=9
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.metadata(joint=joint)

    def test_complete_q_without_any_retained_finite_bound_is_rejected(self):
        # Controlled metadata input: all masters have legitimate no-bound shape,
        # but their aggregate cannot produce a certificate for the old full Q.
        evidence=self.final_pair[1]; cp_ref=evidence['kernel_checkpoint_reference']; actual_read=candidate._read
        cp=deepcopy(actual_read(cp_ref)); state=cp['state']; replacements={}
        for row,ref in zip(state['master_history'],evidence['master_execution_references']):
            body=deepcopy(actual_read(ref))
            row.update(solver_status=1,lower_bound_usd=None)
            body['master_record']={k:v for k,v in row.items() if k!='evidence_reference'}
            body['native_result'].update(status=1,success=False,mip_dual_bound=None)
            replacements[ref['path']]=body
        state['lower_bound_usd']=None;cp['sha256']=candidate.canonical_hash(state)
        replacements[cp_ref['path']]=cp
        def controlled_read(ref):return deepcopy(replacements[ref['path']]) if ref['path'] in replacements else actual_read(ref)
        with patch.object(candidate,'_read',side_effect=controlled_read),self.assertRaisesRegex(ValueError,'at least one retained finite'):
            candidate._bound_candidate(self.problem,'controlled-no-global-bound',evidence)

    def test_finite_old_bound_without_complete_best_q_is_rejected(self):
        evidence=self.final_pair[1];cp_ref=evidence['kernel_checkpoint_reference'];actual_read=candidate._read
        cp=deepcopy(actual_read(cp_ref));cp['state']['best']=None
        cp['sha256']=candidate.canonical_hash(cp['state'])
        def controlled_read(ref):return deepcopy(cp) if ref==cp_ref else actual_read(ref)
        with patch.object(candidate,'_read',side_effect=controlled_read),self.assertRaisesRegex(ValueError,'complete best-plan sweep'):
            candidate._bound_candidate(self.problem,'controlled-no-best-q',evidence)

    def loading_fixture(self):
        # Boundary metadata control, not a claim of running licensed Gurobi.
        execution=deepcopy(candidate._read(self.final_pair[1]['master_execution_references'][-1]))
        nr=deepcopy(execution['native_result'])
        nr.update(message='Gurobi native: model loading exhausted time budget',limit_status='TIME_LIMIT')
        execution['native_result']['message']=nr['message']
        execution['master_record']['message']=nr['message']
        body=dict(native_record=dict(raw_model_status=1,status='MODEL_LOADING_TIME_LIMIT',
            effective_run_time_limit_seconds=0,solver_run_seconds=0,valid_incumbent=False),normalized_result=nr)
        execution['bound_source']='actual_gurobi_result.mip_dual_bound'
        return execution,body

    def test_typed_loading_stop_keeps_no_bound_and_distinct_raw_status(self):
        execution,body=self.loading_fixture()
        execution['native_adapter_reference']=candidate._write(self.root/'loading_boundary_control.json',body)
        self.assertIsNone(masters.native_bound_contribution(execution))
        self.assertEqual(body['native_record']['raw_model_status'],1)
        self.assertEqual(body['native_record']['status'],'MODEL_LOADING_TIME_LIMIT')
        self.assertNotIn('certificate_failures',body['native_record'])
        self.assertNotIn('solver_bound_without_constant_usd',body['native_record'])

    def test_loading_stop_cannot_mask_memory_failure_optimizer_run_or_bound(self):
        for mode in ('memory','optimize_time','run_grant','ran_optimizer','incumbent','bound','gap','nodes','success'):
            execution,body=self.loading_fixture();raw=body['native_record'];normalized=body['normalized_result']
            if mode=='memory':raw['raw_model_status']=17
            elif mode=='optimize_time':raw['raw_model_status']=9
            elif mode=='run_grant':raw['effective_run_time_limit_seconds']=1
            elif mode=='ran_optimizer':raw['solver_run_seconds']=1
            elif mode=='incumbent':raw['valid_incumbent']=True
            elif mode=='bound':normalized['mip_dual_bound']=0
            elif mode=='gap':normalized['mip_gap']=0
            elif mode=='nodes':normalized['mip_node_count']=0
            else:normalized['success']=True
            execution['native_adapter_reference']=candidate._write(self.root/('loading_negative_'+mode+'.json'),body)
            with self.subTest(mode=mode),self.assertRaisesRegex(ValueError,'Typed Gurobi'):
                masters.native_bound_contribution(execution)

    @contextmanager
    def altered_terminal(self):
        ref=self.final_pair[1]['master_execution_references'][-1]
        ep=Path(ref['path']);execution=json.loads(ep.read_text());pp=Path(execution['preparation_reference']['path'])
        prep=json.loads(pp.read_text());mp=Path(prep['matrix_reference']['path'])
        before={p:p.read_bytes() for p in (ep,pp,mp)}
        try:yield ref,ep,execution,pp,prep,mp
        finally:
            for p,raw in before.items():p.write_bytes(raw)

    def test_no_bound_event_still_rejects_physical_matrix_corruption(self):
        with self.altered_terminal() as (old,ep,execution,pp,prep,mp):
            with np.load(mp,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
            arrays['c'][len(self.problem.first_stage.c)]=.5
            with mp.open('wb') as stream:np.savez_compressed(stream,**arrays)
            prep['matrix_reference']=file_reference(mp)
            prep['coefficient_sha256']=masters._hash(masters._load_model(prep['matrix_reference']))
            pp.write_text(json.dumps(prep,indent=2)+'\n')
            execution['preparation_reference']=file_reference(pp)
            execution['submitted_coefficient_sha256']=prep['coefficient_sha256']
            ep.write_text(json.dumps(execution,indent=2)+'\n')
            with self.assertRaises(ValueError):masters.audit_master_evidence(self.problem,old,deadline=time.monotonic()+30)
            # Even a deliberately fresh control pin cannot bypass the original
            # physical objective check; it is not a production trusted pin.
            with self.assertRaisesRegex(ValueError,'empirical objective'):
                masters.audit_master_evidence(self.problem,file_reference(ep),deadline=time.monotonic()+30)

    def test_no_bound_event_changed_bytes_do_not_refresh_caller_authorization(self):
        with self.altered_terminal() as (ref,ep,execution,pp,prep,mp):
            execution['native_result']['message']='tampered event'
            ep.write_text(json.dumps(execution,indent=2)+'\n')
            with self.assertRaises(ValueError):masters.audit_master_evidence(self.problem,ref,deadline=time.monotonic()+30)
            with self.assertRaises(ValueError):self.metadata()

if __name__=='__main__':unittest.main()
