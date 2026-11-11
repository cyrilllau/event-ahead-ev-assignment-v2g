"""Operational integration: tiny synthetic arrays, no server or native license.

Only raw candidate context/atom admission and the native solve are fixture
substitutes. The supervised request, source hashes, exclusive reservation,
actual multi-anchor assembly and its independent audits are exercised.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import ExitStack
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy.sparse import csr_matrix
from evcs_experiments.config import get_paths
from evcs_experiments.evaluation_lock import file_reference
from evcs_experiments.dual_cut_bound import build_dual_cut
from evcs_experiments.epigraph_master import certify_box_floor
from evcs_experiments.probability import ProbabilitySet
from evcs_experiments import candidate_dual_adapter, candidate_dual_audit, native_gurobi, multi_anchor_epigraph
from evcs_experiments import multi_anchor_bound_audit

spec=importlib.util.spec_from_file_location('bound_supervised_driver',get_paths().project_root/'scripts/diagnostics/s05_multi_anchor_bound.py')
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)


class SupervisedBoundTests(unittest.TestCase):
    def fixture(self, stack):
        tmp=Path(stack.enter_context(tempfile.TemporaryDirectory())).resolve();root=get_paths().project_root
        campaign=tmp/'campaign';job=campaign/'jobs'/'tiny';job.mkdir(parents=True)
        def write(path,value):
            path.write_text(json.dumps(value));return file_reference(path)
        from test_s05_multi_anchor_bound_driver import MultiAnchorDriverTests
        req,settlement,review=MultiAnchorDriverTests().inputs()
        sr=write(tmp/'settlement.json',settlement);review['references']['settlement']=sr
        ar=write(tmp/'review.json',review)
        refs={}
        for name in candidate_dual_adapter.REQUIRED:
            refs[name]=write(tmp/(name+'.json'),{'fixture':name})
        refs['candidate_review']=ar
        config=write(tmp/'config.json',{})
        paths=SimpleNamespace(project_root=root,campaign_root=campaign)
        req.update(output_root=str(campaign/'multi_anchor_bounds'/'tiny'),adapter_references=refs)
        req['parent_job'].update(job_id='tiny',directory=str(job),controller_reference=file_reference(root/'src/evcs_experiments/runtime/jobctl.py'))
        # Keep the real driver/root and actual loaded Python file hashes.
        with tempfile.NamedTemporaryFile(dir=root,prefix='.bound-test-',suffix='.json',delete=False) as stream:
            manifest_path=Path(stream.name)
        stack.callback(manifest_path.unlink,missing_ok=True)
        sourcepaths={Path(d.__file__).resolve()}
        for name,module in tuple(sys.modules.items()):
            if name=='evcs_experiments' or name.startswith('evcs_experiments.'):
                if getattr(module,'__file__',None):sourcepaths.add(Path(module.__file__).resolve())
        for name in ('candidate_dual_adapter.py','dual_cut_bound.py','candidate_dual_audit.py','multi_anchor_epigraph.py','multi_anchor_bound_audit.py','native_gurobi.py'):
            sourcepaths.add(root/'src/evcs_experiments'/name)
        manifest_path.write_text(json.dumps({'files':[dict(path=str(path.relative_to(root)),sha256=file_reference(path)['sha256'],size_bytes=path.stat().st_size) for path in sorted(sourcepaths)]}))
        req['references']=dict(settlement=sr,previous_result_review=ar,runtime_config=config,
            source_manifest=file_reference(manifest_path),script=file_reference(d.__file__),
            execution_policy=write(tmp/'policy.json',{'fixture_policy':True}),component_review=write(tmp/'component_review.json',{'fixture_review':True}))
        rr=write(tmp/'request.json',req)
        write(job/'manifest.json',dict(argv=[sys.executable,'-B',str(Path(d.__file__).resolve()),'--request',rr['path'],'--expected-request-sha256',rr['sha256']],
            cwd=str(root),outroot=str(job.parent),job_id='tiny',spec_id='S05',threads=1,timeout_seconds=1000,grace_seconds=5,
            controller_path=req['parent_job']['controller_reference']['path'],created_at_utc=datetime.now(timezone.utc).isoformat()))
        write(job/'status.json',dict(state='RUNNING',command_pid=os.getpid(),supervisor_sid=os.getsid(0)))
        ids=('tiny0','tiny1');P=ProbabilitySet(ids,(),np.empty((2,0)),[],[])
        first=dict(matrix=csr_matrix([[1.]]),row_lower=np.array([0.]),row_upper=np.array([1.]),c=np.array([0.]),
            lower_bounds=np.array([0.]),upper_bounds=np.array([1.]),integrality=np.array([1]),objective_constant_usd=0.)
        groups=[];floors={}
        for anchor in (0.,1.):
            cuts={}
            for i,sid in enumerate(ids):
                template=dict(A_ub=csr_matrix([[-1.],[-1.]]),A_eq=csr_matrix((0,1)),Bx_ub=csr_matrix([[4.],[-4.]]),
                    Bx_eq=csr_matrix((0,1)),Br_ub=csr_matrix((2,0)),Br_eq=csr_matrix((0,0)),b_ub=[-4.,0.],b_eq=[],c=[1.],
                    lower_bounds=[0.],upper_bounds=[10.],scenario_id=sid,template_sha256='template'+sid)
                dual=dict(inequality=[-1.,0.] if anchor==0 else [0.,-1.],equality=[],lower=[0.],upper=[0.])
                cuts[i]=build_dual_cut(template,dual,scenario_constant_usd=0.,anchor_first_stage=[anchor],anchor_q_usd=4.)
                floors[i]=certify_box_floor(template,scenario_constant_usd=0.)
            groups.append(dict(anchor_first_stage=[anchor],cuts=cuts,source_references=[refs['old_pack']]))
        ctx=SimpleNamespace(ids=ids,anchor=np.array([1.]),first_stage_problem=first,probability_set=P,floors=floors,
            binding=dict(plan_sha256=d.PLAN_SHA256,ambiguity_sha256=P.ambiguity_sha256))
        stack.enter_context(patch('evcs_experiments.config.get_paths',return_value=paths))
        stack.enter_context(patch('evcs_experiments.runtime.jobctl.observation',return_value={'supervisor_alive':True}))
        stack.enter_context(patch.object(candidate_dual_adapter,'CandidateDualContext',return_value=ctx))
        stack.enter_context(patch.object(candidate_dual_adapter,'load_old_anchor',return_value=groups[0]))
        stack.enter_context(patch.object(candidate_dual_adapter,'iter_candidate_cuts',side_effect=lambda c:iter([(cut,g,{'fixture':True}) for cut,g in groups[1]['cuts'].values()])))
        stack.enter_context(patch.object(candidate_dual_audit,'audit_derived_atom',return_value={'fixture_atom_audit':True}))
        stack.enter_context(patch.object(candidate_dual_audit,'audit_candidate_axis',return_value={'fixture_axis_audit':True}))
        stack.enter_context(patch.object(d,'SUPPORT_COUNT',2))
        env={key:'1' for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}
        env.update(EVCS_JOB_ID='tiny',EVCS_JOB_DIRECTORY=str(job),EVCS_CONFIG=config['path'])
        stack.enter_context(patch.dict(os.environ,env))
        return tmp,req,rr

    def test_failed_native_phase_preserves_one_attempt_and_raw_evidence(self):
        with ExitStack() as stack:
            tmp,req,rr=self.fixture(stack)
            def failed(*args,**kwargs):
                self.assertEqual(kwargs['options']['threads'],1)
                self.assertLessEqual(kwargs['options']['time_limit'],600)
                Path(kwargs['log_path']).write_text('native fixture failed after model admission')
                raise RuntimeError('fixture native exception')
            solve=stack.enter_context(patch.object(native_gurobi,'solve_milp',side_effect=failed))
            with self.assertRaisesRegex(RuntimeError,'fixture native'):d.execute(rr['path'],rr['sha256'])
            out=Path(req['output_root']);failure=json.loads((out/'failure.json').read_text())
            self.assertEqual(failure['phase'],'native_master')
            self.assertTrue((out/'model_audit.json').exists())
            self.assertTrue((out/'failure_artifact_manifest.json').exists())
            with self.assertRaises(FileExistsError):d.execute(rr['path'],rr['sha256'])
            self.assertEqual(solve.call_count,1)

    def fake_native(self, *args, **kwargs):
        import hashlib
        c,integer,lo,hi,A,rl,ru=args
        opts=kwargs['options'];grant=opts['time_limit']
        self.assertEqual(opts,dict(time_limit=grant,mip_rel_gap=.005,threads=1))
        self.assertTrue(0<grant<=600)
        Path(kwargs['log_path']).write_text('synthetic native TIME_LIMIT bound; no commercial solve')
        eq=np.isfinite(rl)&np.isfinite(ru)&(rl==ru)
        mapping=dict(eq=np.flatnonzero(eq),le=np.flatnonzero(np.isfinite(ru)&~eq),ge=np.flatnonzero(np.isfinite(rl)&~eq))
        rows={name:dict(count=len(v),source_rows_sha256=hashlib.sha256(v.astype('<i8').tobytes()).hexdigest()) for name,v in mapping.items()}
        rows.update(original_rows=A.shape[0],translated_rows=sum(map(len,mapping.values())))
        native=dict(backend='gurobipy',gurobi_version=[12,0,3],original_matrix_shape=list(A.shape),original_matrix_nnz=A.nnz,
            effective_options=dict(Threads=1,MIPGap=.005,Seed=0,FeasibilityTol=1e-8,OptimalityTol=1e-8,IntFeasTol=1e-8),
            requested_time_limit_seconds=grant,raw_model_status=9,status='TIME_LIMIT',certificate_failures=[],is_mip=True,
            bound_source='MIP_ObjBound',solver_bound_consistent=True,raw_solver_bound_without_constant_usd=4.,
            solver_bound_without_constant_usd=4.,row_map_summary=rows,model_rows=rows['translated_rows'],model_columns=A.shape[1],
            valid_incumbent=False,solution_count=0,incumbent_without_constant_usd=None)
        return SimpleNamespace(native_record=native,x=None,status=1,message='TIME_LIMIT',mip_dual_bound=4.,fun=None)

    def test_success_real_assembly_and_independent_native_audit_pending_settlement(self):
        with ExitStack() as stack:
            tmp,req,rr=self.fixture(stack)
            solve=stack.enter_context(patch.object(native_gurobi,'solve_milp',side_effect=self.fake_native))
            d.execute(rr['path'],rr['sha256'])
            out=Path(req['output_root']);result=json.loads((out/'result.json').read_text())
            self.assertEqual(result['status'],'AUDITED_BOUND_PENDING_TERMINAL_PARENT_SETTLEMENT')
            self.assertEqual(result['lower_bound_usd'],d.PREVIOUS_LB)
            self.assertEqual(result['upper_bound_usd'],d.FIXED_UB)
            self.assertTrue(result['cumulative_seconds_provisional'])
            self.assertFalse(result['scientific_acceptance'])
            self.assertEqual(result['native_bound_audit_reference'],file_reference(out/'native_bound_audit.json'))
            self.assertIn('native_master',result['phase_seconds'])
            manifest=json.loads((out/'artifact_manifest.json').read_text())
            self.assertEqual(manifest['result_reference'],file_reference(out/'result.json'))
            self.assertEqual(solve.call_count,1)

    def test_invalid_native_bound_retains_raw_failure_and_no_result(self):
        with ExitStack() as stack:
            tmp,req,rr=self.fixture(stack)
            def invalid(*args,**kwargs):
                answer=self.fake_native(*args,**kwargs)
                answer.native_record['solver_bound_without_constant_usd']=999.
                return answer
            stack.enter_context(patch.object(native_gurobi,'solve_milp',side_effect=invalid))
            with self.assertRaisesRegex(ValueError,'bound differs'):d.execute(rr['path'],rr['sha256'])
            out=Path(req['output_root'])
            self.assertTrue((out/'native_solver.json').exists())
            self.assertTrue((out/'native_return.json').exists())
            self.assertFalse((out/'result.json').exists())
            self.assertEqual(json.loads((out/'failure.json').read_text())['phase'],'independent_native_bound_audit')

    def test_native_overrun_cannot_reset_audit_deadline(self):
        import time
        with ExitStack() as stack:
            tmp,req,rr=self.fixture(stack);clock=[time.monotonic()]
            stack.enter_context(patch('time.monotonic',side_effect=lambda:clock[0]))
            def overrun(*args,**kwargs):
                answer=self.fake_native(*args,**kwargs);clock[0]+=991.;return answer
            stack.enter_context(patch.object(native_gurobi,'solve_milp',side_effect=overrun))
            with self.assertRaises((ValueError,TimeoutError)):d.execute(rr['path'],rr['sha256'])
            out=Path(req['output_root'])
            self.assertTrue((out/'native_return.json').exists())
            self.assertFalse((out/'result.json').exists())
            self.assertEqual(json.loads((out/'failure.json').read_text())['phase'],'independent_native_bound_audit')

    def test_wrong_parent_options_never_enters_numerical_phase(self):
        with ExitStack() as stack:
            tmp,req,rr=self.fixture(stack);path=tmp/'campaign/jobs/tiny/manifest.json'
            value=json.loads(path.read_text());value['threads']=2;path.write_text(json.dumps(value))
            solve=stack.enter_context(patch.object(native_gurobi,'solve_milp'))
            with self.assertRaisesRegex(ValueError,'resource'):d.execute(rr['path'],rr['sha256'])
            solve.assert_not_called()

if __name__=='__main__':unittest.main()
