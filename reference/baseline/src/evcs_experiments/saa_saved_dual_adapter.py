"""Read original SAA fixed-Q records and derive global affine minorants.

No CCG events are synthesized. Original physical acceptance is inherited from
an explicitly bound candidate audit; consumed numerical arrays are rechecked.
Production admission must pin that audit to its independent acceptance and the
current first-stage model. This module does not settle budgets or accept plans.
"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
from scipy.sparse import load_npz
from .campaign_batch import verify_reference
from .dual_cut_bound import build_dual_cut
from .random_streams import canonical_hash
from .recourse_store import MATRICES, ARRAYS, _semantic_sha
from .saa_fixed_plan_audit import HELPER_SHA256


def need(v,m):
    if not bool(v):raise ValueError(m)


def read(ref):
    verify_reference(ref)
    return json.loads(Path(ref['path']).read_text())


def inventory(row,root,expected):
    refs=row['artifacts'];names=[Path(r['path']).name for r in refs]
    need(len(names)==len(set(names)) and set(names)==set(expected),'Original artifact inventory differs')
    need(all(Path(r['path']).is_absolute() and Path(r['path']).parent==root for r in refs),'Foreign original artifact path')
    return dict(zip(names,refs))


def numeric(ref):
    verify_reference(ref)
    with np.load(ref['path'],allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


class SAASavedDualContext:
    def __init__(self,*,candidate_audit_reference,request_reference,checkpoint_reference,
                 template_envelope_reference,expected_first_stage_sha256,deadline):
        self.deadline=deadline;self.tick()
        self.references=dict(candidate_audit_reference=candidate_audit_reference,request_reference=request_reference,
            checkpoint_reference=checkpoint_reference,template_envelope_reference=template_envelope_reference)
        a=read(candidate_audit_reference);q=read(request_reference);cp=read(checkpoint_reference);env=read(template_envelope_reference)
        need(a['status']=='PASS_SAA_CANDIDATE_NUMERICAL_AUDIT' and a['passed'] is True and
             a['risk_certificate_kind']=='fixed_empirical' and a['full_support_recourse_verified'] is True,'Original accepted SAA numerical audit required')
        f=a['fixed_q_audit']
        need(f['status']=='PASS_SAA_FIXED_PLAN_FULL_Q_AND_EMPIRICAL_LAW_AUDIT' and f['passed'] is True and
             f['request_reference']==request_reference and f['checkpoint_reference']==checkpoint_reference and
             f['template_envelope_reference']==template_envelope_reference,'SAA audit original references differ')
        need(q['kind']=='SAA_SERIAL_FIXED_PLAN_Q_V1' and q['domain'] in ('formal','development'),'Original SAA request/domain differs')
        self.ids=tuple(q['scenario_ids']);n=len(self.ids)
        need(n>0 and len(set(self.ids))==n and (q['domain']!='formal' or n==160),'Wrong original support size')
        need(all(len(q[k])==n for k in ('scenario_sha256s','template_sha256s','constants_usd')),'Incomplete original axes')
        need(np.isfinite(q['constants_usd']).all() and f['support_ids']==list(self.ids) and
             np.array_equal(f['empirical_probabilities'],np.full(n,1./n)),'Original empirical law/support differs')
        # The audit recomputes c@y independently; summation roundoff is not an
        # identity change. Keep raw checkpoint/atom equality exact below. This
        # comparison is much tighter than the existing physical/KKT tolerance.
        raw_q=np.asarray(cp['q_usd'],dtype=float);audited_q=np.asarray(f['q_usd'],dtype=float)
        same_q=(raw_q.shape==audited_q.shape==(n,) and np.isfinite(raw_q).all() and
                np.isfinite(audited_q).all() and
                np.all(np.abs(raw_q-audited_q)<=32*np.finfo(float).eps*np.maximum(1.,np.abs(audited_q))))
        need(cp['status']=='COMPLETE_FIXED_Q_PENDING_INDEPENDENT_AUDIT' and cp['request_reference']==request_reference and
             cp['completed_indices']==list(range(n)) and same_q,'Original full Q checkpoint differs')
        need(env['kind']=='SAA_ORIGINAL_TEMPLATE_ENVELOPE_V1' and env['request_reference']==request_reference and
             len(env['template_references'])==n and env['first_stage_sha256']==expected_first_stage_sha256,'Original template/first-stage binding differs')
        need(canonical_hash(q['plan'])==q['plan_sha256']==a['plan_sha256']==f['plan_sha256'],'Original anchor plan identity differs')
        self.plan=q['plan'];self.anchor=np.r_[self.plan['x'],np.asarray(self.plan['r']).ravel()]
        need(np.isfinite(self.anchor).all(),'Nonfinite original anchor')
        need(self.plan['task_sha256']==q['task_sha256'],'Original task/plan differs')
        for ref in q['sources']:verify_reference(ref)
        for ref in env['source_references']:verify_reference(ref)
        helper=f['helper_reference'];verify_reference(helper)
        need(helper['sha256']==HELPER_SHA256,'Original independent KKT source changed')
        spec=importlib.util.spec_from_file_location('_saa_saved_dual_original_kkt',helper['path'])
        self.helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.helper)
        self.attempts={}
        for ref in cp['attempts']:
            self.tick();row=read(ref)
            need(row['request_reference']==request_reference and type(row['completed']) is bool,'Foreign original Q attempt')
            if row['completed']:
                i=row['scenario_index'];need(type(i) is int and 0<=i<n and i not in self.attempts,'Duplicate completed SAA atom')
                self.attempts[i]=(ref,row)
        need(set(self.attempts)==set(range(n)),'Missing original SAA atom')
        self.request=q;self.checkpoint=cp;self.envelope=env;self.first_sha=expected_first_stage_sha256
        self.binding_sha256=canonical_hash(dict(**self.references,first_stage_sha256=expected_first_stage_sha256))
        self.tick()

    def tick(self):
        need(type(self.deadline) in (int,float) and math.isfinite(self.deadline),'Invalid common deadline')
        if time.monotonic()>=self.deadline:raise TimeoutError('SAA saved-dual preparation deadline')

    def derive(self,index):
        self.tick();need(type(index) is int and 0<=index<len(self.ids),'Wrong original scenario index')
        q=self.request;i=index;ref,row=self.attempts[i];need(read(ref)==row,'Original atom bytes changed after admission');trref=self.envelope['template_references'][i];tr=read(trref)
        need(row['scenario_id']==tr['scenario_id']==self.ids[i] and tr['scenario_index']==i and
             tr['kind']=='recourse_template' and row['scenario_sha256']==q['scenario_sha256s'][i] and
             row['template_sha256']==tr['template_sha256']==q['template_sha256s'][i] and
             row['plan_sha256']==q['plan_sha256'] and row['q_usd']==self.checkpoint['q_usd'][i], 'Original SAA atom/template/anchor identity differs')
        ts=inventory(tr,Path(trref['path']).parent,{m+'.npz' for m in MATRICES}|{'arrays.npz','identity.json'})
        raw=inventory(row,Path(ref['path']).parent,{'solver.json','plan.json','fixed_rhs.npz','primal.npy','physical_check.json','dual_kkt.npz'})
        # Full physical/map payload was already independently accepted by the
        # pinned audit. Only its numerical header is used and recomputed below;
        # do not rehash gigabytes of unused variable maps on every derivative.
        meta=self.helper.metadata_header(ts['identity.json']['path'])
        arrays=numeric(ts['arrays.npz']);need(set(arrays)==set(ARRAYS),'Original array schema differs')
        matrices={}
        for name in MATRICES:
            verify_reference(ts[name+'.npz']);matrices[name]=load_npz(ts[name+'.npz']['path']).tocsr()
        need(meta['template_sha256']==q['template_sha256s'][i] and
             _semantic_sha(meta,matrices,arrays,self.first_sha)==q['template_sha256s'][i],'Original semantic matrix identity differs')
        need(meta['task_sha256']==q['task_sha256'] and meta['config_hash']==self.plan['config_hash'],'Original numerical header task/config differs')
        need(np.array_equal(arrays['c'],arrays['component_objectives'].sum(axis=0)),'Original objective decomposition differs')
        need(read(raw['plan.json'])==self.plan,'Saved original plan differs')
        solver=read(raw['solver.json']);dual=numeric(raw['dual_kkt.npz']);fixed=numeric(raw['fixed_rhs.npz'])
        verify_reference(raw['primal.npy']);z=np.load(raw['primal.npy']['path'],allow_pickle=False)
        need(set(dual)=={'inequality','equality','lower','upper'} and set(fixed)=={'b_ub','b_eq'},'Original SAA dual/RHS schema differs')
        need(row['native_status']==solver['status']=='OPTIMAL' and solver['certificate']['template_sha256']==meta['template_sha256'],'Original LP native/template differs')
        plan=SimpleNamespace(x=np.asarray(self.plan['x']),r=np.asarray(self.plan['r']))
        lo,hi=arrays['lower_bounds'],arrays['upper_bounds'];fl,fu=np.isfinite(lo),np.isfinite(hi)
        bu=arrays['b_ub']+matrices['Bx_ub']@plan.x+matrices['Br_ub']@plan.r.ravel()
        be=arrays['b_eq']+matrices['Bx_eq']@plan.x+matrices['Br_eq']@plan.r.ravel()
        self.helper.same(fixed['b_ub'],bu,'Original SAA RHS differs');self.helper.same(fixed['b_eq'],be,'Original SAA RHS differs')
        aux=dict(dual,stationarity=arrays['c']-matrices['A_ub'].T@dual['inequality']-matrices['A_eq'].T@dual['equality']-dual['lower']-dual['upper'],
            inequality_slack=bu-matrices['A_ub']@z,equality_residual=matrices['A_eq']@z-be,
            inequality_complementarity=dual['inequality']*(bu-matrices['A_ub']@z),
            lower_complementarity=dual['lower'][fl]*(z-lo)[fl],upper_complementarity=dual['upper'][fu]*(hi-z)[fu])
        # Independent original-array KKT audit, separate from the cut builder.
        kkt=self.helper.audit_kkt(matrices,arrays,plan,z,aux,solver,fixed)
        constant=q['constants_usd'][i];expected=kkt['primal_objective_usd']+constant
        self.helper.close(expected,row['q_usd'],'Original SAA constant/Q differs',1e-4+1e-7*abs(expected))
        cut,g=build_dual_cut(dict(matrices,**arrays,scenario_id=self.ids[i],template_sha256=meta['template_sha256']),dual,
            scenario_constant_usd=constant,anchor_first_stage=self.anchor,anchor_q_usd=row['q_usd'])
        need(cut['passed'],'Original SAA dual cannot certify a global cut')
        self.tick()
        evidence=dict(kind='SAA_SAVED_ORIGINAL_DUAL_CUT_V1',passed=True,index=i,scenario_id=self.ids[i],
            context_binding_sha256=self.binding_sha256,atom_reference=ref,template_reference=trref,
            cut_sha256=cut['cut_sha256'],kkt=kkt,physical_acceptance_inherited_from=self.references['candidate_audit_reference'],
            unused_descriptive_variable_map_rehashed=False,original_numeric_arrays_checked=True,
            solver_calls=0,full_candidate_acceptance=False,production_admission=False)
        return cut,g,evidence
