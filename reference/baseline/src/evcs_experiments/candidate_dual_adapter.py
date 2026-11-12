"""Derive additional affine cuts from an already audited candidate, without LPs.

This is an input adapter, not a planning runner or acceptance gate. An external
parent must admit and charge its absolute deadline before iterating full support.
The original cut pack is retained; no checkpoint, source or evidence is written.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.sparse import load_npz

from . import dual_cut_bound
from .probability import ProbabilitySet
from .random_streams import canonical_hash

MATRICES = ('A_ub', 'A_eq', 'Bx_ub', 'Bx_eq', 'Br_ub', 'Br_eq')
REQUIRED = ('old_pack', 'old_pack_review', 'original_checkpoint',
            'original_raw_request', 'original_first_record', 'candidate_audit', 'candidate_review',
            'candidate_checkpoint', 'candidate_plan', 'candidate_snapshot')


def need(value, message):
    if not bool(value):
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def metadata_header(path, limit=4*1024*1024):
    """Read numerical identity fields only after the entire file was hashed."""
    with Path(path).open() as stream:
        text = ''
        while '\n  "variable_map":' not in text:
            part = stream.read(8192)
            need(part and len(text)+len(part) <= limit, 'Unknown/oversized template metadata')
            text += part
    return json.loads(text.split('\n  "variable_map":', 1)[0].rstrip().removesuffix(',')+'}')


def template_digest(meta, matrices, values, first_sha):
    identity = dict(task_sha256=meta['task_sha256'], config_hash=meta['config_hash'],
        scenario_sha256=meta['scenario_sha256'], first_stage_sha256=first_sha,
        shapes=meta['variable_shapes'], task_ids=meta['task_ids'], bus_ids=meta['bus_ids'],
        line_ids=meta['line_ids'], station_ids=meta['station_ids'], periods=meta['periods'],
        references=meta['reference_bus_ids_by_period'], cmax_usd=meta['cmax_usd'])
    h = hashlib.sha256(json.dumps(identity, sort_keys=True).encode())
    assembled = {**{k.lower().replace('bx', 'x').replace('br', 'r'): v for k,v in matrices.items()},
                 'b_ub': values['b_ub'], 'b_eq': values['b_eq']}
    for key in sorted(assembled):
        value = assembled[key]
        components = (value.data,value.indices,value.indptr) if hasattr(value,'indptr') else (value,)
        h.update(key.encode())
        for a in components:
            a = np.asarray(a, dtype='<i8' if a.dtype.kind in 'iu' else '<f8')
            h.update(str(a.shape).encode()); h.update(a.tobytes())
    for key in ('lower_bounds', 'upper_bounds', 'component_objectives'):
        h.update(np.asarray(values[key], dtype='<f8').tobytes())
    return h.hexdigest()


def first_digest(meta, matrix, values):
    body = {k: meta[k] for k in ('task_sha256','config_hash','task_ids','periods',
                               'variable_map','constraint_map','units')}
    h = hashlib.sha256(json.dumps(body,sort_keys=True).encode())
    for x in (matrix.data,matrix.indices,matrix.indptr,values['row_lower'],values['row_upper'],
              values['lower_bounds'],values['upper_bounds'],values['integrality'],values['c']):
        a = np.asarray(x, dtype='<i8' if np.asarray(x).dtype.kind in 'iu' else '<f8')
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def scenario_hash_axis(identity, raw):
    """Resolve eager or original input-bound lazy scenario identity without assembly."""
    if 'lazy_input_binding' in identity:
        lazy = identity['lazy_input_binding']
        need(lazy['kind']=='input_bound_lazy_v1' and lazy['binding']==raw['input_binding']
             and lazy['first_stage_sha256']==identity['first_stage_sha256']
             and lazy['scenario_ids']==identity['scenario_ids'], 'Lazy scenario source binding differs')
        hashes = lazy['scenario_sha256s']
        if 'scenario_sha256s' in identity:
            need(identity['scenario_sha256s']==hashes, 'Eager/lazy scenario axes disagree')
    else:
        hashes = identity['scenario_sha256s']
    need(len(hashes)==len(identity['scenario_ids']) and all(isinstance(x,str) and len(x)==64 for x in hashes),
         'Incomplete original scenario hash axis')
    return tuple(hashes)


class CandidateDualContext:
    """Source-bound metadata admission; atom arrays are loaded only on demand.

    ``source_references`` must pin this module and the unchanged dual-cut kernel.
    ``deadline`` is an absolute monotonic deadline owned/charged by the caller.
    Admission may hash large original checkpoint/first-stage files, but does not
    open every candidate LP or template. ``binding`` is a serializable receipt.
    """
    def __init__(self, refs, *, source_references, deadline):
        self.deadline = deadline
        self.used = {}
        self.check_deadline()
        need(set(refs) == set(REQUIRED), 'Wrong candidate adapter reference schema')
        self.refs = json.loads(json.dumps(refs))
        need(isinstance(source_references,list) and source_references, 'Missing adapter sources')
        self.sources = json.loads(json.dumps(source_references))
        for ref in self.sources:
            self.verify(ref)
        for path in (Path(__file__).resolve(), Path(dual_cut_bound.__file__).resolve()):
            need(str(path) in self.used, 'Unbound executable adapter/kernel source')
        for ref in refs.values():
            self.verify(ref)
        pack = self.read(refs['old_pack']); review = self.read(refs['old_pack_review'])
        need(review['status']=='PASS_FULL_SUPPORT_DUAL_CUT_FLOOR_PACK_REVIEW' and review['passed'] is True
             and refs['old_pack'] in review['references'].values(), 'Old pack not independently accepted')
        for ref in review['references'].values():
            self.verify(ref)
        old = self.read(refs['original_checkpoint']); state = old['state']; identity = state['identity']
        need(old['sha256']==canonical_hash(state) and state['problem_sha256']==canonical_hash(identity),
             'Original checkpoint semantic identity differs')
        self.identity = identity
        self.ids = tuple(identity['scenario_ids']); n = len(self.ids)
        need(n == 160 and len(set(self.ids)) == n, 'Original full160 support required')
        need(pack['status']=='DERIVED_FULL_SUPPORT_DUAL_CUT_FLOOR_PACK'
             and pack['problem_sha256']==state['problem_sha256']
             and pack['support_ids']==list(self.ids) and pack['constants_usd']==identity['constants_usd']
             and pack['first_stage_sha256']==identity['first_stage_sha256'], 'Old pack scientific identity differs')
        need(pack['source_references']['checkpoint']==refs['original_checkpoint']
             and pack['source_references']['original_first_record']==refs['original_first_record']
             and pack['source_references']['raw_request']==refs['original_raw_request'],
             'Old pack original source binding differs')
        self.verify(pack['gradients']); self.verify(pack['used_original_references'])
        need(pack['anchor_plan']==state['pending']['plan'], 'Old anchor differs from original checkpoint')
        self.pack = pack
        self.floors = {i: row for i,row in enumerate(pack['floors'])}
        need(len(self.floors)==n and len(pack['cuts'])==n, 'Old full-axis cuts/floors incomplete')
        pi = pack['effective_ambiguity']
        self.probability_set = ProbabilitySet(self.ids,tuple(pi['feature_ids']),np.asarray(pi['features']),
                                               np.asarray(pi['lower']),np.asarray(pi['upper']))
        need(self.probability_set.ambiguity_sha256==identity['effective_ambiguity_sha256'], 'Changed original P')
        weights = np.asarray(pack['probabilities'],dtype=float)
        need(weights.shape==(n,) and np.isfinite(weights).all() and np.all(weights>=0), 'Invalid original P witness')
        moments = self.probability_set.features.T@weights
        need(max(abs(weights.sum()-1),np.max(self.probability_set.lower-moments,initial=0),
                 np.max(moments-self.probability_set.upper,initial=0))<=1e-8, 'Original P witness infeasible')
        raw = self.read(refs['original_raw_request'])
        need(raw['domain']=='formal' and raw['formal_OOS'] is False and raw['problem_identity']==identity,
             'Original raw problem identity differs')
        accepted = self.read(refs['candidate_review'])
        need(accepted['status']=='PASS_INDEPENDENT_FULL_SUPPORT_CANDIDATE_NUMERICAL_RESULT_REVIEW'
             and accepted['passed'] is True and accepted['complete_lp_count']==n
             and accepted['references']['recourse_audit']==refs['candidate_audit']
             and accepted['original_problem_sha256']==state['problem_sha256']
             and accepted['formal_OOS'] is False and accepted['whole_S05_PASS'] is False,
             'Candidate independent review binding/scope differs')
        self.scenario_hashes = scenario_hash_axis(identity,raw)
        audit = self.read(refs['candidate_audit'])
        need(audit['status']=='PASS_INDEPENDENT_CANDIDATE_160_RECOURSE_AUDIT' and audit['passed'] is True
             and audit['complete_lp_count']==n and audit['original_problem_sha256']==state['problem_sha256']
             and audit['original_first_stage_sha256']==identity['first_stage_sha256']
             and audit['support_ids']==list(self.ids) and audit['first_stage_physical_passed'] is True
             and audit['formal_OOS'] is False and audit['scientific_acceptance'] is False,
             'Candidate numerical audit identity/scope differs')
        for key in ('original_checkpoint','original_raw_request'):
            need(audit['references'][key]==refs[key], 'Candidate audit original source mismatch')
        for ref in audit['references'].values():
            self.verify(ref)
        need(audit['checkpoint_reference']==refs['candidate_checkpoint']
             and audit['raw_snapshot_reference']==refs['candidate_snapshot'], 'Candidate evidence unbound by audit')
        cp = self.read(refs['candidate_checkpoint']); snapshot = self.read(refs['candidate_snapshot'])
        self.plan = self.read(refs['candidate_plan']); plan_sha = canonical_hash(self.plan)
        need(cp['plan_reference']==refs['candidate_plan'] and cp['plan_sha256']==audit['plan_sha256']==accepted['plan_sha256']==plan_sha,
             'Candidate plan changed')
        need(cp['status']=='FULL_Q_PENDING_INDEPENDENT_AUDIT_AND_PROBABILITY'
             and cp['completed_count']==cp['next_index']==n and cp['support_ids']==list(self.ids)
             and cp['formal_OOS'] is False and cp['q_usd']==audit['q_usd'], 'Incomplete/wrong candidate Q')
        self.q = np.asarray(cp['q_usd'],dtype=float)
        self.constants = np.asarray(identity['constants_usd'],dtype=float)
        need(self.q.shape==self.constants.shape==(n,) and np.isfinite(self.q).all()
             and np.isfinite(self.constants).all(), 'Nonfinite Q/constants')
        self.events = snapshot['events']
        need(cp['recourse_event_references']==self.events and len(self.events)==n
             and len({r['path'] for r in self.events})==n and snapshot['templates']==[]
             and snapshot['formal_OOS'] is False, 'Candidate raw coverage differs')
        need(self.read(cp['raw_history_reference'])==self.events, 'Candidate raw history changed')
        need(self.read(snapshot['request'])==raw, 'Candidate raw kernel/input identity differs')
        binding = self.read(cp['candidate_binding_reference'])
        need(binding['plan_sha256']==plan_sha and binding['problem_sha256']==state['problem_sha256'],
             'Candidate native plan binding differs')
        vector = arrays(self.verify(binding['vector_reference']))[binding['vector_key']]
        self.verify(binding['original_binding_reference'])
        self.anchor = np.r_[np.asarray(self.plan['x'],dtype=float),np.asarray(self.plan['r'],dtype=float).ravel()]
        need(np.isfinite(self.anchor).all() and binding['first_stage_slice']==[0,len(self.anchor)]
             and np.array_equal(vector[:len(self.anchor)],self.anchor), 'Candidate native vector differs')
        first_record = self.record(refs['original_first_record'])
        first_artifacts = self.artifacts(first_record, refs['original_first_record'])
        fm = self.read(first_artifacts['identity.json']); fv = arrays(self.verify(first_artifacts['arrays.npz']))
        A = load_npz(self.verify(first_artifacts['matrix.npz'])).tocsr()
        need(first_digest(fm,A,fv)==fm['matrix_sha256']==identity['first_stage_sha256'], 'Original first-stage coefficients differ')
        need(fm['task_sha256']==self.plan['task_sha256'] and fm['config_hash']==self.plan['config_hash']
             and len(fv['c'])==len(self.anchor), 'Candidate first-stage axes differ')
        ax = A@self.anchor
        residual = max(np.max(fv['row_lower']-ax,initial=0),np.max(ax-fv['row_upper'],initial=0),
                       np.max(fv['lower_bounds']-self.anchor,initial=0),
                       np.max(self.anchor-fv['upper_bounds'],initial=0))
        integers = np.asarray(fv['integrality'])!=0
        need(residual<=1e-7 and np.max(abs(self.anchor[integers]-np.rint(self.anchor[integers])),initial=0)<=1e-8,
             'Candidate violates original first-stage arrays')
        fpre = float(fv['c']@self.anchor)
        need(math.isfinite(fpre) and abs(fpre-audit['Fpre_usd'])<=1e-4+1e-7*max(1.,abs(fpre)),
             'Candidate original first-stage objective differs')
        self.first_metadata = fm
        self.first_stage_problem = dict(fv,matrix=A,objective_constant_usd=0.,
                                        source_model_sha256=fm['matrix_sha256'],formal_problem_sha256=state['problem_sha256'])
        self.template_hashes = tuple(r['template_sha256'] for r in state['pending']['recourse_certificates'])
        need(len(self.template_hashes)==n, 'Original template axis incomplete')
        inventory = cp['template_reuse_inventory']
        need(inventory['references']['old_checkpoint']==refs['original_checkpoint'], 'Template inventory origin changed')
        self.inventory = inventory['templates']
        need(len(self.inventory)==n, 'Original template inventory incomplete')
        self.binding = dict(kind='SOURCE_BOUND_CANDIDATE_DUAL_CONTEXT_V1', references=self.refs,
            source_references=self.sources, support_ids=list(self.ids), plan_sha256=plan_sha,
            problem_sha256=state['problem_sha256'], first_stage_sha256=fm['matrix_sha256'],
            ambiguity_sha256=self.probability_set.ambiguity_sha256, solver_calls=0,
            scientific_acceptance=False, formal_OOS=False, normal_ccg_checkpoint_compatible=False)
        self.binding['binding_sha256'] = canonical_hash(self.binding)
        self.check_deadline()

    def check_deadline(self):
        if type(self.deadline) not in (int,float) or not math.isfinite(self.deadline) or time.monotonic()>=self.deadline:
            raise TimeoutError('Candidate dual adapter absolute deadline expired')

    def verify(self, ref):
        self.check_deadline()
        need(isinstance(ref,dict) and set(ref)=={'path','sha256','size_bytes'}, 'Invalid strong reference')
        p = Path(ref['path'])
        need(p.is_absolute() and p.is_file() and p.stat().st_size==ref['size_bytes'], 'Referenced file missing/size changed')
        h = hashlib.sha256()
        with p.open('rb') as stream:
            while block := stream.read(1024*1024):
                self.check_deadline(); h.update(block)
        need(h.hexdigest()==ref['sha256'], 'Referenced file hash changed')
        self.used[ref['path']] = dict(ref)
        self.check_deadline()
        return p

    def read(self, ref):
        answer = read_json(self.verify(ref)); self.check_deadline(); return answer

    def artifacts(self, record, ref):
        root = Path(ref['path']).parent
        rows = record['artifacts']
        names = [Path(r['path']).name for r in rows]
        need(len(set(names))==len(names) and all(Path(r['path']).parent==root for r in rows),
             'Duplicate/foreign artifact inventory')
        return dict(zip(names,rows))

    def record(self, ref):
        record = self.read(ref)
        for artifact in self.artifacts(record,ref).values():
            self.verify(artifact)
        return record


def derive_atom_arrays(*, scenario_id, template_sha256, first_stage_sha256, meta,
                       matrices, values, dual, fixed, solver, plan, constant, q):
    """Validate numerical template/RHS/dual semantics and derive one cut; no I/O."""
    need(template_digest(meta,matrices,values,first_stage_sha256)==template_sha256==meta['template_sha256'],
         'Original template semantic digest differs')
    need(meta['task_sha256']==plan['task_sha256'] and meta['config_hash']==plan['config_hash'],
         'Template task/config identity differs')
    need(np.array_equal(values['c'],np.asarray(values['component_objectives']).sum(axis=0)),
         'Original component objective aggregation changed')
    need(solver['status']=='OPTIMAL' and solver['optimal'] is True and solver['solver_status']==0
         and solver['certificate']['template_sha256']==template_sha256, 'Uncertified original dual LP')
    for key in ('inequality','equality','lower','upper'):
        need(np.array_equal(dual[key],np.asarray(solver['certificate']['dual_'+key+'_marginals'])),
             'Saved dual differs from bound solver certificate')
    x = np.asarray(plan['x'],dtype=float); r = np.asarray(plan['r'],dtype=float).ravel()
    for suffix in ('ub','eq'):
        rhs = values['b_'+suffix]+matrices['Bx_'+suffix]@x+matrices['Br_'+suffix]@r
        need(np.array_equal(rhs,fixed['b_'+suffix]), 'Candidate fixed RHS differs from original base/coupling')
    need(math.isfinite(constant) and math.isfinite(q) and q==solver['objective_usd']+constant,
         'Candidate Q or scenario constant differs')
    cut, gradient = dual_cut_bound.build_dual_cut(dict(values,**matrices,scenario_id=scenario_id,
        template_sha256=template_sha256),dual,scenario_constant_usd=float(constant),
        anchor_first_stage=np.r_[x,r],anchor_q_usd=float(q))
    need(cut['passed'] is True, 'Candidate affine minorant certificate failed')
    return cut, gradient


def derive_candidate_atom(context, index):
    """Return (cut, gradient, provenance) for exactly one source-bound atom."""
    c = context; c.check_deadline()
    need(type(index) is int and 0<=index<len(c.ids), 'Invalid candidate scenario index')
    er = c.events[index]; event = c.record(er); ea = c.artifacts(event,er)
    need(event['event']=='recourse' and event['domain']=='formal' and event['iteration']==1
         and event['status']=='RAW_EVIDENCE_RETAINED_PENDING_AUDIT' and event['optimal_certificate_valid'] is True
         and event['scenario_index']==index and event['scenario_id']==c.ids[index]
         and event['plan_sha256']==c.binding['plan_sha256'] and event['q_usd']==c.q[index],
         'Candidate event source/scenario/plan differs')
    need(c.read(ea['plan.json'])==c.plan, 'Raw event plan differs')
    tr = c.read(ea['template_reference.json']); inventory = c.inventory[index]
    need(inventory['index']==index and inventory['scenario_id']==c.ids[index]
         and inventory['template_reference']==tr and inventory['template_sha256']==c.template_hashes[index],
         'Original reused template pointer differs')
    oldevent = c.read(inventory['event_reference']); olda = c.artifacts(oldevent,inventory['event_reference'])
    need(olda['template_reference.json']==inventory['template_pointer_reference']
         and c.read(inventory['template_pointer_reference'])==tr, 'Template differs from original raw event')
    template = c.record(tr); ta = c.artifacts(template,tr)
    need(template['scenario_index']==index and template['scenario_id']==c.ids[index]
         and template['template_sha256']==c.template_hashes[index], 'Original template event axis differs')
    meta = metadata_header(c.verify(ta['identity.json']))
    values = arrays(c.verify(ta['arrays.npz']))
    matrices = {key:load_npz(c.verify(ta[key+'.npz'])).tocsr() for key in MATRICES}
    need(meta['scenario_sha256']==c.scenario_hashes[index]
         and meta['task_ids']==c.first_metadata['task_ids'] and meta['periods']==c.first_metadata['periods'],
         'Original physical task/period axes differ')
    dual = arrays(c.verify(ea['dual_kkt.npz'])); fixed = arrays(c.verify(ea['fixed_rhs.npz']))
    solver = c.read(ea['solver.json'])
    cut, gradient = derive_atom_arrays(scenario_id=c.ids[index],template_sha256=c.template_hashes[index],
        first_stage_sha256=c.identity['first_stage_sha256'],meta=meta,matrices=matrices,values=values,
        dual=dual,fixed=fixed,solver=solver,plan=c.plan,constant=float(c.constants[index]),q=float(c.q[index]))
    provenance = dict(kind='SOURCE_BOUND_CANDIDATE_DUAL_ATOM_V1',index=index,scenario_id=c.ids[index],
        expected_template_sha256=c.template_hashes[index],original_first_stage_sha256=c.identity['first_stage_sha256'],
        scenario_constant_usd=float(c.constants[index]),anchor_plan_reference=c.refs['candidate_plan'],
        event_reference=er,template_reference=tr,source_references=c.sources,
        context_binding=c.binding,context_binding_sha256=c.binding['binding_sha256'],
        input_references=[er,tr,*ea.values(),*ta.values()], solver_calls=0,scientific_acceptance=False,formal_OOS=False)
    c.check_deadline()
    return cut, gradient, provenance


def iter_candidate_cuts(context):
    """Ordered bounded iterator; exhaustion alone does not grant acceptance."""
    for i in range(len(context.ids)):
        yield derive_candidate_atom(context,i)
    context.check_deadline()


def load_old_anchor(context):
    """Load the unchanged accepted old gradients, retaining every original cut."""
    c = context
    matrix = load_npz(c.verify(c.pack['gradients'])).tocsr()
    plan = c.pack['anchor_plan']
    anchor = np.r_[np.asarray(plan['x'],dtype=float),np.asarray(plan['r'],dtype=float).ravel()]
    need(canonical_hash(plan)==c.pack['anchor_plan_sha256']
         and hashlib.sha256(anchor.astype('<f8').tobytes()).hexdigest()==c.pack['anchor_first_stage_vector_sha256']
         and matrix.shape==(len(c.ids),len(c.anchor)), 'Old pack anchor/gradient axes changed')
    cuts = {i:(cut,matrix.getrow(i).toarray().ravel()) for i,cut in enumerate(c.pack['cuts'])}
    for i,(cut,gradient) in cuts.items():
        c.check_deadline()
        body = dict(cut); digest = body.pop('cut_sha256')
        need(canonical_hash(body)==digest and cut['scenario_id']==c.ids[i]
             and cut['anchor_first_stage_sha256']==c.pack['anchor_first_stage_vector_sha256']
             and cut['gradient_sha256']==hashlib.sha256(gradient.astype('<f8').tobytes()).hexdigest(),
             'Old cut coefficient/anchor identity changed')
    c.check_deadline()
    return dict(anchor_first_stage=anchor,cuts=cuts,source_references=[c.refs['old_pack'],c.refs['old_pack_review']])
