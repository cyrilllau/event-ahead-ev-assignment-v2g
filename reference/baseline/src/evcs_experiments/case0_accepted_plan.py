"""Read-only bridge for one explicitly accepted, nonconverged Case0.1 EA plan.

It verifies existing evidence and returns a distinctly typed comparison view.
No solver, campaign mutation, normal-result rewriting, or general gap override.
"""
import json,math
from pathlib import Path
import numpy as np
from evcs_experiments.campaign_batch import file_reference,verify_reference
from evcs_experiments.random_streams import canonical_hash

ACCEPTANCE_SHA='33676d2fa62d2b27dc2b2065ffb93d0d0449e413a4f41cb215006208822695ba'
DECISION_SHA='ddf925651782ea9c9b213164bd8ad17b0333eb336497172cc24655a4f618c7ff'
PLAN_SHA='02e3671c7faad520073d27a68c97fe40b8c9976152f388a3b984bfb24b9d40cf'
MODEL_SHA='6c342f2d438806b26c29b1cc6c7f07a92025c6efda0215cc30a6dcba5fa0ff2f'
UPPER=219430.97701450257
LOWER=217872.73443185713
GAP=(UPPER-LOWER)/max(1.,abs(UPPER))
KEY='size0/EA_DRO'
CAMPAIGN='ieee33-v54-q1-case0-expanded'


def _need(ok,msg):
    if not bool(ok):raise ValueError(msg)

def _close(a,b,msg):
    _need(type(a) in (int,float) and type(b) in (int,float) and math.isfinite(a) and math.isfinite(b) and
          abs(a-b)<=1e-4+1e-7*max(1.,abs(a),abs(b)),msg)


def load_accepted_plan(root,state=None,*,method='EA_DRO'):
    """Return None unless this method has the exact registered EA acceptance.

    The returned result_reference is the unchanged UB evidence. The separately
    typed comparison_result is an in-memory view, not a forged normal result.
    Callers must keep its acceptance and artifact references when persisting it.
    """
    if method!='EA_DRO':return None
    root=Path(root).resolve()
    if state is None:state=json.loads((root/'state.json').read_text())
    ref=state.get('accepted_approximate_plans',{}).get(KEY)
    if ref is None:return None
    _need(root.name==CAMPAIGN and state['size_index']==0,'Exact expanded EA campaign/size required')
    _need(Path(ref['path']).resolve().is_relative_to(root/'operations') and ref['sha256']==ACCEPTANCE_SHA,
          'Only the exact user-accepted EA record is admissible')
    checked={}
    def verify(r):
        key=r['path']
        if key in checked:_need(checked[key]==r,'Conflicting accepted artifact references')
        else:verify_reference(r);checked[key]=r
    def read(r):verify(r);return json.loads(Path(r['path']).read_text())
    def direct_refs(d):
        for r in d.values():
            if isinstance(r,dict) and set(r)=={'path','sha256','size_bytes'}:verify(r)
    a=read(ref);refs=a['references'];expected={'saved_candidate_ub_reference','original_extensive_closure_result_reference',
        'original_extensive_closure_settlement_reference','partial_chain_master_bound_result_reference',
        'partial_chain_master_bound_settlement_reference','source_release_reference','active_config_reference'}
    _need(set(refs)==expected and a['kind']=='CASE0_EXACT_PLAN_APPROXIMATE_ACCEPTANCE_V1' and a['accepted_for_comparison'] is True and
          a['normal_gap_passed'] is False and a['formal_OOS'] is False and a['plan_sha256']==PLAN_SHA and a['model_sha256']==MODEL_SHA and
          (a['upper_bound_usd'],a['lower_bound_usd'],a['gap'])==(UPPER,LOWER,GAP),'Exact acceptance identity/precision differs')
    verify(a['decision_document_reference']);_need(a['decision_document_reference']['sha256']==DECISION_SHA and a['user_instruction']=='就用0.71 可以',
          'Explicit exact-plan decision changed')
    for k in expected:_need(state.get(k)==refs[k],'Accepted historical state reference changed: '+k)
    config=read(refs['active_config_reference']);manifest=read(refs['source_release_reference'])
    _need(config['profile']=='expanded48-v1' and config['target_event']=='zeta_2020' and config['backend']=='gurobi' and
          config['sizes'][0]=={'fleet_per_depot':12,'planning_variants':1} and config['variant_start']==30000 and
          config['gap_target']==.005 and config['configuration_seconds']==1800 and config['master_seconds']==1200,
          'Accepted original scientific configuration differs')
    inventory={r['path']:r['sha256'] for r in manifest['files']};_need(len(inventory)==len(manifest['files']),'Duplicate producer inventory')
    producer_root=Path(refs['source_release_reference']['path']).parent
    for name,sha in config['source_sha256'].items():
        _need(inventory.get(name)==sha,'Accepted producer manifest/config changed');r=file_reference(producer_root/name);_need(r['sha256']==sha,'Accepted original producer source changed');verify(r)
    ub=read(refs['saved_candidate_ub_reference']);binding=ub['source_binding'];direct_refs(binding)
    _need(ub['kind']=='CASE0_SAVED_CANDIDATE_UB_AUDIT_V1' and ub['passed'] is True and ub['method_id']=='EA_DRO' and
          ub['plan_sha256']==PLAN_SHA==canonical_hash(ub['plan']) and ub['model_sha256']==MODEL_SHA and ub['upper_bound_usd']==UPPER and
          ub['gap_passed'] is False and ub['planning_acceptance'] is False and ub['full_numerical_acceptance'] is False and
          ub['first_stage_physical_passed'] is True and ub['full_support_fixed_Q_and_KKT_passed'] is True and ub['probability_primal_dual_passed'] is True and
          binding['config']==refs['active_config_reference'] and binding['source_manifest']==refs['source_release_reference'],
          'Original saved candidate plan or numerical scope differs')
    identity=read(binding['input_identity']);header=read(binding['model'])
    _need(identity['profile']=='expanded48-v1' and identity['target']=='zeta_2020' and identity['lead_h']==1 and len(identity['vehicle_ids'])==48 and
          len(set(identity['vehicle_ids']))==48 and identity['variant_offset']==30000 and identity['variants_per_base']['planning']==1 and
          identity['formal_OOS'] is False and ub['plan']['task_sha256']==identity['task_sha256'] and header['model_sha256']==MODEL_SHA,
          'Accepted EA input information contract differs')
    verify(identity['task_file']);ids=ub['scenario_ids'];q=ub['q_usd'];entries=ub['completed_scenario_references']
    _need(len(ids)==len(set(ids))==len(q)==len(entries)==8 and ids==header['scenario_ids'] and all(type(x) in (float,int) and math.isfinite(x) for x in q),
          'Complete original eight-support axis required')
    for i,(sid,entry) in enumerate(zip(ids,entries)):
        _need(entry['index']==i and entry['scenario_id']==sid and entry['template_sha256']==header['recourse_template_sha256s'][i] and entry['q_usd']==q[i] and
              set(entry['files'])=={'fixed_solver','primal','checks'},'Saved original eight-Q evidence pairing differs')
        direct_refs(entry['files']);checks=read(entry['files']['checks'])
        _need(checks['physical']['passed'] is True and checks['kkt']['passed'] is True,'Original saved Q physical/KKT audit failed')
        _close(checks['kkt']['qop_usd'],q[i],'Original saved Q objective differs');_close(checks['physical']['costs']['Qop'],q[i],'Physical Q objective differs')
    closure=read(refs['original_extensive_closure_result_reference']);closure_sett=read(refs['original_extensive_closure_settlement_reference'])
    _need(closure['kind']=='CASE0_ASSEMBLED_ORIGINAL_EXTENSIVE_WITNESS_CHECK_V1' and closure['passed'] is True and closure['plan_sha256']==PLAN_SHA and
          closure['model_sha256']==MODEL_SHA and closure['original_array_check']['passed'] is True and closure['original_native_primal_recovered'] is False and
          closure['assembled_from_independent_Q_primal_and_P_dual'] is True and closure['planning_acceptance'] is False and closure['new_Q_calls']==closure['new_master_calls']==0,
          'Original extensive witness closure missing or relabeled')
    for k in ('max_row_violation','max_bound_violation','max_integrality_violation'):
        _need(type(closure['original_array_check'][k]) in (float,int) and 0<=closure['original_array_check'][k]<=1e-8,'Original extensive physical residual failed')
    _close(closure['objective_usd'],UPPER,'Original extensive witness objective differs');direct_refs(closure['source_binding']);verify(closure['assembled_primal_reference'])
    _need(closure['source_binding']['ub_result']==refs['saved_candidate_ub_reference'] and closure_sett['result_reference']==refs['original_extensive_closure_result_reference'] and
          closure_sett['original_best_ub_reference']==refs['saved_candidate_ub_reference'] and closure_sett['passed'] is True,'Extensive closure settlement source differs')
    ctr=read(closure_sett['terminal_review_reference']);direct_refs(ctr['references'])
    _need(ctr['kind']=='CASE0_ORIGINAL_EXTENSIVE_CLOSURE_TERMINAL_REVIEW_V1' and ctr['passed'] is True and ctr['numerical_recheck_passed'] is True and
          ctr['references']['result']==refs['original_extensive_closure_result_reference'],'Independent extensive numerical terminal acceptance missing')
    probability=read(closure['source_binding']['probability'])
    _need(probability['certified_optimal'] is True and probability['support_ids']==ids and probability['ambiguity_sha256']==header['ambiguity_sha256'],
          'Original full probability certificate differs')
    p=np.asarray(probability['probabilities'],float)
    _need(p.shape==(8,) and np.isfinite(p).all() and np.min(p)>=-1e-8 and abs(p.sum()-1)<=1e-8,'Original complete probability vector differs')
    _close(float(p@np.asarray(q)),probability['primal_objective_usd'],'Original paired probability objective differs')
    _close(probability['primal_objective_usd'],probability['dual_objective_usd'],'Original probability primal/dual differs')
    _close(ub['Fpre_usd']+probability['dual_objective_usd'],UPPER,'Original risk upper bound differs')
    bound=read(refs['partial_chain_master_bound_result_reference']);sett=read(refs['partial_chain_master_bound_settlement_reference'])
    _need(bound['kind']=='CASE0_PARTIAL_CHAIN_MASTER_BOUND_RESULT_V1' and bound['model_sha256']==MODEL_SHA and bound['old_plan_sha256']==PLAN_SHA and
          bound['lower_bound_usd']==LOWER and bound['upper_bound_usd']==UPPER and bound['gap']==GAP and bound['gap_passed'] is False and
          bound['planning_acceptance'] is False and bound['source_binding']['old_ub']==refs['saved_candidate_ub_reference'] and
          sett['passed'] is True and sett['result_reference']==refs['partial_chain_master_bound_result_reference'] and
          sett['best_ub_reference']==refs['saved_candidate_ub_reference'] and sett['best_lower_bound_usd']==LOWER and sett['best_upper_bound_usd']==UPPER and
          sett['best_gap']==GAP,'Exact separately audited auxiliary lower bound differs')
    btr=read(sett['terminal_review_reference']);direct_refs(btr['references']);native=read(bound['native_audit_reference'])
    _need(btr['passed'] is True and btr['numerical_recheck_passed'] is True and btr['references']['result']==refs['partial_chain_master_bound_result_reference'] and
          native['passed'] is True and native['native_status']==bound['native_status'] and native['lower_bound_usd']==bound['new_native_lower_bound_usd'],
          'Independent native auxiliary-bound terminal acceptance missing')
    _need(a['opening_compute_seconds']==sett['closing_compute_seconds'] and a['opening_configuration_seconds']==sett['closing_configuration_seconds'] and
          a['closing_compute_seconds']==a['opening_compute_seconds']+a['metadata_seconds'] and
          a['closing_configuration_seconds']==a['opening_configuration_seconds']+a['metadata_seconds'] and a['master_seconds']==0.,'Acceptance fee history differs')
    spent=state['configuration_spent_seconds'][KEY];master=state['configuration_master_spent_seconds'][KEY]
    _need(spent==a['closing_configuration_seconds']<=1800 and sett['closing_master_seconds']==master<=1200 and state['compute_seconds']>=a['closing_compute_seconds'],
          'Accepted EA cumulative budget was reset or refined again')
    result=dict(kind='CASE0_EXPLICITLY_ACCEPTED_EA_COMPARISON_RESULT_V1',method_id='EA_DRO',plan=ub['plan'],plan_sha256=PLAN_SHA,model_sha256=MODEL_SHA,
        scenario_ids=ids,q_usd=q,upper_bound_usd=UPPER,lower_bound_usd=LOWER,gap=GAP,gap_passed=False,accepted_for_comparison=True,
        normal_gap_target=.005,numerical_valid=True,full_support_independently_evaluated=True,backend=config['backend'],
        configuration_wall_seconds=spent,acceptance_reference=ref,original_ub_result_reference=refs['saved_candidate_ub_reference'],
        original_bound_result_reference=refs['partial_chain_master_bound_result_reference'],original_native_primal_recovered=False,
        solver_trajectory='Saved original incumbent; independently evaluated eight Q/P; assembled extensive witness; separately audited auxiliary lower bound.',
        normal_planning_acceptance=False,formal_OOS=False,Case1_acceptance=False)
    return dict(kind='CASE0_ACCEPTED_COMPARISON_PLAN_EVIDENCE_V1',acceptance_reference=ref,result_reference=refs['saved_candidate_ub_reference'],
        input_identity_reference=binding['input_identity'],producer_config_reference=refs['active_config_reference'],
        producer_source_manifest_reference=refs['source_release_reference'],producer_source_sha256=config['source_sha256'],
        comparison_result=result,artifact_references=[checked[p] for p in sorted(checked)])
