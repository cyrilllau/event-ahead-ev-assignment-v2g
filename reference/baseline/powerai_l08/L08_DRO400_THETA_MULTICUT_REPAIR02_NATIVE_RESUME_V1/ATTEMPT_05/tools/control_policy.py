"""Shared production/test state decisions. Evidence validity is checked upstream."""
import math,json,hashlib
from gap_phase_reference import gap,route_after_bound_update,candidate_route,request_handover

def finite_bound(value):
 if value is None:return None
 value=float(value)
 return value if math.isfinite(value) and abs(value)<1e90 else None

def json_safe(value):
 if hasattr(value,'item'):return json_safe(value.item())
 if isinstance(value,dict):return {str(k):json_safe(v) for k,v in value.items()}
 if isinstance(value,(list,tuple)):return [json_safe(v) for v in value]
 if isinstance(value,float) and not math.isfinite(value):return None
 return value

def bound_state(lb,ub,strict=None):
 lb,ub,strict=map(finite_bound,(lb,ub,strict))
 return dict(LB=lb,UB_audited=ub,strict_UB=strict,audited_gap=gap(lb,ub),strict_gap=gap(lb,strict))

def classify_candidate(raw_sha,rho,known):
 record=known.get(raw_sha)
 if record is None:return 'VALIDATE_AND_EVALUATE_FULL400'
 return candidate_route(complete_cached=True,certified_cut_violates=record['activity']-rho>1e-6,oracle_tight=record['tightness']<=1e-4,master_proof_complete=True)

def pending_candidate(raw_sha,rho,known):
 return classify_candidate(raw_sha,rho,known) in ('VALIDATE_AND_EVALUATE_FULL400','SEPARATE_CACHED_SUPPORT','REFINE_EXISTING_ORACLE_EVIDENCE')

def check_science(config,identity,expected):
 if config.get('identity',config.get('science'))!=expected or identity.get('science',identity.get('scientific_id',identity.get('identity')))!=expected:raise ValueError('SCIENCE_IDENTITY_MISMATCH')
 return True

def schema_adapter(cache,source_proof,matrix_proof):
 if cache.get('nx')!=60780 or not source_proof.get('d_zero') or not source_proof.get('kappa_zero') or not source_proof.get('continuous') or not matrix_proof.get('passed'):
  raise ValueError('SCHEMA_ADAPTER_PROOF_MISSING')
 return dict(passed=True,legacy_missing_flag='d_s_and_constant_zero_verified' not in cache,source_proof=source_proof,matrix_proof=matrix_proof)

def cut_identity(alpha,beta):
 return hashlib.sha256(alpha.tobytes()+beta.tobytes()).hexdigest()

def commit_phase(path,state,publish,verify):
 for r in state.get('cuts',[]):verify(r['aggregate'])
 for k in ('x_eval','x_master','x_best'):
  if state.get(k):verify(state[k])
 publish(path,state)
 restored=json.loads(path.read_text())
 if restored!=json_safe(state):raise ValueError('CHECKPOINT_READBACK_MISMATCH')
 return restored

def drive(prepare,test,execute):
 prepare();test();return execute()
