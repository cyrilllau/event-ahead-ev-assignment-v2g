"""Hand-checkable one-scenario strengthening; no scenario1 recourse call."""
import copy,tempfile,time,unittest,json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import milp,Bounds,LinearConstraint
from evcs_experiments import case0_partial_cut as partial
from evcs_experiments.dual_cut_bound import build_dual_cut
from evcs_experiments.epigraph_master import certify_box_floor
from evcs_experiments.multi_anchor_epigraph import build_multi_anchor_epigraph_master
from evcs_experiments.multi_anchor_bound_audit import audit_multi_anchor_model
from evcs_experiments.probability import ProbabilitySet
from evcs_experiments.campaign_batch import file_reference
from evcs_experiments.random_streams import canonical_hash

class PartialCutTests(unittest.TestCase):
 def setUp(self):
  tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
  p=self.root/'original.json';p.write_text('{"fixture":"analytical original two-scenario recourse"}');self.refs=[file_reference(p)]
  self.P=ProbabilitySet(('s0','s1'),('f',),np.array([[0.],[1.]]),np.array([.5]),np.array([.5]))
  self.first=dict(matrix=csr_matrix(np.eye(2)),c=np.array([.1,.1]),integrality=np.ones(2,dtype=int),lower_bounds=np.zeros(2),upper_bounds=np.ones(2),row_lower=np.zeros(2),row_upper=np.ones(2),objective_constant_usd=0.)
  self.templates=[]
  for i in range(2):
   self.templates.append(dict(scenario_id='s'+str(i),template_sha256=str(i)*64,A_ub=csr_matrix([[-1.]]) if i==0 else csr_matrix([[0.]]),A_eq=csr_matrix((0,1)),
    Bx_ub=csr_matrix([[1.,1.]]) if i==0 else csr_matrix((1,2)),Bx_eq=csr_matrix((0,2)),Br_ub=csr_matrix((1,0)),Br_eq=csr_matrix((0,0)),
    b_ub=np.array([-1.]) if i==0 else np.zeros(1),b_eq=np.zeros(0),c=np.array([1.]) if i==0 else np.zeros(1),lower_bounds=np.zeros(1),upper_bounds=np.full(1,np.inf)))
  groups=[];self.cut_sets=[]
  for anchor in (np.array([1.,0.]),np.array([0.,1.])):
   cuts={i:build_dual_cut(t,dict(inequality=np.zeros(1),equality=np.zeros(0),lower=t['c'],upper=np.zeros(1)),anchor_first_stage=anchor,anchor_q_usd=0.) for i,t in enumerate(self.templates)}
   self.cut_sets.append(cuts);groups.append(dict(anchor_first_stage=anchor,cuts=cuts,source_references=self.refs))
  floors={i:certify_box_floor(t) for i,t in enumerate(self.templates)}
  self.base,self.construction=build_multi_anchor_epigraph_master(self.first,self.P.support_ids,groups,probability_set=self.P,floors=floors)
  self.audit=audit_multi_anchor_model(self.base,self.first,self.P.support_ids,groups,probability_set=self.P,floors=floors,deadline=time.monotonic()+5)
  self.prob=dict(probabilities=[.5,.5],support_ids=['s0','s1'],feature_ids=['f'],ambiguity_sha256=self.P.ambiguity_sha256)
  self.v=np.zeros(len(self.base['c']))
 def select(self,v=None):
  return partial.select_scenario(self.base,self.audit,self.P,self.cut_sets,self.prob,self.v if v is None else v,source_references=self.refs,deadline=time.monotonic()+5)
 def newcut(self,z=None):
  z=self.v[:2] if z is None else z
  return build_dual_cut(self.templates[0],dict(inequality=np.array([-1.]),equality=np.zeros(0),lower=np.zeros(1),upper=np.zeros(1)),anchor_first_stage=z,anchor_q_usd=float(1-z.sum()))
 def append(self,selection=None,cut=None,v=None):
  c,g=self.newcut() if cut is None else cut
  return partial.append_partial_cut(self.base,self.audit,self.P,self.select() if selection is None else selection,c,g,self.v if v is None else v,source_references=self.refs,deadline=time.monotonic()+5)
 def test_one_true_partial_row_strengthens_integer_bound(self):
  s=self.select();self.assertEqual(s['ranking'],[0,1]);self.assertEqual(s['selected_index'],0)
  m,check,gate=self.append(s);self.assertTrue(gate['master_call_required']);self.assertEqual(check['appended_rows'],1)
  self.assertEqual(check['newly_evaluated_scenario_indices'],[0]);self.assertFalse(check['full_support_new_anchor'] or check['new_UB_verified'])
  result=milp(m['c'],integrality=m['integrality'],bounds=Bounds(m['lower_bounds'],m['upper_bounds']),constraints=LinearConstraint(m['matrix'],m['row_lower'],m['row_upper']))
  self.assertAlmostEqual(result.fun,.1);self.assertEqual(float(self.base['c']@self.v),0.)
 def test_full_epigraph_slack_skips_redundant_master(self):
  # Both scenario features coincide, so the dual epigraph is one common theta.
  # Scenario1 has constant cost2. It certifies theta>=2, so v=(0,0,2,0,0)
  # is genuinely optimal, even though scenario0 old cuts are0 and its newQ is1.
  self.P=ProbabilitySet(('s0','s1'),('f',),np.zeros((2,1)),np.zeros(1),np.zeros(1))
  self.prob.update(ambiguity_sha256=self.P.ambiguity_sha256)
  groups=[];self.cut_sets=[]
  for anchor in (np.array([1.,0.]),np.array([0.,1.])):
   cuts={i:build_dual_cut(t,dict(inequality=np.zeros(1),equality=np.zeros(0),lower=t['c'],upper=np.zeros(1)),
       anchor_first_stage=anchor,anchor_q_usd=2. if i==1 else 0.,scenario_constant_usd=2. if i==1 else 0.) for i,t in enumerate(self.templates)}
   self.cut_sets.append(cuts);groups.append(dict(anchor_first_stage=anchor,cuts=cuts,source_references=self.refs))
  floors={i:certify_box_floor(t,scenario_constant_usd=2. if i==1 else 0.) for i,t in enumerate(self.templates)}
  self.base,_=build_multi_anchor_epigraph_master(self.first,self.P.support_ids,groups,probability_set=self.P,floors=floors)
  self.audit=audit_multi_anchor_model(self.base,self.first,self.P.support_ids,groups,probability_set=self.P,floors=floors,deadline=time.monotonic()+5)
  v=self.v.copy();v[2]=2.;s=self.select(v);self.assertEqual(s['selected_index'],0)
  m,check,gate=self.append(s,v=v)
  self.assertGreater(1.,max(s['scores'][0]['cut_values_usd']))
  self.assertEqual(float(self.base['c']@v),2.);self.assertEqual(floors[1]['floor_usd'],2.)
  self.assertFalse(gate['master_call_required']);self.assertEqual(check['new_row_activity_usd'],2.)
 def test_missing_incumbent_skips_without_requesting_q(self):
  s=partial.select_scenario(self.base,self.audit,self.P,self.cut_sets,self.prob,None,source_references=self.refs,deadline=time.monotonic()+5)
  self.assertEqual(s['status'],'SKIP_NO_AUXILIARY_INCUMBENT');self.assertFalse(s['master_call_required'])
 def test_coherently_rehashed_selection_cannot_change_selected_scenario(self):
  s=self.select();s['scores'][1]['cut_values_usd']=[0.,3.];s['scores'][1]['score_usd']=1.5;s['ranking']=[1,0];s['selected_index']=1;s['selected_scenario_id']='s1';s['selected_template_sha256']='1'*64
  body={k:v for k,v in s.items() if k!='selection_sha256'};s['selection_sha256']=canonical_hash(body)
  with self.assertRaisesRegex(ValueError,'accepted rows'):self.append(s)
 def test_wrong_gradient_constant_and_anchor_rejected(self):
  c,g=self.newcut();bad=g.copy();bad[0]=0.
  with self.assertRaisesRegex(ValueError,'gradient'):self.append(cut=(c,bad))
  for k,val in [('scenario_constant_usd',1.),('anchor_first_stage_sha256','x'*64),('scenario_id','s1')]:
   c,g=self.newcut();c[k]=val;body={a:b for a,b in c.items() if a!='cut_sha256'};c['cut_sha256']=canonical_hash(body)
   with self.assertRaises(ValueError):self.append(cut=(c,g))
 def test_mutated_retained_floor_or_domain_rejected(self):
  s=self.select();c,g=self.newcut();m,_,_=self.append(s)
  for key in ('row_lower','c','integrality','lower_bounds','upper_bounds'):
   bad=copy.deepcopy(m);bad[key][0]+=1
   with self.assertRaises(ValueError):partial.audit_partial_model(bad,self.base,self.audit,self.P,s,c,g,self.v,source_references=self.refs,deadline=time.monotonic()+5)
 def test_wrong_probability_or_deadline_rejected(self):
  self.prob['probabilities']=[.4,.6]
  with self.assertRaisesRegex(ValueError,'feasible'):self.select()
  with self.assertRaises(TimeoutError):partial.select_scenario(self.base,self.audit,self.P,self.cut_sets,self.prob,self.v,source_references=self.refs,deadline=time.monotonic()-1)

if __name__=='__main__':unittest.main()
