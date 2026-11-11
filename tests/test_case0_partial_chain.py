"""Hand-checkable chain: Q0=max(0,1-z0-z1,.5-z0), Q1=0."""
import copy,json,tempfile,time,unittest
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from evcs_experiments import case0_partial_cut as old,case0_partial_chain as chain
from evcs_experiments.campaign_batch import file_reference
from evcs_experiments.dual_cut_bound import build_dual_cut
from evcs_experiments.epigraph_master import certify_box_floor
from evcs_experiments.multi_anchor_epigraph import build_multi_anchor_epigraph_master
from evcs_experiments.multi_anchor_bound_audit import audit_multi_anchor_model
from evcs_experiments.probability import ProbabilitySet
from evcs_experiments.random_streams import canonical_hash

class ChainTests(unittest.TestCase):
 def setUp(self):
  tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name);p=self.root/'original.json';p.write_text('{"analytical_Q0":"max(0,1-z0-z1,.5-z0)","Q1":0}');self.refs=[file_reference(p)]
  self.P=ProbabilitySet(('s0','s1'),('f',),np.array([[0.],[1.]]),np.array([.5]),np.array([.5]))
  self.first=dict(matrix=csr_matrix(np.eye(2)),c=np.array([.1,.1]),integrality=np.ones(2,dtype=int),lower_bounds=np.zeros(2),upper_bounds=np.ones(2),row_lower=np.zeros(2),row_upper=np.ones(2),objective_constant_usd=0.)
  self.ts=[]
  for i in range(2):
   self.ts.append(dict(scenario_id='s'+str(i),template_sha256=str(i)*64,A_ub=csr_matrix([[-1.],[-1.]]) if i==0 else csr_matrix((2,1)),A_eq=csr_matrix((0,1)),
    Bx_ub=csr_matrix([[1.,1.],[1.,0.]]) if i==0 else csr_matrix((2,2)),Bx_eq=csr_matrix((0,2)),Br_ub=csr_matrix((2,0)),Br_eq=csr_matrix((0,0)),b_ub=np.array([-1.,-.5]) if i==0 else np.zeros(2),b_eq=np.zeros(0),c=np.array([1.]) if i==0 else np.zeros(1),lower_bounds=np.zeros(1),upper_bounds=np.full(1,np.inf)))
  self.cuts=[];groups=[]
  for z in (np.array([1.,0.]),np.array([1.,1.])):
   cuts={i:self.cut(i,z,np.zeros(2),0.) for i in range(2)};self.cuts.append(cuts);groups.append(dict(anchor_first_stage=z,cuts=cuts,source_references=self.refs))
  floors={i:certify_box_floor(t) for i,t in enumerate(self.ts)}
  self.original,_=build_multi_anchor_epigraph_master(self.first,self.P.support_ids,groups,probability_set=self.P,floors=floors)
  self.oa=audit_multi_anchor_model(self.original,self.first,self.P.support_ids,groups,probability_set=self.P,floors=floors,deadline=time.monotonic()+5)
  self.prob=dict(probabilities=[.5,.5],support_ids=['s0','s1'],feature_ids=['f'],ambiguity_sha256=self.P.ambiguity_sha256)
  v=np.zeros(len(self.original['c']));c,g=self.cut(0,v[:2],[-1.,0.],1.)
  s=old.select_scenario(self.original,self.oa,self.P,self.cuts,self.prob,v,source_references=self.refs,deadline=time.monotonic()+5)
  self.model,self.audit,_=old.append_partial_cut(self.original,self.oa,self.P,s,c,g,v,source_references=self.refs,deadline=time.monotonic()+5)
  self.retained=[dict(index=0,cut=c,gradient=g.tolist(),anchor=v[:2].tolist(),source_references=self.refs)]
  self.v=v.copy();self.v[1]=1.
 def cut(self,i,z,y,q):
  t=self.ts[i];y=np.asarray(y,float);lower=t['c']-t['A_ub'].T@y
  return build_dual_cut(t,dict(inequality=y,equality=np.zeros(0),lower=lower,upper=np.zeros(1)),anchor_first_stage=z,anchor_q_usd=q)
 def kwargs(self):return dict(original_model=self.original,original_audit=self.oa,retained_cuts=self.retained,source_references=self.refs,deadline=time.monotonic()+5)
 def select(self,v=None):return chain.select_scenario(self.model,self.audit,self.P,self.cuts,self.prob,self.v if v is None else v,**self.kwargs())
 def append(self,selection=None,cut=None):
  c,g=self.cut(0,self.v[:2],[0.,-1.],.5) if cut is None else cut
  return chain.append_partial_cut(self.model,self.audit,self.P,self.select() if selection is None else selection,c,g,self.v,cut_sets=self.cuts,**self.kwargs())
 def test_retains_every_prior_row_and_adds_only_selected_scenario(self):
  s=self.select();self.assertEqual(s['status'],'NEW_Q_REQUIRED');self.assertEqual(s['selected_index'],0)
  m,a,gate=self.append(s);self.assertTrue(gate['master_call_required']);self.assertEqual(gate['raw_violation'],.5)
  n=self.model['matrix'].shape[0];self.assertEqual((m['matrix'][:n]!=self.model['matrix']).nnz,0);self.assertEqual(m['matrix'].shape[0],n+1)
  self.assertEqual(a['total_partial_rows'],2);self.assertEqual(a['newly_evaluated_scenario_indices'],[0]);self.assertFalse(a['full_support_new_anchor'] or a['new_UB_verified'])
  self.assertTrue(np.array_equal(m['integrality'],self.original['integrality']));self.assertTrue(np.array_equal(m['c'],self.original['c']))
 def test_exact_partial_anchor_and_original_anchor_reuse(self):
  for z,theta in (([0.,0.],1.),([1.,0.],0.)):
   v=self.v.copy();v[:2]=z;v[2]=theta;s=self.select(v)
   self.assertEqual(s['status'],'REUSE_EXACT_ANCHOR_SCENARIO');self.assertFalse(s['new_Q_required'] or s['master_call_required']);self.assertIsNotNone(s['reuse_cut_sha256'])
   with self.assertRaisesRegex(ValueError,'reused'):chain.append_partial_cut(self.model,self.audit,self.P,s,*self.cut(0,z,[0.,-1.],max(0.,.5-z[0])),v,cut_sets=self.cuts,**self.kwargs())
 def test_retained_chain_can_be_reopened_without_erasing_rows(self):
  m,a,_=self.append();c,g=self.cut(0,self.v[:2],[0.,-1.],.5)
  self.retained.append(dict(index=0,cut=c,gradient=g.tolist(),anchor=self.v[:2].tolist(),source_references=self.refs));self.model,self.audit=m,a
  v=self.v.copy();v[2]=.5;s=self.select(v);self.assertEqual(s['status'],'REUSE_EXACT_ANCHOR_SCENARIO');self.assertEqual(len(s['retained_cut_identities']),2)
 def test_rehashed_selection_cannot_choose_false_score_or_remove_prior_cut(self):
  s=self.select();s['scores'][1]['score_usd']=10.;s['ranking']=[1,0];s['selected_index']=1;s['selected_scenario_id']='s1';s['selection_sha256']=canonical_hash({k:v for k,v in s.items() if k!='selection_sha256'})
  with self.assertRaisesRegex(ValueError,'preselection'):self.append(s)
  for key in ('row_lower','integrality'):
   saved=copy.deepcopy(self.model);self.model[key][0]+=1
   with self.assertRaises(ValueError):self.select()
   self.model=saved
  self.retained=[]
  with self.assertRaisesRegex(ValueError,'previously accepted'):self.select()
 def test_wrong_retained_anchor_scenario_and_new_global_minorant_rejected(self):
  saved=copy.deepcopy(self.retained)
  for k,val in [('index',1),('anchor',[1.,0.])]:
   self.retained=copy.deepcopy(saved);self.retained[0][k]=val
   with self.assertRaises(ValueError):self.select()
  self.retained=saved;c,g=self.cut(0,self.v[:2],[0.,-1.],.5);c['max_dual_stationarity_residual']=1e-4;c['cut_sha256']=canonical_hash({k:v for k,v in c.items() if k!='cut_sha256'})
  with self.assertRaisesRegex(ValueError,'numerical identity'):self.append(cut=(c,g))
 def test_full_incumbent_slack_skips_master_and_no_new_q_on_missing_incumbent(self):
  # A supplied full incumbent with epigraph slack satisfies the new row. The
  # component requires the driver to establish prior optimality before skip.
  self.v[2]=1.;_,a,gate=self.append();self.assertFalse(gate['master_call_required']);self.assertTrue(a['skip_requires_previously_accepted_auxiliary_optimality'])
  s=chain.select_scenario(self.model,self.audit,self.P,self.cuts,self.prob,None,**self.kwargs());self.assertFalse(s['new_Q_required']);self.assertEqual(s['status'],'SKIP_NO_AUXILIARY_INCUMBENT')
 def test_probability_and_source_tamper_fail(self):
  self.prob['probabilities']=[.4,.6]
  with self.assertRaises(ValueError):self.select()
  self.prob['probabilities']=[.5,.5];Path(self.refs[0]['path']).write_text('changed')
  with self.assertRaises(ValueError):self.select()

if __name__=='__main__':unittest.main()
