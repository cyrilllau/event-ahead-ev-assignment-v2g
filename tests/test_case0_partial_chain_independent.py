"""Independent coherent-tamper tests for retained row ancestry."""
import copy,unittest
from evcs_experiments.random_streams import canonical_hash
from tests.test_case0_partial_chain import ChainTests

class IndependentChainTests(ChainTests):
 def test_resealed_prior_audit_cannot_substitute_retained_anchor(self):
  self.retained[0]['cut']['anchor_first_stage_sha256']='f'*64
  c=self.retained[0]['cut'];c['cut_sha256']=canonical_hash({k:v for k,v in c.items() if k!='cut_sha256'})
  self.audit['cut_sha256']=c['cut_sha256'];self.audit['audit_sha256']=canonical_hash({k:v for k,v in self.audit.items() if k!='audit_sha256'})
  with self.assertRaises(ValueError):self.select()
 def test_resealed_long_chain_cannot_reorder_distinct_rows(self):
  m,a,_=self.append();c,g=self.cut(0,self.v[:2],[0.,-1.],.5)
  self.retained.append(dict(index=0,cut=c,gradient=g.tolist(),anchor=self.v[:2].tolist(),source_references=self.refs))
  self.model,self.audit=m,a;self.v[2]=.5
  self.retained.reverse();self.audit['retained_cut_identities'].reverse()
  self.audit['audit_sha256']=canonical_hash({k:v for k,v in self.audit.items() if k!='audit_sha256'})
  with self.assertRaisesRegex(ValueError,'row/P'):self.select()
 def test_no_new_row_for_existing_original_anchor_even_resealed(self):
  self.v[:2]=[1.,0.];s=self.select();s['status']='NEW_Q_REQUIRED';s['new_Q_required']=True;s['reuse_cut_sha256']=None
  s['selection_sha256']=canonical_hash({k:v for k,v in s.items() if k!='selection_sha256'})
  with self.assertRaisesRegex(ValueError,'preselection'):self.append(s)

if __name__=='__main__':unittest.main()
