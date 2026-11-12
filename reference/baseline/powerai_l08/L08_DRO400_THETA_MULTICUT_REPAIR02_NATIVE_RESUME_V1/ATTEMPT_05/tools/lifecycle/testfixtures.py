"""Small genuine LP fixtures. Never address a production recourse instance."""
from pathlib import Path
import numpy as np
from scipy import sparse as sp
import engine as e

class TinyOracle(e.Oracle):
 def __init__(self,sid,root,env=None,context=None):
  self.sid=int(sid);self.root=Path(root);self.out=self.root/'oracles'/f'{sid:04d}';self.out.mkdir(parents=True,exist_ok=True);n=256
  self.a={'c':np.ones(n),'h':np.zeros(n),'sense':np.full(n,'>'),'lo':np.zeros(n),'hi':np.full(n,2.),'finite_lo':np.zeros(n),'finite_hi':np.full(n,2.),'xlo':np.zeros(1),'xhi':np.ones(1)}
  self.W=sp.eye(n,format='csr');self.T=sp.csr_matrix(([-1.],([0],[0])),shape=(n,1));self.qbar=2.;self.model=None;self.y=None;self.rows=None;self.owned_env=env;self.lifecycle_context=context or {};self.dispatch=None;self.memory_provider=None;self.native_kind='TEST_LIFECYCLE_RECOURSE'
 def test_soft_limit(self):
  return 1e-8 if self.lifecycle_context.get('slot')==0 and self.lifecycle_context.get('generation')==0 else 3.

class FakeOracle:
 def __init__(self,sid,env):self.sid=sid;self.model=object();self.y=object();self.rows=object();self.W=object();self.T=object();self.a={};self.disposed=0
 def close(self,discard=False):
  if self.model is not None:self.disposed+=1
  self.model=None;self.y=None;self.rows=None
  if discard:self.W=None;self.T=None;self.a=None
