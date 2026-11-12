import numpy as np
from scipy import sparse
import gurobipy as gp
import pytest
from ev_assignment.l08.point import canonical
from ev_assignment.domain.planning import Control
from ev_assignment.l08.certificates import row_audit


def test_real_tiny_projection_preserves_discrete_choice_and_tolerances():
    first=dict(A=sparse.csr_matrix([[1.,1.,1.]]),rhs=np.array([1.+1.8e-9]),sense=np.array(['=']),
               lo=np.zeros(3),hi=np.array([1.,2.,2.]),vtype=np.array(['B','C','C']))
    raw=np.array([9e-10,9e-10,1.]);events=[]
    with gp.Env(params={'OutputFlag':0}) as env:
        accepted=canonical(raw,first,control=Control(5,event=events.append),env=env)
    assert accepted[0]==0 and accepted[1]==0
    assert abs(accepted[2]-(1+1.8e-9))<=1e-12
    assert max(row_audit(first['A'],first['rhs'],first['sense'],first['lo'],first['hi'],accepted).values())<=1e-9
    assert max(abs(accepted-raw))<=1e-7
    assert any(e['kind']=='native_end' and e['native_status']==gp.GRB.OPTIMAL for e in events)
    with pytest.raises(ValueError):canonical(raw+1e-3,first)
