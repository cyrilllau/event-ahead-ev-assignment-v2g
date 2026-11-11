"""Check a stored MIP hint on the original matrix and its exact projection."""
import hashlib
import numpy as np


def audit_mip_start(matrix, arrays, vector, solver):
    v=np.asarray(vector,dtype=float)
    if v.shape!=arrays['c'].shape or not np.isfinite(v).all():raise ValueError('MIP start shape')
    activity=matrix@v
    integer=v[arrays['integrality']==1]
    errors=[np.max(np.maximum(arrays['row_lower']-activity,activity-arrays['row_upper']),initial=0),
            np.max(np.maximum(arrays['lower_bounds']-v,v-arrays['upper_bounds']),initial=0),
            np.max(abs(integer-np.rint(integer)),initial=0)]
    if max(errors)>1e-8:raise ValueError('MIP start original-row/bound/integrality failure')
    raw=solver['raw_solver'];native=raw['native_interface']
    reduction=raw.get('zero_reduction')
    cols=np.arange(len(v)) if reduction is None else np.asarray(reduction['retained_columns'],dtype=int)
    if reduction is not None and np.any(v[np.asarray(reduction['zero_columns'],dtype=int)]!=0):
        raise ValueError('Start is incompatible with zero proof')
    projected=v[cols]
    digest=hashlib.sha256(projected.astype('<f8').tobytes()).hexdigest()
    if digest!=native['start_sha256']:raise ValueError('Native start projection hash')
    objective=float(arrays['c']@v)
    tolerance=1e-4+1e-7*max(1.,abs(objective))
    if abs(objective-native['start_objective_without_constant_usd'])>tolerance:raise ValueError('Start objective mismatch')
    if not native['start_is_not_fixed_decision']:raise ValueError('Start mislabeled as fixed decision')
    return dict(status='PASS_ORIGINAL_MIP_START',max_original_violation=float(max(errors)),
                original_objective_without_constant_usd=objective,projection_sha256=digest,
                is_global_lower_bound=False,is_full_support_upper_bound=False)
