"""Hash-bound mmap reload of an already assembled SAA model; no recompile.

This recovers coefficients, not a solver search tree or scientific acceptance.
Every process/recovery boundary verifies its immutable files and exact inputs.
"""
from pathlib import Path
import json
import numpy as np
from scipy.sparse import csr_matrix
from .campaign_batch import verify_reference
from .random_streams import canonical_hash
from .saa_streaming import StreamingSAA,ALGORITHM_ID,_need,_deadline


def load_streaming_saa(manifest_reference,*,first_stage,expected_input_binding,
                       scenario_ids,scenario_sha256s,deadline):
    _deadline(deadline);verify_reference(manifest_reference)
    path=Path(manifest_reference['path']);root=path.parent
    _need(path.name=='manifest.json' and path.is_absolute() and path.resolve()==path,
          'Canonical SAA manifest path required')
    body=json.loads(path.read_text())
    _need(body['manifest_sha256']==canonical_hash({k:v for k,v in body.items() if k!='manifest_sha256'}),
          'SAA manifest semantic identity differs')
    _need(body['schema_version']==1 and body['risk_model']=='SAA' and body['algorithm_id']==ALGORITHM_ID,
          'SAA model schema/algorithm differs')
    _need(body['input_binding']==expected_input_binding and body['first_stage_sha256']==first_stage.matrix_sha256,
          'SAA restored physical/source binding differs')
    _need(body['scenario_ids']==list(scenario_ids) and body['scenario_sha256s']==list(scenario_sha256s),
          'SAA restored support differs')
    names={'data','indices','indptr','row_lower','row_upper','lower_bounds','upper_bounds','c','integrality'}
    refs=body['numeric_artifacts']
    _need(len(refs)==len(names) and {Path(r['path']).stem for r in refs}==names,
          'SAA numeric array inventory differs')
    _need(all(Path(r['path'])==root/(Path(r['path']).stem+'.npy') for r in refs),
          'SAA verified numeric path differs from loaded array')
    for ref in [*refs,*(r for item in body['scenarios'] for r in item['artifacts'])]:
        _deadline(deadline);p=Path(ref['path'])
        _need(p.resolve()==p and p.is_relative_to(root),'SAA array/shard outside original model directory')
        verify_reference(ref)
    _need({p.name for p in root.glob('*.npy')}=={name+'.npy' for name in names},'SAA extra numeric array')
    arrays={name:np.load(root/(name+'.npy'),mmap_mode='r',allow_pickle=False) for name in names}
    rows,cols=body['shape'];nnz=body['nnz']
    for name in names:
        expected=nnz if name in ('data','indices') else rows+1 if name=='indptr' else rows if name in ('row_lower','row_upper') else cols
        _need(arrays[name].shape==(expected,) and not arrays[name].dtype.hasobject,'SAA restored numeric shape differs')
    _need(np.issubdtype(arrays['indices'].dtype,np.integer) and np.issubdtype(arrays['indptr'].dtype,np.integer),
          'SAA sparse index types differ')
    indptr=arrays['indptr'];indices=arrays['indices']
    _need(indptr[0]==0 and indptr[-1]==nnz and np.all(np.diff(indptr)>=0)
        and np.all((indices>=0)&(indices<cols)),'SAA restored CSR structure differs')
    matrix=csr_matrix((arrays['data'],indices,indptr),shape=(rows,cols),copy=False)
    _deadline(deadline)
    return StreamingSAA(matrix,arrays['row_lower'],arrays['row_upper'],arrays['lower_bounds'],arrays['upper_bounds'],
        arrays['c'],arrays['integrality'],body['objective_constant_usd'],first_stage,body,root)
