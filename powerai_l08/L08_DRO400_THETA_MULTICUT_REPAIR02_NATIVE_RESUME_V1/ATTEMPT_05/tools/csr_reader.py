from pathlib import Path
import numpy as np
from scipy import sparse as sp

def read_csr(path):
 with np.load(Path(path),allow_pickle=False) as z:
  if not {'data','indices','indptr','shape'}.issubset(z.files):raise ValueError('CSR_REQUIRED_ARRAY_MISSING')
  if 'format' in z.files and z['format'].item() not in (b'csr','csr'):raise ValueError('NOT_CSR_FORMAT')
  shape=tuple(int(v) for v in z['shape']);data=z['data'];idx=z['indices'];ptr=z['indptr']
  if len(shape)!=2 or min(shape)<0 or data.ndim!=1 or idx.ndim!=1 or ptr.ndim!=1:raise ValueError('CSR_DIMENSION')
  if not np.issubdtype(idx.dtype,np.integer) or not np.issubdtype(ptr.dtype,np.integer):raise ValueError('CSR_INDEX_TYPE')
  if len(ptr)!=shape[0]+1 or ptr[0]!=0 or ptr[-1]!=len(data) or len(idx)!=len(data) or np.any(np.diff(ptr)<0):raise ValueError('CSR_POINTER')
  if np.any(idx<0) or np.any(idx>=shape[1]) or not np.isfinite(data).all():raise ValueError('CSR_PAYLOAD')
  a=sp.csr_matrix((data.copy(),idx.copy(),ptr.copy()),shape=shape);a.check_format(full_check=True);return a
