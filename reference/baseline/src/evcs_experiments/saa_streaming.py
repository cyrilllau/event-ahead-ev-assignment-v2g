"""Exact disk-backed SAA extensive assembly with one physical template resident.

This is the original empirical-expectation MILP, not C&CG or an adversarial law.
The assembler starts no solver. Solver import/presolve memory is NOT bounded by
its disk-backed coefficient size; a production resource probe is still required.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse import csr_matrix, hstack, load_npz, save_npz, vstack

from .evaluation_lock import file_reference
from .first_stage import FirstStagePlan
from .random_streams import canonical_hash

ALGORITHM_ID = 'SAA_EXACT_STREAMING_EXTENSIVE_ZERO_V1'


def _need(value, message):
    if not bool(value):
        raise ValueError(message)


def _deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError('SAA assembly shared deadline exhausted')


def _reduce_recourse(matrix, rl, ru, lower, upper, nfirst):
    """Exact zero propagation; shared first-stage columns are NEVER eliminated."""
    a = csr_matrix(matrix, copy=True)
    a.sum_duplicates(); a.eliminate_zeros(); a.sort_indices()
    zero = (lower == 0) & (upper == 0)
    zero[:nfirst] = False
    proof = [{'kind': 'original_zero_bounds', 'columns': np.flatnonzero(zero).tolist()}]
    while True:
        cols = np.flatnonzero(~zero)
        b = a[:, cols].tocsr()
        rows = np.flatnonzero(np.diff(b.indptr) == 1)
        slots = b.indptr[rows]
        variables, coefficient = cols[b.indices[slots]], b.data[slots]
        upper_zero = upper == 0; lower_zero = lower == 0
        force_up = ((coefficient > 0) & (ru[rows] == 0)) | ((coefficient < 0) & (rl[rows] == 0))
        force_lo = ((coefficient > 0) & (rl[rows] == 0)) | ((coefficient < 0) & (ru[rows] == 0))
        upper_zero[variables[force_up]] = True
        lower_zero[variables[force_lo]] = True
        new = np.flatnonzero(upper_zero & lower_zero & ~zero & (lower <= 0) & (upper >= 0)
                             & (np.arange(len(lower)) >= nfirst))
        if not len(new):
            break
        proof.append({'kind': 'homogeneous_singleton', 'columns': new.tolist(),
                      'witness_rows': rows[np.isin(variables, new) & (force_up | force_lo)].tolist()})
        zero[new] = True
    keep = ~((np.diff(b.indptr) == 0) & (rl <= 0) & (ru >= 0))
    return b[keep].tocsr(), np.flatnonzero(keep), cols, proof


def _write(path, body):
    with Path(path).open('x') as out:
        json.dump(body, out, indent=2, allow_nan=False)
        out.write('\n')


@dataclass(frozen=True)
class StreamingSAA:
    A: object
    row_lower: object
    row_upper: object
    lower_bounds: object
    upper_bounds: object
    c: object
    integrality: object
    objective_constant_usd: float
    first_stage: object
    manifest: dict
    root: Path

    def plan(self, vector):
        v = np.asarray(vector)
        _need(v.shape == self.c.shape and np.isfinite(v).all(), 'Invalid SAA solver vector')
        h = len(self.first_stage.task_ids)
        return FirstStagePlan(v[:h], v[h:len(self.first_stage.c)].reshape(h, -1),
                              self.first_stage.task_sha256, self.first_stage.config_hash)

    def original_recourse_vector(self, index, vector):
        """Expand one witness for checks against its freshly verified template."""
        v = np.asarray(vector)
        _need(v.shape == self.c.shape and np.isfinite(v).all(), 'Invalid SAA solver vector')
        row = self.manifest['scenarios'][index]
        path = self.root / row['mapping_file']
        _need(file_reference(path) == row['mapping_reference'], 'SAA reconstruction map changed')
        cols = np.load(path, allow_pickle=False)
        full = np.zeros(row['original_recourse_columns'])
        full[cols] = v[slice(*row['column_slice'])]
        return full


def assemble_streaming_saa(first_stage, recourses, *, scenario_ids, scenario_sha256s,
                           empirical_probabilities, output_directory,
                           scenario_cost_constants_usd=None, max_coefficient_bytes,
                           deadline=None, domain='development', input_binding=None):
    """Compile once per atom into shards, then fill final CSR memmaps in order.

    ``recourses`` is an indexable bounded provider; never converted to a tuple.
    ``max_coefficient_bytes`` caps FINAL numeric arrays, not solver RSS. The
    caller counts all wall time and binds original sources and template admission.
    Failures leave an incomplete directory for diagnosis; no resume is implied.
    All original tasks, support atoms and supplied probabilities are retained.
    """
    started = time.monotonic(); _deadline(deadline)
    ids, hashes = tuple(scenario_ids), tuple(scenario_sha256s)
    S, nfirst = len(ids), len(first_stage.c)
    _need(S > 0 and len(hashes) == S and len(set(ids)) == S and len(recourses) == S,
          'SAA complete explicit support axis required')
    _need(all(isinstance(v, str) and v for v in ids + hashes), 'Invalid SAA identities')
    pi = np.asarray(empirical_probabilities, dtype=float)
    _need(pi.shape == (S,) and np.isfinite(pi).all() and (pi >= 0).all()
          and abs(float(pi.sum()) - 1) <= 1e-8, 'Invalid explicit empirical weights; never normalized')
    constants = np.asarray(np.zeros(S) if scenario_cost_constants_usd is None else scenario_cost_constants_usd, dtype=float)
    _need(constants.shape == (S,) and np.isfinite(constants).all(), 'Invalid scenario constants')
    _need(type(max_coefficient_bytes) is int and max_coefficient_bytes > 0, 'Explicit positive coefficient byte ceiling required')
    _need(domain in ('formal', 'development'), 'Explicit SAA domain required')
    _need(domain != 'formal' or isinstance(input_binding, dict) and bool(input_binding), 'Formal SAA source binding required')
    _need(domain != 'formal' or S == 160 and np.array_equal(pi, np.full(S, 1/S)),
          'Frozen formal SAA requires all 160 atoms with exact uniform weights')
    root = Path(output_directory).resolve(); root.mkdir(parents=True, exist_ok=False)
    shards = root / 'shards'; shards.mkdir()
    rows = first_stage.A.shape[0]; cols = nfirst; nnz = first_stage.A.nnz
    metadata = []
    for i in range(S):
        _deadline(deadline)
        template = recourses[i]
        _need(template.scenario_sha256 == hashes[i] and template.first_stage.matrix_sha256 == first_stage.matrix_sha256
              and template.task_sha256 == first_stage.task_sha256 and template.config_hash == first_stage.config_hash,
              'SAA physical template identity differs')
        matrices = []; lowers = []; uppers = []
        for sense in ('ub', 'eq'):
            a, bx, br, rhs = (getattr(template, name + '_' + sense) for name in ('A', 'Bx', 'Br', 'b'))
            matrices.append(hstack([-bx, -br, a], format='csr'))
            lowers.append(np.full(len(rhs), -np.inf) if sense == 'ub' else rhs)
            uppers.append(rhs)
        rl, ru = np.concatenate(lowers), np.concatenate(uppers)
        lower = np.r_[first_stage.lower_bounds, template.lower_bounds]
        upper = np.r_[first_stage.upper_bounds, template.upper_bounds]
        matrix = vstack(matrices, format='csr'); del matrices
        reduced, ri, ci, proof = _reduce_recourse(matrix, rl, ru, lower, upper, nfirst)
        del matrix
        _need(np.array_equal(ci[:nfirst], np.arange(nfirst)), 'Shared first-stage domain was reduced')
        local = ci[nfirst:] - nfirst
        folder = shards / f'{i:03d}'; folder.mkdir()
        save_npz(folder / 'matrix.npz', reduced)
        np.savez(folder / 'arrays.npz', row_lower=rl[ri], row_upper=ru[ri],
                 lower_bounds=template.lower_bounds[local], upper_bounds=template.upper_bounds[local],
                 c=pi[i]*template.c[local])
        np.save(folder / 'recourse_columns.npy', local, allow_pickle=False)
        _write(folder / 'zero_proof.json', dict(proof_steps=proof, retained_rows=ri.tolist(),
               retained_columns=ci.tolist(), first_stage_columns_protected=True,
               coefficient_threshold=None, original_matrix_checks_required=True))
        item = dict(index=i, scenario_id=ids[i], scenario_sha256=hashes[i],
                    template_sha256=template.template_sha256, empirical_probability=float(pi[i]),
                    original_recourse_columns=len(template.c), column_slice=[cols, cols+len(local)],
                    row_slice=[rows, rows+reduced.shape[0]], nnz=int(reduced.nnz),
                    mapping_file=str((folder/'recourse_columns.npy').relative_to(root)),
                    mapping_reference=file_reference(folder/'recourse_columns.npy'),
                    artifacts=[file_reference(p) for p in sorted(folder.iterdir())])
        metadata.append(item); cols += len(local); rows += reduced.shape[0]; nnz += reduced.nnz
        # This is a monotone lower bound for final arrays. Fail before loading
        # further atoms once the final coefficient cap is already impossible.
        width = 4 if max(rows, cols, nnz) < np.iinfo(np.int32).max else 8
        partial_bytes = int(nnz*(8+width)+(rows+1)*width+2*rows*8+4*cols*8)
        if partial_bytes > max_coefficient_bytes:
            _write(root/'resource_limit.json', dict(status='COEFFICIENT_LIMIT_BEFORE_SOLVE',
                completed_atoms=i+1, total_atoms=S, coefficient_bytes_lower_bound=partial_bytes,
                max_coefficient_bytes=max_coefficient_bytes, solver_calls=0,
                complete_support_assembled=False, scientific_spec_pass=False, formal_OOS=False))
            raise ValueError(f'SAA final coefficient byte ceiling exceeded after {i+1}/{S} atoms; no solver started')
        del template, reduced, rl, ru, lower, upper, ri, ci, local, a, bx, br, rhs, lowers, uppers
        _deadline(deadline)
    index_dtype = np.int32 if max(rows, cols, nnz) < np.iinfo(np.int32).max else np.int64
    estimated = int(nnz*(8+np.dtype(index_dtype).itemsize)+(rows+1)*np.dtype(index_dtype).itemsize
                    +2*rows*8+4*cols*8)
    _need(estimated <= max_coefficient_bytes, 'SAA final coefficient byte ceiling exceeded; no solver started')
    def allocate(name, length, dtype=float):
        return np.lib.format.open_memmap(root/(name+'.npy'), mode='w+', dtype=dtype, shape=(length,))
    data=allocate('data',nnz); indices=allocate('indices',nnz,index_dtype); indptr=allocate('indptr',rows+1,index_dtype)
    rl=allocate('row_lower',rows); ru=allocate('row_upper',rows)
    lo=allocate('lower_bounds',cols); up=allocate('upper_bounds',cols); c=allocate('c',cols); integ=allocate('integrality',cols,np.int64)
    a=first_stage.A.tocsr(); r0=a.shape[0]; k0=a.nnz
    data[:k0]=a.data; indices[:k0]=a.indices; indptr[:r0+1]=a.indptr
    rl[:r0]=first_stage.row_lower; ru[:r0]=first_stage.row_upper
    lo[:nfirst]=first_stage.lower_bounds; up[:nfirst]=first_stage.upper_bounds
    c[:nfirst]=first_stage.c; integ[:nfirst]=first_stage.integrality; integ[nfirst:]=0
    for item in metadata:
        _deadline(deadline)
        for ref in item['artifacts']:
            _need(file_reference(ref['path']) == ref, 'SAA shard changed during assembly')
        folder=shards/f"{item['index']:03d}"; a=load_npz(folder/'matrix.npz')
        c0,c1=item['column_slice']; r1=item['row_slice'][1]; k1=k0+a.nnz
        shifted=np.asarray(a.indices,dtype=index_dtype).copy(); mask=shifted>=nfirst; shifted[mask]+=c0-nfirst
        data[k0:k1]=a.data; indices[k0:k1]=shifted; indptr[r0+1:r1+1]=a.indptr[1:]+k0
        with np.load(folder/'arrays.npz',allow_pickle=False) as arrays:
            rl[r0:r1]=arrays['row_lower']; ru[r0:r1]=arrays['row_upper']
            lo[c0:c1]=arrays['lower_bounds']; up[c0:c1]=arrays['upper_bounds']; c[c0:c1]=arrays['c']
        r0,k0=r1,k1
    for array in (data,indices,indptr,rl,ru,lo,up,c,integ): array.flush()
    _deadline(deadline)
    manifest=dict(schema_version=1, algorithm_id=ALGORITHM_ID, risk_model='SAA', domain=domain,
        input_binding=input_binding, first_stage_sha256=first_stage.matrix_sha256,
        scenario_ids=list(ids), scenario_sha256s=list(hashes), empirical_probabilities=pi.tolist(),
        scenario_cost_constants_usd=constants.tolist(), objective_constant_usd=float(pi@constants),
        scenarios=metadata, shape=[rows,cols], nnz=int(nnz), coefficient_bytes=estimated,
        max_coefficient_bytes=max_coefficient_bytes, preparation_seconds=time.monotonic()-started,
        numeric_artifacts=[file_reference(p) for p in sorted(root.glob('*.npy'))],
        status='ASSEMBLED_PENDING_RESOURCE_AND_INDEPENDENT_NUMERICAL_AUDIT',
        solver_calls=0, scientific_spec_pass=False, formal_OOS=False,
        solver_peak_rss_measured=False, shared_first_stage_unchanged=True)
    manifest['manifest_sha256']=canonical_hash(manifest); _write(root/'manifest.json',manifest)
    matrix=csr_matrix((data,indices,indptr),shape=(rows,cols),copy=False)
    return StreamingSAA(matrix,rl,ru,lo,up,c,integ,float(pi@constants),first_stage,manifest,root)
