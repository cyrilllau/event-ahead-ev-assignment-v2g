"""Exact zero-column propagation for the original extensive MILP.

Only bounds equal to zero and homogeneous singleton rows are used. There is
no small-coefficient threshold, rounding, task deletion or incumbent fixing.
The solver vector is expanded and checked against every ORIGINAL constraint.
"""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from .random_streams import canonical_hash


@dataclass(frozen=True)
class ZeroReduction:
    matrix: object
    row_indices: np.ndarray
    column_indices: np.ndarray
    zero_columns: np.ndarray
    proof_steps: tuple
    original_columns: int

    def expand(self, vector):
        value = np.asarray(vector, dtype=float)
        if value.shape != self.column_indices.shape:
            raise ValueError('Reduced solution shape mismatch')
        full = np.zeros(self.original_columns)
        full[self.column_indices] = value
        return full

    def as_dict(self):
        body = dict(kind='exact_zero_singleton_v1', zero_columns=self.zero_columns.tolist(),
                    retained_rows=self.row_indices.tolist(), retained_columns=self.column_indices.tolist(),
                    proof_steps=list(self.proof_steps), original_columns=self.original_columns,
                    reduced_shape=list(self.matrix.shape), reduced_nnz=self.matrix.nnz,
                    coefficient_threshold=None, original_constraint_check_required=True)
        return {**body, 'sha256': canonical_hash(body)}


def reduce_zero_columns(matrix, row_lower, row_upper, lower, upper):
    a = csr_matrix(matrix, copy=True)
    a.sum_duplicates(); a.eliminate_zeros(); a.sort_indices()
    lo, hi = np.asarray(lower), np.asarray(upper)
    rl, ru = np.asarray(row_lower), np.asarray(row_upper)
    if (lo.shape != hi.shape or a.shape != (len(rl), len(lo)) or ru.shape != rl.shape
            or np.isnan(lo).any() or np.isnan(hi).any() or np.any(lo > hi)):
        raise ValueError('Invalid zero-reduction dimensions or bounds')
    zero = (lo == 0) & (hi == 0)
    proof = [{'kind': 'original_zero_bounds', 'columns': np.flatnonzero(zero).tolist()}]
    while True:
        cols = np.flatnonzero(~zero)
        b = a[:, cols].tocsr()
        rows = np.flatnonzero(np.diff(b.indptr) == 1)
        if not len(rows):
            break
        slots = b.indptr[rows]
        variables, coeff = cols[b.indices[slots]], b.data[slots]
        # a*x <= 0 or a*x >= 0, intersected with an ORIGINAL zero bound.
        force_upper_zero = ((coeff > 0) & (ru[rows] == 0)) | ((coeff < 0) & (rl[rows] == 0))
        force_lower_zero = ((coeff > 0) & (rl[rows] == 0)) | ((coeff < 0) & (ru[rows] == 0))
        upper_zero, lower_zero = hi == 0, lo == 0
        upper_zero[variables[force_upper_zero]] = True
        lower_zero[variables[force_lower_zero]] = True
        new = np.flatnonzero(upper_zero & lower_zero & ~zero & (lo <= 0) & (hi >= 0))
        if not len(new):
            break
        proof.append({'kind': 'homogeneous_singleton', 'columns': new.tolist(),
                      'witness_rows': rows[np.isin(variables, new) & (force_upper_zero | force_lower_zero)].tolist()})
        zero[new] = True
    # A contradictory constant row is deliberately retained for the solver.
    empty = np.diff(b.indptr) == 0
    keep = ~(empty & (rl <= 0) & (ru >= 0))
    return ZeroReduction(b[keep].tocsr(), np.flatnonzero(keep), cols,
                         np.flatnonzero(zero), tuple(proof), len(lo))
