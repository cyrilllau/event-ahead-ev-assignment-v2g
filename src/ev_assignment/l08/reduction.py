"""Exact PF elimination, preserving original row and column evidence."""
import hashlib
import json
import numpy as np
from scipy import sparse as sp
from .matrix import Matrix

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def identity_map(mat, reason):
    n = len(mat.cost)
    return (mat, dict(applicable=False, reason=reason, S=sp.eye(n, format='csr'), keep=np.arange(n), dropped_eq=[], original_columns=n, reduced_columns=n))

def pf_reduce(mat, net, scene, rho=0.33):
    if list(net['from']) != list(net['line_from']) or list(net['to']) != list(net['line_to']):
        return identity_map(mat, 'NOT_APPLICABLE_ENDPOINT_METADATA_MISMATCH')
    buses = net['buses']
    nb = len(buses)
    nl = len(net['line_ids'])
    bi = {b: i for i, b in enumerate(buses)}
    if rho != 0.33 or nl != nb - 1 or 1 not in bi:
        return identity_map(mat, 'NOT_APPLICABLE_NONRADIAL_OR_RHO')
    if any((u == v for u, v in zip(net['line_from'], net['line_to']))):
        return identity_map(mat, 'NOT_APPLICABLE_SELF_LOOP')
    incidence = np.zeros((nb, nl + 1))
    for j, (u, v) in enumerate(zip(net['line_from'], net['line_to'])):
        incidence[bi[u], j] -= 1
        incidence[bi[v], j] += 1
    incidence[bi[1], -1] = 1
    if np.linalg.matrix_rank(incidence) != nb:
        return identity_map(mat, 'NOT_APPLICABLE_INCIDENCE_NULLSPACE')
    if not np.array_equal(np.asarray(scene['load_q_kvar']), rho * np.asarray(scene['load_p_kw'])):
        return identity_map(mat, 'NOT_APPLICABLE_NONUNIFORM_LOAD_PF')
    q = np.r_[mat.index['line_q'].ravel(), mat.index['grid_q'].ravel()].astype(int)
    p = np.r_[mat.index['line_p'].ravel(), mat.index['grid_p'].ravel()].astype(int)
    keep = np.setdiff1d(np.arange(len(mat.cost)), q)
    oldnew = np.full(len(mat.cost), -1, int)
    oldnew[keep] = np.arange(len(keep))
    S = sp.coo_matrix((np.r_[np.ones(len(keep)), np.full(len(q), rho)], (np.r_[keep, q], np.r_[np.arange(len(keep)), oldnew[p]])), shape=(len(mat.cost), len(keep))).tocsr()
    hs = mat.H @ S
    names = mat.row_names['eq']
    lookup = {n: i for i, n in enumerate(names)}
    drop = []
    for i, n in enumerate(names):
        if n.startswith('Q_balance/'):
            pn = 'P_balance/' + n[len('Q_balance/'):]
            if pn not in lookup:
                return identity_map(mat, 'NOT_APPLICABLE_MISSING_P_ROW')
            j = lookup[pn]
            delta = hs[i] - rho * hs[j]
            zg = mat.G[i] - rho * mat.G[j]
            if np.any(delta.data != 0) or np.any(zg.data != 0) or mat.d[i] != rho * mat.d[j]:
                return identity_map(mat, 'NOT_APPLICABLE_REACTIVE_BALANCE_NOT_REDUNDANT')
            drop.append(i)
    if len(drop) != 96 * nb:
        return identity_map(mat, 'NOT_APPLICABLE_BALANCE_COUNT')
    lo = mat.lo[keep].copy()
    hi = mat.hi[keep].copy()
    for oq, op in zip(q, p):
        j = oldnew[op]
        lo[j] = max(lo[j], mat.lo[oq] / rho)
        hi[j] = min(hi[j], mat.hi[oq] / rho)
    if np.any(lo > hi):
        return identity_map(mat, 'NOT_APPLICABLE_EMPTY_BOUND_INTERSECTION')
    eqkeep = np.setdiff1d(np.arange(len(mat.d)), drop)
    md = dict(mat.metadata, representation='R7_PF_EXACT', rho=rho, original_rows_sha=digest(mat.row_names))
    red = Matrix(np.asarray(S.T @ mat.cost).ravel(), np.asarray(mat.parts @ S), mat.A @ S, mat.B, mat.b, hs[eqkeep], mat.G[eqkeep], mat.d[eqkeep], lo, hi, {'retained_original_columns': keep}, dict(ub=mat.row_names['ub'], eq=[names[i] for i in eqkeep]), md)
    return (red, dict(applicable=True, reason='TREE_UNIFORM_PF_REDUNDANCY_PROVED', S=S, keep=keep, q_columns=q, p_columns=p, dropped_eq=drop, original_columns=len(mat.cost), reduced_columns=len(keep), bound_intersections=len(q), voltage_coefficients=(np.asarray(net['mp']) + rho * np.asarray(net['mq'])).tolist(), rho=rho))

def lift(y, mapping):
    return np.asarray(mapping['S'] @ y).ravel()
