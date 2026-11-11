"""Explicit outward relaxation of tiny bounded-column row contributions.

The original model is not changed. This relaxation is suitable only for an
auxiliary minimization lower bound; it is not the exact original formulation.
No solver is called and no feasibility/certificate tolerance is changed.
"""
from fractions import Fraction
import math
import time

import numpy as np
from scipy.sparse import csr_matrix

from .random_streams import canonical_hash
from .relaxation_bound import coefficient_sha256


def _need(ok, message):
    if not bool(ok):
        raise ValueError(message)


def _exact(value):
    return Fraction.from_float(float(value))


def _fraction(value):
    return None if value is None else dict(numerator=str(value.numerator),denominator=str(value.denominator))


def _number(value):
    return float(value) if math.isfinite(value) else ('inf' if value>0 else '-inf')


def _outward(value, *, lower):
    """Enclose an exact rational on the safe side of a binary64 RHS."""
    try:
        rounded = float(value)
    except OverflowError as exc:
        raise ValueError('Relaxed RHS exceeds finite binary64 range') from exc
    _need(math.isfinite(rounded), 'Relaxed RHS exceeds finite binary64 range')
    represented = Fraction.from_float(rounded)
    if (lower and represented>value) or (not lower and represented<value):
        rounded = math.nextafter(rounded, -math.inf if lower else math.inf)
    _need(math.isfinite(rounded), 'Outward RHS exceeds finite binary64 range')
    return rounded


def _range(value, size, label):
    _need(isinstance(value,(tuple,list)) and len(value)==2
          and all(type(v) is int for v in value) and 0<=value[0]<=value[1]<=size,
          'Explicit half-open '+label+' range required')
    return tuple(value)


def relax_bounded_rows(model, *, allowed_rows, allowed_columns, threshold=1e-13, deadline=None,
                       bound_witnesses=None, witness_rows=None):
    """Remove only whitelisted tiny entries and conservatively relax their rows.

    For removed contribution t, each finite side uses exact bound arithmetic:
    L' = round_down(L - max(t)), U' = round_up(U - min(t)). A needed
    infinite column endpoint is rejected. Existing infinite row sides persist.
    Returned domain/objective arrays are the original objects; only matrix and
    row bounds are copied. Caller must independently audit and bind provenance.
    """
    def alive():
        if deadline is not None and (type(deadline) not in (float,int) or not math.isfinite(deadline)
                                     or time.monotonic()>=deadline):
            raise TimeoutError('Bounded row relaxation deadline expired')
    alive()
    _need(type(threshold) in (float,int) and threshold==1e-13, 'Tiny threshold must remain 1e-13')
    A=csr_matrix(model['matrix'],copy=True)
    _need(A.has_canonical_format and np.isfinite(A.data).all(), 'Canonical finite original CSR required')
    m,n=A.shape; ar=_range(allowed_rows,m,'row'); ac=_range(allowed_columns,n,'column')
    lower=np.asarray(model['lower_bounds'],dtype=float);upper=np.asarray(model['upper_bounds'],dtype=float)
    rl=np.asarray(model['row_lower'],dtype=float).copy();ru=np.asarray(model['row_upper'],dtype=float).copy()
    _need(lower.shape==upper.shape==(n,) and rl.shape==ru.shape==(m,), 'Original bound axes differ')
    _need(not any(np.isnan(v).any() for v in (lower,upper,rl,ru))
          and np.all(lower<=upper) and np.all(rl<=ru)
          and not np.isposinf(lower).any() and not np.isneginf(upper).any()
          and not np.isposinf(rl).any() and not np.isneginf(ru).any(), 'Invalid original bounds')
    effective_upper=upper.copy();witness_records=[]
    bound_witnesses={} if bound_witnesses is None else bound_witnesses
    _need(isinstance(bound_witnesses,dict), 'Bound witnesses must map columns to original row indices')
    wr=None
    if bound_witnesses:
        wr=_range(witness_rows,m,'witness row')
        _need(wr[1]<=ar[0], 'Bound witnesses must be in the unchanged original prefix')
    elif witness_rows is not None:
        wr=_range(witness_rows,m,'witness row')
    for column,source_row in sorted(bound_witnesses.items()):
        alive()
        _need(type(column) is int and ac[0]<=column<ac[1] and type(source_row) is int
              and wr[0]<=source_row<wr[1], 'Bound witness outside declared original prefix/columns')
        _need(math.isfinite(ru[source_row]), 'Upper-bound witness needs a finite upper row side')
        row=A.getrow(source_row);target=None;other_min=Fraction(0)
        for j,a in zip(row.indices,row.data):
            if j==column:
                target=float(a);continue
            if a==0:continue
            endpoint=lower[j] if a>0 else upper[j]
            _need(math.isfinite(endpoint), 'Bound witness needs an unbounded other-variable endpoint')
            other_min+=_exact(a)*_exact(endpoint)
        _need(target is not None and target>0, 'Upper-bound witness needs a positive target coefficient')
        exact_upper=(_exact(ru[source_row])-other_min)/_exact(target)
        derived_upper=_outward(exact_upper,lower=False)
        _need(derived_upper>=lower[column], 'Derived upper bound contradicts explicit column lower bound')
        effective_upper[column]=min(effective_upper[column],derived_upper)
        witness_records.append(dict(column=column,source_row=source_row,target_coefficient=target,
            upper_bound_exact=_fraction(exact_upper),upper_bound=derived_upper))
    removed=[];rows=[];drop=np.zeros(len(A.data),dtype=bool)
    for i in range(m):
        alive();start,stop=A.indptr[i:i+2]
        positions=np.flatnonzero((abs(A.data[start:stop])>0)&(abs(A.data[start:stop])<threshold))+start
        if not len(positions):
            continue
        _need(ar[0]<=i<ar[1], 'Tiny coefficient outside allowed rows')
        need_max=math.isfinite(rl[i]);need_min=math.isfinite(ru[i])
        total_min=Fraction(0) if need_min else None;total_max=Fraction(0) if need_max else None
        for p in positions:
            j=int(A.indices[p]);a=float(A.data[p])
            _need(ac[0]<=j<ac[1], 'Tiny coefficient outside allowed columns')
            min_endpoint=lower[j] if a>0 else effective_upper[j]
            max_endpoint=effective_upper[j] if a>0 else lower[j]
            _need(not need_min or math.isfinite(min_endpoint), 'Unbounded endpoint required for upper row side')
            _need(not need_max or math.isfinite(max_endpoint), 'Unbounded endpoint required for lower row side')
            tmin=_exact(a)*_exact(min_endpoint) if need_min else None
            tmax=_exact(a)*_exact(max_endpoint) if need_max else None
            if need_min:total_min+=tmin
            if need_max:total_max+=tmax
            removed.append(dict(row=i,column=j,coefficient=a,
                min_contribution_exact=_fraction(tmin),max_contribution_exact=_fraction(tmax)))
            drop[p]=True
        oldl,oldu=float(rl[i]),float(ru[i])
        if need_max:rl[i]=_outward(_exact(oldl)-total_max,lower=True)
        if need_min:ru[i]=_outward(_exact(oldu)-total_min,lower=False)
        rows.append(dict(row=i,original_lower=_number(oldl),original_upper=_number(oldu),
            relaxed_lower=_number(rl[i]),relaxed_upper=_number(ru[i]),
            min_sum_exact=_fraction(total_min),max_sum_exact=_fraction(total_max)))
    cumulative=np.r_[0,np.cumsum(drop)]
    A=csr_matrix((A.data[~drop],A.indices[~drop],A.indptr-cumulative[A.indptr]),shape=A.shape)
    result=dict(model,matrix=A,row_lower=rl,row_upper=ru)
    record=dict(kind='BOUNDED_TINY_ROW_RELAXATION_V1',
        original_coefficient_sha256=coefficient_sha256(model),relaxed_coefficient_sha256=coefficient_sha256(result),
        allowed_rows=list(ar),allowed_columns=list(ac),threshold=threshold,
        bound_witnesses=witness_records,witness_rows=None if wr is None else list(wr),
        removed_entries=removed,rows=rows,removed_entry_count=len(removed),changed_row_count=len(rows),
        objective_and_variable_domain_unchanged=True,original_model_modified=False,
        original_feasible_set_contained_in_relaxed=True,exact_original_formulation=False,
        bound_scope='Auxiliary minimization lower bound only; independent native bound audit required',
        endpoint_arithmetic='Exact binary64 values as rational products/sums; outward binary64 RHS rounding',
        solver_calls=0,scientific_acceptance=False,formal_OOS=False)
    record['relaxation_sha256']=canonical_hash(record)
    alive()
    return result,record
