"""Small result-handling helpers. No solver import, no change to model or cuts.

These are reference implementations, not a deployed patch to the remote runner.
The remote runner must integrate them into the actual call and evidence paths.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Callable, Mapping
import math
import os
import uuid
import numpy as np


def save_npz_immutable(root: Path, path: str | Path, /, **arrays: Any) -> Path:
    """Save named arrays without the path/p probability keyword collision.

    Positional-only root/path permit array keys such as 'p' and 'path'.
    Publication refuses overwrite, and temporary writes are kept off the final name.
    """
    base=Path(root).resolve(); target=(base/Path(path)).resolve()
    if base not in target.parents:
        raise ValueError('OUTPUT_OUTSIDE_ROOT')
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists(): raise FileExistsError(target)
    temp=target.with_name(target.name+'.tmp.'+uuid.uuid4().hex)
    try:
        with temp.open('xb') as f:
            np.savez_compressed(f,**arrays)
            f.flush();os.fsync(f.fileno())
        # Hard-link publication atomically refuses an already existing target.
        # A deployment must test filesystem support, rather than assuming it.
        os.link(temp,target)
        temp.unlink()
        fd=os.open(target.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:
        if temp.exists(): temp.unlink()
    return target


def read_optional_numeric(getter: Callable[[], Any], *, infinity: float = math.inf) -> dict[str,Any]:
    """Record unavailable results explicitly; never turn them into a valid bound.

    An ERROR record is diagnostic, not permission to treat arbitrary solver failure
    as success. The caller still handles the actual native state/exception.
    """
    try: raw=getter()
    except Exception as exc:
        return {'value':None,'classification':'UNAVAILABLE','error_type':type(exc).__name__,'error':str(exc)}
    if raw is None: return {'value':None,'classification':'UNAVAILABLE'}
    try: value=float(raw)
    except (TypeError,ValueError,OverflowError) as exc:
        return {'value':None,'classification':'INVALID_TYPE','error_type':type(exc).__name__}
    if math.isnan(value): return {'value':None,'classification':'NAN'}
    if not math.isfinite(value): return {'value':None,'classification':'POS_INF' if value>0 else 'NEG_INF'}
    if math.isfinite(infinity) and abs(value)>=infinity:
        return {'value':None,'classification':'SOLVER_POS_INFINITY' if value>0 else 'SOLVER_NEG_INFINITY'}
    return {'value':value,'classification':'FINITE'}


def lp_comparison_gate(a: Mapping[str,Any] | None,b: Mapping[str,Any] | None,improvement:float=1.0) -> dict[str,Any]:
    """Preserve the original numerical engineering gate, with a total return schema."""
    if a is None or b is None:
        return {'run_mip':False,'status':'LP_COMPARISON_INCOMPLETE','reason':'missing_certificate'}
    if a.get('numerical_gate') is not True:
        return {'run_mip':False,'status':'LP_COMPARISON_INCOMPLETE','reason':'baseline_primal_not_verified'}
    av=a.get('native_objective'); bv=b.get('LB')
    if av is None or bv is None or not math.isfinite(float(av)) or not math.isfinite(float(bv)):
        return {'run_mip':False,'status':'LP_COMPARISON_INCOMPLETE','reason':'finite_bounds_not_available'}
    gain=float(bv)-float(av)
    return {'run_mip':gain>improvement,
            'status':'NUMERICAL_GATE_PASSED' if gain>improvement else 'NO_ROOT_ADVANTAGE',
            'engineering_margin':gain,
            'note':'Numerical baseline-primal gate, not independently rigorous primal UB.'}


def certificate_record(*, native_status: Any, model_id: str,
                       status: str = 'INCOMPLETE', lower: float | None = None,
                       lower_verified: bool = False,
                       native_objective: float | None = None,
                       numerical_gate: bool = False,
                       primal_audit: Mapping[str, Any] | None = None,
                       strict_primal_upper: float | None = None,
                       reason: str | None = None) -> dict[str, Any]:
    """Total certificate schema; does not itself prove a lower bound or feasibility.

    The caller must set verified/gate flags ONLY after checking the real matrix,
    dual, primal and model identity. Missing results do not become zeros.
    """
    if not isinstance(model_id, str) or not model_id:
        raise ValueError('MODEL_ID_REQUIRED')
    if lower_verified and (lower is None or not math.isfinite(float(lower))):
        raise ValueError('VERIFIED_LOWER_MUST_BE_FINITE')
    if lower is not None and not lower_verified:
        raise ValueError('UNVERIFIED_VALUE_IS_NOT_A_CERTIFIED_LOWER')
    if numerical_gate and (native_objective is None or not math.isfinite(float(native_objective)) or primal_audit is None):
        raise ValueError('NUMERICAL_GATE_NEEDS_PRIMAL_EVIDENCE')
    for value in (native_objective, strict_primal_upper):
        if value is not None and not math.isfinite(float(value)):
            raise ValueError('NONFINITE_PRIMAL_FIELD')
    return dict(status=status, native_status=native_status, model_id=model_id,
                matrix_ref=None, LB=None if lower is None else float(lower),
                lower_certificate_verified=bool(lower_verified),
                numerical_gate=bool(numerical_gate),
                native_objective=None if native_objective is None else float(native_objective),
                primal_audit=None if primal_audit is None else dict(primal_audit),
                strict_primal_upper=None if strict_primal_upper is None else float(strict_primal_upper),
                reason=reason, raw_solution_ref=None, dual_ref=None, error_details=[])


def checked_lp_gate(a: Mapping[str, Any] | None,
                    b: Mapping[str, Any] | None,
                    improvement: float = 1.0) -> dict[str, Any]:
    """Engineering gate for this pilot. No native optimization or certificate creation."""
    incomplete = lambda why: dict(run_mip=False, status='LP_COMPARISON_INCOMPLETE', reason=why)
    if a is None or b is None:
        return incomplete('missing_certificate')
    if b.get('lower_certificate_verified') is not True or not b.get('model_id'):
        return incomplete('B_lower_not_verified')
    lb = b.get('LB')
    if lb is None or not math.isfinite(float(lb)):
        return incomplete('B_lower_not_finite')
    upper = a.get('strict_primal_upper')
    if upper is not None:
        mode = 'STRICT_PRIMAL_BOUND_GATE'
    elif a.get('numerical_gate') is True:
        upper = a.get('native_objective')
        mode = 'NUMERICAL_ENGINEERING_GATE'
    else:
        return incomplete('A_primal_not_verified')
    if upper is None or not math.isfinite(float(upper)):
        return incomplete('A_primal_not_finite')
    if not math.isfinite(improvement) or improvement < 0:
        raise ValueError('INVALID_GATE_MARGIN')
    advantage = float(lb) - float(upper)
    return dict(run_mip=advantage > improvement,
                status=mode if advantage > improvement else 'NO_ROOT_ADVANTAGE',
                margin=advantage, evidence_mode=mode,
                original_DRO_UB_created=False)


def safe_native_observations(model: Any, *, infinity: float) -> dict[str, Any]:
    """Inspect optional attributes without letting an unavailable one erase others.

    Does NOT classify mathematical success or replace the actual solver exception,
    and does not write END. Caller must preserve both solver and evidence errors.
    """
    fields = ('Status', 'SolCount', 'Runtime', 'Work', 'MaxMemUsed', 'IsMIP')
    output = {name: read_optional_numeric(lambda n=name: getattr(model, n), infinity=infinity)
              for name in fields}
    is_mip = output['IsMIP'].get('value')
    if is_mip == 1:
        for name in ('NodeCount', 'ObjBoundC', 'ObjBound'):
            output[name] = read_optional_numeric(lambda n=name: getattr(model, n), infinity=infinity)
    else:
        output['ObjBoundC'] = dict(value=None, classification='NOT_REQUESTED_FOR_LP_OR_UNKNOWN_TYPE')
    sol_count = output['SolCount'].get('value')
    if sol_count is not None and sol_count > 0:
        output['ObjVal'] = read_optional_numeric(lambda: getattr(model, 'ObjVal'), infinity=infinity)
    else:
        output['ObjVal'] = dict(value=None, classification='NO_READABLE_SOLUTION')
    return output
