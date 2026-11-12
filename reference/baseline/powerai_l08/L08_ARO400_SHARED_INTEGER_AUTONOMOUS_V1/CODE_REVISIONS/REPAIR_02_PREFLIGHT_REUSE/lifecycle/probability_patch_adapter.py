"""Side-effect-isolated adapter for the existing R42/R44 moment/check functions.
This file does not import or execute the server driver.  All dependencies and
owned output paths must be supplied explicitly by the calling R44 attempt.
"""
from __future__ import annotations
import json,os
from pathlib import Path
from typing import Callable,Mapping,Any,Sequence
import numpy as np
from safe_probability_certificate import certify_with_feasible_anchor

def nontrivial_margin_hint(dd: Mapping[str,Any], p0: np.ndarray, enclosure: Callable) -> float | None:
    """Restore the R42 masks for a step-size hint ONLY, not for validation."""
    raw=dd['raw_unique'] if 'raw_unique' in dd else dd['raw']
    a=dd['a']
    rl,rh=enclosure(raw,p0); sl,sh=enclosure(a['phi'],p0)
    choices=[(rl-dd['lower'])[dd['lower']>dd['raw_min']],
             (dd['upper']-rh)[dd['upper']<dd['raw_max']],
             (sl-a['lower'])[a['lower']>dd['scaled_min']],
             (a['upper']-sh)[a['upper']<dd['scaled_max']]]
    minima=[float(np.min(x)) for x in choices if len(x)]
    hint=min(minima) if minima else None
    return hint if hint is not None and np.isfinite(hint) and hint>0 else None

def certify_candidate_compat(p: Sequence[float],qbars: Sequence[float],ids: Sequence[int],out: Path,
    *,p0: np.ndarray,dd: Mapping[str,Any],check: Callable,enclosure: Callable,owned_root: Path):
    """Return the existing R44 record schema including epsilon; atomic evidence.

    Integrate only after saving the raw probability/dual snapshot.  The caller
    must confirm theta lower bounds >=0, theta upper bounds <=qbars, original
    moment identity and original-anchor SHA.  This function never changes them.
    """
    out=Path(out);root=Path(owned_root).resolve();target=out.resolve()
    if target==root or root not in target.parents:
        raise ValueError('STRICT_CERTIFICATE_OUTPUT_OUTSIDE_OWNED_SUBDIRECTORY')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('STRICT_CERTIFICATE_OUTPUT_ALREADY_NONEMPTY')
    hint=nontrivial_margin_hint(dd,np.asarray(p0,float),enclosure)
    star,rec=certify_with_feasible_anchor(p,p0,qbars,ids,check,nontrivial_margin_hint=hint)
    out.mkdir(parents=True,exist_ok=True)
    temp=out/'PROBABILITY.npz.tmp'
    with temp.open('xb') as f:
        np.savez_compressed(f,p=np.asarray(p,float),pstar=star,Qbar=np.asarray(qbars,float),ids=np.asarray(ids,int))
        f.flush();os.fsync(f.fileno())
    os.replace(temp,out/'PROBABILITY.npz')
    temp=out/'CERTIFICATE.json.tmp'
    with temp.open('x') as f:
        f.write(json.dumps(rec,indent=2,sort_keys=True,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,out/'CERTIFICATE.json')
    return rec
