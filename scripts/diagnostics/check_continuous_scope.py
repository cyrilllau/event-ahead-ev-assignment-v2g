"""No-solve scope check for the inherited frozen93 continuous bound."""

from pathlib import Path
import sys
import time

import numpy as np
from scipy import sparse as sp

from ev_assignment_experiments.analysis.diagnostic import npz, read, reference, save, digest
from ev_assignment_experiments.analysis.diagnostic_replay import FIRST, DROOT


root, out = map(Path, sys.argv[1:])
start = time.monotonic()
history_path = root / DROOT / "ATTEMPT_05/HISTORICAL_LB_SOURCE.json"
history = read(history_path)
for ref in [history["certificate"], history["upstream"], history["prior_replay"]]:
    assert digest(ref["path"]) == ref["sha256"], ref["path"]
base = root / "L08_DRO_R44_POOLED_ROOT_CONTINUATION"
certpath = base / "AUTONOMOUS_CONVERGENCE/SOLVE_RECORDS/163_1/CERTIFICATE.json"
cert = read(certpath)
initial = base / "CONTINUE_01/LAST_SOLVED"
A = sp.load_npz(initial / "A.npz")
a = npz(initial / "ARRAYS.npz")
names = read(initial / "NAMES.json")
F = sp.load_npz(root / FIRST / "FIRST_A.npz")
f = npz(root / FIRST / "FIRST_ARRAYS.npz")
fn = read(root / FIRST / "FIRST_NAMES.json")
nr, nc = F.shape
checks = dict(
    first_matrix_equal=(A[:nr, :nc] != F).nnz == 0,
    first_columns_equal=names["columns"][:nc] == fn,
    active_base_prefix=cert["mapping"]["active"][:nr] == [f"base/{i}" for i in range(nr)],
    cost_equal=np.array_equal(a["cost"][:nc], f["cost"]),
    lower_equal=np.array_equal(a["lo"][:nc], f["lo"]),
    upper_equal=np.array_equal(a["hi"][:nc], f["hi"]),
    rhs_equal=np.array_equal(a["rhs"][:nr], f["rhs"]),
    sense_equal=np.array_equal(a["sense"][:nr], f["sense"]),
)
assert all(checks.values()), checks
# These are inherited native numerical evidence, not a new rational proof.
sources = read(history["upstream"]["path"])
save(
    out,
    dict(
        checks=checks,
        history=history,
        certificate_kind=cert.get("kind"),
        certificate_scalar_fields={
            k: v for k, v in cert.items() if isinstance(v, (str, int, float, bool)) or v is None
        },
        sources=sources,
        bound_grade="INHERITED_NATIVE_CONTINUOUS_BOUND_WITH_SOURCE_CHAIN",
        bound=12536.584767617334,
        use_current_domain=True,
        missing_reason="Upstream risk-cut proof chain inherited by exact source references, not independently regenerated in full in this round; numerical native level only.",
        inputs=[
            reference(p)
            for p in [
                history_path,
                certpath,
                initial / "A.npz",
                initial / "ARRAYS.npz",
                initial / "NAMES.json",
                root / FIRST / "FIRST_A.npz",
                root / FIRST / "FIRST_ARRAYS.npz",
            ]
        ],
        elapsed_readback_seconds=time.monotonic() - start,
    ),
)
print(checks, digest(out))
