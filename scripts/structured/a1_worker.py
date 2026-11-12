"""Remote A1 worker: ``python a1_worker.py OUT_DIR SECONDS [PREVIOUS_OUT]``.

OUT_DIR already exists outside Git. The parent owns the sole budget ledger and
charges import/build/solve/audit wall time. This worker caps its remaining allowance
at 1200 s, gives each native optimize <=60 s, and fails closed on non-optimal status.
It writes RESULT.json exactly once and returns nonzero unless all gates pass.

Scope: two deterministic, two-bus, 96-quarter-hour physical fixtures, not IEEE33,
full400, or OOS evidence. Port case has 3 available vehicles and 1 shared port.
Energy case has 2 vehicles, a healthy and a failed-route scene with fixed .8/.2
probabilities: physical energy binds in the healthy scene, shortage in the failed
scene. Both retain an unchosen alternative with different departure_dc. Three
INTEGER formulations are optimized per case: count, perspective individual, and
the independent ORIGINAL matrix.assemble per-vehicle-contract model. Fixed native
integer primals are audited, canonically expanded, and reverse mapped. All original
Big-M rows are retained. This script does not assert LP-relaxation equivalence.

No work occurs on import. Local development tests only construct coefficients and
map analytical witnesses; invoking main is reserved for the remote budget wrapper.
"""

import json
import hashlib
from pathlib import Path
import sys
import time


def build_cases():
    """Return explicit deterministic A1 inputs, shared by no-solver unit tests."""
    import numpy as np
    from ev_assignment.l08.physics import ebar

    net = dict(buses=[1, 2], station_buses=[2], **{"from": [1], "to": [2]},
               p_limit=[1000.], q_limit=[1000.], u_min=.9, u_max=1.1,
               u_ref=1., mp=[.00001], mq=[.00001], big_m=1., shed_cost=[10., 50.])

    def package(dc, d, k, r, arrival):
        return dict(g=0, c=0, d=d, k=k, r=r, departure_dc=dc, distance=10.,
                    arrival=arrival, Ebar=ebar(dc, 10., arrival, k, r))

    def scene(k, r, load):
        off = np.zeros((96, 1), bool)
        off[k:r + 1] = True
        p = np.zeros((96, 2))
        p[:, 0] = 10.
        p[k:r + 1, 1] = load
        return dict(line_operating_outages=off, load_p_kw=p, load_q_kvar=.33 * p)

    port = dict(name="port_binding", net=net, type_counts=[3], ports=1,
                packages=[package(300., 0, 4, 7, 1.), package(220., 1, 4, 7, 1.)],
                probabilities=[1.], scenes=[scene(4, 7, 180.)],
                route_states=[[dict(success=True, arrival=1.),
                               dict(success=False, arrival=24.)]])
    energy = dict(name="energy_shortfall_binding", net=net, type_counts=[2], ports=20,
                  packages=[package(200., 0, 4, 11, 1.), package(220., 4, 8, 15, 2.)],
                  probabilities=[.8, .2], scenes=[scene(4, 15, 250.), scene(4, 15, 250.)],
                  route_states=[[dict(success=True, arrival=1.),
                                 dict(success=False, arrival=24.)],
                                [dict(success=False, arrival=24.),
                                 dict(success=False, arrival=24.)]])
    return [port, energy]


def _jsonable(value):
    import numpy as np

    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_new(path, value):
    with path.open("x") as stream:
        json.dump(_jsonable(value), stream, indent=2, allow_nan=False)
        stream.write("\n")


def _assemble(first, templates, probabilities):
    import numpy as np
    from scipy import sparse as sp

    sizes = [len(t.data["c"]) for t in templates]
    nx = len(first["cost"])
    rec_W = sp.block_diag([t.W for t in templates], format="csr")
    A = sp.vstack([
        sp.hstack([first["A"], sp.csr_matrix((len(first["rhs"]), sum(sizes)))]),
        sp.hstack([sp.vstack([t.T for t in templates]), rec_W])], format="csr")
    return dict(A=A, rhs=np.concatenate([first["rhs"]] + [t.data["h"] for t in templates]),
                sense=np.concatenate([first["sense"]] + [t.data["sense"] for t in templates]),
                cost=np.concatenate([first["cost"]] +
                                    [p * t.data["c"] for p, t in zip(probabilities, templates)]),
                lo=np.concatenate([first["lo"]] + [t.data["lo"] for t in templates]),
                hi=np.concatenate([first["hi"]] + [t.data["hi"] for t in templates]),
                vtype=np.r_[first["vtype"], np.full(sum(sizes), "C")],
                nx=nx, sizes=sizes)


def _original_template(adapter):
    """Wrap original coefficients only; this does not call the catalogue builder."""
    import numpy as np
    from scipy import sparse as sp
    from types import SimpleNamespace

    m = adapter["matrix"]
    return SimpleNamespace(W=sp.vstack([m.A, m.H], format="csr"),
                           T=sp.vstack([m.B, m.G], format="csr"),
                           data=dict(h=np.r_[m.b, m.d], c=m.cost, lo=m.lo, hi=m.hi,
                                     sense=np.asarray(["<"] * len(m.b) + ["="] * len(m.d))))


def _optimize(data, env, remaining, events, label):
    """Audit raw integer primal; optional LP repair holds rounded integers fixed."""
    import gurobipy as gp
    import numpy as np
    from ev_assignment.l08.certificates import row_audit

    with gp.Model(label, env=env) as model:
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        model.Params.Seed = 0
        model.Params.FeasibilityTol = 1e-9
        model.Params.IntFeasTol = 1e-9
        model.Params.OptimalityTol = 1e-9
        model.Params.MIPGap = 0
        model.Params.MIPGapAbs = 1e-7
        model.Params.Method = 1
        model.Params.NodeMethod = 1
        v = model.addMVar(len(data["cost"]), lb=data["lo"], ub=data["hi"], vtype=data["vtype"])
        model.addMConstr(data["A"], v, data["sense"], data["rhs"])
        model.setObjective(data["cost"] @ v)
        model.Params.TimeLimit = min(60., remaining())
        events.append(dict(label=label, kind="native_begin"))
        model.optimize(lambda m, _: m.terminate() if remaining(False) <= 0 else None)
        native = dict(label=label, kind="native_end", native_status=int(model.Status),
                      runtime=float(model.Runtime), solution_count=int(model.SolCount))
        events.append(native)
        if model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError(f"{label}: native status {model.Status}; A1 not accepted")
        raw = np.asarray(v.X)
        objective, bound = float(model.ObjVal), float(model.ObjBound)
        check = row_audit(data["A"], data["rhs"], data["sense"], data["lo"], data["hi"], raw)
        discrete = data["vtype"] != "C"
        rounded = np.rint(raw[discrete])
        integer_residual = float(np.max(abs(raw[discrete] - rounded), initial=0))
        if integer_residual > 1e-9:
            raise ValueError("Native integer residual exceeds 1e-9")
        accepted = raw.copy()
        accepted[discrete] = rounded
        after = row_audit(data["A"], data["rhs"], data["sense"], data["lo"], data["hi"], accepted)
        repair = None
        if max(after.values()) > 1e-9:
            # No integer reoptimization: repair continuous variables in the exact
            # original rows, within 1e-7 of raw. Native status/bound remain recorded.
            v.VType = np.full(len(raw), "C")
            v.LB = np.maximum(data["lo"], raw - 1e-7)
            v.UB = np.minimum(data["hi"], raw + 1e-7)
            v[discrete].LB = rounded
            v[discrete].UB = rounded
            model.Params.TimeLimit = min(60., remaining())
            events.append(dict(label=label + "/repair", kind="native_begin"))
            model.optimize(lambda m, _: m.terminate() if remaining(False) <= 0 else None)
            events.append(dict(label=label + "/repair", kind="native_end",
                               native_status=int(model.Status), runtime=float(model.Runtime)))
            if model.Status != gp.GRB.OPTIMAL:
                raise ValueError("Continuous repair failed")
            accepted = np.asarray(v.X)
            after = row_audit(data["A"], data["rhs"], data["sense"], data["lo"], data["hi"], accepted)
            repair = dict(native_status=int(model.Status), before=check, after=after,
                          displacement=float(np.max(abs(accepted - raw), initial=0)))
            if (np.max(abs(accepted[discrete] - rounded), initial=0) > 1e-9 or
                    repair["displacement"] > 1e-7):
                raise ValueError("Continuous repair exceeded displacement/integer tolerance")
        if max(after.values()) > 1e-9:
            raise ValueError(("Original row tolerance", after))
        audited_objective = float(data["cost"] @ accepted)
        if abs(audited_objective - objective) > 1e-5 or audited_objective - bound > 1e-5:
            raise ValueError("Audited primal/objective bound gap exceeds 1e-5")
        offset = data["nx"]
        ys = []
        for size in data["sizes"]:
            ys.append(accepted[offset:offset + size])
            offset += size
        return dict(x=accepted[:data["nx"]], ys=ys, raw=raw, accepted=accepted,
                    native=native, native_objective=objective, native_bound=bound,
                    objective=audited_objective, row_audit=after,
                    raw_row_audit=check, integer_residual=integer_residual, repair=repair)


def _replay(data, saved, label):
    """Re-audit a saved native optimum; never optimize or replace its native bound."""
    import numpy as np
    from ev_assignment.l08.certificates import row_audit

    metadata, witnesses = saved
    result = dict(metadata["formulations"][label])
    if result["native"]["native_status"] != 2:
        raise ValueError("Reuse requires an original native OPTIMAL status")
    accepted = np.asarray(witnesses[label + "/accepted"], float)
    raw = np.asarray(witnesses[label + "/raw"], float)
    if (accepted.shape != data["lo"].shape or raw.shape != accepted.shape or
            not np.isfinite(accepted).all() or not np.isfinite(raw).all()):
        raise ValueError("Invalid previous port witness")
    check = row_audit(data["A"], data["rhs"], data["sense"], data["lo"], data["hi"], accepted)
    discrete = data["vtype"] != "C"
    if (max(check.values()) > 1e-9 or
            np.max(abs(accepted[discrete] - np.rint(accepted[discrete])), initial=0) > 1e-9):
        raise ValueError("Previous port witness failed current original-row audit")
    objective = float(data["cost"] @ accepted)
    if (abs(objective - result["objective"]) > 1e-5 or
            abs(objective - result["native_objective"]) > 1e-5 or
            abs(objective - result["native_bound"]) > 1e-5):
        raise ValueError("Previous port objective/bound mismatch")
    offset, ys = data["nx"], []
    for size in data["sizes"]:
        ys.append(accepted[offset:offset + size])
        offset += size
    result.update(x=accepted[:data["nx"]], ys=ys, raw=raw, accepted=accepted,
                  objective=objective, row_audit=check, reused_without_new_solve=True)
    return result


def _previous_port(previous, cases):
    """Require exact ALL-case inputs; read only port checkpoint and primal NPZ."""
    import numpy as np

    paths = [previous / "INPUTS.json", previous / "port_binding_CHECKPOINT.json",
             previous / "port_binding_WITNESSES.npz"]
    if json.loads(paths[0].read_text()) != _jsonable(cases):
        raise ValueError("Previous INPUTS differ: exact all-case inputs required")
    checkpoint = json.loads(paths[1].read_text())
    if checkpoint.get("name") != "port_binding" or checkpoint.get("passed") is not True:
        raise ValueError("Previous port checkpoint did not pass")
    with np.load(paths[2], allow_pickle=False) as loaded:
        witnesses = {key: loaded[key].copy() for key in loaded.files}
    refs = [dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]
    return (checkpoint, witnesses), refs


def _case(case, env, remaining, events, out, replay=None):
    import numpy as np
    from ev_assignment.l08 import catalogue as cat

    remaining()
    packages, net = case["packages"], case["net"]
    cf = cat.build_first(packages, case["type_counts"], len(net["station_buses"]), ports=case["ports"])
    inf = cat.build_first(packages, case["type_counts"], len(net["station_buses"]),
                         ports=case["ports"], individual=True)
    ct, it, adapters = [], [], []
    for scene, states in zip(case["scenes"], case["route_states"]):
        ct.append(cat.build_recourse(net, packages, scene, states, cf))
        it.append(cat.build_recourse(net, packages, scene, states, inf))
        adapters.append(cat.build_original_adapter(net, packages, scene, states, inf))
        remaining()
    original = [_original_template(a) for a in adapters]
    solved = {}
    for label, fs, templates in (("count", cf, ct), ("individual", inf, it),
                                 ("original", adapters[0]["first"], original)):
        data = _assemble(fs, templates, case["probabilities"])
        solved[label] = (_replay(data, replay, label) if replay is not None else
                         _optimize(data, env, remaining, events, case["name"] + "/" + label))
        solved[label]["size"] = dict(variables=data["A"].shape[1], rows=data["A"].shape[0],
                                     nnz=data["A"].nnz)
        remaining()
    objectives = [s["objective"] for s in solved.values()]
    if max(objectives) - min(objectives) > 1e-5:
        raise ValueError(("Integer objective mismatch", objectives))
    witnesses = {label + "/" + key: s[key] for label, s in solved.items()
                 for key in ("raw", "accepted")}
    maps = []
    canonical_x = None
    for sid, (c, i, adapter) in enumerate(zip(ct, it, adapters)):
        remaining()
        xc, yc = solved["count"]["x"], solved["count"]["ys"][sid]
        xi, yi = cat.expand_count(cf, inf, xc, yc, count_recourse=c, individual_recourse=i)
        if canonical_x is not None and not np.array_equal(xi, canonical_x):
            raise ValueError("Canonical first-stage contract changed across scenarios")
        canonical_x = xi.copy()
        zo, yo = cat.to_original(adapter, inf, i, xi, yi)
        xir, yir = cat.from_original(adapter, inf, i, zo, yo)
        xr, yr = cat.sum_individual(inf, cf, xir, yir, individual_recourse=i, count_recourse=c)
        if max(np.max(abs(xr - xc), initial=0), np.max(abs(yr - yc), initial=0)) > 1e-9:
            raise ValueError("Canonical count/original roundtrip exceeds 1e-9")
        # Independently optimized original and individual solutions need not be
        # canonical and need not have the same contracts. Audit both reverse maps.
        checks = []
        for label in ("original", "individual"):
            if label == "original":
                ax, ay = cat.from_original(adapter, inf, i, solved[label]["x"], solved[label]["ys"][sid])
            else:
                ax, ay = solved[label]["x"], solved[label]["ys"][sid]
                cat.to_original(adapter, inf, i, ax, ay)
            bx, by = cat.sum_individual(inf, cf, ax, ay, individual_recourse=i, count_recourse=c)
            delta = abs(float(inf["cost"] @ ax + i.data["c"] @ ay - cf["cost"] @ bx - c.data["c"] @ by))
            if delta > 1e-5:
                raise ValueError("Reverse-map objective mismatch")
            checks.append(dict(source=label, objective_delta=delta))
        old = adapter["matrix"]
        delta = abs(float(cf["cost"] @ xc + c.data["c"] @ yc -
                          adapter["first"]["cost"] @ zo - old.cost @ yo))
        if delta > 1e-5:
            raise ValueError("Canonical original objective mismatch")
        maps.append(dict(sid=sid, original_residual=old.residual(yo, zo),
                         objective_delta=delta, reverse=checks,
                         original_tasks=len(adapter["tasks"]),
                         original_big_m_rows=sum(n.startswith("incident_capability/")
                                                 for n in old.row_names["ub"])))
        for name, value in dict(count_x=xc, count_y=yc, individual_x=xi, individual_y=yi,
                                original_x=zo, original_y=yo).items():
            witnesses[f"canonical/{sid}/{name}"] = value
    xc, yc = solved["count"]["x"], solved["count"]["ys"][0]
    if case["name"] == "port_binding":
        rows = [i for i, n in enumerate(cf["row_names"]) if n.startswith("ports/")]
        slack = cf["rhs"] - cf["A"] @ xc
        binding = dict(port_slack=float(np.min(slack[rows])), selected=float(xc[:len(packages)].sum()))
        if abs(binding["port_slack"]) > 1e-9 or binding["selected"] < 1 - 1e-9:
            raise ValueError("Port fixture did not bind")
    else:
        t = ct[0]
        row = t.data["row_names"].index("discharge_physical/0")
        physical_slack = (t.data["h"][row] - t.W.getrow(row) @ yc - t.T.getrow(row) @ xc).item()
        shortage = float(solved["count"]["ys"][1][ct[1].data["index"]["short"]].sum())
        binding = dict(physical_slack=physical_slack, failed_scene_shortage=shortage)
        if abs(physical_slack) > 1e-9 or shortage <= 1e-9:
            raise ValueError("Energy/shortage fixture did not bind")
    np.savez_compressed(out / (case["name"] + "_WITNESSES.npz"), **witnesses)
    return dict(name=case["name"], passed=True, binding=binding, mappings=maps,
                formulations={label: {k: v for k, v in s.items() if k not in {"x", "ys", "raw", "accepted"}}
                              for label, s in solved.items()})


def main(argv=None):
    # stage.py charges from parent process start, including all imports above.
    started = time.monotonic()
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) not in (2, 3):
        raise ValueError("Usage: python a1_worker.py OUT_DIR SECONDS [PREVIOUS_OUT]")
    out, seconds = Path(argv[0]).resolve(), float(argv[1])
    if not 0 < seconds <= 1200 or not out.is_dir():
        raise ValueError("Require existing OUT_DIR and 0 < SECONDS <= 1200")
    if any((p / ".git").exists() for p in (out, *out.parents)):
        raise ValueError("A1 evidence must be outside Git")
    if (out / "RESULT.json").exists():
        raise FileExistsError("RESULT.json already exists")
    deadline = started + seconds

    def remaining(raise_on_expired=True):
        left = deadline - time.monotonic()
        if raise_on_expired and left <= 0:
            raise TimeoutError("A1 wall budget exhausted")
        return left

    result = dict(stage="A1", passed=False, status="FAILED", cases=[], events=[],
                  scope="two_bus_96_slot_integer_equivalence_only", full400=False,
                  oos_opened=False, row_tolerance=1e-9, objective_tolerance=1e-5)
    try:
        import gurobipy as gp
        from ev_assignment.l08 import catalogue, matrix, physics, energy, certificates, recourse

        cases = build_cases()
        result["core_sources"] = {
            m.__name__: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
            for m in (catalogue, matrix, physics, energy, certificates, recourse)}
        result["worker_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        replay = None
        if len(argv) == 3:
            previous = Path(argv[2]).resolve()
            if previous == out:
                raise ValueError("Previous output must differ from new output")
            replay, refs = _previous_port(previous, cases)
            result["reused_port"] = dict(references=refs, exact_all_case_inputs=True,
                                         current_core_sources=result["core_sources"],
                                         prior_core_identity="same frozen release per caller; witnesses reaudited",
                                         new_port_solver_calls=0)
        _write_new(out / "INPUTS.json", cases)
        remaining()
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            for case in cases:
                saved = replay if case["name"] == "port_binding" else None
                result["cases"].append(_case(case, env, remaining, result["events"], out, replay=saved))
                _write_new(out / (case["name"] + "_CHECKPOINT.json"), result["cases"][-1])
        remaining()
        result.update(passed=True, status="TINY_PHYSICAL_A1_PASS_NOT_FULL_SCALE")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        result["solver_calls"] = sum(e["kind"] == "native_begin" for e in result["events"])
        _write_new(out / "RESULT.json", result)
    print(json.dumps(dict(status=result["status"], passed=result["passed"])))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
