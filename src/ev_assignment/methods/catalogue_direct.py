"""Independent extensive P3-DRO comparator on the same finite catalogue.

Assembly is solver-free. Only solve_direct invokes Gurobi, lazily, in one model
with native slices <=60 s. No Benders controller, cuts, IO or implicit data roots.
Native incumbents are retained as native evidence, NEVER as audited integer UBs.
The caller's Control charges construction, native solves and complete-support
recourse/risk/original-matrix audits. This is the new three-moment domain, not the
old 322311-row ambiguity set and not a claim about LP representation equivalence.
"""

from copy import deepcopy
import time

import numpy as np
from scipy import sparse as sp

from ev_assignment.domain.planning import Control, SolverOptions
from ev_assignment.l08 import catalogue as cat, recourse, risk
from ev_assignment.l08.builders import parameters
from ev_assignment.l08.certificates import enclosure, row_audit


def _domain(ambiguity, support):
    if tuple(support) != ambiguity.support_ids or ambiguity.phi.shape != (3, len(support)):
        raise ValueError("Direct P3 requires identical ordered support and exactly three moments")
    if (ambiguity.feasible_anchor is None or
            not ambiguity.check_original(ambiguity.feasible_anchor).get("passed")):
        raise ValueError("Direct moment dual requires a certified nonempty probability domain")


def assemble_direct(first, templates, ambiguity):
    """Return CSR arrays for [x, beta0, plus[3], minus[3], y_0, ..., y_S-1].

    min F(x)+beta0+upper@plus-lower@minus, with beta0 FREE, plus/minus >=0.
    Each final row is c_s@y_s-beta0-phi_s@plus+phi_s@minus <=0. Every original
    first/recourse row and bound is retained. No nominal scenario weights enter
    this dual epigraph. index['y'] gives half-open (start, stop) column pairs.
    """
    _domain(ambiguity, ambiguity.support_ids)
    ns, nx, m = len(templates), len(first["cost"]), len(ambiguity.lower)
    if ns != len(ambiguity.support_ids) or ns == 0:
        raise ValueError("Missing full-support recourse blocks")
    for template in templates:
        if (template.T.shape != (template.W.shape[0], nx) or
                template.W.shape[1] != len(template.data["c"]) or
                not np.array_equal(template.data["xlo"], first["lo"]) or
                not np.array_equal(template.data["xhi"], first["hi"])):
            raise ValueError("Recourse block/first-stage domain mismatch")
    sizes = [len(t.data["c"]) for t in templates]
    ny, nd = sum(sizes), 1 + 2 * m
    W = sp.block_diag([t.W for t in templates], format="csr")
    T = sp.vstack([t.T for t in templates], format="csr")
    Q = sp.block_diag([sp.csr_matrix(t.data["c"][None, :]) for t in templates], format="csr")
    dual_rows = sp.hstack([-np.ones((ns, 1)), -ambiguity.phi.T, ambiguity.phi.T], format="csr")
    A = sp.vstack([
        sp.hstack([first["A"], sp.csr_matrix((len(first["rhs"]), nd + ny))]),
        sp.hstack([T, sp.csr_matrix((W.shape[0], nd)), W]),
        sp.hstack([sp.csr_matrix((ns, nx)), dual_rows, Q])], format="csr")
    offset, blocks = nx + nd, []
    for size in sizes:
        blocks.append((offset, offset + size))
        offset += size
    names = (tuple(first["names"]) + ("risk/beta0",) +
             tuple(f"risk/plus/{k}" for k in range(m)) +
             tuple(f"risk/minus/{k}" for k in range(m)) +
             tuple(f"scene/{s}/{name}" for s, t in enumerate(templates) for name in t.names))
    row_names = (tuple(first["row_names"]) +
                 tuple(f"scene/{s}/{name}" for s, t in enumerate(templates)
                       for name in t.data["row_names"]) +
                 tuple(f"risk/epigraph/{s}" for s in range(ns)))
    return dict(A=A,
                rhs=np.concatenate([first["rhs"], *[t.data["h"] for t in templates], np.zeros(ns)]),
                sense=np.concatenate([first["sense"], *[t.data["sense"] for t in templates], np.full(ns, "<")]),
                cost=np.r_[first["cost"], 1., ambiguity.upper, -ambiguity.lower, np.zeros(ny)],
                lo=np.concatenate([first["lo"], [-np.inf], np.zeros(2 * m),
                                   *[t.data["lo"] for t in templates]]),
                hi=np.concatenate([first["hi"], np.full(nd, np.inf),
                                   *[t.data["hi"] for t in templates]]),
                vtype=np.r_[first["vtype"], np.full(nd + ny, "C")],
                names=names, row_names=row_names, objcon=0.,
                index=dict(x=(0, nx), beta0=nx, plus=(nx + 1, nx + 1 + m),
                           minus=(nx + 1 + m, nx + nd), y=tuple(blocks)))


def build_direct(case, ambiguity, *, weights=(1, 1, 1, 1), individual=False,
                 control=None, event=lambda row: None):
    """Build all physical blocks and CSR extensive arrays; no optimizer is called."""
    _domain(ambiguity, [s["identity"] for s in case["scenarios"]])
    if control is not None:
        control.check()
    first = cat.build_first(case["packages"], case["type_counts"],
                            len(case["network"]["station_buses"]), weights,
                            ports=case.get("ports", 20), individual=individual)
    templates = []
    for sid, scene in enumerate(case["scenarios"]):
        if control is not None:
            control.check()
        templates.append(cat.build_recourse(case["network"], case["packages"],
                         scene["physical"], scene["route_states"], first))
        event(dict(stage="build", completed=sid + 1, total=len(case["scenarios"])))
    arrays = assemble_direct(first, templates, ambiguity)
    if control is not None:
        control.check()
    return dict(first=first, templates=tuple(templates), arrays=arrays)


def canonical_integer(first, raw):
    """Audit at 1e-9, snap integers, clip contracts, record <=1e-7 displacement.

    This creates a separately evaluated feasible policy; it does not relabel the
    native incumbent or its objective. No continuous optimization/repair occurs.
    """
    raw = np.asarray(raw, float)
    if raw.shape != first["lo"].shape or not np.isfinite(raw).all():
        raise ValueError("Invalid native first-stage vector")
    mask = first["vtype"] != "C"
    integer_residual = float(np.max(abs(raw[mask] - np.rint(raw[mask])), initial=0))
    before = row_audit(first["A"], first["rhs"], first["sense"], first["lo"], first["hi"], raw)
    if integer_residual > 1e-9 or max(before.values()) > 1e-9:
        raise ValueError("Native first stage exceeds original integer/row tolerance")
    x = raw.copy()
    H = len(first["package_map"])
    x[:H] = np.rint(x[:H])
    x[H:] = np.clip(x[H:], 15 * x[:H],
                    np.asarray([p["Ebar"] for p in first["packages"]]) * x[:H])
    after = row_audit(first["A"], first["rhs"], first["sense"], first["lo"], first["hi"], x)
    displacement = float(np.max(abs(x - raw), initial=0))
    if max(after.values()) > 1e-9 or displacement > 1e-7:
        raise ValueError("Canonical policy exceeds original row/displacement tolerance")
    return x, dict(raw=raw.tolist(), integer_residual=integer_residual, before=before,
                   after=after, displacement=displacement, reoptimized_recourse=True)


def _slice_control(control, seconds=60.):
    control.check()
    return Control(min(seconds, control.remaining),
                   stop=lambda: control.stop() or control.remaining <= 0, event=control.event)


def certify_incumbent(case, ambiguity, built, raw_x, control, *, env,
                      event=lambda row: None, witness=lambda *args: None):
    """Complete-support integer audit; incomplete/failed audits return UB=None.

    Witness protocol matches catalogue.solve_structured: (kind, phase, iteration,
    sid, x, answer), kinds recourse/original_audit/risk. Recourse UBs retain their
    NUMERICALLY_AUDITED_ONLY status; the risk upper uses the existing dual guard.
    """
    first, templates = built["first"], built["templates"]
    out = dict(UB=None, best=None, status="AUDIT_INCOMPLETE", original_audit=[],
               original_expansion_audited=False, best_recourse=[], best_risk_certificate=None,
               coverage=dict(expected=len(templates), recourse=0, original=0, risk=False),
               timing=dict(recourse=0., probability=0., audit=0.))
    try:
        _domain(ambiguity, [s["identity"] for s in case["scenarios"]])
        control.check()
        started = time.monotonic()
        x, out["canonicalization"] = canonical_integer(first, raw_x)
        fi = first if first["individual"] else cat.build_first(case["packages"], case["type_counts"],
              len(case["network"]["station_buses"]), first["weights"], ports=first["ports"], individual=True)
        fixed_contracts = None
        out["timing"]["audit"] += time.monotonic() - started
        options = SolverOptions(seconds=60, threads=1, seed=101, absolute_gap=1e-7,
                                soft_memory_gb=20)
        for sid, (scene, template) in enumerate(zip(case["scenarios"], templates)):
            control.check()
            started = time.monotonic()
            try:
                result = recourse.solve(template, x, options, _slice_control(control), env=env)
            finally:
                out["timing"]["recourse"] += time.monotonic() - started
            # Existing recourse.solve calls recourse.audit; preserve its full proof.
            out["coverage"]["recourse"] += 1
            witness("recourse", "integer", 0, sid, x, result)
            record = dict(sid=scene["identity"], qL=result["qL"], qU=result["qU_audited"],
                          qbar=template.qbar, residual=result["residual"],
                          native_status=result["native_status"])
            out["best_recourse"].append(record)
            started = time.monotonic()
            try:
                ti = template if first["individual"] else cat.build_recourse(case["network"],
                     case["packages"], scene["physical"], scene["route_states"], fi)
                xi, yi = ((x, result["y"]) if first["individual"] else
                          cat.expand_count(first, fi, x, result["y"],
                                           count_recourse=template, individual_recourse=ti))
                if fixed_contracts is None:
                    fixed_contracts = xi.copy()
                elif not np.array_equal(fixed_contracts, xi):
                    raise ValueError("Canonical contracts changed across scenarios")
                adapter = cat.build_original_adapter(case["network"], case["packages"],
                          scene["physical"], scene["route_states"], fi)
                zo, yo = cat.to_original(adapter, fi, ti, xi, yi)
                old = adapter["matrix"]
                delta = old.parts @ yo - template.data["parts"] @ result["y"]
                fd = float(adapter["first"]["cost"] @ zo - first["cost"] @ x)
                if max(np.max(abs(delta), initial=0), abs(fd)) > 1e-5:
                    raise ValueError("Original expansion changed objective components")
                check = dict(sid=scene["identity"], residual=old.residual(yo, zo),
                             operating_component_delta=delta.tolist(), first_cost_delta=fd,
                             canonical_contracts_unchanged=True, original_tasks=len(adapter["tasks"]),
                             original_shortfall_big_m_rows=sum(n.startswith("incident_capability/")
                                                              for n in old.row_names["ub"]),
                             original_size=dict(vars=len(old.cost), rows=old.A.shape[0] + old.H.shape[0],
                                                nnz=old.A.nnz + old.B.nnz + old.H.nnz + old.G.nnz))
                witness("original_audit", "integer", 0, sid, zo, dict(y=yo, audit=check))
                out["original_audit"].append(check)
                out["coverage"]["original"] += 1
            finally:
                out["timing"]["audit"] += time.monotonic() - started
            control.check()
            event(deepcopy(out))
        if out["coverage"]["original"] != len(ambiguity.support_ids):
            raise ValueError("Incomplete original support coverage")
        records = out["best_recourse"]
        started = time.monotonic()
        try:
            rr = risk.solve(ambiguity, [r["qL"] for r in records], [r["qU"] for r in records],
                            [r["qbar"] for r in records], _slice_control(control))
        finally:
            out["timing"]["probability"] += time.monotonic() - started
        if (not ambiguity.check_original(rr["pstar"]).get("passed") or
                tuple(rr["all_support_ids"]) != ambiguity.support_ids or
                not np.isfinite([rr["risk_L"], rr["risk_U"]]).all() or
                rr["risk_L"] > rr["risk_U"]):
            raise ValueError("Invalid complete-support risk certificate")
        _, upper = enclosure(sp.csr_matrix(first["cost"][None, :]), x)
        F = float(np.nextafter(float(upper[0]), np.inf))
        value = float(np.nextafter(F + rr["risk_U"], np.inf))
        witness("risk", "integer", 0, -1, x, dict(certificate=rr, Q=records,
                ambiguity_identity=ambiguity.identity, first_cost_upper=F, full_objective_upper=value))
        control.check()
        out.update(UB=value, best=x.tolist(), best_risk_certificate=rr,
                   risk_interval=[rr["risk_L"], rr["risk_U"]], first_cost_upper=F,
                   original_expansion_audited=True, status="FULL_SUPPORT_INTEGER_AUDIT_PASS")
        out["coverage"]["risk"] = True
    except (TimeoutError, RuntimeError, InterruptedError, ValueError) as exc:
        out.update(status="AUDIT_FAILED" if isinstance(exc, ValueError) else "AUDIT_INCOMPLETE",
                   reason=f"{type(exc).__name__}: {exc}")
    return out


def solve_direct(case, ambiguity, control, *, weights=(1, 1, 1, 1), individual=False,
                 optimize_seconds=240., audit_reserve_seconds=30.,
                 event=lambda row: None, witness=lambda *args: None):
    """Run one same-case extensive MIP, then independently certify its best policy.

    All native calls are <=60 s, resume the SAME in-memory model, and share the
    caller's wall budget. optimize_seconds caps cumulative native wall time;
    audit_reserve_seconds is kept out of native solving, not extra authorization.
    Native MIP LB/status/objective are preserved even if audit times out. Native
    objective is never copied into UB. passed uses the original max($1,1%) gate;
    objective_optimality_proven additionally requires audited UB-LB <=1e-5.
    event receives result snapshots. Witness kinds match the existing controller;
    phase native_direct marks native recourse witnesses as UNAUDITED incumbents.
    """
    if (not np.isfinite([optimize_seconds, audit_reserve_seconds]).all() or
            optimize_seconds <= 0 or audit_reserve_seconds < 0):
        raise ValueError("Finite positive solve budget and nonnegative audit reserve required")
    begun = time.monotonic()
    state = dict(method="P3-DRO", algorithm="direct_finite_scenario_moment_dual",
                 science_identity=case["identity"], ambiguity_identity=ambiguity.identity,
                 first_stage_weights=list(weights), ordered_support=list(ambiguity.support_ids),
                 probability_domain=dict(phi=ambiguity.phi.toarray().tolist(), lower=ambiguity.lower.tolist(),
                                         upper=ambiguity.upper.tolist(), feature_rows=3, old_rich_set=False),
                 representation="individual" if individual else "count", LB=None, UB=None,
                 best=None, native_incumbent=None, native_status=None, gap=None, passed=False,
                 status="BUILDING", original_audit=[], original_expansion_audited=False,
                 objective_optimality_proven=False, full_support=len(ambiguity.support_ids),
                 coverage=dict(expected=len(ambiguity.support_ids), built=0, recourse=0, original=0, risk=False),
                 timing=dict(build=0., native=0., recourse=0., probability=0., audit=0.), history=[],
                 lower_evidence="native_solver_full_extensive_model",
                 upper_evidence="NUMERICALLY_AUDITED_RECOURSE_PLUS_CERTIFIED_RISK_DUAL")

    def progress():
        state["elapsed_seconds"] = time.monotonic() - begun
        event(deepcopy(state))

    def built_progress(row):
        state["coverage"]["built"] = row["completed"]
        progress()

    progress()
    try:
        started = time.monotonic()
        try:
            built = build_direct(case, ambiguity, weights=weights, individual=individual,
                                 control=control, event=built_progress)
        finally:
            state["timing"]["build"] += time.monotonic() - started
        a, first = built["arrays"], built["first"]
        state["model_size"] = dict(vars=a["A"].shape[1], rows=a["A"].shape[0], nnz=a["A"].nnz,
                                   integer_vars=int(np.count_nonzero(a["vtype"] != "C")),
                                   first_vars=len(first["cost"]), scenarios=len(built["templates"]))
        import gurobipy as gp

        options = SolverOptions(seconds=60, threads=4, seed=101, mip_gap=.01,
                                absolute_gap=1., soft_memory_gb=20, master_seconds=60)
        raw_x = None
        with gp.Env(params={"OutputFlag": 0}) as env, gp.Model("P3_DIRECT_EXTENSIVE", env=env) as model:
            started = time.monotonic()
            v = model.addMVar(len(a["cost"]), lb=a["lo"], ub=a["hi"], vtype=a["vtype"])
            model.addMConstr(a["A"], v, a["sense"], a["rhs"])
            model.setObjective(a["cost"] @ v)
            model.update()
            state["timing"]["build"] += time.monotonic() - started
            available = min(optimize_seconds, max(0., control.remaining - audit_reserve_seconds))
            deadline = time.monotonic() + available
            iteration = 0
            state["status"] = "NATIVE_SOLVING"
            while time.monotonic() < deadline and control.remaining > audit_reserve_seconds:
                control.check()
                seconds = min(60., deadline - time.monotonic(), control.remaining - audit_reserve_seconds)
                if seconds <= 0:
                    break
                parameters(model, options, seconds)
                control.event(dict(kind="native_begin", stage="direct", iteration=iteration, seconds=seconds))
                started = time.monotonic()
                try:
                    model.optimize(lambda mm, _: mm.terminate() if (control.stop() or
                                   control.remaining <= audit_reserve_seconds or time.monotonic() >= deadline) else None)
                finally:
                    state["timing"]["native"] += time.monotonic() - started
                native = dict(iteration=iteration, native_status=int(model.Status), runtime=float(model.Runtime),
                              node_count=float(model.NodeCount), solution_count=int(model.SolCount), seconds_limit=seconds)
                state["native_status"] = native["native_status"]
                control.event(dict(kind="native_end", stage="direct", **native))
                # ObjBoundC is the unrounded full-model native bound, not a
                # fixed-policy risk lower bound and not an independently proved LB.
                bound = float(model.ObjBoundC)
                if np.isfinite(bound) and abs(bound) < 1e90:
                    native["native_bound"] = bound
                    lower = float(np.nextafter(bound - 1e-6, -np.inf))
                    if state["LB"] is None or lower > state["LB"]:
                        certificate = dict(kind="native_MIP_full_extensive_bound", LB=lower,
                                           native_bound=bound, outward_deduction=1e-6,
                                           native_status=native["native_status"], native_independent_proof=False)
                        payload = {key: a[key] for key in ("rhs", "sense", "cost", "lo", "hi", "vtype", "objcon")}
                        state["lower_witness"] = witness("master_bound", "native_direct", iteration, -1,
                            np.zeros(0), dict(A=a["A"], arrays=payload,
                            names=dict(rows=a["row_names"], cols=a["names"]), certificate=certificate, raw_pi=None))
                        state.update(LB=lower, lower_certificate=certificate)
                if model.SolCount:
                    objective = float(model.ObjVal)
                    native["native_objective"] = objective
                    if state["native_incumbent"] is None or objective < state["native_incumbent"]["objective"]:
                        all_values = np.asarray(v.X)
                        raw_x = all_values[:len(first["cost"])].copy()
                        ix = a["index"]
                        state["native_incumbent"] = dict(objective=objective, x=raw_x.tolist(),
                            beta0=float(all_values[ix["beta0"]]),
                            plus=all_values[slice(*ix["plus"])].tolist(), minus=all_values[slice(*ix["minus"])].tolist(),
                            audited_UB=False, residual=row_audit(a["A"], a["rhs"], a["sense"], a["lo"], a["hi"], all_values))
                        for sid, (start, stop) in enumerate(ix["y"]):
                            witness("recourse", "native_direct", iteration, sid, raw_x,
                                    dict(y=all_values[start:stop], native_incumbent_only=True))
                state["history"].append(native)
                iteration += 1
                progress()
                if model.Status != gp.GRB.TIME_LIMIT:
                    break
            if raw_x is not None:
                state["status"] = "AUDITING_INTEGER_POLICY"
                progress()

                def audit_progress(partial):
                    state["coverage"].update(partial["coverage"])
                    for key, value in partial["timing"].items():
                        state["timing"][key] = value
                    progress()

                audited = certify_incumbent(case, ambiguity, built, raw_x, control, env=env,
                                           event=audit_progress, witness=witness)
                state["coverage"].update(audited.pop("coverage"))
                state["timing"].update(audited.pop("timing"))
                state.update(audited)
                if state["UB"] is not None and state["LB"] is not None:
                    gap = state["UB"] - state["LB"]
                    if gap < 0:
                        state["status"] = "INCOMPATIBLE_CERTIFIED_INTERVAL"
                        state["reason"] = "Audited UB below full-model native LB; stop dependent comparison"
                    else:
                        state["gap"] = gap
                        state["passed"] = bool(state["original_expansion_audited"] and
                                               gap <= max(1., .01 * max(1., abs(state["UB"]))))
                        state["objective_optimality_proven"] = bool(state["passed"] and gap <= 1e-5)
                        state["status"] = ("FULL_SUPPORT_GAP_AND_ORIGINAL_AUDIT_PASS" if state["passed"] else
                                           "DIRECT_COMPARISON_PENDING")
            else:
                state["status"] = "NO_NATIVE_INCUMBENT_WITHIN_ALLOCATION"
    except (TimeoutError, RuntimeError, InterruptedError, ValueError) as exc:
        state.update(status="DIRECT_INCOMPLETE" if not isinstance(exc, ValueError) else "DIRECT_INVALID",
                     reason=f"{type(exc).__name__}: {exc}")
    finally:
        progress()
    return state
