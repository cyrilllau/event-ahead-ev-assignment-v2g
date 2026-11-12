"""Catalogue adapter for the existing certified sparse Benders primitives.

No file IO, historical state adoption, or scenario projection. Only complete
support evaluations of canonical INTEGER points may improve the integer UB.
"""

import hashlib
import time
import numpy as np
from scipy import sparse

from ev_assignment.domain.planning import Control, SolverOptions
from ev_assignment.l08 import catalogue as cat, recourse, risk, theta
from ev_assignment.l08.builders import parameters
from ev_assignment.l08.certificates import row_audit, enclosure


def canonical(first, x):
    """Snap native integer counts then clip each contract to that count's bounds."""
    x = np.asarray(x, float).copy()
    H = len(first["package_map"])
    if np.max(abs(x[:H] - np.rint(x[:H])), initial=0) > 1e-7:
        raise ValueError("Native point is not integer")
    x[:H] = np.rint(x[:H])
    x[H:] = np.clip(x[H:], 15 * x[:H],
                    np.asarray([p["Ebar"] for p in first["packages"]]) * x[:H])
    if max(row_audit(first["A"], first["rhs"], first["sense"], first["lo"], first["hi"], x).values()) > 1e-9:
        raise ValueError("Canonical first-stage violates original catalogue")
    return x


def solve_structured(case, ambiguity, control, *, weights=(1, 1, 1, 1),
                     individual=False, prepass=True, event=lambda row: None,
                     witness=lambda *args: None, method_name="P3-DRO",
                     relative_gap=.01, absolute_gap=1., allow_advance=True):
    """Full-DRO for one immutable case; caller owns cumulative budget and IO.

    Template and point caches live only within this invocation and representation.
    Final audit checks original per-vehicle matrix for EVERY support scenario.
    Lower bound uses the existing finite-box LP certificate / native MIP bound;
    upper is outward risk-dual bound of numerically audited recourse primals.
    """
    import gurobipy as gp
    begun, allotted = time.monotonic(), control.remaining
    first = cat.build_first(case["packages"], case["type_counts"],
                            len(case["network"]["station_buses"]), weights,
                            ports=case.get("ports", 20), individual=individual)
    scenes = case["scenarios"]
    if tuple(s["identity"] for s in scenes) != ambiguity.support_ids:
        raise ValueError("Complete case/ambiguity support identity mismatch")
    nx, ns = len(first["cost"]), len(scenes)
    options = SolverOptions(seconds=min(60., max(.01, allotted)), threads=4, seed=101,
                            mip_gap=relative_gap, absolute_gap=absolute_gap, soft_memory_gb=20,
                            master_seconds=60)
    state = dict(method=method_name, science_identity=case["identity"],
                 timing=dict(build=0., master=0., recourse=0., probability=0., certificate_and_io=0., audit=0.),
                 cuts=0, iterations=0, nodes=0., recourse_calls=0, probability_calls=0,
                 recourse_attempts=0, probability_attempts=0,
                 first_stage_weights=list(weights), ambiguity_identity=ambiguity.identity,
                 first_stage_restriction="all_packages" if allow_advance else "latest_departure_only",
                 ordered_support=list(ambiguity.support_ids), probability_rows=[],
                 representation="individual" if individual else "count",
                 LB=None, UB=None, best=None, gap=None, passed=False,
                 original_expansion_audited=False, history=[], full_support=ns,
                 complete_integer_evaluations=0, partial_evaluations=[],
                 lower_evidence="native_solver_or_finite_box_LP_certificate",
                 upper_evidence="NUMERICALLY_AUDITED_RECOURSE_PLUS_CERTIFIED_RISK_DUAL",
                 status="RUNNING", prepass_rounds=0)
    state["first_size"] = dict(vars=nx, rows=first["A"].shape[0], nnz=first["A"].nnz)
    state["recourse_sizes"] = []
    templates, cached, best_y = {}, {}, None
    def progress():
        if state["LB"] is not None and state["UB"] is not None:
            state["gap"] = state["UB"] - state["LB"]
        state["elapsed_seconds"] = time.monotonic() - begun
        event(state.copy())
    def target():
        if state["LB"] is None or state["UB"] is None:
            return False
        gap = state["UB"] - state["LB"]
        if gap < 0:
            raise ValueError("Negative external gap")
        state["gap"] = gap
        return gap <= max(absolute_gap, relative_gap * max(1., abs(state["UB"])))
    with gp.Env(params={"OutputFlag": 0}) as env, gp.Model("CATALOGUE_SPARSE_MASTER", env=env) as model:
        model_hi = first["hi"].copy()
        if not allow_advance:
            banned = np.asarray([p["departure_choice"] < 2 for p in first["packages"]])
            model_hi[np.r_[banned, banned]] = 0
        xx = model.addMVar(nx, lb=first["lo"], ub=model_hi, vtype="C")
        rho = model.addVar(lb=0, obj=1, name="rho")
        model.addMVar(ns, lb=0, name="theta")
        model.addMConstr(first["A"], xx, first["sense"], first["rhs"])
        model.setObjective(first["cost"] @ xx + rho)
        model.update()
        variables = model.getVars()
        tmap = {s: nx + 1 + s for s in range(ns)}
        row_proof = theta.add_probability(model, variables, nx, tmap, ambiguity.feasible_anchor, "anchor")
        state["probability_rows"].append(dict(p=ambiguity.feasible_anchor.tolist(), proof=row_proof,
                                              membership=ambiguity.check_original(ambiguity.feasible_anchor)))
        probabilities = {hashlib.sha256(ambiguity.feasible_anchor.tobytes()).hexdigest()}
        state["timing"]["build"] += time.monotonic() - begun
        def evaluate(x, cc, phase, iteration):
            pid = hashlib.sha256(x.tobytes()).hexdigest()
            if pid in cached:
                return cached[pid]
            records, ys = [], []
            for sid, scene in enumerate(scenes):
                cc.check()
                if sid not in templates:
                    t = time.monotonic()
                    templates[sid] = cat.build_recourse(case["network"], case["packages"],
                                                       scene["physical"], scene["route_states"], first)
                    state["timing"]["build"] += time.monotonic() - t
                    state["recourse_sizes"].append(dict(sid=sid, vars=templates[sid].W.shape[1],
                        rows=templates[sid].W.shape[0], W_nnz=templates[sid].W.nnz, T_nnz=templates[sid].T.nnz))
                t = time.monotonic()
                state["recourse_attempts"] += 1
                try:
                    result = recourse.solve(templates[sid], x, options, cc, env=env)
                finally:
                    state["timing"]["recourse"] += time.monotonic() - t
                state["recourse_calls"] += 1
                t = time.monotonic()
                _, _, support_proof = theta.add_supports(model, variables, nx, tmap[sid],
                                   sparse.csr_matrix(result["beta"][None, :]), [result["alpha"]],
                                   [pid], first["lo"], first["hi"], f"support/{sid}/{phase}/{iteration}")
                state["cuts"] += 1
                records.append(dict(qL=result["qL"], qU=result["qU_audited"],
                                    residual=result["residual"], qbar=templates[sid].qbar,
                                    global_cut_valid=result["global_cut_valid"], stored_cut_proof=support_proof))
                ys.append(result["y"])
                witness("recourse", phase, iteration, sid, x, result)
                state["timing"]["certificate_and_io"] += time.monotonic() - t
                if sid % 8 == 0 or sid + 1 == ns:
                    state["current_evaluation"] = dict(point=pid, phase=phase, completed=sid + 1, total=ns)
                    progress()
            t = time.monotonic()
            state["probability_attempts"] += 1
            try:
                rr = risk.solve(ambiguity, [r["qL"] for r in records], [r["qU"] for r in records],
                                [r["qbar"] for r in records], Control(min(60., cc.remaining),
                                stop=lambda: cc.stop() or cc.remaining <= 0))
            finally:
                state["timing"]["probability"] += time.monotonic() - t
            state["probability_calls"] += 1
            pp = rr["pstar"]
            ph = hashlib.sha256(pp.tobytes()).hexdigest()
            if ph not in probabilities:
                if not ambiguity.check_original(pp)["passed"]:
                    raise ValueError("Risk row is outside this original probability domain")
                proof = theta.add_probability(model, variables, nx, tmap, pp, f"risk/{ph}")
                state["probability_rows"].append(dict(p=pp.tolist(), proof=proof,
                                                       membership=ambiguity.check_original(pp)))
                probabilities.add(ph)
            F = float(enclosure(sparse.csr_matrix(first["cost"][None, :]), x)[1][0])
            value = float(np.nextafter(F + rr["risk_U"], np.inf))
            witness("risk", phase, iteration, -1, x, dict(certificate=rr, first_cost_upper=F,
                        Q=records, ambiguity_identity=ambiguity.identity, full_objective_upper=value))
            cached[pid] = value, records, ys, rr
            return cached[pid]
        audit_begin = None
        try:
            phase = "continuous" if prepass else "integer"
            rounds = 0
            while control.remaining > 0:
                if phase == "continuous" and (rounds >= 10 or time.monotonic() - begun >= .25 * allotted):
                    phase = "integer"
                if phase == "integer":
                    xx.VType = first["vtype"]
                    model.update()
                duration = min(60., control.remaining)
                if phase == "continuous":
                    duration = min(duration, max(.01, .25 * allotted - (time.monotonic() - begun)))
                parameters(model, options, duration)
                t = time.monotonic()
                model.optimize(lambda m, where: m.terminate() if control.stop() else None)
                state["timing"]["master"] += time.monotonic() - t
                state["iterations"] += 1
                state["nodes"] += float(model.NodeCount) if model.IsMIP else 0
                if phase == "continuous" and model.Status == gp.GRB.OPTIMAL:
                    A, a, _ = theta.model_arrays(model)
                    certificate, _ = theta.lagrangian(A, a, np.asarray(model.getAttr("Pi")), nx, nx)
                    lower = certificate.get("LB")
                else:
                    lower = model.ObjBoundC - 1e-6 if model.IsMIP and abs(model.ObjBoundC) < 1e90 else None
                    certificate = dict(kind="native_MIP_bound", native_status=int(model.Status),
                                       native_bound=float(model.ObjBoundC) if lower is not None else None,
                                       outward_deduction=1e-6, LB=lower)
                if lower is not None:
                    if state["LB"] is None or lower > state["LB"]:
                        A, a, names = theta.model_arrays(model)
                        ref = witness("master_bound", phase, rounds, -1, np.zeros(0),
                            dict(A=A, arrays=a, names=names, certificate=certificate,
                                 raw_pi=np.asarray(model.getAttr("Pi")) if not model.IsMIP else None))
                        state["lower_certificate"] = certificate
                        state["lower_witness"] = ref
                    state["LB"] = max(state["LB"] or 0., float(lower))
                state["native_status"] = int(model.Status)
                state.setdefault("master_history", []).append(dict(phase=phase, iteration=rounds,
                    bound=state["LB"], UB=state["UB"], native_status=int(model.Status),
                    native_objective=float(model.ObjVal) if model.SolCount else None,
                    native_nodes=float(model.NodeCount) if model.IsMIP else 0., runtime=float(model.Runtime)))
                if target():
                    break
                if not model.SolCount:
                    control.check()
                    continue
                x = np.asarray(xx.X) if phase == "continuous" else canonical(first, xx.X)
                remaining_prepass = .25 * allotted - (time.monotonic() - begun)
                cc = Control(min(control.remaining, max(.01, remaining_prepass)), stop=control.stop) if phase == "continuous" else control
                try:
                    value, records, ys, rr = evaluate(x, cc, phase, rounds)
                except (TimeoutError, RuntimeError) as exc:
                    if phase != "continuous" or (cc.remaining > .1 and control.remaining > .1):
                        raise
                    state["partial_evaluations"].append(dict(state.get("current_evaluation", {}),
                                                             phase=phase, reason=str(exc)))
                    phase = "integer"
                    progress()
                    continue
                if phase == "integer":
                    state["complete_integer_evaluations"] += 1
                    if state["UB"] is None or value < state["UB"]:
                        state.update(UB=value, best=x.tolist(), best_recourse=records,
                                     risk_interval=[rr["risk_L"], rr["risk_U"]],
                                     best_risk_certificate=rr)
                        best_y = ys
                else:
                    state["prepass_rounds"] += 1
                state["history"].append(dict(phase=phase, round=rounds, LB=state["LB"], UB=state["UB"],
                                             evaluated_objective_upper=value, integer_UB=phase == "integer"))
                rounds += 1
                progress()
                if target():
                    break
            if state["best"] is not None:
                audit_begin = time.monotonic()
                # Build every original big-M row; never audit only the new equations.
                fi = first if individual else cat.build_first(case["packages"], case["type_counts"],
                        len(case["network"]["station_buses"]), weights, ports=case.get("ports", 20), individual=True)
                x = np.asarray(state["best"])
                audits = []
                state["original_audit"] = audits
                canonical_contracts = None
                for sid, scene in enumerate(scenes):
                    control.check()
                    ti = templates[sid] if individual else cat.build_recourse(case["network"], case["packages"],
                                 scene["physical"], scene["route_states"], fi)
                    xi, yi = (x, best_y[sid]) if individual else cat.expand_count(first, fi, x, best_y[sid],
                                 count_recourse=templates[sid], individual_recourse=ti)
                    if canonical_contracts is None:
                        canonical_contracts = xi.copy()
                    elif not np.array_equal(canonical_contracts, xi):
                        raise ValueError("Canonical first-stage contracts changed across scenarios")
                    adapter = cat.build_original_adapter(case["network"], case["packages"],
                                 scene["physical"], scene["route_states"], fi)
                    zo, yo = cat.to_original(adapter, fi, ti, xi, yi)
                    original = adapter["matrix"]
                    if sid == 0:
                        state["individual_recourse_size"] = dict(vars=ti.W.shape[1], rows=ti.W.shape[0],
                                                                  W_nnz=ti.W.nnz, T_nnz=ti.T.nnz)
                        state["original_recourse_size"] = dict(vars=len(original.cost),
                            rows=original.A.shape[0] + original.H.shape[0], W_nnz=original.A.nnz + original.H.nnz,
                            T_nnz=original.B.nnz + original.G.nnz, representative_scenario=scene["identity"])
                    component_delta = original.parts @ yo - templates[sid].data["parts"] @ best_y[sid]
                    first_delta = adapter["first"]["cost"] @ zo - first["cost"] @ x
                    if max(np.max(abs(component_delta), initial=0), abs(first_delta)) > 1e-5:
                        raise ValueError("Original expansion changed objective components")
                    power = yo[original.index["p"]]
                    shed = yo[original.index["shed"]]
                    audits.append(dict(sid=scene["identity"], residual=original.residual(yo, zo),
                        discharging_vehicles=int(np.count_nonzero(power.sum(axis=(1, 2)) > 1e-9)),
                        discharge_AC_kwh=float(.25 * power.sum()),
                        total_ENS_kwh=float(.25 * shed.sum()),
                        critical_ENS_kwh=float(.25 * shed[:, np.asarray(case["network"]["shed_cost"]) == 50].sum()),
                        shortfall_kwh=float(yo[original.index["short"]].sum()),
                        operating_component_delta=component_delta.tolist(), first_cost_delta=float(first_delta),
                        original_tasks=len(adapter["tasks"]), original_shortfall_big_m_rows=sum(
                            name.startswith("incident_capability/") for name in original.row_names["ub"]),
                        canonical_contracts_unchanged=True))
                    witness("original_audit", "integer", rounds, sid, zo, dict(y=yo, audit=audits[-1]))
                    if sid % 8 == 0 or sid + 1 == ns:
                        state["audit_completed"] = sid + 1
                        progress()
                state["original_audit"] = audits
                state["original_expansion_audited"] = True
                state["timing"]["audit"] = time.monotonic() - audit_begin
            state["passed"] = bool(target() and state["original_expansion_audited"])
            state["status"] = "FULL_SUPPORT_GAP_AND_ORIGINAL_AUDIT_PASS" if state["passed"] else "GATE_NOT_REACHED"
        except (TimeoutError, RuntimeError) as exc:
            state.update(status="TIME_LIMIT_OR_ORACLE_INCOMPLETE", reason=str(exc))
        finally:
            if audit_begin is not None:
                state["timing"]["audit"] = time.monotonic() - audit_begin
            model.update()
            state["master_size"] = dict(vars=model.NumVars, rows=model.NumConstrs, nnz=model.NumNZs,
                                          integer_vars=sum(first["vtype"] != "C"))
            progress()
    return state
