"""Remote-only A0 analytical prototype, probability LPs and preregistration."""

from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
import json
import sys
import time

import numpy as np
from scipy.optimize import linprog

from ev_assignment_experiments.runtime.ledger import BudgetLedger
from ev_assignment_experiments.analysis.diagnostic import read, reference, save


root = Path(sys.argv[1]).resolve()
config = read(root / "DESIGN_CONFIG.json")
ledger = BudgetLedger(root / "store", "structured-design-development-v1")
ledger.create("structured-fleet-design-v1", 7200)
snap = ledger.read()
spent = sum(
    e["payload"]["charged_seconds"] for e in snap["events"] if e["payload"]["kind"] == "settlement"
)
out = root / "A0"
out.mkdir(exist_ok=False)
claim = ledger.claim(
    dict(stage="A0", source=reference(__file__), config=reference(root / "DESIGN_CONFIG.json")),
    min(600, 7200 - spent),
    charge_process_start=True,
)
save(out / "CLAIM.json", claim)
started = time.monotonic()
rc = 1
try:
    C = [0, 80, 200]
    R = [3000, 800, 200]
    crossings = [Fraction(C[i + 1] - C[i], R[i] - R[i + 1]) for i in range(2)]
    assert crossings == [Fraction(2, 55), Fraction(1, 5)]
    probabilities = [
        Fraction(0),
        Fraction(2, 25),
        Fraction(1, 10),
        Fraction(1, 4),
        Fraction(1),
        Fraction(1, 50),
    ]
    rows = []
    for q, target in zip(probabilities, [0, 1, 1, 2, 2, 0]):
        values = [Fraction(c) + q * r for c, r in zip(C, R)]
        assert values.index(min(values)) == target
        rows.append(dict(q=str(q), objectives=[str(v) for v in values], best_policy=target))
    G = np.array([0.0, 1.0, 0.0, 1.0])
    B = np.array([0.0, 0.0, 1.0, 1.0])
    h = G * B
    risk = []
    for method in ["full", "marginal"]:
        lp = linprog(
            -h,
            A_ub=h[None, :] if method == "full" else None,
            b_ub=[0.1] if method == "full" else None,
            A_eq=np.stack([np.ones(4), G, B]),
            b_eq=[1.0, 0.25, 0.25],
            bounds=(0, None),
            method="highs",
            options=dict(
                threads=1,
                time_limit=30,
                primal_feasibility_tolerance=1e-9,
                dual_feasibility_tolerance=1e-9,
            ),
        )
        assert lp.success
        value = float(-lp.fun)
        assert abs(value - (0.1 if method == "full" else 0.25)) < 1e-9
        risk.append(
            dict(
                method=method,
                prototype_only=True,
                maximum_joint_probability=value,
                p=lp.x.tolist(),
                native_status=int(lp.status),
                iterations=lp.nit,
            )
        )
    candidates = []
    for gap in config["departure_gaps_h"]:
        for duration in config["early_gap_durations_h"]:
            for joint in config["joint_nominal_probabilities"]:
                candidates.append(
                    dict(
                        id=f"C{len(candidates) + 1:02d}",
                        gap_h=gap,
                        early_duration_h=duration,
                        joint_nominal=joint,
                        seed=101,
                        boundary=(gap == 0.5 or joint == 0.02),
                        neighborhood_axis="adjacent gap/duration/joint within this fixed list",
                    )
                )
    assert len(candidates) == 12
    save(
        root / "PREREGISTERED_CANDIDATES.json",
        dict(
            candidates=candidates,
            max_finalists=2,
            max_revision=1,
            fallback=config["fallback"],
            no_validation_or_oos_read=True,
        ),
    )
    result = dict(
        stage="A0",
        passed=True,
        status="ANALYTICAL_PROTOTYPE_PASS_NOT_PHYSICAL_RESULT",
        crossings=[str(x) for x in crossings],
        reference_table=rows,
        probability_LPs=risk,
        solver_calls=2,
        candidate_count=12,
        elapsed_seconds=time.monotonic() - started,
        original_full400_claim=False,
        source=reference(__file__),
        config=reference(root / "DESIGN_CONFIG.json"),
    )
    save(out / "RESULT.json", result)
    rc = 0
finally:
    save(out / "WAIT.json", dict(returncode=rc))
    ledger.settle(claim["attempt"], rc, reference(out / "WAIT.json"))
    b = ledger.read()
    save(out / "BUDGET_AFTER.json", b)
    event = dict(
        observed_utc=datetime.now(timezone.utc).isoformat(),
        stage="A0",
        result=reference(out / "RESULT.json") if rc == 0 else None,
        returncode=rc,
        budget_used=b["spent_seconds"],
        budget_remaining=b["remaining_seconds"],
        source=reference(__file__),
    )
    with (root / "CASE_SEARCH_LEDGER.jsonl").open("a") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    (root / "CASE_DESIGN_STATUS.md").write_text(
        f"# 合成实例设计状态\n\nA0 {'PASS（仅解析原型）' if rc == 0 else 'FAIL'}；A1实现/原矩阵验证待完成；A2–A5未开始；验证未开放。\n\n开发预算 {b['spent_seconds']:.6f}/7200秒，剩余{b['remaining_seconds']:.6f}秒。验证额度28800秒尚未使用。\n活动任务：无（本wrapper已结算）。\n当前配置：DESIGN_CONFIG.json；候选表PREREGISTERED_CANDIDATES.json。\n下一步：部署A1类型/套餐内核，在端口活跃、能量/短缺活跃两个小例中比较逐车与计数整数最优目标，并展开至原matrix.assemble审计。\nA0输出：{out}；精确证据见RESULT.json、CLAIM.json、BUDGET_AFTER.json。勿重复执行A0。\n"
    )
print(json.dumps(event, ensure_ascii=False))
