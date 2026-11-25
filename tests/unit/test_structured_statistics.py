"""Artificial numerical fixtures only; no scientific inputs or native solvers."""

from copy import deepcopy

import numpy as np
import pytest

from ev_assignment.domain.planning import Control
from ev_assignment_experiments.analysis.structured_statistics import (
    COMPARATORS, METHODS, paired_bootstrap, summarize_paired,
)


def toy_records(counts=(2, 3, 4, 5), masses=(.1, .2, .3, .4)):
    rows = []
    for stratum, (count, mass) in enumerate(zip(counts, masses, strict=True)):
        for i in range(count):
            common = 10 * stratum + i * i
            rows.append(dict(sid=f"toy/{stratum}/{i}", stratum=stratum, weight=mass / count,
                methods={method: dict(qU=float(common + j), qL=float(common + j - .01),
                                      first_cost=float(2 * j), critical_ENS_kwh=float(common + 4 * j))
                         for j, method in enumerate(METHODS)}))
    return rows


def run(rows, **kwargs):
    return paired_bootstrap(rows, 1101, Control(20), expected_n=len(rows), **kwargs)


def test_paired_constant_shifts_have_degenerate_intervals_and_no_input_rng_mutation():
    rows = toy_records()
    original = deepcopy(rows)
    state = np.random.get_state()
    result = run(rows)
    assert result == run(rows)
    assert rows == original
    after = np.random.get_state()
    assert state[0] == after[0] and np.array_equal(state[1], after[1])
    assert state[2:] == after[2:]
    assert result["formal_ci"] and result["fixture_override"]
    assert result["seed"] == 911101 and result["rng"] == "PCG64"
    assert result["completed_replicates"] == result["replicates"] == 500
    for method in COMPARATORS:
        delta = METHODS.index("P3") - METHODS.index(method)
        for metric, factor in (("total_cost_mean", 3), ("total_cost_cvar95", 3),
                               ("critical_ENS_mean", 4)):
            answer = result["comparisons"][method][metric]
            assert answer["difference"] == pytest.approx(factor * delta, abs=1e-12)
            assert answer["ci95"] == pytest.approx([factor * delta] * 2, abs=1e-12)


def test_strata_masses_and_counts_stay_fixed_even_with_unequal_allocations():
    rows = toy_records()
    for row in rows:
        # There is no within-stratum uncertainty, although values differ greatly
        # across strata. An unstratified bootstrap would give a spurious CI width.
        row["methods"]["P3"].update(qU=100. * row["stratum"], qL=100. * row["stratum"],
                                      first_cost=0., critical_ENS_kwh=10. * row["stratum"])
        row["methods"]["DET"].update(qU=0., qL=0., first_cost=0., critical_ENS_kwh=0.)
    result = run(rows)
    assert [s["count"] for s in result["strata"]] == [2, 3, 4, 5]
    assert [s["mass"] for s in result["strata"]] == pytest.approx([.1, .2, .3, .4])
    answers = result["comparisons"]["DET"]
    for metric, expected in (("total_cost_mean", 200.), ("total_cost_cvar95", 300.),
                             ("critical_ENS_mean", 20.)):
        assert answers[metric]["difference"] == pytest.approx(expected)
        assert answers[metric]["ci95"] == pytest.approx([expected, expected])


def test_cvar_splits_boundary_atom_and_subtracts_method_tails_not_paired_losses():
    rows = toy_records((1, 1, 1, 1), (.02, .04, .44, .5))
    for row, p3, det in zip(rows, [100., 50., 0., 0.], [0., 100., 50., 0.], strict=True):
        for method, cost in (("P3", p3), ("DET", det)):
            row["methods"][method].update(qU=cost, qL=cost, first_cost=0.)
    result = run(rows)
    assert result["point_metrics"]["P3"]["total_cost_cvar95"] == pytest.approx(70.)
    assert result["point_metrics"]["DET"]["total_cost_cvar95"] == pytest.approx(90.)
    answer = result["comparisons"]["DET"]["total_cost_cvar95"]
    assert answer["difference"] == pytest.approx(-20.)
    assert answer["ci95"] == pytest.approx([-20., -20.])


def test_random_intervals_match_independent_scalar_paired_bootstrap():
    rows = toy_records((2, 2, 2, 2), (.25, .25, .25, .25))
    for i, row in enumerate(rows):
        row["methods"]["P3"].update(qU=float(i * i), qL=float(i * i), first_cost=3.)
        row["methods"]["DET"].update(qU=float(8 - i), qL=float(8 - i), first_cost=1.)
    result = run(rows)
    rng = np.random.Generator(np.random.PCG64(911101))
    means, tails = [], []
    for _ in range(500):
        sampled = [rows[2 * s + j] for s in range(4) for j in rng.integers(2, size=2)]
        costs = {method: [r["methods"][method]["qU"] + r["methods"][method]["first_cost"]
                          for r in sampled] for method in ("P3", "DET")}
        means.append((sum(costs["P3"]) - sum(costs["DET"])) / 8)
        # Every atom weighs 1/8 > 5%, so each tail is simply that method's max.
        tails.append(max(costs["P3"]) - max(costs["DET"]))
    for metric, draws in (("total_cost_mean", means), ("total_cost_cvar95", tails)):
        lo, hi = result["comparisons"]["DET"][metric]["ci95"]
        assert [lo, hi] == pytest.approx(np.percentile(draws, [2.5, 97.5], method="linear"))
        assert hi > lo


def test_partial_or_missing_methods_never_produce_formal_intervals():
    rows = toy_records()
    result = paired_bootstrap(rows, 1101, Control(20))  # default remains 1000
    assert result["expected_n"] == 1000 and not result["fixture_override"]
    assert result["status"] == "NOT_ELIGIBLE" and not result["formal_ci"]
    assert not result["comparisons"] and not result["point_metrics"]
    del rows[3]["methods"]["ARO"]
    result = run(rows)
    assert result["missing_method_rows"] == {"ARO": 1}
    assert not result["formal_ci"] and not result["comparisons"]
    assert result["completed_replicates"] == 0


@pytest.mark.parametrize("defect", ["duplicate", "missing_id", "stratum", "missing_stratum",
    "unequal_weights", "weight_sum", "zero_weight", "nan", "missing_metric", "reversed_bound"])
def test_invalid_complete_inputs_cannot_produce_intervals(defect):
    rows = toy_records()
    if defect == "duplicate":
        rows[1]["sid"] = rows[0]["sid"]
    elif defect == "missing_id":
        del rows[0]["sid"]
    elif defect == "stratum":
        rows[0]["stratum"] = 4
    elif defect == "missing_stratum":
        for row in rows:
            if row["stratum"] == 3:
                row["stratum"] = 2
    elif defect == "unequal_weights":
        rows[0]["weight"] += .01
        rows[1]["weight"] -= .01
    elif defect == "weight_sum":
        rows[0]["weight"] += .1
    elif defect == "zero_weight":
        rows[1]["weight"] += rows[0]["weight"]
        rows[0]["weight"] = 0.
    elif defect == "nan":
        rows[0]["methods"]["P3"]["qU"] = float("nan")
    elif defect == "missing_metric":
        del rows[0]["methods"]["SAA"]["qL"]
    elif defect == "reversed_bound":
        rows[0]["methods"]["P3"]["qL"] += 1.
    with pytest.raises(ValueError):
        run(rows)


def test_deadline_and_explicit_stop_abort_without_partial_ci():
    rows = toy_records()
    with pytest.raises(TimeoutError):
        paired_bootstrap(rows, 1101, Control(0), expected_n=len(rows))
    checks = 0

    def stop_mid_bootstrap():
        nonlocal checks
        checks += 1
        return checks > 3 * len(rows) + 10

    with pytest.raises(InterruptedError):
        paired_bootstrap(rows, 1101, Control(20, stop=stop_mid_bootstrap), expected_n=len(rows))


def test_full_size_gate_with_artificial_rows_and_optional_ids_absent():
    rows = toy_records((100, 200, 300, 400))
    for row in rows:
        del row["sid"]
    result = summarize_paired(rows, control=Control(20), seed=911103, repetitions=500)
    assert result["formal_ci"] and not result["fixture_override"]
    assert result["observed_n"] == result["expected_n"] == 1000
    assert result["seed"] == 911103


def test_worker_api_uses_supplied_seed_and_fixed_repetition_count():
    rows = toy_records()
    result = summarize_paired(rows, control=Control(20), seed=911101,
                              repetitions=500, expected_n=len(rows))
    assert result == run(rows)
    with pytest.raises(ValueError, match="500 repetitions"):
        summarize_paired(rows, control=Control(20), seed=911101, repetitions=499)
    empty = summarize_paired([], control=Control(20), seed=911101)
    assert empty["status"] == "NOT_ELIGIBLE" and empty["comparisons"] == {}
    assert empty["ineligible_reasons"] == ["INCOMPLETE_SAMPLE"]


@pytest.mark.parametrize("defect", ["duplicate_id", "negative_weight", "unequal_weights"])
def test_inconsistent_partial_input_is_fatal_not_just_ineligible(defect):
    rows = toy_records()[:2]
    if defect == "duplicate_id":
        rows[1]["sid"] = rows[0]["sid"]
    elif defect == "negative_weight":
        rows[0]["weight"] = -.1
    else:
        rows[0]["weight"] *= 2
    with pytest.raises(ValueError):
        summarize_paired(rows, control=Control(20), seed=911101)
