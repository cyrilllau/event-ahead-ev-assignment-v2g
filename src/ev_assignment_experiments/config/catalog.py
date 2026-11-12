"""Bind versioned full-domain recipes without hiding machine paths or budgets."""

from pathlib import Path
import tomllib
from .run import RunPlan


def bind(path, *, input_root, study, budget_group, budget_seconds, seconds, master_seconds):
    template = tomllib.loads(Path(path).read_text())
    if (
        set(template) != {"schema_version", "recipe", "science", "solver"}
        or template["schema_version"] != 1
    ):
        raise ValueError("Unknown catalog schema")
    science = template["science"]
    if science["protocol"] != "L08_FULL_DWELL_FORECAST44_V1" or science["vehicles"] != 30:
        raise ValueError("Unsupported scientific domain")
    method = template["recipe"]["method"]
    expected = "forecast44" if method == "DET" else "train/0000..0399"
    if science["planning_scenarios"] != expected or science["oos_access"]:
        raise ValueError("Recipe changed planning domain/OOS policy")
    recipe = dict(
        template["recipe"],
        schema_version=2,
        study=study,
        budget_group=budget_group,
        budget_seconds=budget_seconds,
        input_root=input_root,
        design=science,
        solver=dict(template["solver"], seconds=seconds, master_seconds=master_seconds),
    )
    return RunPlan.parse(recipe)
