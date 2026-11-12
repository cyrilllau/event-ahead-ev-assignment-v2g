"""Frozen L08 run recipe; implementation, data, resources and budget stay explicit."""

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
import tomllib

from ev_assignment.domain.planning import SolverOptions


@dataclass(frozen=True)
class RunPlan:
    study: str
    title: str
    method: str
    input_kind: str
    budget_group: str
    budget_seconds: float
    options: SolverOptions
    input_manifest: str | None = None
    input_root: str | None = None
    hypothesis: str = ""
    baseline: str = ""
    inherited_seconds: float = 0.0
    inherited_evidence: tuple = ()
    design: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path):
        with Path(path).open("rb") as h:
            return cls.parse(tomllib.load(h))

    @classmethod
    def parse(cls, value):
        allowed = {
            "schema_version",
            "study",
            "title",
            "method",
            "input_kind",
            "budget_group",
            "budget_seconds",
            "solver",
            "input_manifest",
            "input_root",
            "hypothesis",
            "baseline",
            "inherited_seconds",
            "inherited_evidence",
            "design",
        }
        required = {
            "schema_version",
            "study",
            "title",
            "method",
            "input_kind",
            "budget_group",
            "budget_seconds",
            "solver",
        }
        if (
            set(value) - allowed
            or required - set(value)
            or type(value["schema_version"]) is not int
            or value["schema_version"] != 2
        ):
            raise ValueError("Unknown/missing L08 recipe keys")
        if value["method"] not in {"DET", "DRO", "ARO"} or value["input_kind"] not in {
            "synthetic_l08",
            "frozen_l08",
        }:
            raise ValueError("Unsupported L08 method/input")
        for key in ("study", "title", "budget_group"):
            if not isinstance(value[key], str) or not value[key].strip():
                raise ValueError("Missing " + key)
        budget = value["budget_seconds"]
        if isinstance(budget, bool) or not math.isfinite(budget) or budget <= 0:
            raise ValueError("Invalid cumulative budget")
        options = SolverOptions(**value["solver"])
        inherited = value.get("inherited_seconds", 0.0)
        evidence = tuple(value.get("inherited_evidence", ()))
        if (
            isinstance(inherited, bool)
            or not math.isfinite(inherited)
            or not 0 <= inherited <= budget
            or (inherited and not evidence)
        ):
            raise ValueError("Inherited budget requires bounded cost and sealed evidence")
        if options.seconds > budget:
            raise ValueError("Attempt exceeds cumulative budget")
        root = value.get("input_root")
        manifest = value.get("input_manifest")
        if value["input_kind"] == "frozen_l08" and (
            not root or not Path(root).is_absolute() or not manifest
        ):
            raise ValueError(
                "Frozen inputs require explicit absolute root and trusted manifest SHA"
            )
        if value["input_kind"] == "synthetic_l08" and (root or manifest):
            raise ValueError("Synthetic definition cannot hide external inputs")
        return cls(
            value["study"],
            value["title"],
            value["method"],
            value["input_kind"],
            value["budget_group"],
            float(budget),
            options,
            manifest,
            root,
            value.get("hypothesis", ""),
            value.get("baseline", ""),
            float(inherited),
            evidence,
            value.get("design", {}),
        )

    def as_dict(self):
        value = asdict(self)
        value["solver"] = value.pop("options")
        value["schema_version"] = 2
        return value
