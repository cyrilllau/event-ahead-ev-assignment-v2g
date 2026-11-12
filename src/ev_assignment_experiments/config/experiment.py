"""Validated, self-contained development experiment definitions."""

from dataclasses import dataclass
import math
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class Experiment:
    study: str
    title: str
    support_ids: tuple[str, ...]
    feature_ids: tuple[str, ...]
    features: tuple[tuple[float, ...], ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    q: tuple[float, ...]
    seconds: float

    @classmethod
    def load(cls, path):
        with Path(path).open("rb") as handle:
            return cls.parse(tomllib.load(handle))

    @classmethod
    def parse(cls, value):
        keys = {
            "schema_version",
            "study",
            "title",
            "method",
            "support_ids",
            "feature_ids",
            "features",
            "lower",
            "upper",
            "q",
            "seconds",
        }
        if (
            set(value) != keys
            or type(value["schema_version"]) is not int
            or value["schema_version"] != 1
        ):
            raise ValueError("Unknown/missing experiment keys or unsupported schema")
        if value["method"] != "development_probability_oracle":
            raise ValueError("Only the development probability oracle is admitted; L08 is pending")
        for key in ("study", "title"):
            if not isinstance(value[key], str) or not value[key].strip():
                raise ValueError(f"Missing {key}")
        seconds = value["seconds"]
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or not 0 < seconds <= 60
        ):
            raise ValueError("Development solve budget must be finite and within (0,60]")
        # Delegate scientific shape/identity/moment checks to the preserved core.
        from ev_assignment.methods.probability import ProbabilitySet

        problem = ProbabilitySet(
            tuple(value["support_ids"]),
            tuple(value["feature_ids"]),
            value["features"],
            value["lower"],
            value["upper"],
        )
        q = tuple(float(x) for x in value["q"])
        if len(q) != len(problem.support_ids) or not all(math.isfinite(x) for x in q):
            raise ValueError("q must be finite and aligned with support_ids")
        return cls(
            value["study"],
            value["title"],
            problem.support_ids,
            problem.feature_ids,
            tuple(tuple(row) for row in problem.features.tolist()),
            tuple(problem.lower.tolist()),
            tuple(problem.upper.tolist()),
            q,
            float(seconds),
        )

    def as_dict(self):
        from dataclasses import asdict

        return {"schema_version": 1, "method": "development_probability_oracle", **asdict(self)}
