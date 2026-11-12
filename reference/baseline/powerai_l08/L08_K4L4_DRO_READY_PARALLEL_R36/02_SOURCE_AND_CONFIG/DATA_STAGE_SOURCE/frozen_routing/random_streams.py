"""Versioned random stream identities; no method/regime/lead enters a seed.

Each stream is addressed directly, so loop ordering and worker scheduling do
not change a draw. All targets share primitive exogenous draws. Residual donor
selection uses a shared random event ordering with the target filtered out.
Physical payload generation and OOS evaluation are outside this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral

import numpy as np

SCHEMA = 1
ROOTS = {"planning": 20260901, "calibration": 20260902, "oos": 20260903}
DEVELOPMENT_ROOT = 20260911
STATIC_ROAD_ROOT = 20260904
SPLITS = {"planning": 1, "calibration": 2, "oos": 3}
MECHANISMS = {
    "wind_amplitude": 1, "wind_location": 2, "wind_lag": 3,
    "residual_donor_order": 4, "residual_block": 5, "residual_scale": 6,
    "line_gaussian": 7, "road_uniform": 8, "load_innovation": 9,
}


def canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class Stream:
    root_seed: int
    spawn_key: tuple[int, ...]

    @property
    def stream_id(self):
        return canonical_hash({"root_seed": self.root_seed, "spawn_key": self.spawn_key})

    def generator(self):
        return np.random.Generator(np.random.PCG64(
            np.random.SeedSequence(self.root_seed, spawn_key=self.spawn_key)))

    def as_dict(self):
        return {"root_seed": self.root_seed, "spawn_key": list(self.spawn_key),
                "stream_id": self.stream_id, "bit_generator": "PCG64"}


def stream(split, base_index, variant_index, mechanism, *,
           development=False, asset_index=0, component_index=0):
    if split not in SPLITS or mechanism not in MECHANISMS:
        raise ValueError("Unknown split or mechanism")
    for name, value in (("base_index", base_index), ("variant_index", variant_index),
                        ("asset_index", asset_index), ("component_index", component_index)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if base_index >= 8:
        raise ValueError("The frozen case has exactly eight weather bases")
    # All business indices are one-based in keys; zero denotes shared/not used.
    # domain, split, target, base, variant, mechanism, asset, component.
    key = (SCHEMA, 2 if development else 1, SPLITS[split], 0,
           int(base_index) + 1, int(variant_index) + 1, MECHANISMS[mechanism],
           int(asset_index), int(component_index))
    return Stream(DEVELOPMENT_ROOT if development else ROOTS[split], key)


def event_order(event_ids):
    order = tuple(sorted(str(e) for e in event_ids))
    if len(order) != 8 or len(set(order)) != 8:
        raise ValueError("Expected eight unique event IDs")
    return order


def residual_donor(order, target_event, split, base_index, variant_index, *, development=False):
    order = event_order(order)
    if target_event not in order:
        raise ValueError("Unknown target event")
    rng = stream(split, base_index, variant_index, "residual_donor_order",
                 development=development).generator()
    for i in rng.permutation(len(order)):
        if order[int(i)] != target_event:
            return order[int(i)]
    raise AssertionError("No eligible residual donor")


def ledger(event_ids, split, variants_per_base, *, development=False):
    """Metadata only; iteration is variant-major for balanced nested prefixes.

    For formal OOS, donor_event remains unmaterialized. Stream identities and
    pairing IDs can be locked without drawing physical/test outcomes.
    """
    order = event_order(event_ids)
    max_count = {"planning": 20, "calibration": 10, "oos": 50}.get(split)
    if max_count is None or type(variants_per_base) is not int or not 1 <= variants_per_base <= max_count:
        raise ValueError("Invalid variant count for this split")
    rows = []
    domain = "development" if development else "formal"
    for variant_index in range(variants_per_base):
        for base_index, base_event in enumerate(order):
            pair_id = f"{domain}/{split}/{base_event}/v{variant_index+1:03d}"
            streams = {m: stream(split, base_index, variant_index, m,
                                development=development).as_dict() for m in MECHANISMS}
            for target in order:
                donor = (None if split == "oos" and not development else
                         residual_donor(order, target, split, base_index, variant_index,
                                        development=development))
                rows.append({
                    "target_event": target, "base_weather_event": base_event,
                    "variant": variant_index + 1, "split": split, "domain": domain,
                    "sample_id": f"{target}/{pair_id}", "pair_id": pair_id,
                    "error_donor_event": donor,
                    "donor_rule": "shared uniform event ordering with target removed",
                    "streams": streams,
                    "physical_payload_generated": False,
                })
    return rows


def contract(event_ids):
    return {
        "schema_version": SCHEMA, "event_order": list(event_order(event_ids)),
        "formal_roots": ROOTS, "development_root": DEVELOPMENT_ROOT,
        "static_road_root": STATIC_ROAD_ROOT,
        "static_road_policy": "Reuse stored canonical road parameters; never redraw by scenario.",
        "mechanism_ids": MECHANISMS,
        "key_fields": ["schema", "domain", "split", "shared_target_zero",
                       "base_one_based", "variant_one_based", "mechanism", "asset", "component"],
        "absent_from_keys": ["method", "regime", "target_event", "lead_h", "fleet_size", "support_size"],
        "residual_donor_rule": "Uniform shared event permutation; remove target and use first eligible.",
        "reference_time_grid_h": {"start": -2.0, "stop_exclusive": 24.0, "step": 0.25},
        "time_draw_policy": "Generate on fixed reference grid then select times; never shift RNG order with lead.",
        "asset_draw_policy": "Use canonical full asset universe or stable asset IDs before taking fleet/subset views.",
        "asset_component_index_semantics": "Already stable positive IDs; zero reserved for aggregate. Base/variant API alone converts zero-based positions to one-based keys.",
        "native_time_policy": "Reference grid addresses weather/replay. Load AR rho=0.8 is hourly; each physical mechanism must declare native timestep and mapping before implementation.",
        "nested_support": "Per base take first 5/10/15/20 variants; variant-major pairs.",
        "OOS_policy": "Only metadata/seed identities until batch plan lock; no physical outcome generation here.",
    }
