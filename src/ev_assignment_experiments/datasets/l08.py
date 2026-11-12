"""Explicit L08 frozen-input adapter. No source-location inference or OOS access."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from ev_assignment.domain.planning import PlanningProblem, Scenario, AmbiguitySet
from ev_assignment.l08.certificates import enclosure
from ev_assignment_experiments.tracking.transfer import file_digest

R36 = "L08_K4L4_DRO_READY_PARALLEL_R36"
R29 = "L08_K4L4_CALIBRATION_AND_FORECAST_R29"
R28 = "L08_SAME_POOL_BALANCED_SELECT400_R28"
R41 = "L08_DRO_R41_NESTED_BENDERS_ROOT/REPAIR_01"


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class FrozenInputs:
    def __init__(self, root, manifest_sha256):
        self.root = Path(root).expanduser().resolve()
        manifest = self.root / "import-manifest.json"
        if file_digest(manifest) != manifest_sha256:
            raise ValueError("Frozen input manifest changed")
        data = json.loads(manifest.read_text())
        if data.get("schema_version") != 1 or data.get("kind") != "l08-frozen-input-import":
            raise ValueError("Unsupported input manifest")
        self.identity = manifest_sha256
        self.entries = {r["path"]: r for r in data["files"]}
        if len(self.entries) != len(data["files"]):
            raise ValueError("Duplicate input manifest path")
        self._verified = {}

    def path(self, name):
        if name not in self.entries:
            raise ValueError("Undeclared input: " + name)
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Input escapes root")
        path = self.root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(self.root):
            raise ValueError("Input symlink escapes manifest")
        stat = path.stat()
        stamp = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if self._verified.get(name) != stamp:
            ref = self.entries[name]
            if stat.st_size != ref["bytes"] or file_digest(path) != ref["sha256"]:
                raise ValueError("Frozen input digest mismatch: " + name)
            self._verified[name] = stamp
        return path

    def read(self, name):
        return json.loads(self.path(name).read_text(encoding="utf-8-sig"))

    def npz(self, name):
        with np.load(self.path(name), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def verify(self):
        self._verified.clear()
        for name in sorted(self.entries):
            self.path(name)
        return dict(
            passed=True,
            manifest=self.identity,
            files=len(self.entries),
            bytes=sum(v["bytes"] for v in self.entries.values()),
            oos_opened=False,
        )


class L08Dataset:
    def __init__(self, root, manifest_sha256):
        self.inputs = FrozenInputs(root, manifest_sha256)
        self.input_identity = self.inputs.read(R36 + "/02_SOURCE_AND_CONFIG/INPUT_IDENTITY.json")
        if (
            self.input_identity["forecast_source_sid"] != 44
            or self.input_identity["support_size"] != 400
        ):
            raise ValueError("Not the frozen L08 forecast44/support400 input")
        net = self.inputs.read(R36 + "/02_SOURCE_AND_CONFIG/NETWORK_NEW_CLOCK.json")
        with self.inputs.path(R36 + "/02_SOURCE_AND_CONFIG/C__evcs_bus_road_mapping.csv").open(
            encoding="utf-8-sig"
        ) as f:
            stations = list(csv.DictReader(f))
        net["from"] = net["line_from"]
        net["to"] = net["line_to"]
        net["station_buses"] = [int(x["n_c"]) for x in stations]
        critical = {2, 4, 8, 10, 14, 21, 24, 27, 30, 32}
        net["shed_cost"] = [50.0 if b in critical else 10.0 for b in net["buses"]]
        self.network = net
        complete = self.inputs.read(R36 + "/03_FORECAST44_ROUTES/COMPLETE.json")
        route_path = R36 + "/03_FORECAST44_ROUTES/ROUTES.jsonl"
        if self.inputs.entries[route_path]["sha256"] != complete["route_sha"]:
            raise ValueError("Route provenance differs from locked source")
        domain = self.inputs.npz(R36 + "/04_R10_COMPACT_DOMAIN/DOMAIN.npz")
        dc = self.inputs.read(R36 + "/04_R10_COMPACT_DOMAIN/COMPLETE.json")
        if (
            self.inputs.entries[R36 + "/04_R10_COMPACT_DOMAIN/DOMAIN.npz"]["sha256"]
            != dc["domain_sha"]
        ):
            raise ValueError("Domain provenance differs from locked source")
        routes = []
        with self.inputs.path(route_path).open() as handle:
            for line in handle:
                q = json.loads(line)
                if (
                    q["forecast_candidate"] != 44
                    or q["input_identity"] != self.input_identity["identity"]
                ):
                    raise ValueError("Mixed forecast route identity")
                rid = q["route_id"]
                if q["route"] and domain["offset"][rid + 1] > domain["offset"][rid]:
                    routes.append(
                        dict(
                            route_id=rid,
                            e=q["e"],
                            c=q["c"],
                            d=q["d"],
                            departure_dc=q["departure_dc"],
                            distance=q["route"]["distance_km"],
                            arrival=q["route"]["arrival_h"],
                            B=q["arrival_dc"] - 100,
                            ev=q["ev"],
                        )
                    )
        if len(routes) != dc["qualified_route_rows"] or len(routes) != 20100:
            raise ValueError("Frozen compact route count changed")
        self.routes = tuple(routes)
        self.bounds = np.asarray(
            self.inputs.read(R36 + "/04_R10_COMPACT_DOMAIN/GLOBAL_ENERGY_BOUNDS.json")["M"]
        )
        self._ambiguity = None

    def scene(self, identity):
        if identity == "forecast44":
            folder = R29 + "/07_FORECAST512/scenes/0044"
            replay = "FORECAST44"
        elif (
            identity.startswith("train/")
            and identity[6:].isdigit()
            and 0 <= int(identity[6:]) < 400
        ):
            sid = int(identity[6:])
            folder = R28 + f"/04_SELECTED400/scenes/{sid:04d}"
            replay = f"{sid:04d}"
            if identity != f"train/{sid:04d}":
                raise ValueError("Noncanonical support identity")
        else:
            raise ValueError("Only locked forecast/train inputs are admitted; OOS is closed")
        a = self.inputs.npz(folder + "/physical.npz")
        r = self.inputs.npz(folder + "/recovery.npz")
        ix = np.searchsorted(r["times_h"], np.arange(97) / 4)
        if np.any(ix >= len(r["times_h"])) or not np.array_equal(
            r["times_h"][ix], np.arange(97) / 4
        ):
            raise ValueError("Recovery clock mismatch")
        a["source_physical_operation"] = a["line_operating_outages"].copy()
        a["line_operating_outages"] = r["outage_current"][ix[:-1]].copy()
        a["line_terminal_outages"] = r["outage_current"][ix[-1]].copy()
        if not np.array_equal(a["load_q_kvar"], 0.33 * a["load_p_kw"]):
            raise ValueError("Frozen power-factor semantics changed")
        replay_root = R36 + "/05_SUPPORT400_ROUTE_REPLAY/"
        meta = self.inputs.read(replay_root + replay + ".json")
        observed = self.inputs.entries[replay_root + replay + ".npz"]["sha256"]
        expected_energy = self.inputs.entries[R36 + "/tools/energy_ledger.py"]["sha256"]
        self.inputs.path(R36 + "/tools/energy_ledger.py")
        key = digest(
            dict(
                input_identity=self.input_identity["identity"],
                route_sha=self.inputs.entries[R36 + "/03_FORECAST44_ROUTES/ROUTES.jsonl"]["sha256"],
                source=self.inputs.read(folder + "/PRIMITIVES_AND_IDENTITY.json"),
                physical=self.inputs.entries[folder + "/physical.npz"]["sha256"],
                recovery=self.inputs.entries[folder + "/recovery.npz"]["sha256"],
                geometry=self.inputs.entries[replay_root + "FIXED_ROUTE_GEOMETRY.npz"]["sha256"],
                energy=expected_energy,
            )
        )
        if meta["cache_key"] != key or meta["sha256"] != observed:
            raise ValueError("Replay identity mismatch")
        z = self.inputs.npz(replay_root + replay + ".npz")
        states = tuple(
            dict(success=True, arrival=q["arrival"])
            if identity == "forecast44"
            else dict(
                success=bool(z["success"][q["route_id"]]),
                arrival=float(z["arrival_h"][q["route_id"]]),
            )
            for q in self.routes
        )
        return Scenario(identity, a, states)

    def ambiguity(self):
        if self._ambiguity is not None:
            return self._ambiguity
        prefix = R29 + "/05_DRO_MOMENTS/"
        co = self.inputs.read(prefix + "COMPRESSION_COMPLETE.json")
        for filename, key in [
            ("SUPPORT_UNIQUE.npy", "rows_sha"),
            ("SUPPORT_INVERSE.npy", "inverse_sha"),
            ("EFFECTIVE_INTERVALS.npz", "bounds_sha"),
        ]:
            if self.inputs.entries[prefix + filename]["sha256"] != co[key]:
                raise ValueError("Moment compression provenance changed")
        raw = np.load(self.inputs.path(prefix + "X_SUPPORT.npy"), mmap_mode="r", allow_pickle=False)
        phi = np.load(
            self.inputs.path(prefix + "SUPPORT_UNIQUE.npy"), mmap_mode="r", allow_pickle=False
        )
        inv = np.load(self.inputs.path(prefix + "SUPPORT_INVERSE.npy"), allow_pickle=False)
        bounds = self.inputs.npz(prefix + "EFFECTIVE_INTERVALS.npz")
        rb = self.inputs.npz(prefix + "CALIBRATED_RAW_BOUNDS_K4L4.npz")
        if raw.shape != (400, 322311) or phi.shape != (45139, 400):
            raise ValueError("Frozen moment domain changed")
        low = np.full(45139, -np.inf)
        high = np.full(45139, np.inf)
        np.maximum.at(low, inv, rb["lower"])
        np.minimum.at(high, inv, rb["upper"])
        if not np.array_equal(low, bounds["lower"]) or not np.array_equal(high, bounds["upper"]):
            raise ValueError("Compressed moment intervals are not exact")
        for i in range(0, 322311, 2048):
            if not np.array_equal(raw[:, i : i + 2048].T, phi[inv[i : i + 2048]]):
                raise ValueError("Moment compression changed the original ambiguity set")
        scale = np.maximum(
            np.max(np.abs(phi), axis=1), np.maximum(np.maximum(abs(low), abs(high)), 1e-3)
        )
        scaled = sparse.csr_matrix(phi / scale[:, None])
        lo = low / scale
        hi = high / scale
        raw_unique = sparse.csr_matrix(phi)
        rmin = np.asarray(raw_unique.min(axis=1).toarray()).ravel()
        rmax = np.asarray(raw_unique.max(axis=1).toarray()).ravel()
        smin = np.asarray(scaled.min(axis=1).toarray()).ravel()
        smax = np.asarray(scaled.max(axis=1).toarray()).ravel()

        def check(p):
            p = np.asarray(p, float)
            if p.shape != (400,) or not np.isfinite(p).all() or np.any(p < 0):
                return dict(passed=False, reason="ORIGINAL_SIMPLEX")
            den = 2**52
            numerators = np.rint(p * den).astype(np.int64)
            exact = bool(
                np.array_equal(numerators.astype(float) / den, p)
                and sum(map(int, numerators)) == den
                and (numerators >= 0).all()
            )
            al, au = enclosure(raw_unique, p)
            sl, su = enclosure(scaled, p)
            if exact:
                al = np.maximum(al, rmin)
                au = np.minimum(au, rmax)
                sl = np.maximum(sl, smin)
                su = np.minimum(su, smax)
            margin = min(float(np.min(al - low)), float(np.min(high - au)))
            sm = min(float(np.min(sl - lo)), float(np.min(hi - su)))
            return dict(
                passed=exact and margin >= 0 and sm >= 0,
                exact_simplex=exact,
                raw_margin=margin,
                scaled_margin=sm,
                all_effective_rows=45139,
                all_original_rows=322311,
            )

        anchor = self.inputs.npz(
            "L08_DRO_R42_JOINT_SEPARATION_PILOT/PROBABILITY_INTERIOR_STRICT/P_STAR.npz"
        )["p"]
        self._ambiguity = AmbiguitySet(
            tuple(f"train/{i:04d}" for i in range(400)),
            scaled,
            lo,
            hi,
            digest(dict(manifest=self.inputs.identity, moments=co)),
            check,
            anchor,
        )
        return self._ambiguity

    def problem(self, method, *, include_ambiguity=True):
        if method not in {"DET", "DRO", "ARO"}:
            raise ValueError("Unknown method")
        ids = ("forecast44",) if method == "DET" else tuple(f"train/{i:04d}" for i in range(400))
        proof = None
        if method == "DET":
            r32 = "L08_DET_CERTIFIED_SCREENING_AND_LOCAL_SEARCH_R32/"
            r38 = "L08_DET_CAPACITY_CUTS_AND_DRO_RECOVERY_R38/instructions/L08_DET_CAPACITY_CUTS_AND_DRO_RECOVERY_R38/evidence/"
            proof = dict(
                screening=self.inputs.read(r32 + "proof/SCREENING_CERTIFICATE.json"),
                zero_y=self.inputs.npz(r32 + "frozen/ZERO_INPUT_WITNESS.npz")["y"],
                cuts=self.inputs.read(r38 + "DET_INTEGER_CAPACITY_CUTS.json"),
                definitions=self.inputs.read(r38 + "INTEGER_CAPACITY_CUT_DEFINITIONS.json"),
            )
        return PlanningProblem(
            identity=digest(dict(inputs=self.inputs.identity, method=method)),
            protocol="L08_FULL_DWELL_FORECAST44_V1",
            routes=self.routes,
            fleet_size=30,
            network=self.network,
            energy_bounds=self.bounds,
            scenario_ids=ids,
            scenario=self.scene,
            ambiguity=self.ambiguity() if method == "DRO" and include_ambiguity else None,
            scope="L08_FROZEN400",
            det_proof=proof,
        )
