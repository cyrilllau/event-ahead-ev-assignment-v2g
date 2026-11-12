"""Independent behavioral tests of reproducibility, separation and pairing."""
import unittest
from unittest.mock import patch

import numpy as np
from ev_assignment.domain.random_streams import ledger, residual_donor, stream

EVENTS = tuple(f"event_{i}" for i in range(8))


class RandomStreamTests(unittest.TestCase):
    def test_reordered_calls_reproduce_and_split_streams_differ(self):
        addresses = [(b, v, m) for b in range(2) for v in range(3)
                     for m in ["wind_amplitude", "load_innovation", "line_gaussian"]]
        original = {a: stream("planning", *a, development=True).generator().normal(size=12)
                    for a in addresses}
        for a in reversed(addresses):
            np.testing.assert_array_equal(original[a],
                stream("planning", *a, development=True).generator().normal(size=12))
        ids = {stream(s, 0, 0, m, development=d).stream_id
               for s in ["planning", "calibration", "oos"]
               for m in ["wind_amplitude", "load_innovation"] for d in [True, False]}
        self.assertEqual(len(ids), 12)

    def test_nested_support_and_targets_share_exogenous_noise(self):
        small = ledger(EVENTS, "planning", 5)
        large = ledger(EVENTS, "planning", 20)
        self.assertEqual(small, large[:len(small)])
        self.assertEqual(len({r["pair_id"] for r in small}), 40)
        self.assertEqual(len({r["pair_id"] for r in large}), 160)
        for batch in [small[i:i+8] for i in range(0, len(small), 8)]:
            self.assertEqual(len({r["streams"]["road_uniform"]["stream_id"] for r in batch}), 1)
            self.assertEqual(len({r["sample_id"] for r in batch}), 8)

    def test_donor_excludes_target_and_uses_shared_order(self):
        for v in range(50):
            donors = {target: residual_donor(EVENTS, target, "planning", 2, v, development=True)
                      for target in EVENTS}
            self.assertTrue(all(t != d for t, d in donors.items()))
            # One shared ordering means 7 targets use its first event, and
            # that excluded target uses the second. Check the coupling itself.
            counts = sorted(list(donors.values()).count(d) for d in set(donors.values()))
            self.assertEqual(counts, [1, 7])

    def test_formal_oos_locks_metadata_only(self):
        with patch("ev_assignment.domain.random_streams.Stream.generator",
                   side_effect=AssertionError("Formal OOS must not sample")):
            rows = ledger(EVENTS, "oos", 50)
        self.assertEqual(len(rows), 8*400)
        self.assertTrue(all(r["error_donor_event"] is None for r in rows))
        self.assertTrue(all(not r["physical_payload_generated"] for r in rows))

    def test_invalid_addresses_fail(self):
        for kwargs in [{"base_index": 8}, {"variant_index": -1}, {"asset_index": True}]:
            values = dict(split="planning", base_index=0, variant_index=0, mechanism="wind_lag")
            values.update(kwargs)
            with self.assertRaises(ValueError):
                stream(**values)


if __name__ == "__main__":
    unittest.main(verbosity=2)
