# Historical source reference

`baseline/` preserves the 691 Python files from `pre-refactor-v2-20261111` byte for byte.
Original paths/hashes: `docs/architecture/baseline-source-manifest.json`.
Current locations: `docs/architecture/reference-locations.json`.

These files are not shipped, imported by active code, or included in the default test
collection. Do not execute old operational scripts against current inputs/campaigns.
The standalone reference distribution is solely for cross-version scientific tests:
`python -m pip install --no-deps ./reference/baseline` in an explicit validation environment.
It adds no console entry points. Uninstall `ev-assignment-reference` after comparison.
Historical assertions remain intact; missing private evidence is not a passing test.
