# Reproduction export

The private code repository stores no experiment results. Use `ev-assignment bundle`
against an explicitly selected external archive. The generated local package contains
`manifest.json`, `archive/`, exact installable `code/<source SHA>/`, `tables/`,
`README.md` and `reproduce.py`.

`figures`/`verify` use sealed evidence without a solver. `smoke` requires an explicit new
store and Gurobi 12.0.3. `full` requires the selected attempt, original exact source,
an explicit new store and a stated new reproduction budget. It cannot reset an old run.

Publication items bind a label, attempt, audited decision and named generator. Add a
new generator with a test when the paper needs a different figure/table. Use a separate
public repository and versioned release assets for large evidence. Public upload and
visibility changes are not part of engineering bundle creation.
