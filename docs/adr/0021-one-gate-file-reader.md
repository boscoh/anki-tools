# One gate-file reader above the two packages

`CoreFiles`, `CullFiles`, and `AuditInputs` each held the same lazy read cache.
The cache stored `(ok, value)` and re-raised the error, so one missing input
failed only the gates that read it. Each adapter also repeated the pass-through
accessors of the `GateFiles` protocol.

Status: accepted

## Decision

`GateFileReader` in `anki_tools/gate_files.py` holds the cache and the
accessors. It takes one loader table from an accessor name to the callable that
reads the file. `CoreFiles`, `CullFiles`, and `AuditInputs` subclass it and
supply only that table, so no gate function changes. The reader also owns
`read_notes`, which the core and cull adapters repeated.

## Consequences

The cached-error rule has one implementation and one test,
`test_a_missing_file_fails_only_the_gates_that_read_it`. The cache annotation is
`dict[str, tuple[bool, Any]]`, which is the value that it stores.

## Rejected

`anki_tools/ranking/gates.py` as the home. `ranking.gates` imports
`ranking.assemble`, and `assemble` imports `sentence_pack.audit`. A reader in
`gates.py` that `audit.py` imports closes that cycle and fails on the partially
started module. The module above both packages breaks the cycle and holds no
language package, so `tests/test_language_layout.py` stays green.
