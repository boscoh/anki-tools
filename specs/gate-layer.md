# Spec: One gate layer for every flow

| Field   | Value                                         |
| ------- | --------------------------------------------- |
| Status  | Proposed                                      |
| Date    | 2026-11-01                                    |
| Owner   | anki-tools                                    |
| Related | `docs/gate-filter-comparison.md`, `anki-3fc9` |

## 1. Purpose

Four deck builders audit a built deck with a gate table. The cull and the core
use one gate value and one runner. The pack keeps a second runner and a second
row type. This spec fixes the gate layer only. It gives every flow one gate
value, one row type, and one runner. It keeps the present gate ids, the gate
lists, and the report.

The spec does not change a filter, a manifest, or a gate threshold.

## 2. Scope

In scope:

- One `Gate` value and one `GateResult` row.
- One `run_gates` runner for the pack, the cull, and the core.
- One home for the contract, below `ranking` and `sentence_pack`.
- The pack adapter that supplies the profile to a pack check.
- The docstrings and the flow docs.

Out of scope:

- The filter rule order, the filter keys, and the filter marks.
- The core mark source, L3, tracked by `anki-0d3d`.
- The pack manifest shape, L4.
- One gate id namespace, L5, rejected by ADR 0027.
- Identical gate sets, L6, rejected by `docs/gate-filter-comparison.md`.
- The similarity property of pack G6 and rank G6/K6.
- The deck-specific gate failures: `anki-0s3`, `anki-lb1s`, `anki-ann4`,
  `anki-tazk`.

## 3. Current state

The gate seam has five parts. The report and the reader are shared. The runner
and the row type are shared by the cull and the core only.

| Part   | Name                                | Location                            | Pack | Cull | Core |
| ------ | ----------------------------------- | ----------------------------------- | ---- | ---- | ---- |
| Gate   | `Gate(name, check)`                 | `ranking/gates.py`                  | no   | yes  | yes  |
| Row    | `GateResult(name, passed, message)` | `ranking/gates.py`                  | no   | yes  | yes  |
| Runner | `run_gates(gates, files)`           | `ranking/gates.py`                  | no   | yes  | yes  |
| Report | `print_audit(results, title)`       | `sentence_pack/audit.py`            | yes  | yes  | yes  |
| Reader | `GateFiles`, `GateFileReader`       | `ranking/gates.py`, `gate_files.py` | yes  | yes  | yes  |

The pack declares `(name, check)` tuples and runs its own `run_gates`. The pack
row is `GateRow`, a plain tuple. The language wrappers return `list[GateRow]`.

The layering is the constraint. `ranking/gates.py` imports `print_audit` from
`sentence_pack.audit`. The pack module cannot import `Gate` from
`ranking.gates`, because that import returns to a partly started module. ADR
0021 records the same constraint for the file reader. The contract must live
below both packages.

## 4. Target contract

Add `anki_tools/gates.py`, at the same level as `gate_files.py`. It holds four
names and imports neither `ranking` nor `sentence_pack`.

| Name         | Kind             | Shape                                                                            |
| ------------ | ---------------- | -------------------------------------------------------------------------------- |
| `GateResult` | `NamedTuple`     | `name: str`, `passed: bool`, `message: str`                                      |
| `GateFiles`  | `Protocol`       | `profile`, `rows`, `coverage`, `manifest`, `params`, `output`, `notes`, `fields` |
| `Gate`       | frozen dataclass | `name: str`, `check: Callable[[Any], tuple[bool, str]]`                          |
| `run_gates`  | function         | `(gates: Sequence[Gate], files: GateFiles) -> list[GateResult]`                  |

The pack keeps its `(inputs, profile)` check functions. One adapter wraps the
pack table:

```python
def as_gates(gates: Sequence[tuple[str, GateCheck]]) -> tuple[Gate, ...]:
    return tuple(
        Gate(name, lambda files, check=check: check(files, files.profile))
        for name, check in gates
    )
```

`AuditInputs` holds `profile`, so the adapter needs nothing else. `run_pack_audit`
calls `run_gates(as_gates(gates), inputs)`.

## 5. Change steps

1. Add `anki_tools/gates.py`. Move `GateResult`, `GateFiles`, `Gate`, and
   `run_gates` from `ranking/gates.py`. Do not rewrite them.
2. In `ranking/gates.py`, import the four names and delete the local
   definitions. Keep `report`, `read_csv`, `read_json`, `sample`, and the shared
   checks.
3. In `sentence_pack/audit.py`, import `Gate`, `GateResult`, and `run_gates`.
   Delete the local `run_gates` and `GateRow`. Add `as_gates`. Let
   `run_pack_audit` return `list[GateResult]`, and let `print_audit` take
   `Sequence[GateResult]`.
4. Replace `GateRow` with `GateResult` in the Mandarin, Greek, French, Spanish,
   and Cantonese language wrappers.
5. In `gate_files.py`, point the docstring to `anki_tools.gates.GateFiles`.
6. Update `docs/gate-filter-comparison.md` sections 3.2 and 3.4,
   `docs/anki-tools-layout.md`, `docs/cull-deck-flow.md`, and
   `docs/core-deck-flow.md`.

Amend ADR 0021 with the new home. The boundary is the one of ADR 0021, so no
new ADR is necessary.

## 6. What stays different

- The gate table per flow. The pack has 18 to 21 gates per language. The cull
  has G1 to G17. The core has K1 to K21. The vocabulary deck has no gate table.
- The gate ids. ADR 0027.
- The pack check signature `(inputs, profile)`. A `check(files)` form would
  touch about 60 pack functions and gains nothing. Do it only when a pack check
  must serve another flow.
- The pack reader `AuditInputs` keeps its own accessors: `table`, `index`,
  `apkg`, and `attribution_path`.

## 7. Behavior

None. The runner catches the same error. The row fields and the row order are
the same. The report text and the exit code are the same.

## 8. Verification

Run the focused tests:

```bash
uv run pytest tests/test_gates.py tests/test_gate_id_map.py \
  tests/test_mandarin_audit.py tests/test_french_cull.py \
  tests/test_core_deck.py tests/test_shared_coreflow.py
```

Then run the full suite.

Add one test in `tests/test_gates.py`: `run_pack_audit` returns `GateResult`
rows, and a pack gate runs through the adapter.

Compare one report before and after the change:

```bash
uv run anki swadesh audit french > /tmp/audit-after.txt
uv run anki fix fr apkg/french_1000.apkg --select --check
```

The gate table must match the table before the change, line for line.

## 9. Risks

- The new module must stay free of `ranking` and `sentence_pack` imports. A
  cycle fails at import time.
- `print_audit` must accept `GateResult` without a second report. A
  `NamedTuple` unpacks as a tuple, so the present loop works.
- The language wrappers change their return annotation only. The values are the
  same.
- The pack report must not change. Compare the output.

## 10. References

| Path                                    | Content                                          |
| --------------------------------------- | ------------------------------------------------ |
| `docs/gate-filter-comparison.md`        | the gate and filter comparison of the four flows |
| `docs/adr/0021-one-gate-file-reader.md` | the module above both packages                   |
| `docs/adr/0027-gate-id-namespaces.md`   | the per-flow gate ids                            |
| `anki_tools/ranking/gates.py`           | the shared gate checks and the report call       |
| `anki_tools/sentence_pack/audit.py`     | `AuditInputs`, `GENERIC_GATES`, `run_pack_audit` |
| `anki_tools/gate_files.py`              | the lazy file reader                             |
