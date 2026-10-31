# Spec: One gate layer and one gate set for every flow

| Field   | Value                                                                   |
| ------- | ----------------------------------------------------------------------- |
| Status  | Phase 1 and the consolidation done; pack G1 and step 7 open             |
| Date    | 2026-11-01                                                              |
| Owner   | anki-tools                                                              |
| Related | `docs/gate-inventory.md`, `docs/gate-filter-comparison.md`, `anki-3fc9` |

## 1. Purpose

This spec has two phases.

Phase 1 gives every flow one gate value, one row type, and one runner. The pack
keeps a second runner and a second row type today.

Phase 2 gives every gate family one check function. The same family runs in
every flow that activates it. A duplicate, superseded, or report-only row
leaves the table or moves to the report.

The gate ids stay. ADR 0027 records the per-flow namespaces. A change in phase
2 changes the function behind an id, not the id.

## 2. Scope

In scope:

- Phase 1: one `Gate` value, one `GateResult` row, one `run_gates` runner.
- Phase 1: one home for the contract, below `ranking` and `sentence_pack`.
- Phase 1: the pack adapter that supplies the profile to a pack check.
- Phase 2: the gate inventory below.
- Phase 2: one check per gate family, shared by every flow.
- Phase 2: the duplicate, superseded, and report-only rows.
- Phase 2: the function renames.

Out of scope:

- The gate ids and their order. ADR 0027.
- The filter rule order, the filter keys, and the filter marks.
- The core mark source, L3, tracked by `anki-0d3d`.
- New gate thresholds.
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

## 4. Gate inventory across the flows

The four deck builders held 59 gate table rows and about 77 gate functions.
About 21 functions were copies of another function. Some rows only reported
and could not fail. This table gives the state before phase 2. Section 7.1
gives the final state.

| Family            | What the gate checks                       | Flows and ids                                           |
| ----------------- | ------------------------------------------ | ------------------------------------------------------- |
| Word table        | the reviewed table passes every S1 check   | pack G1                                                 |
| Unique forms      | no duplicate word form                     | pack G2                                                 |
| Targets in table  | every target is a table form               | pack G3, G19; Greek G19                                 |
| Unique Tatoeba id | no repeated source sentence                | pack G4                                                 |
| Target once       | the target appears one time                | pack G5                                                 |
| Similarity        | no near-duplicate pair                     | pack G6, cull G6, core K6                               |
| Sentence audio    | every card has a valid sentence MP3        | pack G7                                                 |
| Media count       | the audio file count equals the APKG media | pack G8                                                 |
| Word boundary     | the target matches a whole token           | pack G9 (FR, ES, EL)                                    |
| Coverage          | coverage reaches the gate share            | pack G10, cull G3, cull G4, core K2, core K3            |
| Manifest counts   | counts and hashes agree                    | pack G11, cull G8, cull G13, core K1, core K8, core K12 |
| Attribution       | the author section names every author      | pack G12                                                |
| Output model      | the APKG model, fields, and ordinals       | pack G13, cull G16, core K21                            |
| Filter recheck    | the kept cards pass the filters            | pack G14, cull G5, core K5                              |
| Review file       | no open row in the review CSV              | pack G15                                                |
| Word audio        | every target has a valid word MP3          | pack G16                                                |
| Language tail     | typography, script, gender, license, sense | pack G17 to G21                                         |
| Card cap          | the count is the cap or the pool end       | cull G1                                                 |
| Use cap           | no lemma exceeds the use cap               | cull G2                                                 |
| Audio valid       | every card has a valid MP3                 | cull G7, core K7                                        |
| Media set         | the media map equals the kept cards        | cull G9                                                 |
| Identity          | the deck title equals the manifest         | cull G10, core K9                                       |
| Order             | the kept order digest                      | cull G11, core K10                                      |
| Coverage rows     | the coverage CSV shape and counts          | cull G12, core K11                                      |
| Swadesh           | the reserve coverage                       | cull G14                                                |
| Templates         | the card templates and the CSS             | cull G15, core K14                                      |
| Core word         | the word, the gloss, and the bold          | cull G17, core K19                                      |
| Bold target       | the bold text is the target                | core K15                                                |
| Outside set       | the outside set and rare ratio             | core K4                                                 |
| Target hit        | the card holds its target form             | core K13                                                |
| Sense limit       | the unchecked targets stay below the limit | core K16                                                |
| Name report       | the rejected names and loan words          | core K17                                                |
| Tail              | every tail card brings a lemma             | core K18                                                |
| Unranked          | the unranked targets stay below the limit  | core K20                                                |

## 5. Target contract

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

## 6. Phase 1: one runner

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

**Behavior**: none. The runner catches the same error. The row fields and the
row order are the same. The report text and the exit code are the same.

## 7. Phase 2: one check per family

### 7.1 Actions

Phase 1 and the consolidation are done, except pack G1. The final decisions:

- pack G3 checks the table form and the shared form. The French G19
  `gate_discontinuous` and the Spanish G19 `gate_pair_parts` stay. Greek and
  Cantonese G19 are gone.
- pack G5 checks the target once. G9 stays only as the Cantonese script check
  and the Mandarin pinyin check.
- pack G20 is gone. The count stays in the manifest as `noun_gender_missing`.
- core K3, K15, and K17 are gone. Their values are in the core report:
  Covered twice, Rejected names, and Rejected loan words.
- K1 and K8 are not merged. The two functions stay separate.
- The renames in section 7.2 are done.
- pack G13 stays until the one output reader lands. Then it uses `gate_output_model`.
- pack G1 is open. Five modules define `gate_word_table`. The French and
  Spanish `check_word_table` take no `csv_path`, so the shared check needs a
  profile hook.

A removed row leaves the table. Its value moves to the report, or the check
joins a shared function behind a kept id.

### 7.2 Renames

The renames are done. The function name changed. The gate id and the check
did not. The functions that belong to a removed row are deleted, not renamed:
`gate_covered2`, `gate_bold`, `gate_names`, `gate_no_native`, and `gate_targets`.

| Current                 | Proposed                  | Flow             |
| ----------------------- | ------------------------- | ---------------- |
| `gate_covered1`         | `gate_covered_targets`    | core K2          |
| `gate_hit`              | `gate_target_in_sentence` | core K13         |
| `gate_sense`            | `gate_unchecked_sense`    | core K16         |
| `gate_card_count`       | `gate_card_cap`           | cull G1          |
| `gate_filter`           | `gate_filter_recheck`     | cull G5, core K5 |
| `gate_proper_sentences` | `gate_filter_recheck`     | pack G14         |
| `gate_target_coverage`  | `gate_coverage_once`      | cull G3          |
| `gate_double_coverage`  | `gate_coverage_twice`     | cull G4          |
| `gate_coverage`         | `gate_coverage_agreement` | core K11         |
| `gate_coverage_rows`    | `gate_coverage_shape`     | cull G12         |

Cull G12 and core K11 had close names, but the checks differ. The new names
state the difference.

### 7.3 What stays different

- The gate table per flow. The pack has 18 to 20 gates per language. The cull
  has G1 to G17. The core has K1 to K21, 18 rows. The vocabulary deck has no
  gate table.
- The gate ids. ADR 0027.
- The pack check signature `(inputs, profile)`. A `check(files)` form would
  touch about 60 pack functions and gains nothing. Do it only when a pack check
  must serve another flow.
- The pack reader `AuditInputs` keeps its own accessors: `table`, `index`,
  `apkg`, and `attribution_path`.
- Cull G1 checks the cap or the pool end. Core K1 checks a three-way
  agreement. The two functions stay separate.

### 7.4 Behavior

The kept deck does not change. A gate reads a built file and reports. It does
not write a file.

The pass or fail can change in one place: the core audio check, when cull G7
and core K7 become one check. The cull check reads the media map. The core can
start to fail where it did not.

The report text changes for the removed rows and the renamed functions. The
core table loses K3, K15, and K17. The pack table loses G20. The core report
prints Covered twice, Rejected names, and Rejected loan words.

### 7.5 Verification

After phase 1 and after each phase 2 tier, run the focused tests:

```bash
uv run pytest tests/test_gates.py tests/test_gate_id_map.py \
  tests/test_mandarin_audit.py tests/test_french_cull.py \
  tests/test_core_deck.py tests/test_shared_coreflow.py
```

Then run the gate table of each flow and compare it with the table before the
tier:

```bash
uv run anki swadesh audit french
uv run anki fix fr apkg/french_1000.apkg --select --check
uv run anki fix fr apkg/french_core.apkg --core --check
```

A changed pass or fail is a defect, except for the core audio check in the rank
pair tier. The removed rows change the table length and the message. That is
the expected result.

## 8. Order

| Step | Work                        | Behavior                                 | Risk          | Status |
| ---- | --------------------------- | ---------------------------------------- | ------------- | ------ |
| 1    | the runner and the row type | none                                     | low           | done   |
| 2    | the eight language copies   | none                                     | low           | open   |
| 3    | the six rank pairs          | none, except the core audio check        | low to medium | done   |
| 4    | the superseded pack checks  | none, except G13 after the output reader | medium        | open   |
| 5    | the renames                 | none                                     | low           | done   |
| 6    | the report-only removals    | report and table length                  | low           | done   |
| 7    | the one output reader       | pack G13 changes                         | high          | open   |

Step 1 is tracked by `anki-3fc9`. Steps 1, 3, 5, and 6 are done. Step 4 is
open only for pack G13; G3 and G5 are done. Step 2 is open because of pack G1.
Step 7 is open and is separate work.

## 9. References

| Path                                    | Content                                          |
| --------------------------------------- | ------------------------------------------------ |
| `docs/gate-inventory.md`                | the gate inventory and the detailed refactor     |
| `docs/gate-filter-comparison.md`        | the gate and filter comparison of the four flows |
| `docs/adr/0021-one-gate-file-reader.md` | the module above both packages                   |
| `docs/adr/0027-gate-id-namespaces.md`   | the per-flow gate ids                            |
| `anki_tools/ranking/gates.py`           | the shared gate checks and the report call       |
| `anki_tools/sentence_pack/audit.py`     | `AuditInputs`, `GENERIC_GATES`, `run_pack_audit` |
| `anki_tools/gate_files.py`              | the lazy file reader                             |
