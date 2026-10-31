# Gate and filter comparison across flows

## 1. Scope

This document compares the gates and the filters of the four deck builders.
One builder has no gate table:

| Flow | Command | Gate table | Filter table |
| --- | --- | --- | --- |
| Sentence pack | `anki swadesh pack <lang>` | G1 to G21, 18 to 20 rows per language | F1 to F12, `unknown`, `sense` |
| Cull deck | `anki fix <lang> <input> --select` | G1 to G17 | F1 to F9 |
| Core deck | `anki fix <lang> <input> --core` | K1 to K21, 18 rows | F1 to F9, C1 to C5 |
| Vocabulary deck | `anki swadesh vocab <lang>` | none | none |

A **filter** rejects a sentence or a card before selection. The pack code calls
a filter a guard. A **gate** audits a built file and reports a fault. A gate
never rejects a sentence. The glossary holds both terms.

A rule key and a gate id are valid only inside their owner table. ADR 0027
records the overlap. For example, pack `F8` is the internal-mark rule and rank
`F8` is the name rule. Pack `G5` tests a one-time target and cull `G5`
re-runs the filters.

The claims below match the code after phase 1 and phase 2, except pack G1
and the one output reader. Section 3.5 lists them.

## 2. Filters

### 2.1 Rule table

| Key | Sentence pack (`sentence_pack/pool.py`) | Cull deck (`ranking/cullflow.py`) | Core deck (`ranking/coreflow.py`) |
| --- | --- | --- | --- |
| F1 | an English link exists | English text has 2 to 12 words | English text has 2 to 12 words |
| F2 | unit count inside the window | token count inside `min_words` to `max_words` | token count inside `min_units` to `max_units` |
| F3 | script test: no Latin letter, no digit | no digit, bracket, link, or ellipsis | no digit, bracket, link, or ellipsis |
| F4 | no repeated pattern | the last character is a final mark | the last character is a final mark |
| F5 | the last character is a final mark | exactly one final mark | exactly one final mark |
| F6 | exactly one final mark | no quotation mark or dialogue dash | no quotation mark or dialogue dash |
| F7 | no quotation mark or dialogue dash | at most one internal mark | at most one internal mark |
| F8 | at most one internal mark | no `PROPN` token | no `PROPN` token |
| F9 | predicate tag | the sentence text is unique | the sentence text is unique |
| F10 | no orphan conjunction | - | - |
| F11 | English text has 2 to 12 words | - | - |
| F12 | dedupe key is new | - | - |
| `unknown` | unknown token budget | - | - |
| `sense` | English gloss holds the sense | - | - |
| C1 | - | - | has a `VERB`, `AUX`, or `ADJ` token |
| C2 | - | - | no orphan conjunction |
| C3 | - | - | no whole word of the name list |
| C4 | - | - | no whole word of the loan word list |
| C5 | - | - | the audio file is in the media map |

### 2.2 Rule order

The pack runs two phases. `SHAPE_GUARDS` holds F2 to F8 and runs first, because
those tests are cheap. The measure step reads the tokens and the facts. Then
`FACT_GUARDS` holds F9 to `sense`. `FILTER_KEYS` is the report order. A sentence
that passes no guard becomes a candidate for the lowest-index word it covers.

The cull runs one list in `filter_reason`: F1, F2, F3, the mark rules, F8, then
F9. The mark rules come from `check_sentence_shape` and keep their shape-guard
names in `MARK_RULES`. The cull stores the first failed rule as the card
`reason`.

The core runs `filter_cards` with the Latin default rule function, then C1 to
C5 in `filter_core_cards`. The core stores the first failed rule as the card
`reason`.

### 2.3 Differences

- The pack tests 14 keys, the cull tests 9, the core tests 14. The unique-text
  rule holds in all three. F2 is the length window in all three, but the pack
  measures the text and the rank flows count tokens.
- The pack gives the scaffold path an exemption. A scaffold sentence may fail
  F6 or F8, and it skips the `sense` guard (`SCAFFOLD_SHAPE_FAILURES`,
  `SCAFFOLD_SKIP_GUARDS`).
- The pack records a pass count for every guard before the first failure. The
  cull and core record the first failed rule as the card `reason`.
- The pack and the cull take the marks from the language profile. The core takes
  the Latin mark constants of `ranking/cardfilter.py`, because `CoreProfile`
  holds no mark fields. The Mandarin and Cantonese core adapters convert the
  final mark of each sentence to ASCII in `_sentence`, because the shared F4
  accepts only `.`, `?`, and `!`. The comments in `mandarin/core.py` and
  `cantonese/core.py` name that limit.
- The cull stores the mark-rule names through `MARK_RULES`: pack F5 to F8
  become cull F4 to F7. The pack has no `PROPN` rule. The cull has no
  predicate, orphan, English-word, unique-dedupe, unknown, or sense rule.
- The pack `F1` is a link test. A sentence with no English pair leaves the loop
  before the pool counts a guard. Rank `F1` is a word-window test.

## 3. Gates

### 3.1 Gate tables

Each workflow owns a table. The pack table differs by language. A removed row
leaves a gap in the numbering.

| Workflow                          | Active rows                        | Count |
| --------------------------------- | ---------------------------------- | ----- |
| Sentence pack, French and Spanish | G1 to G8, G10 to G19, G21          | 19    |
| Sentence pack, Greek              | G1 to G8, G10 to G18, G20          | 18    |
| Sentence pack, Mandarin           | G1 to G18                          | 18    |
| Sentence pack, Cantonese          | G1 to G18, G20, G21                | 20    |
| Cull deck                         | G1 to G17                          | 17    |
| Core deck                         | K1, K2, K4 to K14, K16, K18 to K21 | 18    |
| Vocabulary deck                   | none                               | 0     |

The pack rows mean different checks in different languages. Section 3.3 gives
the language tail.

| Workflow        | Table                                             | Scope                                                                                                 | Thresholds                                                                        |
| --------------- | ------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| Sentence pack   | `sentence_pack/audit.py` plus the language module | source word table, review CSV, sentence CSV, audio directory, attribution file, output APKG, manifest | `coverage_min`, for example 0.70 Cantonese to 0.95 French                         |
| Cull deck       | `ranking/cullgates.py`                            | output APKG, cull CSV, coverage CSV, manifest                                                         | `target_coverage_gate` 0.98, `double_coverage_gate` 0.95, `max_uses`, `max_cards` |
| Core deck       | `ranking/coregates.py`                            | output APKG, core CSV, coverage CSV, rank CSV, manifest                                               | `outside_set_ratio`, `outside_rare_ratio`, `max_unchecked_sense`, `max_unranked`  |
| Vocabulary deck | none                                              | -                                                                                                     | -                                                                                 |

### 3.2 Shared code

The gate seam is already shared in part:

| Code | Role |
| --- | --- |
| `anki_tools/gates.py` | `Gate`, `GateResult`, `GateFiles`, and `run_gates` |
| `gate_files.py` | `GateFileReader`, the lazy reader with the cached error |
| `ranking/gates.py` | `report` and the shared checks |
| `sentence_pack/audit.py` | `AuditInputs`, `as_gates`, `print_audit`, shared pack checks |
| `ranking/cullgates.py` | `CullFiles` |
| `ranking/coregates.py` | `CoreFiles` |

`AuditInputs`, `CullFiles`, and `CoreFiles` all extend `GateFileReader`. All
three flows print through `print_audit`, and each report prints a flow title.

Every flow runs `anki_tools.gates.run_gates`. The pack keeps its `(name, check)`
tuple type and its `(inputs, profile)` check form, so `as_gates` wraps the pack
table. Section 3.4 gives the interface, and section 5.1 gives the change.

### 3.3 Differences

- The pack gates read input artifacts. The cull and core gates read the output
  APKG, the output CSVs, and the manifest.
- The pack gate list is not fixed. Mandarin stops at G18. Greek stops at G20.
  French, Spanish, and Cantonese reach G21 with different gates at G17 to G21,
  for example typography, jyutping, discontinuous target, and false friend.
  G19 exists only in French and Spanish. Greek and Cantonese G19 are gone.
- The pack G14 re-applies the shape guards and F9 to F10 only. The cull G5
  re-applies F1 to F9. The core K5 re-applies F1 to F9 and C1 to C5.
- The cull G5 exempts F8 for a reserve card. A Swadesh pack has no name rule,
  and G14 needs those cards (`ranking/cullgates.py`, `gate_filter_recheck`).
- The core rows K3, K15, and K17 are gone. Their values are report lines:
  Covered twice, Rejected names, and Rejected loan words. The bold target of
  K15 is part of K19. Every pack, cull, and core gate can fail.
- The remaining gaps: pack G19 covers the target parts only in French and
  Spanish. Pack G3 covers the table form and the shared form, not the target
  parts. Pack G9 covers the script only in Cantonese and the pinyin only in
  Mandarin. The noun gender count `noun_gender_missing` is a manifest value.
- The pack manifest has a second shape. `sentence_pack/pack.py` writes it, and
  G11 reads it. The cull and core manifests come from
  `ranking/assemble.py`.
- The pack similarity gate G6 uses `profile.similarity` on the sentence text.
  The cull G6 and core K6 use the lemma Jaccard value with the content parts of
  speech. The three checks do not compare the same property.

### 3.4 The harmonized gate interface

The gate seam has four parts in `anki_tools/gates.py`, below `ranking` and
`sentence_pack`. The report lives in `ranking/gates.py`. Every flow uses the
one runner.

| Part | Name | Location | Shared by |
| --- | --- | --- | --- |
| Gate value | `Gate(name, check)` | `anki_tools/gates.py` | pack, cull, core |
| Row | `GateResult(name, passed, message)` | `anki_tools/gates.py` | pack, cull, core |
| Runner | `run_gates(gates, files)` | `anki_tools/gates.py` | pack, cull, core |
| Report | `report(results, title)` | `ranking/gates.py` | pack, cull, core |
| Reader | `GateFiles` and `GateFileReader` | `anki_tools/gates.py`, `gate_files.py` | pack, cull, core |

Each flow supplies three items: a gate table, a file reader, and a title.

| Flow | Gate table | File reader | Title |
| --- | --- | --- | --- |
| Sentence pack | the language tuple plus `GENERIC_GATES` | `AuditInputs` | `Swadesh sentence pack` |
| Cull deck | `GATES` in `ranking/cullgates.py` | `CullFiles` | `Cull deck` |
| Core deck | `GATES` in `ranking/coregates.py` | `CoreFiles` | `Core deck` |
| Vocabulary deck | none | none | none |

The gate ids and the gate sets stay per flow. ADR 0027 records why. The
harmonized parts are the runner, the row type, and the report. Each flow keeps
its own table.

The pack declares `(name, check)` tuples and `as_gates` wraps them, so the pack
supplies the profile through the reader. Section 5.1 records the change.

### 3.5 Gate refactor status

The gate refactor in `specs/gate-layer.md` is done, except two items.

| Item                                                            | State     |
| --------------------------------------------------------------- | --------- |
| One value, one row, one runner in `anki_tools/gates.py`         | done      |
| The pack `as_gates` adapter                                     | done      |
| The language copies (G12, G17, G18, G19, G3/G19, G5/G9)         | done      |
| The rank pairs (G6/K6, G7/K7, G8/K8, G11/K10, G13/K12, G15/K14) | done      |
| The report-only rows (K3, K15, K17, pack G20)                   | done      |
| The function renames                                            | done      |
| Pack G1: five `gate_word_table` copies                          | open      |
| One output reader, then pack G13 uses `gate_output_model`       | open      |
| The optional K1 and K8 merge                                    | not doing |

What was done:

- The pack, the cull, and the core run one gate value and one runner.
- The shared checks moved to `sentence_pack/audit.py` and `ranking/gates.py`.
- The report-only rows left the gate table. Their values are report lines or
  manifest values.
- The function names state the check.

What is left:

- Pack G1 needs a profile hook for the CSV path, then one
  `gate_word_table`. Tracked by `anki-ujvj`.
- One output reader. Then pack G13 uses `gate_output_model`, and the pack
  shares the audio and template checks. Tracked by `anki-ujvj`.

The filter work is separate. The rule order and the core mark source are not
part of the gate refactor.

## 4. Why the flows differ

The pack selects sentences from a Tatoeba pool. It measures concept coverage
and has no card slot. The cull and core rank the cards of an input deck. They
fill card slots, apply a card cap, and have no concept quota. ADR 0027 records
that reason.

Some differences are deliberate. Some are accidents:

| Difference | Class | Reason |
| --- | --- | --- |
| Different filter keys | deliberate | the two jobs differ |
| Different gate numbers per flow | deliberate, recorded | ADR 0027 |
| Core uses Latin mark constants | accidental | `CoreProfile` lacks the mark fields |
| Pack has a second manifest shape | accidental | the pack writer came first |
| Pack has a second gate runner | accidental | the runner type arrived later |

## 5. What "consistent" can mean

The word "consistent" covers six levels. Each level has a different cost. Only
the accidental items are worth a change.

| Level | Change | Behavior change | Cost | Risk |
| --- | --- | --- | --- | --- |
| L1 | one gate runner | no | small | low |
| L2 | one rule record schema | no | medium | low |
| L3 | one mark source and one rank rule function | yes | medium | medium |
| L4 | one manifest shape | yes | medium | medium |
| L5 | one gate id namespace | no | large | high |
| L6 | identical rule sets | yes | large | high |

### 5.1 L1: one gate runner

**Change**: every flow declares its gates as `Gate` values and calls
`ranking.gates.run_gates`. Delete the second `run_gates` in
`sentence_pack/audit.py`, and replace `GateRow` with `GateResult`. A small
adapter converts the present pack table:

```python
def as_gates(gates: Sequence[tuple[str, GateCheck]]) -> tuple[Gate, ...]:
    return tuple(
        Gate(name, lambda files, check=check: check(files, files.profile))
        for name, check in gates
    )
```

**Result**: one runner, one row type, one report. Section 3.4 lists the three
items that each flow supplies.

**Residue**: the pack checks keep the `(inputs, profile)` form, and the
adapter supplies the profile. A full `check(files)` form would touch about 60
pack check functions and gain nothing. Do it only when a pack check must
serve another flow.

**Files**: `sentence_pack/audit.py`, the five language audit wrappers, and
`ranking/gates.py` for the shared type.

**Cost**: about 30 lines. **Behavior**: none. The report text and the exit
code must stay the same.

**Test**: `uv run pytest tests/test_gates.py tests/test_mandarin_audit.py`.

**Verdict**: do it. This is the gate harmonization: one seam and four tables.

**Status**: done. See `specs/gate-layer.md`.

### 5.2 L2: one rule record schema

**Change**: add one `Rule` record, for example `(key, name, test)`. Build
`SHAPE_GUARDS`, `FACT_GUARDS`, the cull order, and the core order from it.
Keep the present keys and the present order, so the output does not change.

**Files**: `sentence_pack/pool.py`, `ranking/cullflow.py`,
`ranking/coreflow.py`, `ranking/cardfilter.py`.

**Cost**: about 100 lines touched. **Behavior**: none if the order is exact.

**Verdict**: optional. The current tables are readable. Do it only when a third
consumer needs the same rules. Until then it is speculative.

### 5.3 L3: one mark source and one rank rule function

**Change**:

1. Add `final_marks`, `quote_marks`, and `internal_marks` to `CoreProfile`.
2. Move `CardShape` and `filter_reason` from `ranking/cullflow.py` to
   `ranking/cardfilter.py`. Give `filter_reason` a mark set, not a
   `CullProfile`.
3. The cull and the core call the one function with their own marks and their
   own token counts.
4. The four core adapters pass the marks of the language pack profile. Greek
   has no core command, so the change touches French, Spanish, Mandarin, and
   Cantonese.

**Cost**: about 80 lines moved, 3 fields added, 4 core profiles changed, 2 call
sites changed.

**Behavior change**: yes. A core card can end in a language mark or hold a
language quote that the Latin set rejects. The kept set can grow, and the
selection can pick different cards. Keep or remove the `_sentence` final-mark
conversion as a deck style choice; the filter no longer forces it.

**Risk**: a new core deck differs from the committed package. Rebuild every
core deck and compare the card count and the gate table.

**Verdict**: do it. This fixes a real gap. The mark source belongs to the
language, not to a flow.

### 5.4 L4: one manifest shape

**Change**: make `sentence_pack/pack.py` write the fix shape of ADR 0018:
`params`, `counts`, `source`, and `files`. Change the pack gate G11 to read
that shape.

**Cost**: about 60 lines in the writer and the gate, plus the pack readers.
**Behavior change**: yes. An old pack manifest fails G11 until a rebuild.

**Verdict**: optional. It removes the second shape, but no current check fails
because of it. Do it with the next pack rebuild.

### 5.5 L5: one gate id namespace

**Change**: give the pack, the cull, and the core one numbering. Move the
shared checks into the table, renumber the shifted rules, and update every
reference.

**Cost**: large. About 83 lines in the code, docs, specs, and tests hold `F8`.
About 136 quoted `G` ids and 21 quoted `K` ids live in the code. The test
`tests/test_gate_id_map.py` pins the current map. `ranking/gates.py` states
that the gate numbers stay stable. ADR 0027 records the decision and the
reasons.

**Verdict**: rejected. Keep the namespaces. A report already prints the flow
name, and a doc row or a message must name the flow.

### 5.6 L6: identical rule sets

**Change**: one filter table for all flows. This is not a refactor. It changes
what the flows accept:

- Pack F1 is a link test. Rank F1 is a word window. One key cannot hold both.
- Pack F4 is the repeated-sentence rule. Rank F8 is the name rule. Each has no
  counterpart in the other table.
- Pack F9 to `sense` measure the sentence pool. The cull and core have no
  concept quota, and the pack has no card slot or card cap.
- Pack similarity uses the language sentence hook. The rank similarity uses the
  lemma Jaccard value.
- The pack gate reads the source word table and the review file. The rank gates
  read the output identity and the manifest.

**Verdict**: do not do it. The two tables serve different jobs. Keep the
separate sets and make the accidental differences go away.

## 6. Recommended order

The gate harmonization is done. It changed no deck:

1. **L1: one gate runner**. Small, no behavior change. Tracked by
   `anki-3fc9`. ADR 0021 records the one file reader. This step adds the one
   runner and needs no new seam.
2. **L3: one mark source**. Medium, one clear win, one rebuild. This is a
   filter change, not a gate change. Tracked by `anki-0d3d`.
3. **Stop**. L2 and L4 are optional. L5 and L6 are out.

Record one ADR for L3, because it changes the `CoreProfile` interface. ADR
0027 stays in force for the ids.

## 7. Verification

Run the focused tests first:

```bash
uv run pytest tests/test_gate_id_map.py tests/test_gates.py \
  tests/test_mandarin_audit.py \
  tests/test_core_deck.py tests/test_shared_coreflow.py \
  tests/test_mandarin_core.py tests/test_cantonese_core.py \
  tests/test_cull_shared.py tests/test_french_cull.py \
  tests/test_spanish_cull.py tests/test_swadesh_build.py
```

Then run the full suite. Then rebuild the affected decks and run the gates:

```bash
uv run anki fix fr <input.apkg> --core
uv run anki fix fr apkg/french_core.apkg --core --check
uv run anki fix fr <input.apkg> --select
uv run anki fix fr apkg/french_1000.apkg --select --check
uv run anki swadesh audit french
```

Compare the kept count and the gate table before and after. Record the values
in the L3 ADR. A changed count is the expected result of L3. A changed count
from L1, L2, or L4 is a defect.

## 8. References

| Path | Content |
| --- | --- |
| `docs/adr/0027-gate-id-namespaces.md` | the id namespaces and the rejected renumber |
| `docs/swadesh-sentence-flow.md` | pack guard and gate order |
| `docs/cull-deck-flow.md` | cull rule and gate order |
| `docs/core-deck-flow.md` | core rule and gate order |
| `docs/vocab-deck-flow.md` | the vocabulary deck check |
| `docs/gate-inventory.md` | the gate inventory and the refactor plan |
| `anki_tools/sentence_pack/pool.py` | `FILTER_KEYS`, `SHAPE_GUARDS`, `FACT_GUARDS` |
| `anki_tools/sentence_pack/guards.py` | `check_sentence_shape` |
| `anki_tools/sentence_pack/audit.py` | `GENERIC_GATES`, `run_pack_audit` |
| `anki_tools/ranking/cardfilter.py` | `filter_cards`, `_filter_reason` |
| `anki_tools/ranking/cullflow.py` | `FILTER_RULES`, `MARK_RULES`, `filter_reason` |
| `anki_tools/ranking/coreflow.py` | `FILTER_KEYS`, `filter_core_cards` |
| `anki_tools/ranking/cullgates.py` | cull gates G1 to G17 |
| `anki_tools/ranking/coregates.py` | core gates K1 to K21 |
| `anki_tools/gates.py` | the gate value, the row, the runner, and the reader contract |
| `anki_tools/ranking/gates.py` | the report and the shared checks |
| `anki_tools/gate_files.py` | the one lazy file reader |
