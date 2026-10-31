# Gate and filter comparison across flows

## 1. Scope

This document compares the gates and the filters of the four deck builders.
One builder has no gate table:

| Flow | Command | Gate table | Filter table |
| --- | --- | --- | --- |
| Sentence pack | `anki swadesh pack <lang>` | G1 to G21 | F1 to F12, `unknown`, `sense` |
| Cull deck | `anki fix <lang> <input> --select` | G1 to G17 | F1 to F9 |
| Core deck | `anki fix <lang> <input> --core` | K1 to K21 | F1 to F9, C1 to C5 |
| Vocabulary deck | `anki swadesh vocab <lang>` | none | none |

A **filter** rejects a sentence or a card before selection. The pack code calls
a filter a guard. A **gate** audits a built file and reports a fault. A gate
never rejects a sentence. The glossary holds both terms.

A rule key and a gate id are valid only inside their owner table. ADR 0027
records the overlap. For example, pack `F8` is the internal-mark rule and rank
`F8` is the name rule. Pack `G5` tests a one-time target and cull `G5`
re-runs the filters.

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

| Flow | Table | Ids | Scope | Thresholds |
| --- | --- | --- | --- | --- |
| Sentence pack | `sentence_pack/audit.py` plus the language module | G1 to G21, 18 to 21 gates per language | source word table, review CSV, sentence CSV, audio directory, attribution file, output APKG, manifest | `coverage_min`, for example 0.70 Cantonese to 0.95 French |
| Cull deck | `ranking/cullgates.py` | G1 to G17 | output APKG, cull CSV, coverage CSV, manifest | `target_coverage_gate` 0.98, `double_coverage_gate` 0.95, `max_uses`, `max_cards` |
| Core deck | `ranking/coregates.py` | K1 to K21 | output APKG, core CSV, coverage CSV, rank CSV, manifest | `outside_set_ratio`, `outside_rare_ratio`, `max_unchecked_sense`, `max_unranked` |
| Vocabulary deck | none | - | - | - |

### 3.2 Shared code

The gate seam is already shared in part:

| Code | Role |
| --- | --- |
| `ranking/gates.py` | `Gate`, `GateResult`, `GateFiles`, `run_gates`, `report`, and the shared checks |
| `gate_files.py` | `GateFileReader`, the lazy reader with the cached error |
| `sentence_pack/audit.py` | `AuditInputs`, `print_audit`, shared pack checks |
| `ranking/cullgates.py` | `CullFiles` |
| `ranking/coregates.py` | `CoreFiles` |

`AuditInputs`, `CullFiles`, and `CoreFiles` all extend `GateFileReader`. All
three flows print through `print_audit`, and each report prints a flow title.

The pack keeps its own `run_gates` and its `(name, check)` tuple type. The cull
and the core use `ranking.gates.run_gates` and `Gate` values.

### 3.3 Differences

- The pack gates read input artifacts. The cull and core gates read the output
  APKG, the output CSVs, and the manifest.
- The pack gate list is not fixed. Mandarin stops at G18, Greek at G20, and
  French, Spanish, and Cantonese reach G21 with different gates at G17 to G21,
  for example typography, noun gender, jyutping, and false friend.
- The pack G14 re-applies the shape guards and F9 to F10 only. The cull G5
  re-applies F1 to F9. The core K5 re-applies F1 to F9 and C1 to C5.
- The cull G5 exempts F8 for a reserve card. A Swadesh pack has no name rule,
  and G14 needs those cards (`ranking/cullgates.py`, `gate_filter`).
- The core K3 and K17 report only. They do not fail. Every pack and cull gate
  can fail.
- The pack manifest has a second shape. `sentence_pack/pack.py` writes it, and
  G11 reads it. The cull and core manifests come from
  `ranking/assemble.py`.
- The pack similarity gate G6 uses `profile.similarity` on the sentence text.
  The cull G6 and core K6 use the lemma Jaccard value with the content parts of
  speech. The three checks do not compare the same property.

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

**Change**: make the pack gate table a list of `Gate` values and delete the
second `run_gates`. A small closure passes `files.profile` to the pack check:

```python
def as_gates(gates: Sequence[tuple[str, GateCheck]]) -> tuple[Gate, ...]:
    return tuple(
        Gate(name, lambda files, check=check: check(files, files.profile))
        for name, check in gates
    )
```

**Files**: `sentence_pack/audit.py`, the language audit wrappers, and
`ranking/gates.py` for the shared type.

**Cost**: about 30 lines. **Behavior**: none. The report is already shared.

**Verdict**: do it. It removes a duplicate runner and leaves one gate type.

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

1. **L1: one gate runner**. Small, no behavior change.
   Tracked by `anki-3fc9`.
2. **L3: one mark source**. Medium, one clear win, one rebuild.
   Tracked by `anki-0d3d`.
3. **Stop**. L2 and L4 are optional. L5 and L6 are out.

Record one ADR for L3, because it changes the `CoreProfile` interface. ADR 0027
stays in force for the ids.

## 7. Verification

Run the focused tests first:

```bash
uv run pytest tests/test_gate_id_map.py tests/test_gates.py \
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
| `anki_tools/sentence_pack/pool.py` | `FILTER_KEYS`, `SHAPE_GUARDS`, `FACT_GUARDS` |
| `anki_tools/sentence_pack/guards.py` | `check_sentence_shape` |
| `anki_tools/sentence_pack/audit.py` | `GENERIC_GATES`, `run_pack_audit` |
| `anki_tools/ranking/cardfilter.py` | `filter_cards`, `_filter_reason` |
| `anki_tools/ranking/cullflow.py` | `FILTER_RULES`, `MARK_RULES`, `filter_reason` |
| `anki_tools/ranking/coreflow.py` | `FILTER_KEYS`, `filter_core_cards` |
| `anki_tools/ranking/cullgates.py` | cull gates G1 to G17 |
| `anki_tools/ranking/coregates.py` | core gates K1 to K21 |
| `anki_tools/ranking/gates.py` | gate type, runner, report, shared checks |
| `anki_tools/gate_files.py` | the one lazy file reader |
