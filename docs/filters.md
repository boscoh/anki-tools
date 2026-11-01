# Filters

| Field   | Value                                                |
| ------- | ---------------------------------------------------- |
| Status  | The rule sets stay separate; the mark source is open |
| Date    | 2026-11-01                                           |
| Owner   | anki-tools                                           |
| Related | `docs/gates.md`, ADR 0027, `anki-0d3d`               |

## 1. What a filter is

A filter rejects a sentence or a card before selection. The pack code calls a
filter a guard. A gate audits a built file and reports a fault, and
`docs/gates.md` covers the gates.

A rule id names one check. A flow that runs the check uses the id. Each flow
keeps its own subset and its own order. The id shows no position.

The flows differ because their jobs differ. The pack selects sentences from a
Tatoeba pool and measures concept coverage. It has no card slot. The cull and
the core rank the cards of an input deck. They fill card slots and apply a card
cap, and they have no concept quota.

## 2. Rule keys

| Id               | Sentence pack                      | Cull deck                            | Core deck                                                                |
| ---------------- | ---------------------------------- | ------------------------------------ | ------------------------------------------------------------------------ |
| `english_link`   | an English link exists             | -                                    | -                                                                        |
| `length_window`  | unit count inside the window       | the same                             | the same                                                                 |
| `digit_marks`    | no digit, bracket, link, ellipsis  | the same                             | the same                                                                 |
| `script`         | the language alphabet test         | the same                             | the same                                                                 |
| `repeat`         | no repeated pattern                | the same                             | the same                                                                 |
| `marks`          | final, quote, and internal marks   | the same                             | the same                                                                 |
| `unique`         | dedupe key is new                  | the same                             | the same                                                                 |
| `sense`          | English gloss holds the sense      | the same, per target                 | the same, per target                                                     |
| C3 to C5         | -                                  | -                                    | a name test, a loan word test, and the media map                         |

The pack tests 8 ids, the cull 7, and the core 7 plus C3 to C5. The profile
of the language gives the `repeat` test and the `unique` key to all three flows.
The language states its own length rule: `unit_count` and the window
(`min_units`, `max_units`). No flow tests the English length.
The cull and the core apply `sense` to a card and target pair, not to a whole card.
Section 5, L9, states the meaning of `sense` and the status of the code.

## 3. Rule order

The pack runs two phases. `SHAPE_GUARDS` holds `length_window` to
`marks` and runs first, because those tests are cheap. The measure step
reads the tokens and the facts. Then `FACT_GUARDS` holds `unique` and `sense`.
`FILTER_KEYS` is the report order. A
sentence that passes no guard becomes a candidate for the lowest-index word it
covers.

The cull runs one list in `filter_reason`, in the same order as the pack:
`length_window`, `digit_marks`, `script`, `repeat`, the mark rules, then `unique`.
The shape guard `check_sentence_shape` gives all rules except `unique`.
`LENGTH_RULES`, `TEXT_RULES`, and `MARK_RULES` hold their ids.

The core runs `filter_cards` with `filter_reason`, the function of the cull, then
C3 to C5 in `filter_core_cards`.

The cull and the core store the first failed rule as the card `reason`. The
pack records a pass count for every guard before the first failure.

## 4. Differences

- `length_window` is in all three flows. It is a filter, not a gate. The test is
  one function: `_passes_length` in `sentence_pack/pool.py`. The pack counts
  units with `profile.unit_count`. The cull and the core use the same hook from
  their profile, so the three flows count one unit.
- A scaffold sentence skips the one-final-mark and internal-mark checks of `marks`,
  and it skips the `sense` guard (`shape_failure(scaffold=True)`,
  `SCAFFOLD_SKIP_GUARDS`).
- The pack, the cull, and the core take the marks from the language profile. The
  Mandarin and Cantonese core adapters convert the final mark of each sentence to
  ASCII in `_sentence`. That conversion is a deck style choice. A full-width
  mark inside the sentence stays, so a second final mark fails `marks`.
- `marks` is one filter with four checks: the last character is a final mark, the
  sentence has exactly one final mark, no quotation mark or dialogue dash, and at
  most one internal mark. A failure gives no sub-reason. `MARK_RULES` lists the id
  that the cull takes from the pack shape
  guard. The cull has no `english_link`,
  `sense` rule.
- `digit_marks` is the digit, bracket, link, and ellipsis test. It has no language
  input. `script` is the alphabet test of the language profile. Both run in all
  three flows.

### 4.1 Length window by flow

| Flow          | Unit count                                    | Window                                              |
| ------------- | --------------------------------------------- | --------------------------------------------------- |
| Pack          | `profile.unit_count`: words or Han characters | `profile.min_units` to `max_units`                  |
| Pack scaffold | the same                                      | `profile.fallback_min_units` to `max_units`         |
| Pack frames   | the same                                      | the same, no scaffold retry                         |
| Cull          | the same                                      | `profile.min_units` to `max_units`                  |
| Core          | the same                                      | `profile.min_units` to `max_units`                  |

The pack also runs `exceeds_max_units` before the measure step. It rejects a
sentence with more whitespace chunks than `max_units` and records `length_window`.

Windows by language, as minimum to maximum:

| Language  | Pack, cull, and core | Pack scaffold minimum |
| --------- | -------------------- | --------------------- |
| French    | 4 to 10              | 3                     |
| Spanish   | 4 to 10              | 3                     |
| Greek     | 4 to 10              | 3                     |
| Mandarin  | 4 to 7               | `FALLBACK_MIN_HAN`    |
| Cantonese | 4 to 7               | `FALLBACK_MIN_HAN`    |

Each language states the window once, in its pack profile. The pack command has
`--min-words` and `--max-words`. The `fix` select and core modes have the same
two flags. Each flag replaces one end of the profile window for that run. Both
modes write the window that they used to the manifest.

The gate rechecks read the window of the run: K5 and G5 read `min_units` and
`max_units`, or `min_words` and `max_words`, from the stored params. K5 falls
back to the profile for an old manifest. `scaffold_problem` reads the scaffold
window.

### 4.2 Known differences between flows

These differences are deliberate, because the jobs differ. A reader must not
compare the reject counts of two flows directly.

- Frames: a pack frame sentence runs the shape guards and a target-form test
  only. It skips `unique` and the scaffold retry. This is
  deliberate: a frame sentence is template text, not a Tatoeba pair.

The cull and core reports once left out `repeat`. Both report lists now hold it.
`tests/test_gate_id_map.py` checks the rule ids.

## 5. Decisions

| Level | Change                                | Behavior                                       | Verdict                                        |
| ----- | ------------------------------------- | ---------------------------------------------- | ---------------------------------------------- |
| L2    | one rule record, `(key, name, test)`  | none                                           | deferred; do it when a third consumer needs it |
| L3    | one mark source for the core          | a core card can keep a language mark           | done, `anki-0d3d`, ADR 0028                    |
| L4    | one manifest shape                    | an old pack manifest fails G11 until a rebuild | deferred to the next pack rebuild              |
| L5    | one rule key namespace                | none                                           | rejected, ADR 0027                             |
| L6    | one identical rule set for every flow | a different accepted set                       | rejected, the two jobs differ                  |
| L7    | `length_window` to the gate layer     | none                                           | rejected, a gate never rejects a sentence      |
| L8    | `digit_marks`, `script` in all flows  | the cull and the core gain an alphabet test    | done, `anki-sxzu`, ADR 0029                    |
| L9    | one `sense` rule in every flow        | the cull and the core drop a failed target     | done, S1 to S5, ADR 0030 and 0031              |

L3 added `final_marks`, `quote_marks`, and `internal_marks` to `CoreProfile`. It
moved `CardShape` and `filter_reason` to `ranking/cardfilter.py`, and
`filter_reason` takes a mark set. The cull and the core now call one function
with their own marks and their own token counts.

L3 changed the passed set of the Cantonese core deck. ADR 0028 records the
card counts.

L8 split the old `script` test. Each language `script_ok` hook now tests only the
alphabet, and `digit_marks` is one function in `sentence_pack/pool.py` for every
flow. The pack, the cull, and the core run both guards through
`check_sentence_shape`. `CullProfile` and `CoreProfile` hold `script_ok`. L8
changed the Cantonese core deck by one card. ADR 0029 records the counts and the
open pack rebuild.

L9 makes `sense` the one rule for the meaning of a target word. A card shows a
target when three tests hold:

1. The form or the lemma is in the sentence.
2. The form is not inside a compound that gives a different meaning.
3. The English holds a sense word of the target and no false-friend word.

| Flow | Unit                 | A failed test                                          |
| ---- | -------------------- | ------------------------------------------------------ |
| Pack | candidate sentence   | rejects the sentence and counts it under `sense`       |
| Core | card and target pair | drops the target; rejects the card with no target left |
| Cull | card and target pair | the same as the core                                   |

All three flows now run `sense` through `sense_failure`. The pack rejects a
sentence. The core and the cull drop a target, and the cull drops a lemma that
fails the test (ADR 0030). The status of each part is in the work items below.

One record per target word holds the data, in one file per language:

| Field           | Content                                  | Side     |
| --------------- | ---------------------------------------- | -------- |
| `senses`        | English words that show the sense        | English  |
| `false_friends` | English words that show another sense    | English  |
| `compounds`     | sentence strings that give another sense | sentence |

The compound test stays inside the occurrence count (`form_occurrences`,
`single_hits`). The "exactly one hit" rule counts only the occurrences outside a
compound, so the test cannot run after `hits`. `sense` calls that same code and
reads `compounds` from the record. The cull does not use `compounds`: its targets
are tokenizer lemmas, and its languages have no compounds.

`sense` replaces these parts:

| Part                                                     | Change                                    |
| -------------------------------------------------------- | ----------------------------------------- |
| `sense_filter` hook, `make_sense_filter`, `SENSE_CHECKS` | removed; the record holds `false_friends` |
| `false_friend_spans` hook, `FALSE_FRIENDS`               | removed; the record holds `compounds`     |
| `holds_sense`, `holds_tier_s_sense`, `sense_match`       | one function, `sense_failure`             |
| `gate_no_sense_mismatch`, `gate_no_false_friend`         | one `sense` recheck gate in every flow    |
| `gate_unchecked_sense` (K16)                             | a limit on the `sense` rule in every flow |
| `gate_target_in_sentence` (K13)                          | test 1 of the rule                        |

`sense` does not replace `english_link`, the shape rules,
`unique`, C3 to C5, or the deck gates such as `gate_core_word`.

S3 is done. `sense_failure(english, row, gloss)` in
`sentence_pack/english_sense.py` returns `false_friend`, `sense`, or None. The
pack, the core, and the audit call it. A tier S form passes a Swadesh `gloss`;
a tier F lemma passes none. `read_senses` keys a row by tier and word, and the
TSV has the columns `word`, `senses`, `false_friends`, `compounds`, `tier`. A
blank `tier` is `F`. Greek has no core flow, so `vocab/greek_core_senses.tsv`
holds its tier S rows only.

S4 is done. `gate_sense` in `ranking/gates.py` is cull `G18` and core `K22`. It
calls `sense_failure` on the `Core` word and the English of every kept note. The
pack filter, the cull, and the core call the same function. The pack gates
`gate_no_sense_mismatch` and `gate_no_false_friend`, core `K13`, and core `K16`
stay as separate gates: they test the sentence CSV, the form in the sentence, and
the count of unchecked targets. A cull deck built
before S2 can fail `G18`, because it kept cards that no sense test checked.

The English table `vocab/english_senses.json` stays. It gives the sense words of a
Swadesh gloss, and WordNet builds it.

Work items:

| Id | Task                                                                  | Status                 |
| -- | --------------------------------------------------------------------- | ---------------------- |
| S1 | write this section                                                    | done                   |
| S2 | cull: remove a lemma that fails `sense_failure` in `select_cards`     | done, ADR 0030         |
| S3 | one shared function and one record; move the dicts into the TSV files | done                   |
| S4 | one `sense` recheck gate in the cull and the core (G18, K22)          | done                   |
| S5 | sense rows for all 300 cull targets, and a Cantonese file             | done, ADR 0031         |

A compound-only sentence has no hit, so it is not a candidate and the pack counts
no failure for it. In S3, `read_senses` must not read a tier S row as a tier F
target, because the record file then holds both.

Open work after L9:

| Issue       | Problem                                                                      |
| ----------- | ---------------------------------------------------------------------------- |
| `anki-o1vk` | a fresh Mandarin core build fails K6; the build and K6 use two lemma sets    |
| none yet    | Spanish cull G5: one card fails `length_window`                              |

The "Distinct lemmas" line of the cull report still counts a lemma that failed
`sense`. The line has no issue yet.

## 6. Verification

```bash
uv run pytest tests/test_cull_shared.py tests/test_french_cull.py \
  tests/test_spanish_cull.py tests/test_core_deck.py \
  tests/test_shared_coreflow.py tests/test_mandarin_core.py \
  tests/test_cantonese_core.py tests/test_swadesh_build.py
```

Then rebuild the affected decks and compare the kept count with the count before
the change. L3 changed one count (ADR 0028).

## 7. References

| Path                                  | Content                                       |
| ------------------------------------- | --------------------------------------------- |
| `docs/gates.md`                       | the gate layer and the gate tables            |
| `docs/adr/0027-gate-id-namespaces.md` | the rule and gate id namespaces               |
| `docs/swadesh-sentence-flow.md`       | the pack guard order                          |
| `docs/cull-deck-flow.md`              | the cull rule order                           |
| `docs/core-deck-flow.md`              | the core rule order                           |
| `anki_tools/sentence_pack/pool.py`    | `FILTER_KEYS`, `SHAPE_GUARDS`, `FACT_GUARDS`  |
| `anki_tools/sentence_pack/guards.py`  | `check_sentence_shape`                        |
| `anki_tools/ranking/cardfilter.py`    | `filter_cards`, `filter_reason`, `MARK_RULES` |
| `anki_tools/ranking/cullflow.py`      | `FILTER_RULES`                                |
| `anki_tools/ranking/coreflow.py`      | `FILTER_KEYS`, `filter_core_cards`            |
