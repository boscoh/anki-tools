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
| `length_window`  | unit count inside the window       | token count inside the length window | token count inside the length window                                     |
| `digit_marks`    | no digit, bracket, link, ellipsis  | the same                             | the same                                                                 |
| `script`         | the language alphabet test         | the same                             | the same                                                                 |
| `repeat`         | no repeated pattern                | -                                    | -                                                                        |
| `final_mark`     | the last character is a final mark | the same                             | the same                                                                 |
| `one_final_mark` | exactly one final mark             | the same                             | the same                                                                 |
| `quote`          | no quotation mark or dialogue dash | the same                             | the same                                                                 |
| `internal_mark`  | at most one internal mark          | the same                             | the same                                                                 |
| `predicate`      | predicate tag                      | -                                    | -                                                                        |
| `orphan`         | no orphan conjunction              | -                                    | -                                                                        |
| `english_words`  | English text has 2 to 12 words     | the same                             | the same                                                                 |
| `unique`         | dedupe key is new                  | the sentence text is unique          | the same                                                                 |
| `unknown`        | unknown token budget               | -                                    | -                                                                        |
| `sense`          | English gloss holds the sense      | -                                    | -                                                                        |
| `propn`          | -                                  | no `PROPN` token                     | the same                                                                 |
| C1 to C5         | -                                  | -                                    | a verb, an orphan test, a name test, a loan word test, and the media map |

The pack tests 15 ids, the cull 10, and the core 15.

## 3. Rule order

The pack runs two phases. `SHAPE_GUARDS` holds `length_window` to
`internal_mark` and runs first, because those tests are cheap. The measure step
reads the tokens and the facts. Then `FACT_GUARDS` holds `predicate` to `sense`.
`FILTER_KEYS` is the report order. A
sentence that passes no guard becomes a candidate for the lowest-index word it
covers.

The cull runs one list in `filter_reason`: `english_words`, `length_window`,
`digit_marks`, `script`, the mark rules, `propn`, then `unique`. The shape guard
`check_sentence_shape` gives all rules except `english_words`, `propn`, and
`unique`. `LENGTH_RULES`, `TEXT_RULES`, and `MARK_RULES` hold their ids.

The core runs `filter_cards` with `filter_reason`, the function of the cull, then
C1 to C5 in `filter_core_cards`.

The cull and the core store the first failed rule as the card `reason`. The
pack records a pass count for every guard before the first failure.

## 4. Differences

- `length_window` is in all three flows. It is a filter, not a gate. The test is
  one function: `_passes_length` in `sentence_pack/pool.py`. The pack counts
  units with `profile.unit_count`. The cull and the core pass the token count
  through `CardShape.unit_count` in `filter_reason`.
- A scaffold sentence may fail `one_final_mark` or `internal_mark`, and it skips
  the `sense` guard (`SCAFFOLD_SHAPE_FAILURES`, `SCAFFOLD_SKIP_GUARDS`).
- The pack, the cull, and the core take the marks from the language profile. The
  Mandarin and Cantonese core adapters convert the final mark of each sentence to
  ASCII in `_sentence`. That conversion is a deck style choice. A full-width
  mark inside the sentence stays, so a second final mark fails `one_final_mark`.
- `MARK_RULES` lists the four mark ids that the cull takes from the pack shape
  guard. The pack has no `propn` rule. The cull has no `english_link`,
  `repeat`, `predicate`, `orphan`, `unknown`, or `sense` rule.
- `digit_marks` is the digit, bracket, link, and ellipsis test. It has no language
  input. `script` is the alphabet test of the language profile. Both run in all
  three flows.

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
