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

Each flow owns its rule keys (ADR 0027). Pack `F8` is the internal-mark rule.
Rank `F8` is the name rule. A doc row and an issue must name the flow with the
key.

The flows differ because their jobs differ. The pack selects sentences from a
Tatoeba pool and measures concept coverage. It has no card slot. The cull and
the core rank the cards of an input deck. They fill card slots and apply a card
cap, and they have no concept quota.

## 2. Rule keys

| Key       | Sentence pack                          | Cull deck                            | Core deck                                                                |
| --------- | -------------------------------------- | ------------------------------------ | ------------------------------------------------------------------------ |
| F1        | an English link exists                 | English text has 2 to 12 words       | the same                                                                 |
| F2        | unit count inside the window           | token count inside the length window | token count inside the length window                                     |
| F3        | script test: no Latin letter, no digit | no digit, bracket, link, or ellipsis | the same                                                                 |
| F4        | no repeated pattern                    | the last character is a final mark   | the same                                                                 |
| F5        | the last character is a final mark     | exactly one final mark               | the same                                                                 |
| F6        | exactly one final mark                 | no quotation mark or dialogue dash   | the same                                                                 |
| F7        | no quotation mark or dialogue dash     | at most one internal mark            | the same                                                                 |
| F8        | at most one internal mark              | no `PROPN` token                     | the same                                                                 |
| F9        | predicate tag                          | the sentence text is unique          | the same                                                                 |
| F10       | no orphan conjunction                  | -                                    | -                                                                        |
| F11       | English text has 2 to 12 words         | -                                    | -                                                                        |
| F12       | dedupe key is new                      | -                                    | -                                                                        |
| `unknown` | unknown token budget                   | -                                    | -                                                                        |
| `sense`   | English gloss holds the sense          | -                                    | -                                                                        |
| C1 to C5  | -                                      | -                                    | a verb, an orphan test, a name test, a loan word test, and the media map |

The pack tests 14 keys, the cull 9, and the core 14.

## 3. Rule order

The pack runs two phases. `SHAPE_GUARDS` holds F2 to F8 and runs first,
because those tests are cheap. The measure step reads the tokens and the facts.
Then `FACT_GUARDS` holds F9 to `sense`. `FILTER_KEYS` is the report order. A
sentence that passes no guard becomes a candidate for the lowest-index word it
covers.

The cull runs one list in `filter_reason`: F1, F2, F3, the mark rules, F8, then
F9. The mark rules come from `check_sentence_shape`, and `MARK_RULES` holds
their cull names.

The core runs `filter_cards` with the Latin rule function, then C1 to C5 in
`filter_core_cards`.

The cull and the core store the first failed rule as the card `reason`. The
pack records a pass count for every guard before the first failure.

## 4. Differences

- F2 is the length window in all three flows. The pack measures the text and
  the rank flows count tokens.
- A scaffold sentence may fail F6 or F8, and it skips the `sense` guard
  (`SCAFFOLD_SHAPE_FAILURES`, `SCAFFOLD_SKIP_GUARDS`).
- The pack and the cull take the marks from the language profile. The core
  takes the Latin mark constants of `ranking/cardfilter.py`, because
  `CoreProfile` holds no mark fields. The Mandarin and Cantonese core adapters
  convert the final mark of each sentence to ASCII in `_sentence`. That
  conversion is a deck style choice.
- `MARK_RULES` holds the shift of pack F5 to F8 to cull F4 to F7. The pack has
  no `PROPN` rule. The cull has no predicate, orphan, English-word, dedupe,
  unknown, or sense rule.
- Pack F1 is a link test, and rank F1 is a word window. One key cannot hold
  both rules.

## 5. Decisions

| Level | Change                                | Behavior                                       | Verdict                                        |
| ----- | ------------------------------------- | ---------------------------------------------- | ---------------------------------------------- |
| L2    | one rule record, `(key, name, test)`  | none                                           | deferred; do it when a third consumer needs it |
| L3    | one mark source for the core          | a core card can keep a language mark           | do it, tracked by `anki-0d3d`                  |
| L4    | one manifest shape                    | an old pack manifest fails G11 until a rebuild | deferred to the next pack rebuild              |
| L5    | one rule key namespace                | none                                           | rejected, ADR 0027                             |
| L6    | one identical rule set for every flow | a different accepted set                       | rejected, the two jobs differ                  |

L3 adds `final_marks`, `quote_marks`, and `internal_marks` to `CoreProfile`. It
moves `CardShape` and `filter_reason` to `ranking/cardfilter.py`, and
`filter_reason` takes a mark set. The cull and the core then call one function
with their own marks and their own token counts.

L3 changes the kept set of a core deck, so rebuild the four core decks and
record the card counts in ADR 0028.

## 6. Verification

```bash
uv run pytest tests/test_cull_shared.py tests/test_french_cull.py \
  tests/test_spanish_cull.py tests/test_core_deck.py \
  tests/test_shared_coreflow.py tests/test_mandarin_core.py \
  tests/test_cantonese_core.py tests/test_swadesh_build.py
```

Then rebuild the affected decks and compare the kept count with the count before
the change. A changed count is the expected result of L3.

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
| `anki_tools/ranking/cardfilter.py`    | `filter_cards`, the Latin mark constants      |
| `anki_tools/ranking/cullflow.py`      | `FILTER_RULES`, `MARK_RULES`, `filter_reason` |
| `anki_tools/ranking/coreflow.py`      | `FILTER_KEYS`, `filter_core_cards`            |
