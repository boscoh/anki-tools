# Filters

| Field   | Value                                                |
| ------- | ---------------------------------------------------- |
| Status  | The rule sets stay separate; the mark source is open |
| Date    | 2026-11-01                                           |
| Owner   | anki-tools                                           |
| Related | `docs/gates.md`, ADR 0027, `anki-0d3d`               |

## 1. What a filter is

A filter rejects a sentence or a card before selection. The pack code calls a
filter a guard. A gate audits a built file and reports a fault. `docs/gates.md`
covers the gates.

A rule id names one check. Each flow runs its own subset in its own order. The
id shows no position.

The flows differ because their jobs differ:

- The pack selects sentences from a Tatoeba pool and measures concept coverage.
  It has no card slot.
- The cull and the core rank the cards of an input deck. They fill card slots and
  apply a card cap. They have no concept quota.

## 2. Rules

All three flows run the same seven rule ids.

| Id              | Test                                              | Note                                           |
| --------------- | ------------------------------------------------- | ---------------------------------------------- |
| `length_window` | unit count inside the window                      | section 4                                      |
| `digit_marks`   | no digit, bracket, link, or ellipsis              | no language input                              |
| `script`        | the alphabet test of the language                 | the profile hook `script_ok`                   |
| `repeat`        | no repeated pattern                               | the profile gives the test                     |
| `marks`         | final, quote, and internal marks                  | four checks, no sub-reason                     |
| `unique`        | the dedupe key is new                             | section 2.1; the profile gives the key         |
| `sense`         | the English holds the sense of the target         | per target in the cull and the core; section 6 |

The `marks` rule has four checks:

1. The last character is a final mark.
2. The sentence has exactly one final mark.
3. The sentence has no quotation mark and no dialogue dash.
4. The sentence has at most one internal mark.

No flow tests the English length.

### 2.1 The unique rule

The cull and the core reject a card as `unique` when one of two keys is not new:

1. The sentence key from the profile `dedupe_key`.
2. The English key from `english_key` in `ranking/cardfilter.py`: the English
   text, casefolded, with single spaces.

A card with an empty English text never matches on the English key. Only a
passed card adds its keys, so a rejected card blocks no later card. Cull `G5`
and core `K5` call `filter_cards`, so they use the same rule (ADR 0047). The
pack has its own sentence dedupe in `sentence_pack/pool.py` and does not test
the English key.

## 3. Rule order

| Flow | Order                                                                                  |
| ---- | -------------------------------------------------------------------------------------- |
| Pack | `SHAPE_GUARDS`, the measure step, then `FACT_GUARDS`                                   |
| Cull | `filter_reason`: `length_window`, `digit_marks`, `script`, `repeat`, `marks`, `unique` |
| Core | `filter_cards` with the `filter_reason` of the cull, in `filter_core_cards`            |

- `SHAPE_GUARDS` holds `length_window` to `marks`. `FACT_GUARDS` holds `unique`
  and `sense`. The shape tests run first, because they are cheap.
- `FILTER_KEYS` gives the report order of the pack.
- `check_sentence_shape` gives all rules except `unique`.
  `LENGTH_RULES`, `TEXT_RULES`, and `MARK_RULES` hold their ids.
- The cull and the core run `sense` after the filter, on each target of a card.
- The cull and the core read the input deck with `read_input_deck` and
  `drop_silent_cards`. These two functions remove a card with no audio file
  before the filter.
- The cull and the core store the first failed rule as the card `reason`. The
  pack records a pass count for every guard before the first failure.

## 4. Length window

`length_window` is a filter, not a gate. One function holds the test:
`_passes_length` in `sentence_pack/pool.py`. All flows count units with
`profile.unit_count`: words, or Han characters.

| Flow          | Window                                      |
| ------------- | ------------------------------------------- |
| Pack          | `profile.min_units` to `max_units`          |
| Pack scaffold | `profile.fallback_min_units` to `max_units` |
| Pack frames   | the same as the pack, no scaffold retry     |
| Cull          | `profile.min_units` to `max_units`          |
| Core          | `profile.min_units` to `max_units`          |

The pack also runs `exceeds_max_units` before the measure step. It rejects a
sentence with more whitespace chunks than `max_units` and records
`length_window`.

| Language  | Pack, cull, and core | Pack scaffold minimum |
| --------- | -------------------- | --------------------- |
| French    | 4 to 10              | 3                     |
| Spanish   | 4 to 10              | 3                     |
| Greek     | 4 to 10              | 3                     |
| Mandarin  | 4 to 7               | `FALLBACK_MIN_HAN`    |
| Cantonese | 4 to 7               | `FALLBACK_MIN_HAN`    |

Each language states the window once, in its profile. No flag changes it. The
gates G5 and K5 read the same values. The run manifest records the window.

## 5. Deliberate differences

Do not compare the reject counts of two flows directly.

- A scaffold sentence skips the one-final-mark check, the internal-mark check,
  and the `sense` guard (`shape_failure(scaffold=True)`, `SCAFFOLD_SKIP_GUARDS`).
- A pack frame sentence runs the shape guards and a target-form test only. It
  skips `unique` and the scaffold retry, because a frame is template text.
- All flows take the marks from the language profile. The Mandarin and Cantonese
  core adapters convert the final mark to ASCII in `_sentence`. This is a deck
  style choice. A full-width mark inside the sentence stays, so a second final
  mark fails `marks`.
- The pack has no `english_link` rule. The pool reads only the linked sentences
  that have an English text (ADR 0037). The report shows the count as
  `english_paired`.

`tests/test_gate_id_map.py` checks the rule ids.

## 6. The sense rule

`sense` is the one rule for the meaning of a target word. A card shows a target
when three tests hold:

1. The form or the lemma is in the sentence.
2. The form is not inside a compound that gives a different meaning.
3. The English holds a sense word of the target and no false-friend word.

| Flow | Unit                 | A failed test                                          |
| ---- | -------------------- | ------------------------------------------------------ |
| Pack | candidate sentence   | rejects the sentence and counts it under `sense`       |
| Core | card and target pair | drops the target; rejects the card with no target left |
| Cull | card and target pair | the same as the core, and drops the lemma (ADR 0030)   |

All flows call `sense_failure(english, row, gloss)` in
`sentence_pack/english_sense.py`. It returns `false_friend`, `sense`, or None. A
tier S form passes a Swadesh `gloss`. A tier F lemma passes none.

One TSV file per language holds one record per target word. `read_senses` keys a
row by tier and word. A blank `tier` is `F`.

| Column          | Content                                  | Side     |
| --------------- | ---------------------------------------- | -------- |
| `senses`        | English words that show the sense        | English  |
| `false_friends` | English words that show another sense    | English  |
| `compounds`     | sentence strings that give another sense | sentence |
| `tier`          | `S` or `F`                               | -        |

- The compound test stays inside the occurrence count (`form_occurrences`,
  `single_hits`). The "exactly one hit" rule counts only the occurrences outside
  a compound, so `sense` calls that same code.
- The cull does not use `compounds`. Its targets are tokenizer lemmas, and its
  languages have no compounds.
- A compound-only sentence has no hit. It is not a candidate, so the pack counts
  no failure for it.
- Greek has no core flow. `vocab/greek_core_senses.tsv` holds tier S rows only.
- `vocab/english_senses.json` gives the sense words of a Swadesh gloss. WordNet
  builds it from the first synset of each gloss word (ADR 0048).
  `EXTRA_WORDS` in the builder adds the words that the first synset misses, for
  example `slim` for `thin` and `run` for `to flow` (ADR 0049).

`sense` does not replace the shape rules or `unique`. Core `K13` (form and Core
word in the sentence) stays.

## 7. History

| Id | Change                                           | Result                                           | Reference           |
| -- | ------------------------------------------------ | ------------------------------------------------ | ------------------- |
| L2 | one rule record, `(key, name, test)`             | deferred until a third consumer needs it         | -                   |
| L3 | one mark source for the core                     | done; one Cantonese core count changed           | `anki-0d3d`, 0028   |
| L4 | one manifest shape                               | deferred to the next pack rebuild                | -                   |
| L5 | one rule key namespace                           | rejected                                         | ADR 0027            |
| L6 | one identical rule set for every flow            | rejected; the jobs differ                        | -                   |
| L7 | `length_window` in the gate layer                | rejected; a gate never rejects a sentence        | -                   |
| L8 | `digit_marks` and `script` in all flows          | done; one Cantonese core card changed            | `anki-sxzu`, 0029   |
| L9 | one `sense` rule in every flow                   | done, S1 to S5                                   | ADR 0030, 0031      |

Detail of the changes:

- L3 added `final_marks`, `quote_marks`, and `internal_marks` to `CoreProfile`.
  It moved `CardShape` and `filter_reason` to `ranking/cardfilter.py`. The cull
  and the core now call one function with their own marks and token counts.
- L8 split the old `script` test. Each `script_ok` hook now tests only the
  alphabet. `digit_marks` is one function in `sentence_pack/pool.py`. Every flow
  runs both guards through `check_sentence_shape`.
- The cull and core reports once left out `repeat`. Both report lists now hold
  it.

Steps of L9:

| Id | Task                                                                  | Status            |
| -- | --------------------------------------------------------------------- | ----------------- |
| S1 | write the sense section                                               | done              |
| S2 | cull: remove a lemma that fails `sense_failure` in `select_cards`     | done, ADR 0030    |
| S3 | one shared function and one record; move the dicts into the TSV files | done              |
| S4 | one `sense` recheck gate in the cull and the core (G18, K22)          | retired, ADR 0039 |
| S5 | sense rows for all 300 cull targets, and a Cantonese file             | done, ADR 0031    |

L9 removed these parts:

| Part                                                     | Replacement                          |
| -------------------------------------------------------- | ------------------------------------ |
| `sense_filter` hook, `make_sense_filter`, `SENSE_CHECKS` | the `false_friends` column           |
| `false_friend_spans` hook, `FALSE_FRIENDS`               | the `compounds` column               |
| `holds_sense`, `holds_tier_s_sense`, `sense_match`       | `sense_failure`                      |
| `gate_no_sense_mismatch`, `gate_no_false_friend`         | the `sense` filter (ADR 0039)        |
| `gate_sense` (cull G18, core K22)                        | the `sense` filter (ADR 0039)        |

A deck built before S2 can keep a card that no sense test checked, and no gate
reports it. Build the deck again.

## 8. Open work

| Issue       | Problem                                          |
| ----------- | ------------------------------------------------ |
| `anki-xrfk` | Spanish cull G5: one card fails `length_window`  |

## 9. Verification

```bash
uv run pytest tests/test_cull_shared.py tests/test_french_cull.py \
  tests/test_spanish_cull.py tests/test_core_deck.py \
  tests/test_shared_coreflow.py tests/test_mandarin_core.py \
  tests/test_cantonese_core.py tests/test_swadesh_build.py
```

Rebuild the affected decks. Compare the kept count with the count before the
change.

## 10. References

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
