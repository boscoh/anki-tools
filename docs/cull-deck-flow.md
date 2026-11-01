# Cull Deck Flow

This document shows how `anki fix <language> --select` culls a sentence deck
to a small deck of wide cards. It shows the stage order, the rule order, and
the gate order. The shared code lives in `anki_tools/ranking/`. The language
values live in `anki_tools/languages/<language>/cull.py`.

The French and Spanish decks have the `--select` command.

Related: `docs/vocab-deck-flow.md`, `docs/core-deck-flow.md`,
`docs/swadesh-sentence-flow.md`, and `specs/french-1000-sentence-cull.md`.

## 1. Stage map

One command runs the stages in order: `anki fix <language> --select`. The
`--check` flag runs the gates on the output and skips C0 to C5.

| Stage | Name      | Input                    | Output                                      |
| ----- | --------- | ------------------------ | ------------------------------------------- |
| C0    | extract   | input APKG               | one card per note: sentence, English, audio |
| C1    | frequency | Tatoeba file, `wordfreq` | frequency rows and blended ranks            |
| C2    | tokenize  | C0                       | lemma and tag per card                      |
| C3    | filter    | C2, name list, marks     | passed cards, reason per card               |
| C4    | select    | C1, C2, C3, reserve pack | kept cards and reserve cards                |
| C5    | write     | C0, C4                   | APKG and three side files                   |

The gates R1, R2, R5, and R6 run after stage C5. The gate ids R3, R4, and R7 to R18 are retired. A cull gate id starts with `R` (ADR 0057).

The `--cards` and `--reserve` values change the request. The other
`CullRequest` fields, such as `max_uses` and `target_words`, keep their
defaults.

The input fields come from the profile: `input_model_name`,
`input_sentence_field`, `input_english_field`, and `input_audio_field`. C0
keeps the lowest card of each note. It cleans the sentence with
`clean_sentence`.

C0 drops a note with no sound tag. It also drops a card whose audio file is not
in the media map of the input APKG (`drop_silent_cards` in `ranking/scan.py`).

## 2. Flow diagram

```text
C0  input APKG (sentence, English, audio)
        |
        v
C1  frequency: Tatoeba counts blended with written counts, then ranks
        |
        v
C2  tokenize: lemma and tag per card
        |
        v
    target words (content lemmas, in rank order)
        |
        v
C3  filter
        |
        +--> F1 to F9  shared card filter, with the marks of the language
        |
        v
    reserve pack: one Swadesh card per concept, minus the duplicates
        |
        v
C4  selection
        |
        +--> sense: sense_failure per lemma; a failed lemma leaves the card
        |    lemma set (ADR 0030)
        |
        +--> gain of each card, minus the used and capped lemmas
        |
        +--> reject a card too close to a kept card
        |
        v
C5  APKG, .cull.csv, .coverage.csv, .manifest.json
        |
        v
    gates R1, R2, R5, R6
```

## 3. Target words (C1 and C2)

### 3.1 Frequency blend

`load_frequency` reads the cached TSV file of the language. The columns are
`word`, `tatoeba_count`, `tatoeba_pm`, `written_pm`, and `zipf`. When the file
is absent, the language builds it from the Tatoeba file with one tokenize
pass. Thus the first run costs more than the later runs.

```text
blend(row) = written_pm                             if tatoeba_count < 5
blend(row) = weight * tatoeba_pm + (1 - weight) * written_pm
```

`weight` is `spoken_weight`, 0.5 by default. A lemma with fewer than 5
Tatoeba occurrences uses the written term alone. `frequency_ranks` sorts the
rows by the blend value, then by the word, and gives 1-based ranks. A lemma
that the file does not hold gets the rank `UNRANKED_RANK`, 100000.

### 3.2 Target filter

A target is a content lemma that the deck holds. These rules are in
`build_targets` of `anki_tools/ranking/corelist.py`.

| Rule           | Value                                          |
| -------------- | ---------------------------------------------- |
| Part of speech | `NOUN`, `VERB`, `ADJ`, or `ADV`                |
| Deck count     | at least 3 content tokens                      |
| Content share  | at least 0.5 of all occurrences of the lemma   |
| Lemma length   | at least `min_lemma_length`, 2 by default      |
| Name list      | the lemma is not in the name stoplist          |

The targets are the first `target_words` of the rest in rank order, 300 by
default. The value is a `CullRequest` field, not a flag.

The target list sets the coverage report. It does not limit the selection.
The gain uses every ranked content lemma, so the kept cards cover more lemmas
than the target list.

## 4. Sentence rules (C3)

`filter_cards` applies F1 to F9 in order. The card rule is `filter_reason` of
`cardfilter.py`.

| Key | The rule passes when                                            |
| --- | --------------------------------------------------------------- |
| F1  | retired: the English text length (ADR 0034)                     |
| F2  | the unit count is inside the `min_units` to `max_units` window  |
| F3  | the text passes `digit_marks` and `script`                      |
| F4  | the last character is an accepted final mark                    |
| F5  | the text holds exactly one final mark                           |
| F6  | the text holds no `QUOTE_MARKS` character of the language       |
| F7  | the text holds at most one comma, colon, or semicolon           |
| F8  | the text holds no proper noun tag                               |
| F9  | the sentence text is not kept yet                               |

A language with a `register_ok` hook also runs the `register` rule after the
mark rules and before F9. French rejects a sentence with a passe simple verb
(ADR 0055).

The proper-noun rule F8 is a shared rule of `anki_tools/ranking/cardfilter.py`. F2
to F7 come from the sentence shape guard `check_sentence_shape`, with the window,
the marks, and the alphabet test of the language profile. One card filter,
`marks` (`MARK_RULES`), covers F4 to F7.

The unit count of F2 is `profile.unit_count`, and the window is `profile.min_units`
to `profile.max_units`. The language module states both (ADR 0034).

The run stores the first failed rule as the `reason` of the card. A passed
card has the empty reason.

## 5. Reserve pack

The reserve pack is a Swadesh sentence pack of the same language.
`load_reserve` reads one card per target from it, in pack order. The card
gives the sentence, the bold sentence, the English, the target, the gloss,
and the audio file.

`dedupe_reserve_sets` drops a reserve card in three cases:

- its sentence is in the input deck
- its lemma Jaccard value against an input card reaches `similarity`
- its lemma Jaccard value against an earlier reserve card reaches `similarity`

The test visits only the cards that share a lemma with the reserve card.

A reserve card counts against the card cap. Thus the frequency selection keeps
at most `max_cards - len(reserve)` cards. `--no-reserve` turns the reserve
pack off. A language without a reserve pack skips this stage.

## 6. Selection (C4)

### 6.1 Gain, cap, and similarity

```text
gain(card) = sum over content lemmas w of the card of
             0.1 * weight(w)                     if w is exempt
             weight(w) / (uses[w] + 1)           if w is not exempt and uses[w] < max_uses
             0                                   otherwise
weight(w)  = 1 / log2(rank(w) + 1)
```

The top `exempt_top` targets by rank are exempt from the cap. The default is
100. A non-exempt lemma appears in at most `max_uses` kept cards. The default
is 4. A card that holds a non-exempt lemma at its cap is skipped with the
reason `use_cap`.

The selector rejects a card when the lemma Jaccard value against a kept card
that shares a lemma reaches `similarity`, 0.67 by default. It compares only
the kept cards that share a lemma.

`greedy_cover` runs the loop. A tie breaks on fewer tokens, then on the lower
input index. Two runs give the same output.

### 6.2 Stop rule

The selection stops in three cases:

- the kept frequency card count reaches `max_cards - len(reserve)`
- every candidate has a gain of zero
- every remaining candidate is capped or too similar

A card that the selection does not keep gets the reason `max_cards`,
`use_cap`, or `similar`.

### 6.3 Order

`order_cards` sorts the kept cards with this key. The `order` value of
`CullRequest` is `coverage` by default.

1. `hardest_rank`: the rank of the least frequent content lemma, ascending
2. the token count, ascending
3. the input index, ascending

An unranked lemma counts as rank 1000000. The value `input` keeps the input
order.

The cull does not reorder the input deck. The coverage order lives in the cull
CSV and in the manifest. The APKG keeps the input note order of the kept
cards, and the reserve notes follow them.

## 7. Files (C5)

| File                       | Content                                                 |
| -------------------------- | ------------------------------------------------------- |
| `<language>_1000.apkg`       | the deck: the kept cards and the reserve cards          |
| `<stem>.cull.csv`          | one row per input card and reserve card                 |
| `<stem>.coverage.csv`      | per target: word, rank, count, status, gloss            |
| `<stem>.manifest.json`     | params, counts, frequency source, licenses, hashes      |

`identity.default_output(language, "cull")` gives `apkg/<language>_1000.apkg`.
Every name comes from the output file stem (`anki_tools/ranking/identity.py`).

The cull CSV columns are `source`, `sentence`, `english`, `audio`, `kept`,
`reason`, `targets`, `gain`, and `hardest_rank`. The `source` value is
`swadesh` or `frequency`. The rows are the kept reserve cards, then the kept
input cards in coverage order, then the rejected input cards.

The coverage CSV columns are `word`, `rank`, `count`, `status`, and `gloss`.
The gloss comes from the sense file of the profile, `sense_path`.

The coverage status of a target is one of these values:

| Status    | Meaning                              |
| --------- | ------------------------------------ |
| `covered` | the target holds two or more cards   |
| `partial` | the target holds one card            |
| `absent`  | the target holds no card             |

The deck title and the note type name are the output file stem, and so is
`manifest.source.model`. The writer (`anki_tools/apkg/shape.py`) clones the
input note type under that name. The output holds one note type, one field
set, and one card template. The assembler (`anki_tools/ranking/assemble.py`)
writes the side files, the manifest, and the report.

The writer deletes the notes that no kept card uses, inserts the reserve notes,
and deletes the input note type and every other note type. It keeps the `due`
value of each kept card. `reserve_fields` gives the five output values of a
reserve card, so the pack `Target` becomes `Core` and the pack `Gloss` becomes
`CoreGloss`.

The field set is `output_fields(profile)`: `Audio`, `Sentence`, `English`, the
`Romanization` field when the language has a romanization, `Core`, and
`CoreGloss`. `card_cores` takes the first
target lemma of the card that has a sense row, and `coreflow.note_values` writes
that lemma and its first sense. `select_cards` leaves out a card without a
target before the pick, so the cap fills from cards that have a core word. Its
CSV row holds `kept = 0` and `reason = no_core`. `counts.dropped_no_core` is 0
for a cull and stays in the manifest for gate R1.

The report line `Length pool/kept %` gives `length:pool/kept` per sentence
length, as whole percent of the passed cards and of the kept frequency cards.
No gate tests it.

The manifest records the output note type in `source.model` and the input note
type in `source.input_model`. It records the kept order hash of the reserve
sentences and the kept input sentences. It records the two CSV hashes, the
frequency source, the licenses, the reserve pack path, and the Swadesh
exceptions.

## 8. Audit gates

`--check` reads the output files only. The gates fall in these groups:

| Group     | Gates        | Checks                                                    |
| --------- | ------------ | --------------------------------------------------------- |
| size      | R1, R2       | card count against the cap, lemma uses against the cap    |
| rules     | R5, R6       | F1 to F9 on every kept card, similarity                   |

The gate ids R3, R4, R7 to R11, R13 to R17 are retired (ADR 0033, 0035, 0040 to 0045). Their ids are not reused. The
report and the coverage CSV still give the targets with one card and with two
cards, and the Swadesh count of the reserve concepts. No gate tests them.

Gate R18 is retired (ADR 0039). The `sense` filter is the one sense test, and no gate
repeats it.

Gate R2 counts the lemma uses with `sense_lemma_sets`, the same function that
`select_cards` uses. A lemma that fails the sense test on a card is not a use of
that card. Gates R2 and R6 read their limits from the constants `DEFAULT_MAX_USES`,
`EXEMPT_TOP`, and `SIMILARITY`, not from the manifest. Gate R1 reads the cap from the
manifest, because the user sets it with `--cards`. Gate R5 checks the frequency rows
and skips the reserve rows, which have their own length window (ADR 0046).

Gate R1 passes when the kept count plus `counts.dropped_no_core` reaches the cap. It
also passes when the pool has no more cards: no unkept row has the reason
`max_cards`. Every other passed card then has the reason `similar`,
`use_cap`, `no_gain`, or `no_core`. A small input deck does not fail gate R1.

The similarity test of `select_cards` and gate R6 use the full lemma sets of the
cards. A duplicate is a property of the text, so the sense test does not change
it.

## 9. File map

| Path                                      | Content                                        |
| ----------------------------------------- | ---------------------------------------------- |
| `anki_tools/ranking/cullflow.py`          | C0 to C5 flow, profile, selection, values      |
| `anki_tools/ranking/cullgates.py`         | gates R1, R2, R5, R6, 4 rows                   |
| `anki_tools/gates.py`                     | gate value, row, runner, reader contract       |
| `anki_tools/ranking/gates.py`             | report, shared gates                           |
| `anki_tools/ranking/assemble.py`          | deck, side files, manifest, report             |
| `anki_tools/ranking/identity.py`          | deck title, note type, side-file stems         |
| `anki_tools/ranking/deckprofile.py`       | language facts, core view, cull view           |
| `anki_tools/apkg/shape.py`                | the written package shape                      |
| `anki_tools/ranking/fixrun.py`            | the shared `fix` route                         |
| `anki_tools/ranking/coreflow.py`          | input deck reader, output field values         |
| `anki_tools/ranking/corelist.py`          | target word list                               |
| `anki_tools/ranking/cardfilter.py`        | shared filter rules F1 to F9                   |
| `anki_tools/ranking/cover.py`             | the generic greedy cover loop                  |
| `anki_tools/ranking/duplicates.py`        | Jaccard value and the limit test               |
| `anki_tools/languages/<language>/cull.py` | one `CullProfile` per language                 |
| `anki_tools/cli.py`                       | the `--select` mode of `fix <lang>`            |

## 10. Invariants

- A language gives values only. Do not add a language condition to
  `cullflow.py`. A new value goes into `CullProfile`.
- A new language needs one `CullProfile` in `languages/<language>/cull.py`,
  one `concept_forms` hook, and one `reserve_fields` hook.
- The filter reasons are the keys of `FILTER_RULES`, F1 to F9. The shape guard
  names do not go into the cull CSV.
- The reserve cards count against the card cap. The frequency selection limit
  is `max_cards - len(reserve)`.
- The gates read the output files only. A gate function lives in
  `cullgates.py`, and it takes a `CullFiles` reader.
- The selection scans the cards in input order, so two runs give the same
  cards and the same order.
