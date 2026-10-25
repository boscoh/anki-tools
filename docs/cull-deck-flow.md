# Cull Deck Flow

This document shows how `anki fix <language> --select` culls a sentence deck
to a small deck of wide cards. It shows the stage order, the rule order, and
the gate order. The shared code lives in `anki_tools/ranking/`. The language
values live in `anki_tools/languages/<language>/cull.py`.

The French deck has the `--select` command. Spanish has a `CullProfile`, and
it has no command flag yet.

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

The gates G1 to G15 run after C5.

The `--cards`, `--targets`, `--max-uses`, and `--reserve` values change the
request. The other `CullRequest` fields keep their defaults.

The input fields come from the profile: `model_name`, `sentence_field`,
`english_field`, and `audio_field`. C0 keeps the lowest card of each note. It
cleans the sentence with `clean_sentence`.

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
        +--> gain of each card, minus the used and capped lemmas
        |
        +--> reject a card too close to a kept card
        |
        v
C5  APKG, .cull.csv, .coverage.csv, .manifest.json
        |
        v
    gates G1 to G15
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

The targets are the first `--targets` of the rest in rank order. The default
is 300.

The target list sets the coverage report. It does not limit the selection.
The gain uses every ranked content lemma, so the kept cards cover more lemmas
than the target list.

## 4. Sentence rules (C3)

`filter_cards` applies F1 to F9 in order. The card rule is `filter_reason` of
`cullflow.py`.

| Key | The rule passes when                                            |
| --- | --------------------------------------------------------------- |
| F1  | the English text has 2 to 12 words                              |
| F2  | the token count is inside the `min_words` to `max_words` window |
| F3  | the text holds no digit, bracket, link, or ellipsis             |
| F4  | the last character is an accepted final mark                    |
| F5  | the text holds exactly one final mark                           |
| F6  | the text holds no quotation mark and no dialogue dash           |
| F7  | the text holds at most one comma, colon, or semicolon           |
| F8  | the text holds no proper noun tag                               |
| F9  | the sentence text is not kept yet                               |

The English window F1, the script rule F3, and the proper-noun rule F8 are
the shared rules of `anki_tools/ranking/cardfilter.py`. F2 and F4 to F7 come
from the sentence shape guard `check_sentence_shape`, with the marks of the
language profile. The shape guard names the mark rules F5 to F8, and
`MARK_RULES` maps them to the card filter names F4 to F7.

The unit count of F2 is the token count of the card, not a text measure. The
`CardShape` adapter gives the unit count and the three mark sets to the shape
guard.

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
| `<stem>_culled.apkg`       | the deck: the kept cards and the reserve cards          |
| `<stem>.cull.csv`          | one row per input card and reserve card                 |
| `<stem>.coverage.csv`      | one row per target: word, rank, count, status           |
| `<stem>.manifest.json`     | params, counts, frequency source, licenses, hashes      |

`default_output` gives `<stem>_culled.apkg` next to the input file.

The cull CSV columns are `source`, `sentence`, `english`, `audio`, `kept`,
`reason`, `targets`, `gain`, and `hardest_rank`. The `source` value is
`swadesh` or `frequency`. The rows are the kept reserve cards, then the kept
input cards in coverage order, then the rejected input cards.

The coverage status of a target is one of these values:

| Status    | Meaning                              |
| --------- | ------------------------------------ |
| `covered` | the target holds two or more cards   |
| `partial` | the target holds one card            |
| `absent`  | the target holds no card             |

The deck name comes from `profile.default_deck_name`. The note model keeps
the name of the input model. The writer deletes the notes that no kept card
uses, inserts the reserve notes, and sets the templates and the CSS of the
profile.

The manifest records the kept order hash of the reserve sentences and the kept
input sentences. It records the two CSV hashes, the frequency source, the
licenses, the reserve pack path, and the Swadesh exceptions.

## 8. Audit gates

`--check` reads the output files only. The gates fall in these groups:

| Group     | Gates        | Checks                                                    |
| --------- | ------------ | --------------------------------------------------------- |
| size      | G1, G2       | card count against the cap, lemma uses against the cap    |
| coverage  | G3, G4, G12  | covered targets, two-card share, coverage CSV rows        |
| rules     | G5, G6       | F1 to F9 on every kept card, similarity                   |
| files     | G7, G8, G9   | audio files, note count, media set                        |
| identity  | G10, G11     | deck name, model name, kept order                         |
| manifest  | G13          | the two CSV hashes                                        |
| swadesh   | G14          | the merged deck covers the reserve concepts               |
| templates | G15          | the cull templates and the CSS classes                    |

G3 fails below `target_coverage_gate`, 0.98 for French and Spanish. G4 fails
below `double_coverage_gate`, 0.95. The profile holds both values.

G5 skips F8 for a reserve card, because the Swadesh pack already checked that
sentence. G14 needs the Swadesh count minus the exceptions of the profile, for
example `intestins` in French. G15 needs the template set of the profile and
each card class in the CSS.

G1 passes when the card count reaches the cap, or when the pool has no more
cards. A small input deck does not fail G1.

## 9. File map

| Path                                      | Content                                        |
| ----------------------------------------- | ---------------------------------------------- |
| `anki_tools/ranking/cullflow.py`          | C0 to C5 flow, profile, selection, files       |
| `anki_tools/ranking/cullgates.py`         | gates G1 to G15                                |
| `anki_tools/ranking/coreflow.py`          | input deck reader, deck writer, reserve insert |
| `anki_tools/ranking/corelist.py`          | target word list                               |
| `anki_tools/ranking/cardfilter.py`        | shared filter rules F1 to F9                   |
| `anki_tools/ranking/cover.py`             | the generic greedy cover loop                  |
| `anki_tools/ranking/duplicates.py`        | Jaccard value and the limit test               |
| `anki_tools/ranking/coverage.py`          | coverage CSV rows                              |
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
