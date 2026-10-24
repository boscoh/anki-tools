# Spec: Core-Word Small Deck (`anki fix --core`)

| Field   | Value                                                    |
| ------- | -------------------------------------------------------- |
| Status  | draft                                                    |
| Date    | 2026-10-21                                               |
| Owner   | anki-tools                                               |
| Input   | a sentence deck with English and audio, for example `apkg/6000_French_Sentences_w_Audio.apkg` |
| Related | `specs/french-1000-sentence-cull.md` (cull mode)         |
| Related | `specs/yue-sentence-selection.md` (spoken frequency)     |
| Related | `docs/swadesh-sentence-flow.md` (guards and word table)  |
| Issues  | epic `anki-eo6`; milestones E1 to E5 (section 12)        |

## Contents

1. [Summary](#1-summary)
2. [Target set](#2-target-set)
3. [Sentence rules](#3-sentence-rules)
4. [Selection](#4-selection)
5. [Size and presets](#5-size-and-presets)
6. [Card order](#6-card-order)
7. [Pipeline and code](#7-pipeline-and-code)
8. [Card design](#8-card-design)
9. [Interface](#9-interface)
10. [Gates and tests](#10-gates-and-tests)
11. [Risks and open questions](#11-risks-and-open-questions)
12. [Milestones](#12-milestones)
13. [Appendix: evidence](#13-appendix-evidence)
14. [Implementation result](#14-implementation-result)

## 1. Summary

`anki fix fr --select` culls a deck to 1,000 wide cards. This spec adds
`--core`, which builds a small deck from a fixed core word list. Each core
word gets 2 examples. A sentence that brings many words outside the core
costs more, so the deck adds few other words.

### 1.1 Result at a glance

Built on the French 6,000-card deck with the preset of 5.2, which is the CLI
default.
The numbers come from the built deck, not from the prototype.

| Measure                        | 1,000-card cull | This spec |
| ------------------------------ | --------------- | --------- |
| Cards                          | 1,000           | 300       |
| Core targets                   | 300             | 209       |
| Core targets with 2 examples   | 287             | 183       |
| Swadesh coverage, concepts / forms | 206 / 207    | 159 / 203 |
| Words outside the core, core cards | none measured | 247       |
| Words from the tail            | none measured   | 346       |

The deck has three parts: 209 core targets, 247 support words in the core
cards, and 346 new words from the tail. Five of the 209 targets are
`absent_sense`, and 21 have one card, because the pool holds one sentence for
them.

`--core` cannot cover a Swadesh form that the input deck does not hold. The
French input holds 159 of the 203 distinct forms with the default window and
the full filter. Section 3.2 explains the gap, and 5.3 the reserve option.

### 1.2 Key decisions

| Decision        | Choice                                                    |
| --------------- | --------------------------------------------------------- |
| Target order    | Swadesh index first, then spoken frequency rank            |
| Target unit     | a form for Swadesh, a content lemma for frequency          |
| Examples        | 2 per content target; 1 for a form in more than 25% of the candidates |
| Third example   | off. It costs 33 cards and adds no `covered2`              |
| Selection       | greedy cover of a value minus a cost for outside words     |
| Sentence rules  | simple, complete, unique, no name, no loan word, audio     |
| Sense           | a card gets credit only when the English shows the sense    |
| Card design     | audio-only front; the back adds the core word              |
| Output size     | about 1.0 card per target at quota 2                       |
| Swadesh reserve | off. 198 cards cannot fit a 300-card core deck             |

Not in scope: new audio or TTS, a new note model, a grammar syllabus, and a
change to the cull window.

## 2. Target set

### 2.1 Two tiers, one order

| Tier | Source                                                    | Order           |
| ---- | --------------------------------------------------------- | --------------- |
| S    | the reviewed Swadesh word table of the language (`S1`)     | `swadesh_index` |
| F    | the cached blended frequency list, after the tier S forms  | blended rank    |

Tier S comes first. Tier F follows in rank order. Each target has a 1-based
`position` in this list.

A shared Swadesh form gives one target when the two glosses share a sense
word. Otherwise the form gives two targets, and the sense guard keeps their
examples apart (3.3). The prototype found 207 rows and 203 distinct forms.

A target that no candidate holds stays in the list with the status `absent`.
Gates count `targets`, not `targets + absent`.

### 2.2 Match rule

| Tier | Unit   | A card holds the target when                            |
| ---- | ------ | ------------------------------------------------------- |
| S    | a form | the form occurs in the sentence, by the pack match rule |
| F    | a lemma | a token of the card has the lemma and a content tag    |

Tier S reuses `form_occurrences` and `form_offsets` of
`anki_tools/sentence_pack/guards.py`, with the `FALSE_FRIENDS` spans of the
language. Thus 闻 inside 新闻 is not a hit, and 食 counts inside 食飯.

Tier F uses the tokenizer of the language and the tags `NOUN`, `VERB`, `ADJ`,
and `ADV`.

### 2.3 Quota

```text
quota(w) = 1                  if share(w) > --ubiquitous-share (0.25)
quota(w) = --per-word (2)     otherwise
quota(w) = 3                  only for the first --third-examples targets
```

`share(w)` is the number of candidates that hold the target, divided by the
number of candidates.

Reason for the first rule: pronouns, negation, and auxiliaries are Swadesh
targets, and they occur in most sentences. A second example costs a card and
adds no skill. The third example is off (1.2).

### 2.4 Target filters

A tier F candidate is a target when all these rules hold.

| Rule           | Value                                               |
| -------------- | --------------------------------------------------- |
| Part of speech | `NOUN`, `VERB`, `ADJ`, or `ADV`                     |
| Content share  | at least 0.5 of all occurrences of the lemma        |
| Deck count     | at least 3                                          |
| Lemma length   | 2 characters or more                                |
| Name list      | the lemma is not in `vocab/<language>_names.*`      |
| Swadesh overlap| the lemma is not the lemma of a tier S form         |

These rules are `build_targets` of `anki_tools/languages/french/cull.py`.

## 3. Sentence rules

### 3.1 Rules

The mode applies the cull filter and five extra guards. A card must pass
every rule.

| Key    | Rule                                                       |
| ------ | ---------------------------------------------------------- |
| F1–F9  | the cull filter: English 2 to 12 words, the length window, no digit or URL, one final mark, no quote or dialogue dash, at most one internal mark, no `PROPN`, unique text |
| C1     | a predicate: a `VERB`, `AUX`, or `ADJ` token                |
| C2     | a main clause: no orphan conjunction                        |
| C3     | no name from `vocab/<language>_names.*`                     |
| C4     | no word from `vocab/<language>_loanwords.txt`               |
| C5     | the card has an audio file in the input media map           |

- C1 rejects a fragment such as `Très bon.` C2 rejects a clause that starts
  with a conjunction and has no main verb.
- C3 is a whole-word test for French, so the name `tom` does not reject
  `tomber`. A language without a `PROPN` tag uses the substring test of the
  yue selector instead.
- C4 is a whole word match, so `stop` does not reject `stopover`. The list is
  optional; gate K17 reports the count.
- C5 makes this an audio deck. All 5,928 cards of the French input have
  audio, so C5 removes 0 cards. A card without audio is not a candidate.

### 3.2 Length window

| Language  | Unit     | Window | Simple limit | Evidence                                        |
| --------- | -------- | ------ | ------------ | ----------------------------------------------- |
| French    | word     | 5 to 9 | 7            | the pack preferred range; the cull uses 3 to 12 |
| Greek     | word     | 5 to 8 | 7            | the pack preferred range                        |
| Cantonese | Han char | 4 to 7 | 6            | the pack uses 4 to 8; this spec uses 7          |
| Mandarin  | Han char | 4 to 7 | 6            | the pack window                                 |

`--lambda-len` applies to each unit above `--simple-units` (7). `--core-min-words`
and `--core-max-words` set the window.

Measured effect on the French input: the window keeps 3,442 of the 4,958
cards that pass F1 to F9. F2 rejects 1,869 cards, and C1 rejects 39 more. The
window costs 9 Swadesh forms that then have no candidate (164 present forms
become 155).

### 3.3 Sense guard

A target is a word and a sense. A card gets credit only when the English
text holds that sense.

| Tier | Sense words                                              | False-friend words     |
| ---- | -------------------------------------------------------- | ---------------------- |
| S    | the Swadesh gloss, or the `SENSE_CHECKS` row of the form   | the `SENSE_CHECKS` row |
| F    | the sense words of the target row of the sense file        | its false-friend words |

```text
hit(w, c) = the form or lemma of w occurs in c
            and the English holds a sense word of w
            and the English holds no false-friend word without a sense word
```

- Reuse `make_sense_filter` (`sentence_pack/profile.py`), `holds_sense`
  (`sentence_pack/english_sense.py`), and `compound_spans`
  (`sentence_pack/guards.py`).
- A hit that fails does not count. The card stays when another target counts.
- A target with no sense row is `unchecked`. K16 counts it against
  `--max-unchecked-sense`.
- A target whose every candidate fails the sense test is `absent_sense`. It
  does not fail K2.
- Tier F rows live in `vocab/<language>_core_senses.tsv`, with the columns
  `word`, `senses`, and `false_friends`, comma-separated.

The French pack has no `SENSE_CHECKS` rows, so the French run needs the new
file before K16 can pass. The file starts with the top 50 tier F targets.

## 4. Selection

### 4.1 Value and cost

```text
weight(w) = 1 / log2(position(w) + 1)
demand(u) = 1.0, 0.35, 0.12       for example 1, 2, 3
value(c)  = sum of weight(w) * demand(uses[w])
            over the targets w of c with uses[w] < quota[w]

outside(c) = content lemmas of c that are not targets
new(c)     = outside(c) minus introduced
cost(c)    = 0.1 * sum(mult(w) for w in new(c))
             + 0.03 * |outside(c) - new(c)|
             + 0.02 * max(0, units(c) - 7)

mult(w)    = 1.0 if rank(w) <= 1000, 1.8 if rank(w) <= 3000, else 3.0
score(c)   = value(c) - cost(c)
```

- The value falls with the target position and with the example number.
- A new outside word costs. A repeated one costs little. A rare new word
  costs 3 times a frequent one.
- `--lambda-new`, `--lambda-repeat`, `--lambda-len`, and `--simple-units`
  change the constants of the cost. Section 13.2 shows the effect of
  `--lambda-new`.

### 4.2 Passes

| Pass       | Quota rule           | Accept                                   | Goal                    |
| ---------- | -------------------- | ---------------------------------------- | ----------------------- |
| seed       | 1 per target         | `score > 0`, with a small cost (0.01)     | 1 example of each target |
| depth      | `quota(w)`           | `score > 0`                              | the full quota          |
| completion | `quota(w)`           | the best card, even at `score <= 0`, at most 120 cards | fill a short quota |
| tail       | none                 | `score > 0`, gain = the new lemmas only   | width, at most `--tail-cards` |
| top-up     | 3 for the first N    | `score > 0`                              | off by default          |

The tail runs last, and it does not take a card from a target below quota,
because the earlier passes fill the quota first. The tail gain has no cost.

### 4.3 Loop, stop, and determinism

```text
repeat:
    pick the unkept card with the largest score
    over the candidates that have value > 0
        and pass the score rule of the pass
        and are not a duplicate of a kept card
    if no card: stop the pass
    keep the card, then update uses and introduced
```

- Stop a pass when no card qualifies, at `--core-max-cards`, or at
  `--outside-budget` (0 is no cap; the cost does the work).
- `anki_tools/ranking/core.py` gives the passes, the value, the cost, and
  the loop. Each step scans the candidates in input order and keeps the best
  key: `(score, -new words, -units, -index)`, with the largest key first.
- Duplicates: lemma Jaccard at `--similarity` (0.67) or above, tested only
  against kept cards that share a lemma.
- The scan order is the input order. A tie goes to fewer new words, then
  fewer units, then the lower input index. Two runs give the same output.

## 5. Size and presets

### 5.1 Size

The table uses the wide window 3 to 12 words, to show the sizing. The presets
of 5.2 use the default window of 3.2.

| Tier F words | Targets | Cards | `covered2` | `outside` |
| ------------ | ------- | ----- | ---------- | --------- |
| 80           | 244     | 240   | 230        | 227       |
| 150          | 314     | 300   | 300        | 227       |
| 250          | 414     | 389   | 399        | 253       |
| 400          | 564     | 523   | 548        | 253       |

`cards ~ 0.95 * targets` at quota 2. A kept card holds about 2 target hits.

Check the request before the run:

```text
if quota_total > 2.1 * max_cards:
    stop with "the card budget is too small for this quota"
```

Thus a 200-card budget cannot give 2 examples of 314 targets, and the command
says so instead of leaving a gap.

### 5.2 Presets

The rows come from a prototype of the selection with the rules of section 3,
except that the prototype matched tier S by lemma. The tail fills to 300.

| Preset                    | Core words | Cards | Vocab | `spoken_mass` | `covered2` |
| ------------------------- | ---------- | ----- | ----- | ------------- | ---------- |
| Swadesh only, quota 2     | 0          | 179   | 394   | 0.307         | 142 / 155  |
| Swadesh only, tail 300    | 0          | 300   | 833   | 0.459         | 142 / 155  |
| Swadesh + 50, quota 2     | 50         | 216   | 420   | 0.340         | 192 / 205  |
| Swadesh + 50, tail 300    | 50         | 300   | 740   | 0.436         | 192 / 205  |
| Swadesh + 150, quota 2    | 150        | 303   | 531   | 0.377         | 292 / 305  |

The recommended preset is the fourth row:

```bash
uv run anki fix fr "apkg/6000_French_Sentences_w_Audio.apkg" --core
```

These values are the CLI defaults: `--core` alone builds this deck.

The built deck gives 300 cards for 209 targets: 183 targets with 2 cards, 21
with one, and 5 `absent_sense`. The tail adds 346 new words. The prototype
row above reads 192 of 205 because it uses the lemma match and no C3 or C4.
The built deck rejects 25 cards with C3 and 32 with C4; the form match moves a
few targets (section 13.1).

The core deck is not the widest deck for its card count. A breadth selection
at 300 cards covers 1,201 lemmas at mass 0.497, against 740 at 0.436 here.
The tail is how you buy that width. The core wins on mass per lemma, because
its cards go to frequent words. Section 13.5 gives the table.

### 5.3 Swadesh reserve

`--reserve-pack` is a select-mode option. Core mode does not read it. The
default is off and stays off: the pack has about 198 cards, which is two
thirds of a 300-card core deck. Build the reserve as its own deck. The table
below is the design for a later version.

| Rule          | Value                                       |
| ------------- | ------------------------------------------- |
| Duplicate     | the same test as the cull, section 16.3     |
| Quota         | one reserve card counts as one example      |
| Cap           | every reserve card is kept                  |
| Budget        | the frequency pass fills the rest           |
| Outside cost  | a reserve card pays no cost                 |

## 6. Card order

`reorder_deck` sets the `due` values from the ranking CSV, so the Anki code
path does not change.

`assign_targets` gives each kept card one target: the lowest position among
its targets that no earlier card assigned. The sort keys are:

| Order | Key                                                    |
| ----- | ------------------------------------------------------ |
| 1     | the position of the assigned target                     |
| 2     | the example number of that target in this card (1, 2, 3) |
| 3     | the count of new outside words of the card              |
| 4     | the unit count of the card                              |
| 5     | the input index                                         |

The deck teaches the core list in order, and a card follows the card that
teaches its target. `hardest_rank` stays in the CSV as a report value.

## 7. Pipeline and code

| Stage | Name      | Function                                                     |
| ----- | --------- | ------------------------------------------------------------ |
| K0    | Extract   | `cull.load_input_deck`                                       |
| K1    | Tokenize  | `cull.tokenize_deck`                                         |
| K2    | Targets   | `load_frequency`, `frequency_ranks`, `profile.load_word_table`, the sense file |
| K3    | Filter    | `cull.filter_cards`, then C1 to C5 of 3.1                    |
| K4    | Select    | this spec, over `ranking.core.select`                        |
| K5    | Order     | `ranking.core.assign_targets` and `ranking.core.order_cards` |
| K6    | Write     | `cull.write_deck`, `apply_card_templates`, `reorder_deck`    |

Code changes:

- Add `anki_tools/languages/<language>/core.py` for K2, K4, and K5. Keep the
  generic gain, cost, and pass code in `anki_tools/ranking/core.py`.
- Add the filter C1 to C5 as `core_filter_reason` after the cull reason. The
  predicate and orphan tests are the tests of `sentence_pack/pool.py`.
- Share the filter with the cull through `filter_reason(card, tokens, profile)`:
  the cull keeps 3 to 12 words, C1 and C2, C3 only where F8 has no tagger,
  C4 and C5 off, and the sense guard only when a sense file exists. The cull
  part is `anki-0h8`.
- Add the sense guard as `core_sense_hits`, over `make_sense_filter` and
  `holds_sense`.
- Add `vocab/<language>_core_senses.tsv` and
  `vocab/<language>_loanwords.txt`. Add a name list where there is none.
- Put the passes, the value, the cost, the loop, `assign_targets`, and
  `order_cards` in `anki_tools/ranking/core.py`. `ranking/cover.py` keeps the
  cull and sentence pack loop.
- Add the `--core` option block to `anki fix fr` in `cli.py`.
- Reuse the APKG writer and `reorder_deck` unchanged.

## 8. Card design

The note ids do not change. The model gains two fields, `Core` and
`CoreGloss`, and keeps one
card template, so each note has one card. The CSS is the cull value, so the deck
looks like the 1,000-card deck and the Swadesh deck.

| Card             | Front          | Back                                                                     |
| ---------------- | -------------- | ------------------------------------------------------------------------ |
| 1, comprehension | the audio only | the audio, the sentence with the bold target, the English, the core word |

| Part      | Value                                                                  |
| --------- | ---------------------------------------------------------------------- |
| Templates | `templates/french_core/`, one card, written by `_write_core_model`     |
| CSS       | `card_css()`: shared `card.css` plus `sentence_pack/french/card.css`   |
| Classes   | `.front-audio`, `.sentence`, `.english`, `.target`                     |
| Fields    | the input fields, `Core`, and `CoreGloss`                              |
| Audio     | the input MP3 file                                                     |

The `assigned` target of section 6 is bold in the `French` field. A card can
hold more than one target, and only the assigned target is bold. This marks
the word in the sentence.

The `Core` field holds the core word for the back of the card. It is the form of
the assigned target. A card with no assigned target, such as a second example,
gets the held target of the lowest position. A tail card that holds no target
has an empty `Core` field, and the template hides the line.

The `CoreGloss` field holds the English gloss of the core word. A tier S
target uses the Swadesh gloss of the word table, for example `water`. A tier F
target uses the first sense of `vocab/french_core_senses.tsv`, for example
`not` for `pas`. The template shows the word and the gloss with a middle dot,
so the core deck and the Swadesh deck read the same way.

| Field        | Value of a kept card                                                           |
| ------------ | ------------------------------------------------------------------------------ |
| `French`     | the sentence, with the assigned target in `<b>` tags                           |
| `sound`      | `[sound:<input audio file>]`                                                   |
| `Back`       | the English text of the card                                                   |
| `freq ranks` | unchanged                                                                      |
| `Core`       | the core word of the card, as described above; empty for a card with no target |
| `CoreGloss`  | the English gloss of the core word; empty for a card with no core word         |

- `bold_target` is on by default. A false value writes plain text.
- The CSV holds the plain sentence; the markup stays in the APKG.
  `_plain_sentence` removes it on a later extract.
- The `Core` field changes the note model. Anki can ask for a full sync when you
  import the deck over an older copy of it.

## 9. Interface

### 9.1 Options

| Flag          | Default             | Meaning                                 |
| ------------- | ------------------- | --------------------------------------- |
| `--core`      | off                 | Run the core selection                  |
| `--cards`     | 300                 | Card cap                                |
| `--words`     | 50                  | Tier F words after the Swadesh set      |
| `--examples`  | 2                   | Examples per content target, 1 to 3     |
| `--tail`      | 120                 | Cards after the core passes             |
| `--output`    | from the input name | Output APKG                             |
| `--check`     | off                 | Run gates K1 to K19 on the output       |

The other values are fields of `CoreOptions` or `CoreRequest`: the third
examples, the ubiquitous share, the completion cap, the outside budget, the
rank curve, the four lambdas, the free unit count, the word window, the
sense and loan word paths, the unchecked limit, the similarity limit, the
spoken weight, and the bold flag. The defaults are in `core.py` and
`ranking/core.py`.

`--core` and `--select` cannot be used together. Rank mode stays the default.

### 9.2 Output files

| File                                 | Content                                                     |
| ------------------------------------ | ----------------------------------------------------------- |
| `apkg/french_core_300.apkg`          | the kept cards in core order, with the input model and audio |
| `apkg/french_core_300.core.csv`      | one row per input card: `note_id`, `index`, `sentence`, `english`, `audio`, `kept`, `pass`, `reason`, `targets`, `assigned`, `value`, `cost`, `units`, `new_lemmas`, `sense`, `bold` |
| `apkg/french_core_300.coverage.csv`  | one row per target: `key`, `word`, `tier`, `position`, `quota`, `count`, `status` |
| `apkg/french_core_300.manifest.json` | counts, options, frequency source, hashes                   |
| `apkg/french_core_300.rank.csv`      | the ranking CSV of `reorder_deck`                           |

`status` is `covered`, `partial`, `absent`, `absent_sense`, or `unchecked`.

`targets` joins the keys with a vertical bar, not a space, because a form such
as `ne pas` holds a space.

## 10. Gates and tests

### 10.1 Gates

`anki fix fr --core --check` runs the gates on the output. A failure exits
non-zero.

| Gate | Check                                                                |
| ---- | -------------------------------------------------------------------- |
| K1   | the card count is at most `--core-max-cards`, and the CSV, the APKG, and the manifest agree |
| K2   | every coverable target has a kept card, where a target whose every candidate fails the sense test is `absent_sense` and is not counted |
| K3   | report, not a gate: the share of coverable targets with 2 cards (French measured 0.897) |
| K4   | the outside set of the core cards is at most 1.3 times the target count, and the rare outside words are at most 0.1 of the outside set (measured 247 of 209) |
| K5   | every kept card passes F1 to F9 and C1 to C5, and the APKG text equals the CSV text |
| K6   | no kept pair reaches the similarity limit                            |
| K7   | every kept card has a valid MP3 file                                 |
| K8   | the note count equals the kept count and the manifest count, and the media files are exactly the kept files |
| K9   | the deck name and the note model are the requested values             |
| K10  | the kept order equals the recorded hash, and the CSV holds every kept card |
| K11  | the coverage CSV has one row per target, the counts match the cards, and every status is valid |
| K12  | the manifest hashes match the two CSVs                               |
| K13  | each target's cards hold its form or lemma                           |
| K14  | the templates equal `CORE_TEMPLATES` and the CSS equals `card_css()` |
| K15  | the bold text is the assigned target, and the CSV holds plain text    |
| K16  | every target with a card has a sense row, or the unchecked count is within `--max-unchecked-sense` |
| K17  | no kept card holds a name or a loan word, and the manifest gives both counts |
| K18  | every tail card brings at least one new content lemma                 |
| K19  | every note with a `Core` value holds the target gloss of that word, and no other note holds a gloss |

### 10.2 Tests

Add `tests/test_core_deck.py`. Use a small fixture APKG and a small frequency
file. Do not download a file.

| Area        | Test                                                                           |
| ----------- | ------------------------------------------------------------------------------ |
| Core list   | one target per distinct form when the glosses share a sense                    |
| Core list   | two glosses without a shared sense word give two targets                       |
| Core list   | a function word with a low content share is not a tier F target                |
| Core list   | an absent target gets `absent` and does not fail K2                            |
| Quota       | a ubiquitous target gets quota 1; a content target gets 2                      |
| Cost        | a rare new word costs 3 times a frequent one; a repeat costs less              |
| Passes      | the seed pass covers each target once                                          |
| Passes      | the completion pass fills a short target and respects its cap                  |
| Passes      | a tail card brings a new lemma; `--tail-cards 0` keeps the core set            |
| Filter      | one failing card for each rule C1 to C5                                        |
| Window      | a card outside the window fails; `--core-max-words` widens it                  |
| Sense       | a hit without a sense word does not count; a compound span blocks a tier S hit |
| Sense       | a target with no sense row gets `unchecked` and fails K16                      |
| Order       | the assigned positions do not decrease                                         |
| Determinism | two runs give the same set and order                                           |
| Budget      | a small `--core-max-cards` stops with a clear message                          |
| APKG        | the kept notes, their media files, the templates, and the CSS                  |

## 11. Risks and open questions

### 11.1 Risks

| Risk                                       | Effect                            | Mitigation                         |
| ------------------------------------------ | --------------------------------- | ---------------------------------- |
| Greedy is not optimal                      | a few targets stay below quota    | the completion pass; an ILP later  |
| Function words fill the cards              | the deck looks alike              | the ubiquitous rule                |
| The quota needs more cards than the budget | a gap in the core                 | the check of 5.1; K2 and K3        |
| The window 5 to 9 removes 1,869 cards      | 9 Swadesh forms have no candidate | `--core-max-words 12`; the reserve |
| No sense file for a language               | K16 fails or many `unchecked`     | build the top 50 first             |
| A sense row is wrong                       | a good card loses its credit      | review the CSV; `absent_sense`     |
| The loan word list is incomplete           | a loan word stays in a card       | add a row when the review shows it |

### 11.2 Open questions

| #   | Question                                                     | Decide at |
| --- | ------------------------------------------------------------ | --------- |
| 1   | Which spoken list does each language use, and is its license public? | E2 |
| 2   | Should the order spread the second example, or keep it near the first? | E3 |
| 3   | Who writes the sense file, and how many words does it cover? | E2        |

## 12. Milestones

| Milestone | Issue      | Task                                                     | Output                    |
| --------- | ---------- | -------------------------------------------------------- | ------------------------- |
| E1        | `anki-eo6` | this spec and the prototype                               | `specs/core-word-small-deck.md` |
| E2        | `anki-v12` | core list, quota, filter C1 to C5, sense file, name and loan lists | `languages/<lang>/core.py` |
| E3        | `anki-p62` | value, cost, passes, cover key, sense-filtered hits        | selection report          |
| E4        | `anki-7z5` | order, writer, CSV files, manifest, gates K1 to K19        | `.apkg` and CSV files     |
| E5        | `anki-yop` | tests, README, docs entry                                 | release check             |

## 13. Appendix: evidence

### 13.1 Method

A prototype of sections 2 to 4, not in the repository, used:

- the APKG `apkg/6000_French_Sentences_w_Audio.apkg`: 5,928 cards;
- spaCy `fr_core_news_sm` with the parser and the entity model disabled;
- `data/french_frequency.tsv` at `--spoken-weight 0.5`;
- the French word table: 207 rows, 203 distinct lemmas after the first-token
  lemma step;
- the filter of the cull spec, the name list, and the content-share rule;
- a lemma match for tier S. The code uses the form match of 2.2, so a count
  can move by a few targets.

The prototype over-counts `absent`: some gaps are lemmatizer artifacts
(`éper`, `serper`). E2 measures `absent` again with the form match.

### 13.2 Cost sweep

Core of 314 targets, quota 2, wide window. Every other value is the default.

| `--lambda-new` | Cards | `covered1` | `covered2` | `outside` | `outside_rare` |
| -------------- | ----- | ---------- | ---------- | --------- | -------------- |
| 0.0            | 258   | 314        | 291        | 322       | -              |
| 0.05           | 292   | 314        | 300        | 235       | 20             |
| 0.1            | 300   | 314        | 300        | 227       | 19             |
| 0.2            | 302   | 314        | 300        | 224       | 19             |
| 0.5            | 302   | 314        | 300        | 224       | 19             |

At 0 the selection keeps 258 cards and 322 outside lemmas. The cost at 0.1
adds 42 cards and removes 95 outside lemmas. The values from 0.2 to 0.5 give
the same set, so 0.1 is inside the stable range.

### 13.3 Pass and tier sweeps

Core of 314 targets, `--lambda-new 0.1`, wide window.

| Passes                        | Cards | `covered2` | `outside` | `outside_rare` |
| ----------------------------- | ----- | ---------- | --------- | -------------- |
| seed and depth                | 217   | 212        | 146       | 13             |
| seed, depth, completion       | 300   | 300        | 227       | 19             |
| the above, third examples on 60 | 333 | 300        | 233       | 19             |

| `--support-top` / `--hard-top` | Cards | `covered2` | `outside` | `outside_rare` |
| ------------------------------ | ----- | ---------- | --------- | -------------- |
| 1,000,000 / 1,000,000 (flat)   | 286   | 298        | 228       | 32             |
| 1,000 / 3,000 (default)        | 300   | 300        | 227       | 19             |
| 300 / 1,000 (tight)            | 309   | 300        | 217       | 26             |

The default tier range gives the lowest rare count for the same `covered2`.
The completion pass adds 83 cards and lifts `covered2` from 212 to 300. The
third example adds 33 cards and does not change `covered2`.

### 13.4 Quota sweep

`--lambda-new 0.1`, wide window.

| `--per-word` | Tier F words | Targets | Cards | `covered2` | `outside` |
| ------------ | ------------ | ------- | ----- | ---------- | --------- |
| 3            | 0            | 164     | 292   | 150        | 266       |
| 3            | 40           | 204     | 327   | 190        | 243       |
| 3            | 100          | 264     | 401   | 250        | 242       |
| 2            | 80           | 244     | 240   | 230        | 227       |
| 2            | 150          | 314     | 300   | 300        | 227       |

Quota 3 costs about 1.8 cards per target, because a ubiquitous target keeps
quota 1. A 300-card deck with 3 examples needs a core of about 160 targets.

### 13.5 Bang for buck

`spoken_mass` is the share of Tatoeba French content tokens that the kept
content lemmas cover. The buck is one card. The rows use the wide window.

| Deck                 | Cards | Content lemmas | `spoken_mass` | Mass per card |
| -------------------- | ----- | -------------- | ------------- | ------------- |
| Core, quota 2, X=150 | 300   | 518            | 0.372         | 0.0012        |
| Core X=0, then a tail | 300  | 894            | 0.464         | 0.0015        |
| Breadth only         | 300   | 1,201          | 0.497         | 0.0017        |
| Breadth only         | 1,000 | 2,239          | 0.555         | 0.0006        |

- A breadth deck wins on mass per card: 33% more mass and 2.3 times the
  lemmas at 300 cards. It gives no second example.
- The core deck wins on mass per lemma: 7.2e-4 against 4.1e-4.
- The mass saturates. The tail from 300 to 400 cards adds 0.033 mass, and a
  1,000-card deck adds 0.058 mass over a 300-card breadth deck. The knee is
  near 300 to 400 cards.

## 14. Implementation result

The code is built, all 18 gates pass, and the work is committed. Milestones E2
to E5 are closed. Section 14.5 lists what remains.

### 14.1 Code

| Item                                        | State                                                              |
| ------------------------------------------- | ------------------------------------------------------------------ |
| `anki_tools/ranking/core.py`                | value, cost, passes, assignment, order                             |
| `anki_tools/languages/french/core.py`       | targets, filter C1 to C5, sense, driver, CSV files, manifest, bold |
| `anki_tools/languages/french/core_gates.py` | gates K1 to K19                                                    |
| `anki_tools/cli.py`                         | `anki fix fr --core` and its options                               |
| `anki_tools/apkg/package.py`                | `add_model_field` and `remove_model_template` for `anki2` models   |
| `vocab/french_core_senses.tsv`              | 51 rows: the top 50 tier F words and `ne`                          |
| `vocab/french_loanwords.txt`                | 28 words                                                           |
| `tests/test_core_deck.py`                   | 31 tests                                                           |
| `apkg/french_core_300.apkg`                 | 300 cards, all gates pass                                          |

### 14.2 Gates

`anki fix fr --core --check` reads the output files and runs K1 to K19.

| Gate | Value                                                                             |
| ---- | --------------------------------------------------------------------------------- |
| K1   | 300 cards, equal to the card cap of the request; the CSV, the APKG, and the manifest agree |
| K2   | 204 of 204 coverable targets hold a card; 5 targets are `absent_sense`            |
| K3   | report only: 0.897 of the coverable targets hold 2 cards                            |
| K4   | 247 outside words for 209 targets, 1.18 times; 18 are rare                        |
| K5   | every kept card passes F1 to F9 and C1 to C5                                      |
| K6   | no kept pair reaches the lemma Jaccard limit                                      |
| K7   | every kept card has a valid MP3 file                                              |
| K8   | the note, media, and manifest counts agree                                        |
| K9   | the deck name and the note model are the requested values                         |
| K10  | the kept order matches the recorded hash of the rank CSV                          |
| K11  | the coverage CSV agrees with the card CSV                                         |
| K12  | the manifest hashes match the two CSV files                                       |
| K13  | each target's cards hold its form or lemma                                        |
| K14  | the templates equal `CORE_TEMPLATES` and the CSS equals the cull CSS              |
| K15  | 182 cards hold the bold assignment; the CSV holds plain text                      |
| K16  | 0 unchecked targets                                                               |
| K17  | 0 names and 0 loan words in the kept cards                                        |
| K18  | every tail card brings a new content lemma                                        |
| K19  | every note with a `Core` value holds the target gloss of that word                |

### 14.3 Built deck

`uv run anki fix fr "apkg/6000_French_Sentences_w_Audio.apkg" --core` gives:

| Measure                         | Value                                                             |
| ------------------------------- | ----------------------------------------------------------------- |
| Cards                           | 300                                                               |
| Targets                         | 209, of which 159 tier S and 50 tier F                            |
| Targets with 2 cards            | 183                                                               |
| Targets with 1 card             | 21                                                                |
| `absent_sense` targets          | 5                                                                 |
| Support words in the core cards | 247, of which 18 rare                                             |
| New words from the tail         | 346                                                               |
| Bold cards                      | 182, plain 118                                                    |
| Cards with a core word          | 273                                                               |
| Cards by pass                   | seed 99, depth 25, completion 82, tail 94                         |
| Filter reasons                  | F1 163, F2 1,869, F5 22, F7 22, F8 410, C1 39, C2 8, C3 25, C4 32 |
| Other reasons                   | similar 6, card cap 3,032                                         |

### 14.4 Notes on C3 and C4

C3 rejects 25 cards and C4 rejects 32 cards. The two guards remove 57
candidates before the selection. Both lists are short on purpose:
`vocab/french_names.txt` holds 200 names and `vocab/french_loanwords.txt`
holds 28 words. Each test is a whole word test, so `tom` does not reject
`tomber` and `stop` does not reject `stopover`. A review that finds a new name
or loan word adds a row to the list.

### 14.5 Deferred

- Custom card hints, and a name list option.
- A spoken list per language, the spread of the second example, and the sense
  file owner (the three open questions of 11.2).
- The cull adoption of C1, C2, C3, and the sense guard (`anki-0h8`).
- The reserve pack: off by default, as decided.
- The `bold` column of the card CSV stays 0 (`anki-4bh`). Only the manifest
  counts the bold cards.
