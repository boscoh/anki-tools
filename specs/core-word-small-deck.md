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
2. [Goals and scope](#2-goals-and-scope)
3. [Core target set](#3-core-target-set)
4. [Algorithm](#4-algorithm)
5. [Deck size](#5-deck-size)
6. [Card order](#6-card-order)
7. [Pipeline](#7-pipeline)
8. [Swadesh reserve](#8-swadesh-reserve)
9. [Interface](#9-interface)
10. [Quality gates and tests](#10-quality-gates-and-tests)
11. [Risks and open questions](#11-risks-and-open-questions)
12. [Milestones](#12-milestones)
13. [Appendix: prototype evidence](#13-appendix-prototype-evidence)

## 1. Summary

`anki fix <lang>` ranks a whole deck today. `anki fix fr --select` culls a
deck to 1,000 cards (see `specs/french-1000-sentence-cull.md`). Both modes
keep a large deck.

This spec adds a third mode, `--core`. It builds a small deck from a fixed
core word list. Each core word gets 2 to 3 example sentences. A sentence that
brings many words outside the core costs more, so the deck adds few other
words.

### 1.1 The four changes

| Change          | What it does                                                      | Section |
| --------------- | ----------------------------------------------------------------- | ------- |
| Core target set | Swadesh forms, then X spoken-frequency words                      | 3       |
| Example quota   | Every target gets 1 to 3 examples; a function word gets 1         | 3.3     |
| Outside cost    | Each new word outside the core costs; a rare word costs more      | 4.3     |
| Core order      | The card order follows the core list, so the deck is a curriculum | 6       |

The mode uses the same filter, frequency list, tokenizer, similarity test, and
APKG writer as the cull. The new code is the target list, the cost, and the
three selection passes.

### 1.2 Result at a glance

The numbers come from a prototype of sections 3 and 4 on the French 6,000-card
deck. The settings are the defaults of section 9.2.

| Measure                        | 1,000-card cull | This spec |
| ------------------------------ | --------------- | --------- |
| Cards                          | 1,000           | 300       |
| Core targets                   | 300             | 314       |
| Targets with 1 example         | 297             | 314       |
| Targets with 2 examples        | 287             | 300       |
| Swadesh forms covered          | 206 / 207       | 164 / 203 |
| Distinct content lemmas in all | 1,981           | 541       |
| Words outside the core         | none measured   | 227       |
| Words outside and rare         | none measured   | 19        |

The core deck is one third of the cards. It gives 2 examples to a larger share
of its targets, and it holds 541 content lemmas in total: 314 core targets and
227 support words. The support words are the small set that the sentences need
to stay natural.

`--core` cannot cover a Swadesh form that the input deck does not hold. The
French input holds 164 of the 203 distinct forms. Section 8 gives the reserve
option for the other 39 forms.

### 1.3 Key decisions

| Decision            | Choice                                                        | Evidence |
| ------------------- | ------------------------------------------------------------- | -------- |
| Target order        | Swadesh index first, then spoken frequency rank               | 3.1      |
| Target unit         | A form for Swadesh, a content lemma for frequency             | 3.2      |
| Target count        | 150 frequency words after the Swadesh set                     | 5.2      |
| Examples per target | 2 for a content target, 1 for a target in more than 25% of the candidates | 3.3 |
| Selection           | Greedy cover of the value, minus a cost for outside words      | 4        |
| Passes              | Seed, depth, completion, and optional third example           | 4.4      |
| Outside cost        | A new rare word costs 3 times a new frequent word             | 4.3      |
| Order               | The order of the first target that a card covers              | 6        |
| Output size         | About 1.0 card per target at quota 2                          | 5.1      |

## 2. Goals and scope

### 2.1 Goals

1. Build a deck of 250 to 400 cards from a core list of 250 to 400 words.
2. Cover every core word that the input deck holds.
3. Give 2 examples of every core word, and 3 examples when the caller asks.
4. Keep the words outside the core near 230 lemmas, and keep rare outside
   words near 20.
5. Order the cards as a curriculum: the easiest core words come first, and
   each card adds few new words.
6. Reuse the input model, fields, audio, and the `reorder_deck` code path.

### 2.2 Measures

| Measure            | Meaning                                                              |
| ------------------ | -------------------------------------------------------------------- |
| `cards`            | Sentences kept                                                       |
| `targets`          | Core words that the input deck holds                                 |
| `covered1`         | Targets with at least 1 kept card                                    |
| `covered2`         | Targets with at least 2 kept cards                                   |
| `full`             | Targets that reach their quota                                       |
| `outside`          | Distinct content lemmas outside the core in the kept cards           |
| `outside_rare`     | Outside lemmas with a blended rank above `--hard-top` (3000)         |
| `outside_per_card` | `outside / cards`                                                    |
| `absent`           | Core words that no candidate sentence holds                          |
| `hardest_rank`     | Rank of the least frequent content lemma of a card                   |

### 2.3 In scope

- One `--core` mode of `anki fix <lang>` for a sentence deck with audio.
- A core list from the reviewed Swadesh word table and the cached frequency
  list of the language.
- The cost-aware selection, the card order, a report, and gates.

### 2.4 Out of scope

- New audio or TTS.
- A new note model or a new template. The mode changes the card order only.
- A curated grammar syllabus, and a new corpus download.
- Changes to the 1,000-card cull and to the Swadesh sentence packs.

## 3. Core target set

### 3.1 Two tiers, one order

| Tier | Name      | Source                                                   | Order               |
| ---- | --------- | -------------------------------------------------------- | ------------------- |
| S    | Swadesh   | the reviewed word table of the language (`S1`)            | `swadesh_index`     |
| F    | Frequency | the cached blended frequency list, after the Swadesh forms | blended rank        |

The core list is one list. Tier S comes first. Tier F follows tier S in rank
order. Each target has a 1-based `position` in this list.

A shared Swadesh form gives one target. The reviewed word table can give two
concepts the same form, for example index 1 and a later index. The target
keeps the lowest index and the set of glosses. The list has one entry, not two.
The prototype found 207 rows and 203 distinct forms.

### 3.2 Match rule

| Tier | Target unit | A card holds the target when                             |
| ---- | ----------- | -------------------------------------------------------- |
| S    | a form      | the form occurs in the sentence, with the pack match rule |
| F    | a lemma     | the card token list holds the lemma with a content tag     |

Tier S uses the same form match as the Swadesh sentence pack:
`form_occurrences` and `form_offsets` of `anki_tools/sentence_pack/guards.py`,
with the `FALSE_FRIENDS` spans of the language. Thus 闻 inside 新闻 is not a
hit, and 食 counts inside 食飯.

Tier F uses the tokenizer of the language and the content tags `NOUN`, `VERB`,
`ADJ`, and `ADV`.

### 3.3 Quota

```text
quota(w) = 1                       if share(w) > --ubiquitous-share
quota(w) = --per-word              otherwise
quota(w) = 3                       for the first --third-examples targets
```

- `share(w)` is the number of candidate sentences that hold the target,
  divided by the number of candidates.
- `--per-word` is 2 by default, and the accepted range is 1 to 3.
- `--ubiquitous-share` is 0.25 by default.
- `--third-examples` is 0 by default. It raises the quota of the highest
  targets that are not ubiquitous.

Reason for the ubiquitous rule: pronouns, negation, and auxiliaries are
Swadesh targets. A form such as `je`, `ne`, or `pas` is in many candidate
sentences. A second example costs a card and adds no new skill. With quota 1
for these targets, the cards go to the content words.

### 3.4 Target filters

A tier F candidate is a target when all these rules hold.

| Rule        | Value                                                    |
| ----------- | -------------------------------------------------------- |
| Part of speech | `NOUN`, `VERB`, `ADJ`, or `ADV`                       |
| Content share  | at least 0.5 of all occurrences of the lemma          |
| Deck count     | at least `--min-deck-count` (3)                       |
| Lemma length   | 2 characters or more                                  |
| Name list      | the lemma is not in `vocab/<language>_names.txt`      |
| Swadesh overlap | the lemma is not the lemma of a tier S form          |

The rules match `build_targets` of `anki_tools/languages/french/cull.py`.

A target that no candidate sentence holds stays in the core list and gets the
status `absent` in the coverage CSV. The gates count `targets`, not
`targets + absent`.

## 4. Algorithm

### 4.1 State

| Name         | Meaning                                                        |
| ------------ | -------------------------------------------------------------- |
| `uses[w]`    | Kept cards that hold the target `w`, at most `quota[w]`         |
| `introduced` | Content lemmas outside the core that a kept card holds          |
| `kept`       | Card indices in pick order                                      |

### 4.2 Value

```text
weight(w) = 1 / log2(position(w) + 1)
demand(u) = 1.0, 0.35, 0.12   for example 1, 2, 3
value(s)  = sum of weight(w) * demand(uses[w])
            over the targets w of s with uses[w] < quota[w]
```

- The value falls with the position. A Swadesh word before a frequency word,
  and a frequent frequency word before a rare one.
- The value falls with the example number. The first example of a word is
  worth more than the third.
- The value of a card does not count a target that already has its quota.

### 4.3 Cost

```text
outside(s) = the content lemmas of s that are not core targets
new(s)     = outside(s) minus introduced
hold(s)    = outside(s) intersect introduced
mult(w)    = 1.0 if rank(w) <= --support-top    (1000)
             1.8 if rank(w) <= --hard-top       (3000)
             3.0 otherwise
cost(s)    = --lambda-new * sum(mult(w) for w in new(s))
             + --lambda-repeat * len(hold(s))
             + --lambda-len * max(0, units(s) - 8)
```

- A new outside word costs. A repeated outside word costs little.
- A rare new outside word costs 3 times a frequent one. A frequent support
  word is a useful word. A rare support word is noise.
- `--lambda-new` is 0.1 by default. Section 13.2 shows the effect of other
  values.

The cost makes the selection prefer a sentence whose words are core words or
words that an earlier card already introduced. This is the rule that keeps the
outside set small.

### 4.4 The passes

```text
seed:        stop[w] = 1 for every w, demand = 1.0, lambda = --lambda-seed
             accept a card only when value - cost > 0
             goal: 1 example of every target
depth:       stop[w] = quota[w], demand as 4.2, lambda = --lambda-new
             accept a card only when value - cost > 0
             goal: the full quota, without a noisy word
completion:  stop[w] = quota[w], demand as 4.2, lambda = --lambda-new
             accept the best card even when value - cost <= 0
             goal: fill a target that the depth pass leaves short
top-up:      stop[w] = 3 for the first --third-examples targets
             goal: a third example for the top of the core list
```

- The seed pass cannot end with a gap, unless the pool has no sentence for the
  target.
- The depth pass is the quality pass. It pays for outside words.
- The completion pass protects the quota. It is the only pass that accepts a
  card with a negative score. It stops at `--completion-cards` (120) and at
  `--max-cards`.
- `--lambda-seed` is 0.01. A small cost in the seed pass is a tie-break between
  cheap and expensive candidates.

### 4.5 Loop and stop rules

```text
for each pass:
    repeat:
        pick the unkept card with the largest value(card) - cost(card)
        over the candidates that
            have value(card) > 0, and
            pass the score rule of the pass, and
            are not a duplicate of a kept card
        if no card: stop the pass
        keep the card; update uses and introduced
```

The selection stops a pass when one of these rules holds:

| Rule           | Condition                                               |
| -------------- | ------------------------------------------------------- |
| no card        | every candidate has a value of 0, or fails the score rule |
| card cap       | `len(kept) == --max-cards`                              |
| outside budget | a new word would put `len(introduced)` above `--outside-budget` |

`--outside-budget` is 0 (no cap) by default. The cost keeps the outside set
small, and the budget is a backstop for a run with a small `--lambda-new`.

### 4.6 Reuse of the greedy cover

`anki_tools/ranking/cover.py` gives the loop, the heap, the stale-entry test,
the limit, and the reason labels.

| Cover parameter | Value in this mode                                    |
| --------------- | ----------------------------------------------------- |
| `gain`          | `value(card) - cost(card)`                            |
| `key`           | `(-score, new_words, units, index)`                   |
| `reject`        | `"similar"` when the card is a duplicate              |
| `precheck`      | `"no_slot"` when no target of the card needs an example |
| `limit`         | `--max-cards - len(kept)`                             |
| `on_pick`       | update `uses` and `introduced`                        |

The stale-entry test compares the stored score with the new score. The cost
enters the score, so a change of `introduced` invalidates the entry. No change
to `cover.py` is necessary beyond the heap key order.

### 4.7 Duplicate test

Two cards are duplicates when the Jaccard value of their content-lemma sets
reaches `--similarity` (0.67). Reuse `SimilarityIndex` of
`anki_tools/sentence_pack/select.py` and the bucket rule: a new card compares
only with the kept cards that share a lemma.

### 4.8 Determinism

- The scan order is the input order.
- The tie-break is the score, then fewer new words, then fewer units, then the
  input index.
- Two runs with the same input, options, and frequency file give the same
  kept set and the same order.

## 5. Deck size

### 5.1 Measured size

The prototype measured these sets. Every row uses the defaults of section 9.2
and quota 2, except the last three rows.

| Tier F words | Core targets | Cards | `covered1` | `covered2` | `outside` | `outside_rare` |
| ------------ | ------------ | ----- | ---------- | ---------- | --------- | -------------- |
| 80           | 244          | 240   | 244        | 230        | 227       | 17             |
| 150          | 314          | 300   | 314        | 300        | 227       | 19             |
| 250          | 414          | 389   | 414        | 399        | 253       | 24             |
| 400          | 564          | 523   | 564        | 548        | 253       | 31             |

The card count is close to the target count:

```text
cards ~ 0.95 * targets            at quota 2
cards ~ 1.5 to 1.8 * targets      at quota 3

At quota 3 the function targets keep quota 1, so the factor is below 2. The
measured rows at quota 3 are 292 cards for 164 targets, 327 for 204, and 401
for 264.

A kept card holds about 2 target hits. Thus the slot count is
`sum(quota) / 2`, and the card count follows from the quota.

### 5.2 Target count for a card budget

| Card budget | Tier F words | Core targets | `covered2` |
| ----------- | ------------ | ------------ | ---------- |
| 240         | 80           | 244          | 230        |
| 300         | 150          | 314          | 300        |
| 390         | 250          | 414          | 399        |
| 520         | 400          | 564          | 548        |

The default is `--core-words 150`. It gives 300 cards on the French input,
which is the smallest set that covers the Swadesh forms of the deck with 2
examples.

The command checks the request before it starts:

```text
if quota_total > 2.1 * max_cards:
    stop with "the card budget is too small for this quota"
```

A budget of 200 cards cannot give 2 examples of 314 targets. The check turns a
silent gap into a clear message.

### 5.3 The outside set

| Core targets | `outside` | `outside_rare` | `outside / card` |
| ------------ | --------- | -------------- | ---------------- |
| 244          | 227       | 17             | 0.95             |
| 314          | 227       | 19             | 0.76             |
| 414          | 253       | 24             | 0.65             |
| 564          | 253       | 31             | 0.48             |

The outside set is almost constant at about 230 lemmas. A larger core absorbs
the frequent support words, so the total vocabulary grows by the core size and
not by twice the core size.

## 6. Card order

The mode writes the order into the ranking CSV and lets `reorder_deck` set the
`due` values. The Anki code path does not change.

`assign_target` gives each kept card one target: the target with the lowest
position among the targets of the card that no earlier card assigned.

The cards are sorted by these keys:

| Order | Key                                                           |
| ----- | ------------------------------------------------------------- |
| 1     | the position of the assigned target                            |
| 2     | the example number of the assigned target in this card (1, 2, 3) |
| 3     | the count of new outside words of the card                     |
| 4     | the unit count of the card                                     |
| 5     | the input index                                                |

Thus the deck teaches the core list in order. A card comes after the card that
teaches its target. The second example of a word comes near the first.

The mode does not use the `hardest_rank` order of the cull. The core position
is the curriculum key. `hardest_rank` stays in the CSV as a report value.

## 7. Pipeline

| Stage | Name       | Input                          | Output                                    |
| ----- | ---------- | ------------------------------ | ----------------------------------------- |
| K0    | Extract    | input APKG                     | cards, English, audio                     |
| K1    | Tokenize   | K0, the tokenizer of the language | lemma and tag per card                 |
| K2    | Frequency  | the cached frequency list, the word table | core list, quota, `absent` list |
| K3    | Filter     | K1                             | passed cards and reasons F1 to F9         |
| K4    | Select     | K1, K2, K3                     | kept cards, uses, reasons                 |
| K5    | Order      | K4                             | kept cards in core order                  |
| K6    | Write      | K0, K5                         | `.apkg`, `.core.csv`, `.coverage.csv`, manifest |

K1, K3, and K6 are the existing functions:

| Stage | Existing function                                    |
| ----- | ---------------------------------------------------- |
| K0    | `cull.load_input_deck`                               |
| K1    | `cull.tokenize_deck`                                 |
| K2    | `cull.load_frequency`, `cull.frequency_ranks`, `profile.load_word_table` |
| K3    | `cull.filter_cards`                                  |
| K4    | this spec, over `ranking.cover.greedy_cover`         |
| K5    | this spec                                            |
| K6    | `cull.write_deck` and `reorder.reorder_deck`         |

`anki fix fr --core` runs K0 to K6. A second run with the same input gives the
same output. The audio cache is read only.

## 8. Swadesh reserve

A core word with no source sentence in the input deck is `absent`. The French
input deck holds 164 of the 203 distinct Swadesh forms. The 39 gaps are mostly
rare nouns such as `pou`, `bâton`, and `graine`.

The mode can add the Swadesh sentence pack of section 16 of the cull spec:

| Rule                | Value                                                  |
| ------------------- | ------------------------------------------------------ |
| Option              | `--reserve-pack apkg/swadesh_french_sentences.apkg`    |
| Default             | off, because the pack has about 198 cards               |
| Duplicate check     | the same test as the cull, section 16.3                 |
| Quota               | one reserve card counts as one example of its concept   |
| Cap                 | every reserve card is kept                              |
| Card budget         | the frequency pass fills `--max-cards - reserve cards`  |
| Outside cost        | a reserve card pays no cost                             |

With the reserve on and a budget of 300 cards, the deck holds about 198 reserve
cards and 100 frequency cards. That fails the 2-example goal for the frequency
words. Use the reserve with `--max-cards 500`, or use it alone as a Swadesh
deck.

## 9. Interface

### 9.1 Command

```bash
uv run anki fix fr "apkg/6000_French_Sentences_w_Audio.apkg" --core \
    --core-words 150 \
    --per-word 2 \
    --max-cards 340 \
    --output apkg/french_core_300.apkg
```

`--core` and `--select` cannot be used together. `--core` is not the default;
rank mode stays the default.

### 9.2 Options

| Option              | Default      | Meaning                                              |
| ------------------- | ------------ | ---------------------------------------------------- |
| `--core`            | off          | Run the core-word selection                          |
| `--core-words`      | 150          | Tier F words after the Swadesh set                   |
| `--per-word`        | 2            | Examples per content target, 1 to 3                  |
| `--third-examples`  | 0            | Targets that get a third example                     |
| `--ubiquitous-share`| 0.25         | Share above which a target gets quota 1              |
| `--max-cards`       | 400          | Card cap                                             |
| `--completion-cards`| 120          | Card cap of the completion pass                      |
| `--outside-budget`  | 0            | Cap on the outside set; 0 is no cap                  |
| `--support-top`     | 1000         | Outside rank that costs 1.0                          |
| `--hard-top`        | 3000         | Outside rank that costs 1.8, above is 3.0            |
| `--lambda-new`      | 0.1          | Cost of one new outside word                         |
| `--lambda-repeat`   | 0.03         | Cost of one repeated outside word                    |
| `--lambda-len`      | 0.02         | Cost per unit above 8                                |
| `--lambda-seed`     | 0.01         | Cost factor of the seed pass                         |
| `--min-deck-count`  | 3            | Minimum deck count of a tier F target                |
| `--spoken-weight`   | 0.5          | Weight of the Tatoeba term in the blend              |
| `--min-words`       | 3            | Minimum word count of a card                         |
| `--max-words`       | 12           | Maximum word count of a card                         |
| `--similarity`      | 0.67         | Lemma Jaccard limit                                  |
| `--reserve-pack`    | none         | Swadesh pack to add, or none                         |
| `--name`            | from the language | Output deck name                                |
| `--output`          | from the input name | Output APKG                                  |
| `--check`           | off          | Run gates K1 to K15 on the output                    |

### 9.3 Output files

| File                                  | Content                                                     |
| ------------------------------------- | ----------------------------------------------------------- |
| `apkg/french_core_300.apkg`           | The kept cards in core order, with the input model and audio |
| `apkg/french_core_300.core.csv`       | One row per input card: `kept`, `pass`, `reason`, `targets`, `assigned`, `value`, `cost` |
| `apkg/french_core_300.coverage.csv`   | One row per target: `word`, `tier`, `position`, `quota`, `count`, `status` |
| `apkg/french_core_300.manifest.json`  | Counts, options, frequency source, and hashes               |
| `apkg/french_core_300.rank.csv`       | The ranking CSV that `reorder_deck` reads                    |

`status` is `covered`, `partial` (below the quota), or `absent`.

### 9.4 Code changes

- Add `anki_tools/languages/<language>/core.py` with K2, K4, and K5. Keep the
  generic gain, cost, and pass code in `anki_tools/ranking/core.py`, so two
  languages share it.
- Add the `--core` option and the option block to `anki fix fr` in `cli.py`.
- Extend `ranking/cover.py` with the `key` order `(-score, new, units, index)`.
- Reuse `ANKI` write and `reorder_deck` without a change.
- Reuse the gates file layout of `cull.py`.

## 10. Quality gates and tests

### 10.1 Gates

`anki fix fr --core --check` runs the gates on the output. A failure stops the
command with a non-zero exit code.

| Gate | Check                                                                |
| ---- | -------------------------------------------------------------------- |
| K1   | The card count is at most `--max-cards`, and the count matches the CSV and the manifest |
| K2   | Every present target has at least 1 kept card (`covered1 == targets`) |
| K3   | At least 0.95 of the targets reach 2 examples                         |
| K4   | The outside set is at most 0.9 times the target count, and the rare outside words are at most 0.1 of the outside set |
| K5   | Every kept card passes the filter rules F1 to F9                      |
| K6   | No kept pair reaches the lemma Jaccard limit                          |
| K7   | Every kept card has its input audio, and the file is a valid MP3      |
| K8   | The APKG note count equals the kept count and the manifest count      |
| K9   | The APKG holds exactly the media files of the kept cards              |
| K10  | The deck name and the note model are the requested values             |
| K11  | A second run gives the same kept set and the same order               |
| K12  | The coverage CSV has one row per target, and `count` matches the cards |
| K13  | The manifest hashes match the core CSV and the coverage CSV           |
| K14  | Every target with a kept card holds the form or lemma in that card    |
| K15  | The templates and the CSS are the input values                        |

### 10.2 Tests

Add `tests/test_core_deck.py`. Use a small fixture APKG and a small frequency
file. Do not download a file.

| Area         | Test                                                              |
| ------------ | ----------------------------------------------------------------- |
| Core list    | One target per distinct Swadesh form; two shared forms merge      |
| Core list    | A function word with a low content share is not a tier F target    |
| Core list    | A tier F lemma equal to a Swadesh lemma is not added               |
| Core list    | An absent target gets the status `absent` and does not fail K2     |
| Quota        | A ubiquitous target gets quota 1; a content target gets 2          |
| Value        | The first example is worth more than the second                    |
| Cost         | A new rare word costs 3 times a new frequent word                  |
| Cost         | A repeated outside word costs less than a new one                  |
| Seed pass    | Every present target has one card after the seed pass              |
| Completion   | The completion pass fills a short target and respects its cap      |
| Order        | The assigned positions do not decrease                             |
| Determinism  | Two runs give the same card set and order                          |
| Budget check | A small `--max-cards` stops with a clear message                   |
| APKG         | The output holds the kept notes and only their media files         |

## 11. Risks and open questions

### 11.1 Risks

| Risk                                              | Effect                                    | Mitigation                                    |
| ------------------------------------------------- | ----------------------------------------- | --------------------------------------------- |
| Greedy is not optimal                             | A few targets stay below quota            | The completion pass; an ILP is a later option |
| Function words fill the cards                     | The deck looks alike                      | The ubiquitous rule; `--ubiquitous-share`     |
| The quota needs more cards than the budget        | A gap in the core                          | The size check of section 5.2; K2 and K3      |
| The outside set is 227 lemmas at 300 cards        | The deck is not pure core                 | Raise `--core-words`, lower `--per-word`, raise `--lambda-new` |
| No spoken list for some languages                 | Tier F is a written list                  | Fall back to the blend of section 4 of the cull spec; `--spoken-weight` |
| The input deck lacks a Swadesh form               | `absent` target                           | `--reserve-pack`; K2 excludes `absent`        |
| The prototype matched lemmas, the code matches forms | A count can move by a few targets      | Re-measure at E2; the gates use the output    |
| Lemmatizer artifacts (`taire`, `fait`, `éper`)    | A bad target                              | The word review file; the `absent` status     |

### 11.2 Open questions

| #   | Question                                                              | Decide at |
| --- | --------------------------------------------------------------------- | --------- |
| 1   | Is quota 2 right for the Swadesh forms, or should a function form always get 1? | E2 |
| 2   | Which spoken list does each language use, and is its license public?   | E2        |
| 3   | Does the output bold the target token, as the Swadesh pack does?        | E4        |
| 4   | Does the output need a `Target` field to show the assigned word?        | E4        |
| 5   | Does the order interleave tier S and tier F, or follow the core list?   | E3        |
| 6   | Is a second example of the same word in the same week useful, or should the order spread it? | E3 |

## 12. Milestones

| Milestone | Issue      | Task                                                     | Output                    |
| --------- | ---------- | -------------------------------------------------------- | ------------------------- |
| E1        | `anki-eo6` | This spec and the prototype analysis                      | `specs/core-word-small-deck.md` |
| E2        | `anki-v12` | Core list, quota, `absent` list, size check                | `languages/<lang>/core.py` |
| E3        | `anki-p62` | Value, cost, three passes, cover key                       | selection report          |
| E4        | `anki-7z5` | Core order, writer, CSV files, manifest, gates K1 to K15   | `.apkg` and CSV files     |
| E5        | `anki-yop` | Tests, README, docs entry                                  | release check             |

## 13. Appendix: prototype evidence

### 13.1 How the numbers were made

A prototype of sections 3 and 4, not in the repository, used:

- The APKG `apkg/6000_French_Sentences_w_Audio.apkg`: 5,928 cards, 4,958 pass
  the filter of section 7 of the cull spec.
- spaCy `fr_core_news_sm` with the parser and the named-entity model disabled.
- `data/french_frequency.tsv` at `--spoken-weight 0.5`.
- The French word table `vocab/english-french.csv`: 207 rows, 203 distinct
  lemmas after the first-token lemma step.
- The filter of the cull spec, the name list, and the content-share rule.
- The value and cost of sections 4.2 and 4.3, with the seed, depth, and
  completion passes.
- A lemma match for tier S. The implementation uses the form match of section
  3.2, so a count can move by a few targets.

The prototype over-counts `absent`: some gaps are lemmatizer artifacts of the
prototype (`éper`, `serper`) and not absent forms. The implementation must
measure `absent` again with the form match.

### 13.2 Cost sweep

Core of 314 targets, quota 2. Every other value is the default.

| `--lambda-new` | Cards | `covered1` | `covered2` | `outside` | `outside_rare` |
| -------------- | ----- | ---------- | ---------- | --------- | -------------- |
| 0.0            | 258   | 314        | 291        | 322       | -              |
| 0.05           | 292   | 314        | 300        | 235       | 20             |
| 0.1            | 300   | 314        | 300        | 227       | 19             |
| 0.2            | 302   | 314        | 300        | 224       | 19             |
| 0.3            | 302   | 314        | 300        | 224       | 19             |
| 0.5            | 302   | 314        | 300        | 224       | 19             |

At `--lambda-new 0` the selection keeps 258 cards and 322 outside lemmas. The
cost at 0.1 adds 42 cards and removes 95 outside lemmas. The values from 0.2 to
0.5 give the same set, so 0.1 is inside the stable range and is the default.

The `outside_rare` value is the count of outside lemmas with a blended rank
above 3000. The 0.0 row is not measured with the tier multiplier.

### 13.3 Pass sweep

Core of 314 targets, quota 2, `--lambda-new 0.1`.

| Passes                                        | Cards | `covered2` | `outside` | `outside_rare` |
| --------------------------------------------- | ----- | ---------- | --------- | -------------- |
| Seed and depth                                | 217   | 212        | 146       | 13             |
| Seed, depth, completion                       | 300   | 300        | 227       | 19             |
| Seed, depth, completion, third examples on 60 | 333   | 300        | 233       | 19             |

The completion pass is the cost of the quota. It adds 83 cards, 81 outside
lemmas, and 6 rare lemmas, and it lifts `covered2` from 212 to 300.

### 13.4 Tier multiplier sweep

Core of 314 targets, `--lambda-new 0.1`.

| `--support-top` / `--hard-top` | Cards | `covered2` | `outside` | `outside_rare` |
| ------------------------------ | ----- | ---------- | --------- | -------------- |
| 1,000,000 / 1,000,000 (flat)   | 286   | 298        | 228       | 32             |
| 1,000 / 3,000 (default)        | 300   | 300        | 227       | 19             |
| 300 / 1,000 (tight)            | 309   | 300        | 217       | 26             |

The default range gives the lowest rare count for the same `covered2`. The
tight range trades 14 cards for 10 outside lemmas.

### 13.5 Quota sweep

`--lambda-new 0.1`.

| `--per-word` | Tier F words | Targets | Cards | `covered2` | `outside` |
| ------------ | ------------ | ------- | ----- | ---------- | --------- |
| 3            | 0            | 164     | 292   | 150        | 266       |
| 3            | 40           | 204     | 327   | 190        | 243       |
| 3            | 100          | 264     | 401   | 250        | 242       |
| 2            | 80           | 244     | 240   | 230        | 227       |
| 2            | 150          | 314     | 300   | 300        | 227       |

At quota 3 the Swadesh set alone needs 292 cards. Quota 3 costs about 1.8
cards per target, because the ubiquitous targets keep quota 1. A caller who
wants a deck of 300 cards with 3 examples must cut the core to about 160
targets.

### 13.6 First cards of the default order

The prototype order follows section 6. The first cards hold the first Swadesh
targets, and the support words arrive one or two at a time.

| # | Reason to place the card                     |
| - | -------------------------------------------- |
| 1 | `je` (Swadesh index 1), few new outside words |
| 2 | `tu` (index 2)                               |
| 3 | `il` (index 3)                               |
| 4 | `nous` (index 4) or the first free target     |
| 5 | the first frequency word that fits            |

The prototype did not store the exact sentences. E4 records the list in the
manifest for the review.
