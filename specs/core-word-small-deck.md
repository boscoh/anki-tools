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

### 1.1 The five changes

| Change          | What it does                                                      | Section |
| --------------- | ----------------------------------------------------------------- | ------- |
| Core target set | Swadesh forms, then X spoken-frequency words                      | 3       |
| Example quota   | Every target gets 1 to 3 examples; a function word gets 1         | 3.3     |
| Outside cost    | Each new word outside the core costs; a rare word costs more      | 4.3     |
| Sentence filter | Simple, complete, no name, no loan word, and the right sense      | 7.1, 7.2 |
| Core order      | The card order follows the core list, so the deck is a curriculum | 6       |

The mode uses the same filter, frequency list, tokenizer, similarity test, and
APKG writer as the cull. The new code is the target list, the cost, and the
three selection passes.

### 1.2 Result at a glance

The numbers come from a prototype of sections 3, 4, and 7 on the French
6,000-card deck. The settings are the recommended preset of section 5.5:
`--core-words 50 --per-word 2 --max-cards 300 --tail-cards 120`.

| Measure                        | 1,000-card cull | This spec |
| ------------------------------ | --------------- | --------- |
| Cards                          | 1,000           | 300       |
| Core targets                   | 300             | 205       |
| Targets with 1 example         | 297             | 205       |
| Targets with 2 examples        | 287             | 192       |
| Swadesh forms covered          | 206 / 207       | 155 / 203 |
| Distinct content lemmas in all | 1,981           | 740       |
| Words outside the core, core cards | none measured | 215     |
| Words from the tail            | none measured   | 320       |

The core part is one third of the cards of the cull. It gives 2 examples to
192 of its 205 targets, and it holds 215 support words. The tail adds 320 new
lemmas, so the deck has 740 content lemmas in total.

This deck is not the best choice for the most words per card. Section 5.4
compares it with a breadth deck and with a core deck plus a breadth tail.
Section 5.5 gives the measured presets with the filter of section 7.

`--core` cannot cover a Swadesh form that the input deck does not hold. With
the window of 7.1.1, the French input holds 155 of the 203 distinct forms.
Section 8 gives the reserve option for the other 48 forms.

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
6. Keep every card simple and complete: inside the length window, with a
   predicate, and without a name or an English loan word.
7. Give a card credit for a core word only when the sense is the same.
8. Reuse the input model, fields, audio, and the `reorder_deck` code path.

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
| `units`            | Units in a card: words or Han characters, per language               |
| `name_hits`        | Kept cards with a name from the name list                            |
| `loan_hits`        | Kept cards with a word from the loan list                            |
| `no_predicate`     | Kept cards with no verb or adjective token                           |
| `unchecked`        | Targets with no sense row                                            |
| `absent_sense`     | Targets whose every candidate fails the sense test                    |
| `tail_words`       | Content lemmas that only the tail cards add                          |

### 2.3 In scope

- One `--core` mode of `anki fix <lang>` for a sentence deck with audio.
- A core list from the reviewed Swadesh word table and the cached frequency
  list of the language.
- The cost-aware selection, the card order, a report, and gates.

### 2.4 Out of scope

- New audio or TTS.
- A new note model, a new field, or a new card design. The mode reuses the
  cull templates and CSS (7.3).
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

A shared Swadesh form gives one target when the two glosses share a sense
word. The target keeps the lowest index and the gloss set. When the glosses
do not share a sense word, the form gives two targets, and the sense guard of
7.2 keeps their examples apart. This rule replaces a plain merge. The
prototype found 207 rows and 203 distinct forms, and the duplicate forms are
the subject of `anki-daz`.

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
tail:        gain counts only the content lemmas that no kept card holds
             no quota, no cost, at most --tail-cards cards
             goal: the widest vocabulary in the remaining card budget
```

- The seed pass cannot end with a gap, unless the pool has no sentence for the
  target.
- The depth pass is the quality pass. It pays for outside words.
- The completion pass protects the quota. It is the only pass that accepts a
  card with a negative score. It stops at `--completion-cards` (120) and at
  `--max-cards`.
- The tail pass is the breadth option of section 5.4. It runs last, and it
  keeps a card only when the card brings a new content lemma. It does not
  take a card from a target below quota, because the earlier passes fill the
  quota first.
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

Sections 5.1 to 5.4 use the wide window 3 to 12 words of the cull, to compare
the objectives and the sizing. Section 5.5 uses the default window of 7.1.1.

The prototype measured these sets. Every row uses quota 2, except the last
three rows of 5.1.

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

The default is `--core-words 50`. With the wide window it gives 216 cards on
the French input, and the tail fills the card budget. With the default window
of 7.1.1 it gives 216 cards for 205 targets, as section 5.5 shows.

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

### 5.4 Bang for buck

The bang is `spoken_mass`: the share of the Tatoeba French content tokens
that the kept content lemmas cover. The buck is one card. The prototype
measured these decks on the same pool and with the same filter.

| Deck                    | Cards | Distinct content lemmas | `spoken_mass` | Mass per card |
| ----------------------- | ----- | ----------------------- | ------------- | ------------- |
| Core, quota 2, X=150    | 300   | 518                     | 0.372         | 0.0012        |
| Core, quota 2, X=0      | 186   | 398                     | 0.310         | 0.0017        |
| Core X=0, then a breadth tail | 300 | 894                  | 0.464         | 0.0015        |
| Breadth only            | 300   | 1,201                   | 0.497         | 0.0017        |
| Breadth only            | 1,000 | 2,239                   | 0.555         | 0.0006        |

- A breadth selection wins on mass per card. At 300 cards it covers 33%
  more mass and 2.3 times the lemmas. It gives no second example of a word.
- The core mode wins on mass per lemma: 7.2e-4 against 4.1e-4. Its cards go
  to frequent words, and its support set is small.
- The tail is the middle choice. `--tail-cards` adds cards that bring only
  new words. The split at 300 cards holds the Swadesh depth, and it adds 0.09
  mass and 376 lemmas to the 300-card core deck of the first row.
- The mass saturates. The tail from 300 to 400 cards adds 0.033 mass. A
  1,000-card deck adds 0.058 mass over the 300-card breadth deck. The knee
  is near 300 to 400 cards.

The rows above use the wide window 3 to 12 words, so they compare the
objectives only. The default is `--tail-cards 0`, because the mode targets a
core list. A caller who wants the largest frequency mass per card sets a
larger value.

### 5.5 Measured presets with the default filter

These rows use the filter of section 7: the window 5 to 9 words, the
predicate guard, the name guard, and the sense guard. The pool is 3,403
cards. The tail fills to 300 cards.

| Preset                     | Core words | Cards | Vocab | `spoken_mass` | `covered2` |
| -------------------------- | ---------- | ----- | ----- | ------------- | ---------- |
| Swadesh only, quota 2      | 0          | 179   | 394   | 0.307         | 142 / 155  |
| Swadesh only, tail 300     | 0          | 300   | 833   | 0.459         | 142 / 155  |
| Swadesh + 50, quota 2      | 50         | 216   | 420   | 0.340         | 192 / 205  |
| Swadesh + 50, tail 300     | 50         | 300   | 740   | 0.436         | 192 / 205  |
| Swadesh + 150, quota 2     | 150        | 303   | 531   | 0.377         | 292 / 305  |

The recommended preset is the fourth row: `--core-words 50 --per-word 2
--max-cards 300 --tail-cards 120`. The core part teaches 192 of its 205
targets twice, and the tail adds the wide vocabulary of a breadth deck.

The tighter window costs 9 Swadesh forms and about 0.02 `spoken_mass` against
the wide window. It removes 1,869 cards of 4,958. The gain is a deck of
simple sentences. Set `--max-units 12` to use the wide window.

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
| K2    | Frequency  | the cached frequency list, the word table, the sense file | core list, quota, sense rows, `absent` list |
| K3    | Filter     | K1                             | passed cards and reasons F1 to F9 and C1 to C4 |
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

### 7.1 K3 sentence filter

The mode applies the cull filter F1 to F9 and four extra guards. A card must
pass every rule. A card that fails a rule is not a candidate.

| Key | Rule                                                       | Source        |
| --- | ---------------------------------------------------------- | ------------- |
| F1  | the English text has 2 to 12 words                          | cull F1       |
| F2  | the unit count is inside the window of 7.1.1                | cull F2       |
| F3  | no digit, bracket, URL, or ellipsis                         | cull F3       |
| F4  | a final mark `.`, `?`, or `!` at the end                    | cull F4       |
| F5  | exactly one final mark                                     | cull F5       |
| F6  | no quotation mark and no dialogue dash                      | cull F6       |
| F7  | at most one internal comma, colon, or semicolon             | cull F7       |
| F8  | no proper name token                                       | cull F8       |
| F9  | the sentence text is unique                                 | cull F9       |
| C1  | a predicate: a `VERB`, `AUX`, or `ADJ` token                | pool F9       |
| C2  | a main clause: no orphan conjunction                        | pool F10      |
| C3  | no name from `vocab/<language>_names.*`, as a substring     | yue Y8        |
| C4  | no word from `vocab/<language>_loanwords.txt`               | new           |

C1 makes the sentence complete. A fragment such as `Très bon.` has no
predicate, so it is not a core card. C2 rejects a clause that starts with a
conjunction and has no main verb.

C3 is the rule of the yue selector. A name can split into two tokens, so the
test is a substring test on the sentence. F8 covers the languages whose
tokenizer gives a `PROPN` tag. F3 covers a Latin name inside a Chinese
sentence.

C4 rejects an English loan word. The list is one word per line, for example
`week-end`, `parking`, `sandwich`, `email`, `internet`, `job`, and `top` for
French. A deck author adds a word when the deck shows it. The rule is a whole
word match, not a substring match, so `stop` does not reject `stopover`.
The list is optional. Gate K18 reports the count, so a missing list is
visible and does not fail the run.

#### 7.1.1 Length window

The window is the preferred range of the Swadesh pack of the language. A
short card is a fragment, and a long card is hard work in a core deck. The
window also sets the length cost: `--lambda-len` applies to each unit above
the simple limit.

| Language  | Unit     | Window | Simple limit | Evidence                                        |
| --------- | -------- | ------ | ------------ | ----------------------------------------------- |
| French    | word     | 5 to 9 | 7            | the pack preferred range; the cull uses 3 to 12 |
| Greek     | word     | 5 to 8 | 7            | the pack preferred range                        |
| Cantonese | Han char | 4 to 7 | 6            | the pack uses 4 to 8; this spec uses 7          |
| Mandarin  | Han char | 4 to 7 | 6            | the pack window                                 |

The Cantonese window is 7 and not 8, for parity with Mandarin. The yue
selector uses 3 to 12, because a Cantonese sentence is short. A caller can
widen the window with `--max-units`.

Measured effect on the French 6,000-card deck: the window 5 to 9 keeps 3,442
of the 4,958 cards that pass F1 to F9. F2 rejects 1,869 cards. The predicate
guard C1 rejects 39 more. The tighter window costs 9 Swadesh forms that have
no candidate (164 present forms become 155).

### 7.2 Sense guard

A target is a word and a sense. A card gets credit for a target only when the
English text holds that sense. The rule is the rule of the Swadesh pool.

| Tier | Sense words                                             | False-friend words        |
| ---- | ------------------------------------------------------- | ------------------------- |
| S    | the Swadesh gloss, or the `SENSE_CHECKS` row of the form | the `SENSE_CHECKS` row    |
| F    | the sense words of the target row                        | the false-friend words    |

The tier F rows live in `vocab/<language>_core_senses.tsv`, with the columns
`word`, `senses`, and `false_friends`. The two word columns are
comma-separated. Reuse `make_sense_filter` of
`sentence_pack/profile.py`, `holds_sense` of `sentence_pack/english_sense.py`,
and `compound_spans` of `sentence_pack/guards.py`.

```text
hit(w, card) is True when
    the form or lemma of w occurs in the card, and
    the English holds a sense word of w, and
    the English does not hold a false-friend word without a sense word
```

- A hit that fails the test does not count. The card stays when another
  target counts, so one bad sense does not remove a good card.
- The compound rule of the pack applies to a tier S form. For example, 闻
  inside 新闻 is not a hit, and 聽 inside 聽日 is not a hit.
- A target with no sense row is `unchecked`. Gate K17 fails when the count is
  above `--max-unchecked-sense` (0 by default). The coverage CSV gives the
  status `unchecked` for the target.
- A target whose every candidate fails the sense test is `absent_sense`. It
  stays in the coverage CSV, and it does not fail K2.

The French pack has no `SENSE_CHECKS` rows, so the French core run needs a new
sense file before K17 can pass. That file is the E2 task. A rough first pass
is enough: the sentence pairs of the deck show the senses.

### 7.3 K6 card design

K6 writes the deck with the card design of the cull. The note model, its
fields, and the note ids do not change. The model takes the templates and the
CSS of `languages/french/cull.py`, so the core deck looks like the 1,000-card
deck and the Swadesh deck.

| Card | Front                                                    | Back                                                      |
| ---- | -------------------------------------------------------- | --------------------------------------------------------- |
| 1, comprehension | the audio, and the hint `Tap to show the sentence` | the audio, the sentence with the bold target, and the English |
| 2, production    | the English, and the hint `Tap to show the French`  | the English, the sentence with the bold target, and the audio |

| Part    | Value                                                                    |
| ------- | ------------------------------------------------------------------------ |
| Templates | `CARD_TEMPLATES` of `cull.py`, written by `apply_card_templates`       |
| CSS     | `card_css()`: `templates/card.css` plus `EXTRA_CSS` of the language, through `pack.load_css` |
| Classes | `.front-audio`, `.sentence`, `.english`, `.hint`, `.target`, `.source`   |
| Fields  | the input fields, no new field                                           |
| Audio   | the input MP3 file, not a new TTS file                                   |
| Deck    | `--name`, or the input deck name                                         |

The model keeps the number of card directions that it has. The cull template
set has two, so a model with two directions gets both.

### 7.3.1 Field mapping

| Input field | Value of a kept card                                  |
| ----------- | ----------------------------------------------------- |
| `French`    | the sentence, with the assigned target in `<b>` tags   |
| `sound`     | `[sound:<input audio file>]`                          |
| `Back`      | the English text of the card                          |
| `freq ranks`| unchanged                                             |

The target is the `assigned` target of section 6. A card can hold more than
one target, and only the assigned target is bold. The bold word shows the
learner what the card teaches, and it needs no new field.

- `--bold-target` is on by default. `--no-bold-target` writes the plain
  sentence.
- The core CSV holds the plain sentence. The bold markup stays in the APKG
  only.
- `_plain_sentence` of `cull.py` removes the markup on a later extract, so a
  second run on this output is safe.
- `--front-hint` and `--back-hint` change the hint text of the two cards.

Gate K15 checks the template and CSS values. Gate K21 checks the bold target
and the plain CSV text.

### 7.4 Relation to the cull filter

The four guards and the window can improve the cull. They cannot replace it.
The two modes want different pools, and one measurement shows why: the cull
selection at 1,000 cards, with the same targets and options.

| Cull pool                    | Pool | Cards | Content lemmas | `spoken_mass` |
| ---------------------------- | ---- | ----- | -------------- | ------------- |
| wide window 3 to 12 words    | 4,912 | 1,000 | 2,079         | 0.544         |
| tight window 5 to 9 words    | 3,403 | 1,000 | 1,856         | 0.523         |

- The tight window costs the cull 223 lemmas and 0.021 mass, and it puts the
  lemma count below the 1,900 aim of the cull spec. The cull keeps the wide
  window.
- C1 and C2 cost little and remove fragments. The predicate guard removes 46
  cards of 4,958. The cull adopts C1 and C2.
- C3 adds a name guard for a language without a `PROPN` tag. The cull keeps
  F8 and adds C3 for the languages of F8 that have no tagger.
- C4 is a taste rule with a manual list. It stays in the core mode.
- The sense guard improves the credit of a target, so the cull adopts it as
  an option. The cull counts a hit today without a sense test.

The shared code is one function with a profile:

| Value            | `core` profile          | `cull` profile          |
| ---------------- | ----------------------- | ----------------------- |
| Length window    | the 7.1.1 window        | 3 to 12 words, or the pack window |
| C1, C2           | on                      | on                      |
| C3               | on for every language   | on when F8 has no tagger |
| C4               | on when a list exists   | off                     |
| Sense guard      | on, K17 fails          | on when a sense file exists |

This relation is a decision for E2. It needs a small refactor of
`cull.filter_cards` into `filter_reason(card, tokens, profile)`. The cull
keeps its output files and its gates. The cull part is `anki-0h8`.

## 8. Swadesh reserve

A core word with no source sentence in the input deck is `absent`. The French
input deck holds 155 of the 203 distinct Swadesh forms with the window of
7.1.1. The 48 gaps are mostly rare nouns such as `pou`, `bâton`, and `graine`.

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
| `--core-words`      | 50           | Tier F words after the Swadesh set                   |
| `--per-word`        | 2            | Examples per content target, 1 to 3                  |
| `--third-examples`  | 0            | Targets that get a third example                     |
| `--ubiquitous-share`| 0.25         | Share above which a target gets quota 1              |
| `--max-cards`       | 300          | Card cap                                             |
| `--completion-cards`| 120          | Card cap of the completion pass                      |
| `--outside-budget`  | 0            | Cap on the outside set; 0 is no cap                  |
| `--tail-cards`      | 0            | Cards after the core passes that only bring new words |
| `--support-top`     | 1000         | Outside rank that costs 1.0                          |
| `--hard-top`        | 3000         | Outside rank that costs 1.8, above is 3.0            |
| `--lambda-new`      | 0.1          | Cost of one new outside word                         |
| `--lambda-repeat`   | 0.03         | Cost of one repeated outside word                    |
| `--lambda-len`      | 0.02         | Cost per unit above 8                                |
| `--lambda-seed`     | 0.01         | Cost factor of the seed pass                         |
| `--min-deck-count`  | 3            | Minimum deck count of a tier F target                |
| `--spoken-weight`   | 0.5          | Weight of the Tatoeba term in the blend              |
| `--min-words`       | per language | First unit count of the window of 7.1.1              |
| `--max-words`       | per language | Last unit count of the window of 7.1.1               |
| `--simple-units`    | per language | Unit count above which the length cost applies        |
| `--names`           | per language | Name list of rule C3, or none                        |
| `--loanwords`       | per language | Loan word list of rule C4, or none                   |
| `--senses`          | per language | Sense file of section 7.2                            |
| `--max-unchecked-sense` | 0        | Unchecked targets that gate K17 accepts              |
| `--similarity`      | 0.67         | Lemma Jaccard limit                                  |
| `--reserve-pack`    | none         | Swadesh pack to add, or none                         |
| `--name`            | from the language | Output deck name                                |
| `--output`          | from the input name | Output APKG                                  |
| `--check`           | off          | Run gates K1 to K20 on the output                    |

### 9.3 Output files

| File                                  | Content                                                     |
| ------------------------------------- | ----------------------------------------------------------- |
| `apkg/french_core_300.apkg`           | The kept cards in core order, with the input model and audio |
| `apkg/french_core_300.core.csv`       | One row per input card: `kept`, `pass`, `reason`, `targets`, `assigned`, `value`, `cost`, `units`, `sense` |
| `apkg/french_core_300.coverage.csv`   | One row per target: `word`, `tier`, `position`, `quota`, `count`, `status` |
| `apkg/french_core_300.manifest.json`  | Counts, options, frequency source, and hashes               |
| `apkg/french_core_300.rank.csv`       | The ranking CSV that `reorder_deck` reads                    |

`status` is `covered`, `partial` (below the quota), `absent` (no candidate),
`absent_sense` (every candidate fails the sense test), or `unchecked` (no
sense row).

### 9.4 Code changes

- Add `anki_tools/languages/<language>/core.py` with K2, K4, and K5. Keep the
  generic gain, cost, and pass code in `anki_tools/ranking/core.py`, so two
  languages share it.
- Add the `--core` option and the option block to `anki fix fr` in `cli.py`.
- Add the filter C1 to C4 as `core_filter_reason` after the cull reason. The
  predicate and orphan tests are the tests of `sentence_pack/pool.py`.
- Add the sense guard as `core_sense_hits`, over `make_sense_filter` and
  `holds_sense`.
- Add `vocab/<language>_core_senses.tsv` and
  `vocab/<language>_loanwords.txt`. Add a name list for a language that has
  none.
- Extend `ranking/cover.py` with the `key` order `(-score, new, units, index)`.
- Reuse the APKG writer and `reorder_deck` without a change.
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
| K5   | Every kept card passes the filter rules F1 to F9 and C1 to C4         |
| K6   | No kept pair reaches the lemma Jaccard limit                          |
| K7   | Every kept card has its input audio, and the file is a valid MP3      |
| K8   | The APKG note count equals the kept count and the manifest count      |
| K9   | The APKG holds exactly the media files of the kept cards              |
| K10  | The deck name and the note model are the requested values             |
| K11  | A second run gives the same kept set and the same order               |
| K12  | The coverage CSV has one row per target, and `count` matches the cards |
| K13  | The manifest hashes match the core CSV and the coverage CSV           |
| K14  | Every target with a kept card holds the form or lemma in that card    |
| K15  | The two templates equal `CARD_TEMPLATES`, and the CSS equals `card_css()` |
| K16  | Every tail card brings at least one new content lemma                  |
| K17  | Every target with a card has a sense row, or the unchecked count is within `--max-unchecked-sense` |
| K18  | No kept card holds a name from the name list or a word from the loan list |
| K19  | Every kept card has a predicate, and its unit count is inside the window |
| K20  | The coverage CSV gives the status `absent`, `partial`, `covered`, or `unchecked`, and the manifest counts agree |
| K21  | Every kept card has the assigned target in `<b>` when `--bold-target`, and the CSV holds the plain sentence |

### 10.2 Tests

Add `tests/test_core_deck.py`. Use a small fixture APKG and a small frequency
file. Do not download a file.

| Area         | Test                                                              |
| ------------ | ----------------------------------------------------------------- |
| Core list    | One target per distinct Swadesh form when the glosses share a sense |
| Core list    | Two glosses without a shared sense word give two targets          |
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
| Tail pass    | A tail card brings a new lemma, and no tail card repeats a core card |
| Tail pass    | `--tail-cards 0` keeps the core set and the core order             |
| Filter C1    | A fragment with no verb or adjective is not a candidate            |
| Filter C2    | A sentence that starts with a conjunction and has no main clause fails |
| Filter C3    | A name from the list removes the card, also when it splits in two tokens |
| Filter C4    | A loan word removes the card; a word that only contains it does not |
| Window       | A card outside the language window fails, and `--max-units` widens it |
| Sense        | A hit whose English holds no sense word does not count             |
| Sense        | A false-friend English word removes the hit, and the card stays for another target |
| Sense        | A target with no sense row gets the status `unchecked` and fails K17 |
| Sense        | A compound span blocks a tier S hit (闻 inside 新闻)                  |

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
| The window 5 to 9 removes 1,869 cards             | 9 Swadesh forms have no candidate         | `--max-units 12`; `--reserve-pack`            |
| A sense file does not exist for most languages    | K17 fails, or many targets are `unchecked` | E2 builds the file for the top 50 words first |
| A sense row is wrong                             | A good card loses its credit              | Review the CSV; the `absent_sense` status     |
| The loan word list is incomplete                  | A loan word stays in a card               | Add a row when the review shows the word; K18 |

### 11.2 Open questions

| #   | Question                                                              | Decide at |
| --- | --------------------------------------------------------------------- | --------- |
| 1   | Is quota 2 right for the Swadesh forms, or should a function form always get 1? | E2 |
| 2   | Which spoken list does each language use, and is its license public?   | E2        |
| 3   | Does the output bold the target token, as the Swadesh pack does?        | E4        |
| 4   | Does the output need a `Target` field to show the assigned word?        | E4        |
| 5   | Does the order interleave tier S and tier F, or follow the core list?   | E3        |
| 6   | Is a second example of the same word in the same week useful, or should the order spread it? | E3 |
| 7   | Should `--tail-cards` have a default above 0 for a deck of 300 cards? | E3 |
| 8   | Is the window 5 to 9 for French right, or should it be 4 to 10? | E2 |
| 9   | Does a curated loan word list cover the deck, or does it need a rule? | E2 |
| 10  | Who writes the sense file, and how many words does it cover? | E2 |

## 12. Milestones

| Milestone | Issue      | Task                                                     | Output                    |
| --------- | ---------- | -------------------------------------------------------- | ------------------------- |
| E1        | `anki-eo6` | This spec and the prototype analysis                      | `specs/core-word-small-deck.md` |
| E2        | `anki-v12` | Core list, quota, `absent` list, size check, filter C1 to C4, sense file, name and loan lists | `languages/<lang>/core.py` |
| E3        | `anki-p62` | Value, cost, passes, cover key, sense-filtered hits        | selection report          |
| E4        | `anki-7z5` | Core order, writer, CSV files, manifest, gates K1 to K20   | `.apkg` and CSV files     |
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

### 13.7 Bang-for-buck comparison

The comparator of section 5.4 is a breadth greedy selection. It uses the same
pool, the same filter, and the same similarity test. Its gain is the sum of
`1 / log2(rank + 1)` over the content lemmas of a card that no kept card
holds yet. There is no quota and no outside cost, so it never repeats a
word.

The tail rows start from the kept core cards. The tail then adds cards with
the breadth gain until the card limit. `spoken_mass` comes from
`cull.spoken_mass`, with the Tatoeba counts of `data/french_frequency.tsv`.
It counts the content lemmas only, because a function word is not a target.

| Core                      | Cards | Vocab | `spoken_mass` |
| ------------------------- | ----- | ----- | ------------- |
| Swadesh only, quota 2     | 186   | 398   | 0.310         |
| Swadesh, then tail to 300 | 300   | 894   | 0.464         |
| Swadesh, then tail to 400 | 400   | 1,194 | 0.497         |
| Core X=50, quota 2        | 211   | 418   | 0.339         |
| Core X=50, then tail to 300 | 300 | 809   | 0.455         |
| Core X=150, quota 2       | 300   | 518   | 0.372         |
