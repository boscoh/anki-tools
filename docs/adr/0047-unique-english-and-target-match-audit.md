# Reject the same English as unique; keep the target match rules

Status: accepted.

## Decision

1. Filter `unique` also rejects a card whose English text matches an earlier
   passed card. The key is `english_key` in `ranking/cardfilter.py`: casefolded,
   with single spaces. An empty English text never matches. The sentence key
   test stays. `filter_cards` runs both tests, so the cull, the core, cull `G5`,
   and core `K5` use one rule. Cull `G5` now calls `filter_cards` and no longer
   holds a copy of the dedupe loop.
2. The Mandarin and Cantonese `hits` stay as they are. A fix that requires a
   whole token costs more than it gives (see below).

## Evidence

Same English with a different sentence in the kept rows of the committed core
decks: French 3, Mandarin 2, Spanish 0, Cantonese 0.

| Run                                   | `Rejected unique` before | after | Kept cards before | after |
| ------------------------------------- | ------------------------ | ----- | ----------------- | ----- |
| French core, 50 words                 | 1                        | 224   | 276               | 275   |
| Spanish cull (`--select`), 9263 input | 0                        | 2     | 695               | 695   |

The French input has many cards with the same English. They now fail before
the selection, so the pool is smaller but the kept count changes by one.

## Target match audit

Causes:

- Mandarin 得到 in 听得到 (tier F): the tokenizer. jieba gives the token 得到
  (`tokens_with_pos`, `languages/mandarin/profile.py`).
- Cantonese 蟲 in 昆蟲 (tier S): `form_offsets` in
  `languages/cantonese/profile.py`. A form inside a longer token counts when
  that token is not a target form.
- French `quelques` in `quelque chose` (tier S): `form_keys` holds the lemma
  `quelque`, and the token `quelque` matches it (`hits` in
  `languages/french/profile.py`).

Decisions:

- Mandarin: no change. The tier F target matches the jieba token, so a fix
  needs a new tokenizer rule or a user dictionary entry that splits 听得到.
  Mandarin tier S `hits` match the character stream on purpose, so 水 hits
  inside 喝水.
- Cantonese: the `elif` branch of `form_offsets` is deliberate. The sense table
  once credited 蟲 inside 昆蟲; a tier S `compounds` row for 蟲 then blocked
  that credit (the dead tier F row listing `insect` was deleted). ADR 0053
  removed the row; the sense test rejects the card instead. A strict
  whole-token rule was tried and reverted. The Cantonese tokenizer gives tokens
  such as 蟲好 and 好忙, so a whole-token rule rejects true hits. It changed the
  core deck from 116 targets and 109 cards to 90 targets and 89 cards, and it
  broke the preview card for 忙. A fix needs a better tokenizer, or
  an English-side sense entry (ADR 0053 removed the `compounds` column).
- French: not changed. The lemma key is the cause.

## Reason

- Two cards with the same English teach the same sentence twice and use two
  card slots.
- A gate must run the rule of the producer. One function gives that, with no
  copy.
