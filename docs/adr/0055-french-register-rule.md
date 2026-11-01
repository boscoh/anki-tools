# Reject French passe simple sentences with a register rule

Status: accepted

The hand review of the French cards found sentences in the passe simple (`posa`,
`arrivèrent`, `éclata`), a literary tense that spoken French does not use and
that a learner should not meet as a first example (`anki-03d9`). The shared card
filter now has a `register` rule. `filter_reason` in `ranking/cardfilter.py`
runs it after the shape guards, before `unique`, so the cull (G5) and the core
(K5) re-run it through `filter_cards`.

The language profile gives the test, as `script_ok` and `is_repeated` do. The
hook is `register_ok(sentence) -> bool`, a field of `LanguageDeckProfile`,
`CullProfile`, and `CoreProfile`, with the default None. A language with no hook
skips the rule. Only French sets it.

The French test reads the tokens that `french_tokens` already holds. It rejects
a VERB or AUX token with `Mood=Ind`, `Tense=Past`, and `VerbForm=Fin` whose
surface ends in `a`, `it`, `ut`, `int`, or `rent`.

## Consequences

The French core grows from 277 to 280 cards, and covers the same 211 targets
(K23 0.821, floor 0.80). The French cull keeps 1000 cards and covers 298 of 300
targets, one more than before. The input holds 141 more rejects for `register`
among the cards that pass the shape guards. The kept frequency cards held 7
(core) and 21 (cull) passe simple sentences, and now hold none.

The pack does not run the rule. `SHAPE_GUARDS` and `FILTER_KEYS` are
unchanged, so the French pack output is the same. One kept reserve card of the
cull, a pack card, still has a passe simple verb (`fendit`), and G5 skips the
reserve rows.

## Known limits

- The small model gives the passe simple tags to any unknown form, for example
  `bleu`, `soupe`, `senti`, and `continué`. The ending test removes these.
- It still rejects three good sentences of the input: `deviendra` (a future) and
  `interdit` (two participles).
- The model often tags a passe simple as `Tense=Pres` (`quitta`, `partirent`,
  `vit`), so the rule misses some of them (about 55 more in the input end like one
  in `-a`, `-èrent`, or `-irent`). A form in `-it` or `-ut` cannot be
  separated from the present (`choisit`, `peut`) by its ending alone.

## Rejected

Putting the rule in `SHAPE_GUARDS`. The guards run in the pack, which would
change the French pack output and add a French rule to the shared guard table.

A rule with a French word list of every passe simple form. It does not scale,
and the model already tags most of them.

Reading `Tense=Pres` forms by ending as well. It adds about 55 rejects at the
cost of a second ending table and one false reject (`meurent`).
