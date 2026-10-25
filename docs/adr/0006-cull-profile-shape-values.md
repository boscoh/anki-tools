# Put the cull shape values in the CullProfile

The cull filter read three Latin assumptions from module constants.
`ranking/cardfilter.py` held `FINAL_MARKS = ".?!"` and the Latin internal marks,
`ranking/corelist.py` held `CONTENT_POS` and `MIN_LEMMA_LENGTH = 2`, and
`ranking/cullflow.py` read them for every language. A Han language ends a
sentence on `。`, has no part-of-speech tagger, and teaches one-character
lemmas. The shared flow deleted every such card and every such lemma without a
message.

Status: accepted

## Decision

`CullProfile` holds five more values: `content_pos`, `min_lemma_length`,
`final_marks`, `quote_marks`, and `internal_marks`. An empty `content_pos`
counts every token, which a language without a tagger needs.

The cull filter keeps the rule names of the shared card filter, `F1` to `F9`.
`ranking/cullflow.py` `filter_reason` composes them: the English window F1, the
script rule F3, and the proper-noun rule F8 come from `ranking/cardfilter.py`;
the mark rules and the length window come from the shape guard
`sentence_pack.guards.check_sentence_shape`. The guard names its keys `F5` to
`F8`, and `MARK_RULES` maps them to the cull names `F4` to `F7`, so the
`reason` column of the cull CSV stays the same. `filter_reason` gives the guard
a `CardShape`: the marks of the profile and the token count of the card, so the
length window still counts tokens.

`cardfilter.filter_cards` takes an optional `card_rule` hook and defaults to its
own Latin rules, so the core flow keeps its behaviour. `cullgates.gate_filter`
calls `cullflow.filter_reason`, so gate G5 and the flow apply one rule table.
`build_targets`, `lemma_sets`, `hardest_rank`, `order_cards`, `select_cards`,
and the two reserve dedupe functions take `content_pos`; `build_targets` also
takes `min_lemma_length`.

## Consequences

A language supplies the marks, the content parts of speech, and the shortest
lemma in its `CullProfile`, and no flow code. The French profile passes the
values of the French language profile and `CONTENT_POS`, so the two French CSV
files stay byte-identical. The helper signatures grew one argument each.

## Rejected

Putting a `LanguageProfile` in the `CullProfile`. The shape guard needs six
values and the cull needs a token count for the length window, so the cull
would have to wrap the profile anyway. The `CardShape` adapter gives the guard
the marks and the token count that it reads.

Adding a Protocol for the shape guard input. One adapter and one guard key map
cover the cull, and a Protocol would add a type seam with one extra
implementation.

Reading the marks from `cardfilter` and translating them per language. Two
tables of marks drift, and gate G5 would read the other one.
