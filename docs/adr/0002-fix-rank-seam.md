# Put one seam in front of the fix rank skeleton

The four `anki fix <lang>` rank commands (`zh`, `fr`, `yue`, `es`) repeated
one skeleton: rank the paths, print the banner and "Step 1", call
`run_ranked_deck`, stop on an empty extraction, then print the DONE table.
Only the label, the ranker, the keyword arguments, and a few prints differed.

Status: accepted

## Decision

`_run_rank_mode` in `anki_tools/cli.py` owns the skeleton. Each command passes
the paths, the banner label, its ranker, and the extra `run_ranked_deck`
keyword arguments. Three optional hooks carry the rest: `target` for the extra
banner line, `count` for the ranked count, and `after` for the language steps
that print and return the extra DONE lines. The hook takes the whole
`RankedDeck`, because Spanish prints `len(deck.sentences)`.

## Consequences

Anki output and printed text stay byte-identical. A hook owns the order of
its own prints, so each language keeps its "Step 2", "Fixing pinyin", and
"Fixing jyutping" lines. A new language needs one call, not one copy.

## Rejected

A `RankRequest` dataclass with one module function per language. It needs more
files and types, and it moves the Cantonese and Spanish closures out of their
commands without changing a line of output.
