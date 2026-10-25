# Put one CullProfile in front of the cull flow

`anki_tools/languages/french/cull.py` held 2179 lines. It forked the shared deck
flow that moved into `anki_tools/ranking/coreflow.py`, so 26 function names
existed twice: the deck helpers, the card filter, the selection, the file
writers, and the gate readers. The cull is not French-only logic. A new language
needed a second 2000-line copy.

Status: accepted

## Decision

`anki_tools/ranking/cullflow.py` owns the cull flow: the request and run types,
the frequency blend, the reserve pack, the selection, the order, the cull CSV,
the coverage CSV, the manifest, and `run_cull(request, profile)`. The module
reads every language value from a frozen `CullProfile` dataclass. The profile
holds the scan adapters (`tokenize_texts`, `load_frequency`, `frequency_ranks`,
`read_names`, `concept_forms`), the model fields and the sentence cleaner, the
card templates and the CSS, the reserve pack and its hooks (`reserve_fields`),
and the file paths and licenses.

`anki_tools/ranking/cullgates.py` owns gates G1 to G15 behind a `CullFiles`
reader, mirroring `ranking/coregates.py`. `anki_tools/languages/french/cull.py`
keeps only the French values, `FRENCH_CULL_PROFILE`, thin bindings, and the
re-exports that `cli.py` and the tests read.

The deck writer stays in `anki_tools/ranking/coreflow.py`. `write_deck` grew
`templates`, `card_css`, `reserve`, `reserve_pack_path`, and `reserve_fields`
keywords, and `write_manifest` grew `licenses`, `reserve_pack`, and
`swadesh_exceptions`. The core flow passes none of the new keywords, so its
files stay byte-identical.

## Consequences

A new language needs one `CullProfile`, one `reserve_fields` hook, and one
`concept_forms` hook, not a second copy of the flow. The cull and the core flow
apply one card filter (`ranking/cardfilter.py`) and one reserve insertion. The
printed report of `run_cull` and every gate message stay byte-identical, because
the flow is a move, not a rewrite.

The seam has a cost: `CullProfile` has 23 fields, and a language that needs one
more hook edits the dataclass. The benefit is one module for every cull, so the
cost is paid once instead of once per language.

## Rejected

A third module for the profile (`ranking/cullscan.py`) next to
`ranking/scan.py`. It splits one flow over two files and buys no test seam,
because `tests/test_cull_shared.py` drives the profile through the public
`run_cull`.

A cull-only deck writer that copies `coreflow.write_deck`. The two writers
differ by the template set, the CSS, and the reserve cards, so the copy keeps
two card-deletion paths and two note-count checks in step by hand.

Putting `templates` and `card_css` into the profile as loaded strings. The core
profile already holds `card_css` as a callable, and one form for one concept is
the rule.
