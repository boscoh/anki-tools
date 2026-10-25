# One core module per language package

The French package held the core deck in three files. `core_types.py` held the
French values, the `CoreProfile`, and the bound wrappers. `core_gates.py` held a
`CoreFiles` subclass and `check_core`. `core.py` held no logic: it re-exported
both files, with one star import, one private import, and one import of
`check_core`, so that `from anki_tools.languages.french import core` kept
working. That shim grew a 50-name `__all__` list, and the surface existed twice:
once in `core_types.__all__` and once in `core.py`.

The other packages never split the file. Spanish, Mandarin, and Cantonese each
have one `core.py` of 213 to 325 lines. French was the outlier at 363 lines
across three files, and the split was a leftover of the extraction that moved
the shared flow into `ranking/coreflow.py`.

Status: accepted

## Decision

One `core.py` per language package. It holds the profile values, the
`LanguageProfile` hooks where the language needs them, the `CoreProfile`, the
bound wrappers (`run_core`, `read_senses`, and the rest), the language-bound
`CoreFiles` reader, and `check_core`. `__all__` declares the surface once.

`languages/french/core_types.py` and `languages/french/core_gates.py` are
deleted. Callers keep using `from anki_tools.languages.french import core`; the
two test call sites that reached for `core_gates.CoreFiles` now use
`core.CoreFiles`.

## Consequences

The import graph loses a hop, and the surface is declared once. A reader who
opens `core.py` sees every French core value, wrapper, and gate in one place,
which is where they already look for the other languages. The file is 239 lines,
between Spanish at 213 and Mandarin at 325.

The cost is one long explicit import list from `ranking/coreflow`, which the
module already needed. No behaviour changes: the French core gate table prints
the same 20 rows.

## Rejected

Keeping `core.py` as a shim and merging nothing. It costs a file, a 50-name
list, and a `__all__` built by unpacking that static tools cannot read.

Splitting every language package the same way. Two seams for one concept buy no
test seam: the gates are already shared in `ranking/coregates.py`, and the
language-bound reader is 12 lines.
