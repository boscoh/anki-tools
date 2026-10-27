# Delete the fix rank mode

The four `anki fix` commands had a rank mode. ADR 0016 made the mode explicit
with `--rank`. No user flow calls it, and the mode keeps a second output shape
and a second set of language modules alive.

Status: accepted

## Decision

Delete the rank mode of the four `anki fix` commands.

- `--rank` is gone. A command needs `--select` or `--core`.
- Delete `anki_tools/ranking/pipeline.py`, the Mandarin `fix_steps.py`, and the
  four language rank modules: Cantonese, French, Mandarin, and Spanish.
- The Spanish core profile keeps its own `SIMILARITY_THRESHOLD`, because the core
  flow reads it.
- `anki_tools/ranking/rank.py` and `anki_tools/ranking/reorder.py` stay. The
  core and cull flows read the shared sentence helpers and the reorder step.

## Consequences

- `anki fix <lang>` and `anki fix <lang> --rank` fail. The output is only a cull
  deck or a core deck.
- The `*_reordered.apkg` files and their rank CSVs are not written. The `apkg/`
  directory drops them.
- The `anki_tools` package no longer exports `rank_sentences_zh`,
  `rank_sentences_es`, `complexity_score_zh`, or `frequency_score_zh`.
- The Mandarin rank steps no longer fix the pinyin field or apply
  `templates/default/`. `anki fix zh --core` writes the core templates, and
  `anki style` applies the default templates to any deck.
- ADR 0016 recorded the explicit flag. This ADR supersedes it.

## Alternatives

Keep the rank mode behind `--rank`. The mode has no user, and it keeps the large
language score modules and their tests alive.

Delete `anki_tools/ranking/rank.py`. The reorder step and the shared helpers of
the cull and core flows live there, so that deletion moves code that this request
does not touch.
