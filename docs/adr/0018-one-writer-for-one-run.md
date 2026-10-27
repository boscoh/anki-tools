# Put the output shape in the assembler and return one run value

The core flow and the cull flow each built the same seven-line `OutputShape`
block from the output identity and the language profile, and each returned its
own run type. `coreflow.CoreRun` and `cullflow.CullRun` held the same four
paths, renamed, and four count fields that no production caller reads. The
output shape could not move into `assemble.py` before this decision, because
`coreflow` imports `assemble`, so a reverse import of `DeckProfile`,
`output_fields`, and `deck_templates` would close a cycle.

The report lines, the manifest `params`, the manifest `counts`, and the
`frequency` block stay with the mode. They state different facts per mode: the
cull prints ten sample cards, and its frequency block holds a `Path` where the
core block holds a label string.

Status: accepted. Supersedes the row of ADR 0004 that keeps the deck writer in
`ranking/coreflow.py`. The other decisions of ADR 0004 stand.

## Decision

`anki_tools/ranking/assemble.py` owns the output shape and the one run value of
a fix run.

- `assemble.write_run(run, profile)` builds the `OutputShape` from
  `run.identity` and the deck profile interface, then writes the package, the
  side files, the manifest, and the report. `DeckRun` holds no `shape`.
- `DeckProfile`, `output_fields`, `deck_templates`, `OUTPUT_FIELDS`,
  `OUTPUT_ROMANIZATION_FIELD`, and `ROMANIZATION_BLOCK` live in `assemble.py`.
  The flows, the three gate modules, and the tests import them from there.
  `coreflow` keeps the field-name constants that its own readers use and reads
  `assemble.OUTPUT_ROMANIZATION_FIELD` where it needs the shared one.
- Both flows return `assemble.RunFiles`. `CoreRun` and `CullRun` are gone, and
  `finalize` takes that value.
- The mode-only fields of `DeckRun` stay as they are: `rank_rows` is one
  decision (write a rank CSV and reorder), and an empty `reserve` is one
  decision.
- `DeckProfile` gains the `highlight` member. `note_values` already read it
  from a value typed `DeckProfile`, so the contract was incomplete.

## Consequences

One place states the output shape, so the note type of a core deck and a cull
deck cannot drift. The six packages in `apkg/` rebuild with the same side files
apart from the manifest `build_date`, and every gate keeps its earlier result.

`RunFiles` is the return value of both flows, so a caller that wants the card
CSV reads `card_csv` for either mode. The counts that the deleted run types
held are read from the manifest, where the gates already read them.

The seam is real: two flows and six tests cross it.

## Rejected

A third module `ranking/shape.py` for the shape builders. It adds a file for
forty lines that the writer owns, and the writer is the only reader.

`write_run(run, shape)` with the flow passing a built shape. That is the
duplication this decision removes, and it keeps the shape rule in two flows.

One `mode_extras` field for `rank_rows` and the reserve trio. The wrapper hides
two decisions that `write_run` already reads correctly.

One report builder for both modes. The four head lines do not earn a function,
and the bodies differ: the cull appends ten sample cards.
