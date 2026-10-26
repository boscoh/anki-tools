# Remove three fix flags that only restate a module default

The four `anki fix` commands took three flags that carried no decision of their
own. Each flag forwarded a value that already lives in the module that owns it.

- `--examples` forwarded to `CoreOptions.per_word`. The default is 2 for every
  language, and the CLI default was `DEFAULT_OPTIONS.per_word`.
- `--tail` forwarded to `CoreOptions.tail_cards`. The default is 0 for Mandarin,
  Cantonese, and Spanish, and 120 for French. The CLI default was
  `DEFAULT_OPTIONS.tail_cards`.
- `--keep-filtered` inverted one boolean. Its only effect was to keep the
  high-similarity cards in the reordered deck. No document, spec example, or
  script passed it.

Both CLI defaults equalled the language-module defaults, so the two count flags
only restated them. ADR 0009 removed the tuning flags that had no owner and kept
a reduced surface. These three flags pass the same test: the value belongs to the
module, not to the run.

Status: accepted

## Decision

Delete `--examples`, `--tail`, and `--keep-filtered`, and delete the code and the
documents that exist only to carry them.

- `_run_core_command` takes no `examples` and no `tail`. It uses
  `profile.default_options`, and the existing `replace` when `cards is not None`
  sets `max_cards`. No other `CoreOptions` field changes.
- `run_ranked_deck` takes no `keep_filtered`. It calls `reorder_deck` without
  `remove_filtered`, so the `remove_filtered=True` default applies. Rank mode
  still removes the high-similarity cards, as it always did.
- The CLI keeps `--cards` and `--words`. Both choose a mode, and both differ from
  the module default in a documented case.
- The removed values stay as the `CoreOptions` defaults of the language module
  and as the `reorder_deck` default.

## Consequences

- A default build gives the same output, byte for byte. The removed flags were
  always set to the module default, or to a value that reproduced it.
- Each fix command takes three parameters fewer. `_run_core_command` takes two
  fewer. `run_ranked_deck` takes one fewer.
- A core deck that needs other counts is a code edit of the language module,
  like the other measured presets. No flag opens it.
- `anki fix <lang> --examples 3`, `--tail 0`, and `--keep-filtered` fail. The
  README and the specs drop them from the option tables.
- ADR 0009 stays as the record of the option set of that date. This ADR removes
  three of its survivors.

## Alternatives

Keep the three flags for a rarely measured deck. A flag that only restates a
module default hides that default, and a reader cannot tell which value the run
uses.

Keep `--keep-filtered` alone, because it inverts a boolean and is not a count. No
caller ever set it. A deck that must keep the filtered cards needs the ranking
CSV, which the run writes, and a separate reorder.

Add one generic override flag, for example `--option per_word=3`. That is a new
interface with a parser, and it opens every measured value again. The values of a
language belong to the language module.
