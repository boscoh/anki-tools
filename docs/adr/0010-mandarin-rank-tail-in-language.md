# Put the Mandarin rank tail in the Mandarin package

`anki fix zh` rank mode ended with two steps: fix the pinyin field, then write
the card templates into the deck. The `zh` command held both as a closure, and
two flags, `pinyin_fix` and `style`, could turn each one off. ADR 0009 kept
them as the two options of one language, which broke the rule that the four
`fix` commands share one option set.

The style step also had no place of its own. `apply_style` and its three
helpers sat in `cli.py`, 140 lines of package work, next to the command that
called them. `anki style`, the manual door, called the same code.

Status: accepted

## Decision

The command has no tail. The tail belongs to the language.

- `anki_tools/apkg/templates.py` holds `apply_templates` and its three
  helpers: `extract_template_fields`, `find_best_match`, and
  `apply_field_mapping`. It also holds the three default file paths. The
  module already held the loaders of the same files.
- `anki_tools/languages/mandarin/fix_steps.py` holds `after_rank`. It fixes
  the pinyin field, then calls `apply_templates`. The function returns the
  extra DONE lines.
- The `zh` command passes `after=mandarin_fix_steps.after_rank` to the
  existing hook of ADR 0002, as the other commands pass their own hook.
- `pinyin_fix` and `style` are deleted. Both steps always run in rank mode.
  A deck that needs an untouched pinyin field or an untouched model is not a
  rank-mode case; `anki style` and the pinyin module stay available on their
  own.

The tag "style" goes with the flag. The step writes templates and CSS, so the
function is `apply_templates` and the printed line is "Applying card
templates...". The `anki style` command keeps its name, because it is the
documented manual door for any deck.

## Consequences

- `cli.py` loses the four template helpers and 140 lines. It imports
  `apply_templates` for `anki style` and the hook for `zh`.
- `anki fix zh --no-pinyin-fix` and `anki fix zh --no-style` fail. The README
  drops them from the option tables.
- The rank mode of `zh` prints "Templates applied:" in the DONE table instead
  of "Styles applied:".
- A second language with a rank tail adds one module beside its `rank.py` and
  one argument in its command.

## Rejected

Keep the closure and pass two booleans. Two flags of one language stay, and
the package work stays in `cli.py`. The user asked for one option set and for
the step to belong to the language.

Put `apply_templates` in `cli.py` and import it from the Mandarin package. The
import runs from a language package back into the command tree, and the import
graph test of `languages/` fails.

Name the new module `style.py`. The step writes templates, so the old name
would carry the concept that the decision removes.
