# One option set for every fix command

The four `fix <lang>` commands grew four hand-written parameter lists. They
drifted:

- `zh` required the input path. `fr`, `es`, and `yue` had a default.
- `--verbose` reached only the Mandarin pinyin step.
- `--text-field` sat on `fr` and `es` only.
- `--select` was opt-in in `fr` and opt-out in `yue` (`--no-select`), because
  the two commands had different defaults.
- The card cap was `--cards` everywhere except `es` rank mode, which used
  `--max-sentences`.
- `--max-uses`, `--no-pinyin-fix`, and `--no-style` named values of one
  language.

A user reads one help page and expects the other three to match.

Status: accepted

## Decision

The four commands take the same options, in the same order, with the same
meaning:

`apkg-path`, `--output`, `--model-id`, `--text-field`, `--keep-filtered`,
`--select`, `--core`, `--cards`, `--words`, `--examples`, `--tail`, `--check`.

- Every command has a default input path. `zh` adds
  `apkg/Neri's Chinese Sents Part 2.apkg`, the deck that the core documents
  name.
- `--cards` is the card cap of the chosen mode for every command. The `es`
  rank reduction uses it instead of `--max-sentences`.
- Rank mode is the default for `yue`, as for the other three. `--select` runs
  the coverage selection.
- `--text-field` reaches the shared rank pipeline for `zh` and `yue`. `None`
  means "infer the field", which is what those two commands did before.
- `--verbose` is deleted. It printed the details of one step in one language.
- `--max-uses` is deleted. The value stays in `SelectRequest` and
  `CullRequest` with its measured default.

A value that one language owns stays on that command: `--targets` and
`--reserve` (`fr`). ADR 0010 moves the two Mandarin steps out of the command.

## Consequences

A typed `anki fix yue <deck>` now writes the rank-only deck. The coverage
selection needs `--select`. The stored LTL artifacts came from the selection,
so `README.md` and `specs/yue-sentence-selection.md` name the flag.

A typed `anki fix es <deck> --max-sentences 3000` fails. The new name is
`--cards 3000`, and it also caps the core deck.

`--max-uses 3` on `fr` or `yue` fails. The value is a request field, and a
change to it needs a code edit, as the specs already describe the measured
presets.

## Later changes

`yue` lost `--model-id`, `--freq`, and `--freq-report`. `es` lost
`--similarity-threshold`. The value of each one stays in the language module, so
no flag can drift from the measured default.

`yue` lost `--select` with the coverage selection. The frequency list and the
tokenizer stay, and `specs/cantonese-frequency-list.md` names them. `--select`
now belongs to `fr` and `es`.

`es` gained `--select` with the cull of `SPANISH_CULL_PROFILE`. Its rank mode no
longer reads `--cards`, so the flag is a cap in every command.

`--model-id` is deleted from the four commands, `extract_sentences`, and
`run_ranked_deck`. No deck holds cards in more than one note type, and the field
inference picks the sentence field per model.

The four commands are declared from `LANGUAGES`. A `FixCommand` value holds the
default path, the mode views, the word count, and the help text of each
language, and two factories state the option set. The signature keeps each real
default, so the help page still shows the value.

## Rejected

Give each command a shared options dataclass. Cyclopts flattens a dataclass
into the help page, but the per-language defaults of `words`, `examples`,
`tail`, and `apkg-path` cannot live in one object. The help would then show
"language default" instead of the value, and the README would lose the list of
expected files.
