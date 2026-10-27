# Require an explicit mode on the fix commands

The four `anki fix` commands ranked the deck when no mode flag was set. A bare
command started the longest run and rewrote the deck, and the user had no
request that made the work happen.

Status: superseded by ADR 0017

## Decision

Require exactly one mode flag: `--rank`, `--select`, or `--core`.

- `run_fix` exits non-zero when no mode is set, and when two modes are set.
- `--rank` selects the route that was the default. The output, the rank CSV, the
  pinyin fix, and the jyutping fix do not change.
- `--check` still needs `--select` or `--core`. It does not accept `--rank`.
- A bare `anki fix <lang>` prints the mode list of the command and exits
  non-zero.

## Consequences

- A bare command no longer ranks a large deck by accident. The run states what
  it builds.
- `--rank` is a mode flag of every command. The README examples add it.
- The rank flow, the rank CSV, and the rank tests do not change.
- The mode list of a command comes from `FixLanguage.modes`. The `--check` list
  comes from `FixLanguage.check_modes`, because `--check` does not apply to
  rank mode.

## Alternatives

Keep the implicit rank and add `--no-rank`. That keeps a default that the user
does not want, and a second flag restates it.

Delete rank mode. The core and cull flows read `reorder_deck` and the similarity
constants of the language rank modules, so that deletion moves shared code. It
is a larger change than this request needs.
