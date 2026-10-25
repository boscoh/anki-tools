# Name each deck once and give it one door

The `anki swadesh` group held eight commands. Two defects came from the names.
`swadesh core` and `fix <lang> --core` called one function, so the core deck had
two doors. `build` wrote the vocabulary deck, but the verb names no deck.
`list-languages` printed the vocabulary languages and omitted French, which
every sentence command accepts.

Status: accepted

## Decision

One command for one deck, and the deck name is the command name.

| Deck | Command |
| --- | --- |
| Vocabulary deck | `anki swadesh vocab <language>` |
| Sentence pack | `anki swadesh pack <language>` |
| Core deck | `anki fix <language> --core` |

- `build` is renamed to `vocab`. The registry field `build` and the helper
  `swadesh_languages()` become `vocab` and `vocab_languages()`.
- `swadesh core` is deleted. The command `fix <lang> --core` keeps the default
  input path of `fr`, `yue`, and `es`, so the user types no file name.
- The registry loses the `core` capability. Every language with a core module
  already has a `fix <lang>` command, so the field answered no question that
  the command set does not answer.
- `list-languages` is deleted. The error message of each command lists the
  languages that the command accepts.

## Consequences

ADR 0003 is superseded. Its `core` capability and its `swadesh core` command
are gone. The four `fix <lang> --core` paths stay.

ADR 0005 names `anki swadesh core`. That command is now
`anki fix <lang> --core`. The decision of ADR 0005 does not change.

The `vocab` rename breaks a typed command and gets no alias. README.md names
the command, so the change reaches the user.

## Not decided here

The stage commands `fetch`, `sentences`, and `audio` stay. A maintainer stops
after S3 to read the review CSV, and a separate command marks that stop.
Folding them into `pack` is a separate decision (anki-g3x1).

The sentence-pack tier value `core` keeps its name. A `core` card covers the
target within the window. The core deck is a different concept. A rename to
`direct` must rebuild the five packs and their manifests (anki-4x1s).

## Rejected

Delete `fix <lang> --core` and keep `swadesh core`. It loses the default input
path of three decks, and the `fix` group already holds the other two transforms
(`--select` and rank). That is more loss than the second name costs.

Keep both doors and rename the old one. Two names for one deck stay in the
help text, which is the defect.
