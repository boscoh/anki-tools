# Title case the deck title

Anki shows the deck title and the note type name in its deck list, its card
browser, and its note type list. The fix decks held the output file stem, so the
user read `spanish_core` and `mandarin_core`. The vocabulary decks already held
`Swadesh Spanish Vocabulary`. One collection showed two styles.

ADR 0015 made the output file stem the source of truth, because five writers
had drifted apart. That decision stands. Only the written form of the title
changes. The stem still names the file and the side files.

Status: accepted. Supersedes the title row of ADR 0015. Its other decisions
stand: one stem is the source of truth, the note type name equals the deck
title, and one module builds every name.

## Decision

`DeckIdentity.title` is the title case of the output file stem. It splits the
stem on `_`, capitalizes each part, and joins the parts with one space.
`DeckIdentity.model_name` equals the title, because ADR 0013 makes the note type
name equal the deck title.

| Artifact | Value |
| --- | --- |
| Deck title | `French Core`, `Spanish 1000` |
| Note type name | the deck title |
| `manifest.source.deck`, `manifest.source.model`, `manifest.params.name` | the deck title |
| Output file name and side-file stems | the output file stem, `french_core` |

- Gate K9 and gate G10 compare the deck and the note type with
  `manifest.params.name` and `manifest.source.model`. All three values come from
  the identity module, so the title case cannot break the comparison.
- Gate K21 and gate G16 read the model name from the manifest.

## Consequences

A new deck shows a readable name in Anki, and the fix decks match the vocabulary
decks.

Anki matches a deck by title and a note type by name at import. A collection
that holds an older snake-case deck keeps it, and the import adds the title-case
deck beside it. The migration is one manual step per deck: rename the deck in
Anki before the import, or delete the old deck and its note type. Section 14.11
of `specs/core-deck-all-languages.md` records the step.

A title that is not valid in a file name, for example a space, stays in the deck
and out of the file.

## Alternatives

Keep the snake-case title. Anki then shows the file stem to the user, and the
fix decks differ in style from the vocabulary decks.

Add a display title beside the stem, and keep the stem as the note type name.
Anki matches a deck by title and a note type by name, so the two values are one
identity. Two forms give two identities, and the drift of ADR 0015 returns.

Title case the file stem too. A file name with a space is awkward in a shell,
and the side files inherit the space.
