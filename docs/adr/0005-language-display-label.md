# Put the language display label in the registry

The deck commands printed the language name as a literal. `anki fix zh --core`
printed "Processing Chinese Anki Deck", the four rank banners printed
"Chinese", "French", "Cantonese", and "Spanish", and `anki swadesh core`
printed `name.title()`. The same language had two display names, and a new
literal could drift from the registry.

Status: accepted

## Decision

`CliLanguage` holds a `label: str` field: the display name of the deck
commands. The field has no default, so a new registry entry must state it. The
four `fix <lang>` commands read `LANGUAGES["<name>"].label` for the core banner
and the rank banner, and `anki swadesh core` reads `entry.label` of the
resolved entry.

## Consequences

One place holds the display name, so both deck commands print the same name.
`anki fix zh` and `anki fix zh --core` now print "Mandarin" where they printed
"Chinese". A language whose display name differs from its registry name records
that difference once.

## Rejected

A `name.title()` helper. It cannot express "Chinese" for mandarin, and it hides
the intent of the label behind a string transform.
