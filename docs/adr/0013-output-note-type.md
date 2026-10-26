# Give every written deck a note type of its own

The core deck and the cull deck rewrote the note type of the input sentence deck.
`coreflow._write_core_model` added `Core` and `CoreGloss` to the input model and
left every other field in place, and the cull added no field at all. An audit of
the built decks in `apkg/` found four faults:

- The output name was the input name. `apkg/mandarin_core.apkg` holds the note
  type `0 2024 Sentences Deck`, and `apkg/LTL Cantonese Deck Level 1 -
  Short_core.apkg` holds `LTL Note Type - Cantonese`.
- The output held the input fields. The Mandarin deck carries 21 fields and the
  card reads 3 of them.
- The output held an unused note type. `apkg/french_core_300.apkg` and
  `apkg/mandarin_core.apkg` also hold `Basic` and `TEXT CARD`.
- The cull deck held no core word. `templates/french_cull/back.html` has no
  `Core` block, so 27 of the 300 cards of `apkg/french_core_300.apkg` showed no
  word, and the cull showed none on any card.

ADR 0001 says a builder that rewrites an existing deck keeps the input note type.
That row is the one this ADR reverses.

Status: accepted. Supersedes the core-deck row of ADR 0001. The rest of ADR 0001
stands.

## Decision

The core deck and the cull deck create their note type. They clone the input
model, they move the kept notes to the clone, they delete every other note type,
and they write our fields, our templates, and our CSS.

- The note type name is the deck title, the rule of ADR 0001 for a builder that
  creates the note type: `French Core Sentences`, `Spanish Core Sentences`,
  `Mandarin Core Sentences`, `Cantonese Core Sentences`, and `French 1000 Core
  Sentences` and `Spanish 1000 Core Sentences` for the two culls.
- The field set is `Audio`, `Sentence`, `English`, `Core`, and `CoreGloss`, plus
  `Romanization` for Mandarin and Cantonese. The names are shared constants, not
  profile values, so a gate cannot lose the field that it checks.
- The profile splits its two identities: `input_model_name` and the three
  `input_*_field` names locate the source notes; `output_model_name` and
  `romanization_source` shape the result.
- The build route is a clone in the input package. Every note id, note guid,
  card id, `due` value, and scheduling column stays, so a cull keeps the
  schedule of the cards that the user already studies. A package from scratch
  cannot keep a card id.
- `anki_tools/apkg/package.py` gains `clone_model`, `update_note`,
  `prune_extra_cards`, and `delete_model`, and `add_note_and_card` takes a
  `model_id`. `add_model_field` and `remove_model_template` lose their last
  caller and go.

Every written card shows the core word and its English gloss, in core mode and
in cull mode. A kept card with no core word is not written, and the manifest
counts it. The output holds no frequency field: the four inputs hold four
different rank fields, and the rank is a selection value that stays in the
coverage CSV.

## Consequences

A note type is identity in Anki. A new name is a new note type, so a user who
imports the new deck beside the old deck gets two note types and duplicate
cards. The user must delete the old note type in Tools > Manage Note Types, or
import into a fresh profile. The release note must say this.

The deck title does not change, so Anki updates the deck in place. A note keeps
its guid, so Anki matches the note and changes its note type on import. A cull
import does not delete the notes that the cull drops; that limit holds today.

The committed `apkg/*.apkg` files and their side files hold the old note type
until a rebuild. `manifest.source.model` now holds the output name and
`manifest.source.input_model` the input name, and the cull coverage CSV gains a
`gloss` column.

Gate K9 and gate G10 compare the output name with `profile.output_model_name`.
New gates K21 and G16 check the single note type, the field set, and one card per
note. Gate K19 and new gate G17 require a `Core` and a `CoreGloss` on every
note.

## Alternatives

Keep the input model and add fields, as the code does today. The output keeps 21
fields for 3 values, the input name, and an unused note type, and a user cannot
tell our deck from the source deck in the note-type list.

Add `remove_model_field` and `rename_model` and keep the input model id. Anki
keys a note type by name, so a rename gives the same duplicate problem, and a
field delete adds a method that the next deck would also need. A clone is one
method for the whole shape.

Build the package from scratch with `AnkiPackage.create`. The cull would reset
the schedule and the card ids of a deck that the user studies.

Name the note type after the input deck with a suffix, for example `French
Sentences Core`. The name then carries the source name of one input, and the
four inputs disagree. The deck title is the name that the user already reads.
