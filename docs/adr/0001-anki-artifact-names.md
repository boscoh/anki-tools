# Name the Anki deck, note type, and file from one scheme

A read-only review found that the three deck builders name their Anki artifacts
inconsistent ways. Deck titles, note-type names, and file names disagree inside
one package. We record one scheme here so the implementer has one source of
truth. Anki matches a deck by title and a note type by name, so the name is
identity, not a label. The rule: a builder that creates the note type sets the
model name equal to the deck title. A builder that rewrites an existing deck
keeps the input note type.

Status: accepted

| Builder | Deck title | Note type (model) name | Output file |
| --- | --- | --- | --- |
| Vocabulary (`anki swadesh build <lang>`) | `Swadesh <Language> Vocabulary` | same as the deck title | `apkg/swadesh_<lang>.apkg` |
| Sentence pack (`anki swadesh pack <lang>`) | `Swadesh <Language> Sentences` | same as the deck title | `apkg/swadesh_<lang>_sentences.apkg` |
| Core deck (`anki fix <lang> --core`) | `<Language> Core Sentences` | the note type of the input deck, unchanged | `<input stem>_core.apkg` |

`<Language>` is the title case of the canonical name in
`anki_tools/languages/registry.py`, for example Mandarin or Cantonese.

The core deck does not get the `Swadesh` prefix. The core flow rewrites an
existing sentence deck and must keep its notes. A `Swadesh` prefix would claim a
provenance that the deck does not have. The core gate table and the field
mapping depend on the input model.

The vocabulary deck loses the entry count. Today it is `Swadesh Mandarin 207`,
so every rebuild changes the deck identity, and the name goes stale when the
word table grows. The count belongs in the CLI output and in the manifest, not
in the title.

## Consequences

A changed deck title or note-type name makes a new deck or note type. Anki
keeps the old scheduling and duplicates the cards. The five sentence packs and
the vocabulary decks need a manual rename in Anki, or a rebuild plus a fresh
import. The user must do the rename in Browse and in Tools > Manage Note Types.
The release note must warn the user.

`anki_tools/languages/french/cull.py` reads the French pack by the model name
`SWADESH_MODEL_NAME`. It must change with the pack model.

The committed `apkg/*.apkg` files keep the old names until a rebuild.
`anki swadesh audit <lang>` fails on an old package after the rename, and that
is expected.

## Not decided here

`anki-nqn` applies the deck titles, the note types, and the manifest keys. The
file and directory level names are deferred to `anki-5r2`: side-file stems,
mode suffixes, `pack --output`, audio and template directories, and language
codes.
