# Name the Swadesh artifacts from one output stem

The five Swadesh sentence pack profiles hold four output names each:
`deck_name`, `model_name`, `output_apkg`, and `manifest_path`
(`sentence_pack/profile.py:124-127`). The five constructors repeat the four
values (`languages/french/profile.py:1086-1089` and four more). The vocabulary
route states three names in a dict (`swadesh_vocab.py:_deck_config`).

ADR 0015 decided that the output file stem is the source of truth and that "the
profile holds no output name". The fix routes follow that rule. The two Swadesh
routes do not. This is unfinished adoption, not a new decision.

The drift has the cost ADR 0015 removed for the fix routes: a rename is five
edits per language, gate K9 and gate G10 exist only to catch the mismatch, and
`audit.py:527` reads `profile.model_name` as a second source for the note type
name.

Status: accepted. Extends ADR 0015 to the two Swadesh routes. Supersedes the row
of ADR 0001 that named the Swadesh file suffixes.

## Decision

`anki_tools/ranking/identity.py` gains two rows in the one default table:

| Mode | Default output |
| --- | --- |
| `pack` | `apkg/swadesh_sentences_<language>.apkg` |
| `vocab` | `apkg/swadesh_vocab_<language>.apkg` |

The rules of ADR 0015 and ADR 0019 then apply without a second table:

- The deck title is the title case of the stem: `Swadesh Sentences French` and
  `Swadesh Vocab French`.
- The note type name equals the deck title.
- The manifest is `<stem>.manifest.json`.

The four members leave `LanguageProfile`, and the five constructors lose their
values. Each route states one output path, builds one `DeckIdentity`, and gives
its `title`, `model_name`, and side paths to the deck writer, the manifest, and
the audit.

The writer still states no name. This change moves the naming into the callers
that already hold the path, so the rejected row of ADR 0024 still stands.

`gate_identity` shrinks to the deck-title check. `gate_output_model` already
compares the note type set with `manifest.source.model`, so the gate keeps only
the check that no other gate makes.

## Consequences

All 13 committed packages change their names and their deck titles. The eight
vocabulary decks change from `Swadesh Mandarin Vocabulary` to
`Swadesh Vocab Mandarin`; the five sentence packs change the word order from
`Swadesh French Sentences` to `Swadesh Sentences French`.

A user who imports a new package gets a second deck, because Anki matches a deck
by title. The old deck keeps its cards. The migration is a rename or a delete of
the old deck in Anki.

A rename of one default now moves the file, the title, the note type, the
manifest, and the side files together, in one edit of one table.

The pack audit, `mandarin/stages.py`, and the vocabulary module stop holding
names. Three readers of a name become one.

## Rejected

Put the language first, `apkg/french_swadesh_sentences.apkg`. The titles read
better and match `french_core.apkg`. The file names that the caller chose win.

Give the title its own word order, so `swadesh_sentences_french` renders as
`Swadesh French Sentences`. That is the drift ADR 0015 removed: two rules for
one name.

Add an `--output` flag to the Swadesh commands. Nobody needs a second name, and
ADR 0025 removes the flag from `anki fix` for the same reason.
