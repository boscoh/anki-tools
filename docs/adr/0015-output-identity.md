# Name every artifact from the output file stem

The deck title, the note type name, the output file stem, and the side-file
stems came from five places. Each language profile held `output_model_name`,
`deck_name`, or `default_deck_name`, and each mode held its own default output
name. The title and the note type drifted, and three recent faults came from
that drift: the cull file naming, the core title, and the rank title.

ADR 0001 named the artifacts, and ADR 0013 put the note type name equal to the
deck title, but no code owned the whole scheme. This ADR makes one value the
source of truth and gives it one module.

Status: accepted. Supersedes the naming rows of ADR 0001 and ADR 0013. Their
other decisions stand, in particular that a builder which creates the note type
sets its name to the deck title, and that a builder which rewrites an existing
deck keeps the input note type.

## Decision

The output file stem of a run is the source of truth.
`anki_tools/ranking/identity.py` is the only place that builds these names:

| Artifact | Value |
| --- | --- |
| Deck title | output file stem |
| Note type name | output file stem |
| `manifest.source.deck` and `manifest.source.model` | output file stem |
| `.core.csv`, `.cull.csv`, `.coverage.csv`, `.manifest.json`, `.rank.csv` stems | output file stem |

- `DeckIdentity(output_path)` gives the stem and the derived names. It also
  gives the side paths.
- `identity.default_output(language, mode)` gives the default path of a mode:
  `apkg/<language>_core.apkg`, `apkg/<language>_1000.apkg`, and
  `apkg/<language>_reordered.apkg`. The mode suffix table is the one default
  stem per language and mode.
- `--output <path>` sets the stem for every name of the run.
- The core flow, the cull flow, and the rank flow cross the seam. The rank flow
  keeps the input note type and renames the deck to the stem.
- The input identity stays in the profile: `input_model_name`,
  `input_sentence_field`, `input_english_field`, and `input_audio_field`. The
  profile holds no output name, and `CoreRequest` and `CullRequest` hold no
  `name`.
- Gate K9 compares the deck and the note type with `manifest.params.name` and
  `manifest.source.model`. Gate G10 is the same check. Gate K21 and gate G16
  read the model name from the manifest.

## Consequences

A rename is one edit in one module. The title, the note type, the file, and the
side files cannot disagree, because no second writer exists.

The seam is real: three flows and four commands already name the same
artifacts. The interface is a dataclass and two functions, and the implementation
behind it is nine name builders, so the module is deep.

A changed default output path changes the committed decks, the deck titles, the
note type names, and the side-file stems together. The migration steps are in
section 14.11 of `specs/core-deck-all-languages.md`.

## Alternatives

Keep one name field per profile and add a gate that compares the names. The
gate then reports a fault that the writer made, and the fault returns with the
next profile. The identity module removes the fault class.

Derive the output path from the input path, as the old `_core` and `_culled`
suffixes did. The name of the input deck then leaks into the output title, and
the default output of a language does not exist until a run starts.

Name the note type with a language prefix and the deck with a display title.
Anki matches a deck by title and a note type by name, so the two values are
identity. Two display forms give two identities.
