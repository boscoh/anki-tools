# Vocab Deck Flow

This document shows how `anki swadesh vocab <language>` turns one word list
and a set of audio files into a vocabulary deck. It shows the stage order, the
file naming rules, and the card layout. The shared code lives in
`anki_tools/cli.py` and `anki_tools/apkg/`.

Related: `docs/swadesh-sentence-flow.md`, `docs/core-deck-flow.md`, and
`docs/cull-deck-flow.md`.

## 1. Stage map

One command runs the stages in order: `anki swadesh vocab <language>`. The
`--prune` value is on by default. `--no-prune` keeps the unused audio files.
The value `all` runs every language of the registry that has a vocabulary
deck. The languages are mandarin, spanish, greek, german, cantonese, arabic,
and hindi.

| Stage | Name   | Input                          | Output                                   |
| ----- | ------ | ------------------------------ | ---------------------------------------- |
| V0    | read   | `vocab/english-<language>.csv` | word rows, audio file name per row       |
| V1    | audio  | V0, gTTS                       | `audio/audio_<language>/<form>.mp3`      |
| V2    | layout | V0, `templates/vocab/`         | fields, card templates, CSS              |
| V3    | prune  | V0, the audio directory        | the audio directory without unused files |
| V4    | pack   | V0, V1, V2                     | `apkg/swadesh_<language>.apkg`           |

The stage names are labels of this document. One builder, `_build_apkg`,
runs the stages inline.

The vocabulary deck has no gate table and no manifest. The `anki swadesh
audit` command serves the sentence packs only. The check is a test, not a gate:
`tests/test_swadesh_build.py` builds a deck and compares the card count with the
CSV row count and the media set with the audio names.

## 2. Flow diagram

```text
V0  vocab/english-<language>.csv
        |
        +--> one row per Swadesh concept
        |
        v
V1  audio files: gTTS makes the missing file of each row
        |
        v
V2  layout: field order, front and back template, CSS layers
        |
        v
V3  prune: delete the audio files that no row uses
        |
        v
V4  read every audio file
        |
        +--> a missing file stops the run with no APKG
        |
        v
    apkg/swadesh_<language>.apkg
```

## 3. Word list (V0)

`_parse_vocab_entries` reads `vocab/english-<language>.csv`. The path is
relative to the repository root.

The file has one row per concept. The first column is `english`. The target
column has the language name, for example `mandarin`. A `transliteration`
column is optional. A column with another name, for example `gender` in the
Greek file, is not a card field.

| Language  | Columns                             |
| --------- | ----------------------------------- |
| Mandarin  | english, mandarin, transliteration  |
| Cantonese | english, cantonese, transliteration |
| Arabic    | english, arabic, transliteration    |
| Hindi     | english, hindi, transliteration     |
| Greek     | english, greek, gender              |
| Spanish   | english, spanish                    |
| German    | english, german                     |

The builder gives each row an `audio_filename`:

```text
text  = the target field value
text  = text without each "(...)" group
text  = the part before the first "/"
text  = text with each space as "_"
name  = text + ".mp3"
```

Thus `أَنْتَ/أَنْتِ` gives `أَنْتَ.mp3`, and one file serves both forms of the
row.

## 4. Audio (V1)

`_generate_audio_files` reads the `tts` code of the language from
`anki_tools/languages/registry.py`: `zh-cn` for Mandarin, `es` for Spanish,
`el` for Greek, `de` for German, `yue` for Cantonese, `ar` for Arabic, and
`hi` for Hindi.

| Rule                | Value                           |
| ------------------- | ------------------------------- |
| Engine              | gTTS, the installed package     |
| Existing file       | skipped, not generated again    |
| Space between files | 0.5 seconds                     |
| Failed file         | reported, and the run continues |
| Network             | needed by gTTS                  |

The audio directory is `audio/audio_<language>/`. The builder creates it when
it is absent. A failed file gives a warning at V1. V4 then fails, because it
needs every file. Run the command again to retry the missing files.

## 5. Layout (V2)

### 5.1 Field order

`_content_field_order` puts the target field first. Then it adds
`transliteration` when the file holds it, and `english` when the file holds
it. Every other column stays out of the card. The builder appends the `audio`
field.

Thus the mandarin card holds `mandarin`, `transliteration`, `english`, and
`audio`. The Greek card holds `greek`, `english`, and `audio`.

### 5.2 Templates

| File                                  | Content                                |
| ------------------------------------- | -------------------------------------- |
| `templates/vocab/front.html`          | the audio of the card                  |
| `templates/vocab/back.html`           | the audio, a divider, then every field |
| `templates/vocab/<language>/card.css` | the CSS of one language, when present  |
| `templates/shared/card.css`           | the shared CSS layer                   |
| `templates/shared/char_links.html`    | the Han character link script          |

The CSS layers join in this order: shared, language. A later layer overrides an
earlier layer. A missing language layer adds no text. The shared layer centers
the card. No layer sets a direction or an alignment: a script that reads right
to left, for example Arabic, follows the Unicode bidirectional algorithm.
The back template replaces `{{fields}}` with one `<div>` per content field.
The field name gives the shared class:

| Field name                                                | Class                 |
| --------------------------------------------------------- | --------------------- |
| `characters`, `hanzi`, `cantonese`, `mandarin`, `chinese` | `target char-links`   |
| `transliteration`                                         | `romanization`        |
| `english`                                                 | `english`             |
| any other field                                           | `target`              |

The character link script makes each Han character a link to the CC-Canto
dictionary.

## 6. Prune (V3)

`_prune_outdated_audio` deletes each file of the audio directory that no row
of V0 uses. The run prints the count of the deleted files. `--no-prune` keeps
them.

The default is on, so a removed word removes its audio file on the next run.
Thus the audio directory holds the files of the current word list only.

## 7. Pack (V4)

`_pack_apkg` reads every audio file of the list. A missing file prints a
warning and adds the English word to the `missing` list. The run then stops
before it writes the APKG, and it prints the count of the missing files.

| Value        | Rule                                              |
| ------------ | ------------------------------------------------- |
| Deck title   | `Swadesh <Language> Vocabulary`                   |
| Note model   | the same name as the deck title                   |
| Output file  | `apkg/swadesh_<language>.apkg`                    |
| Sort field   | field 0, the target form                          |
| Writer       | a temporary file, then a rename to the output     |

The deck title and the note model name follow `docs/adr/0001-anki-artifact-names.md`.
They hold no word count, because a count in the title changes the deck
identity at each rebuild.

A run writes the APKG only. It writes no manifest, no attribution file, and
no side CSV.

## 8. Check

The vocabulary deck has no gate table. Use the inspect commands on the output
file:

```bash
uv run anki inspect cards apkg/swadesh_vocab_mandarin.apkg
uv run anki inspect decks apkg/swadesh_vocab_mandarin.apkg
uv run anki inspect models apkg/swadesh_vocab_mandarin.apkg --verbose
uv run anki inspect audio apkg/swadesh_vocab_mandarin.apkg
```

A card count that differs from the row count of the CSV is a defect. So is an
audio file that is absent from the media set of the APKG.

That comparison lives in `tests/test_swadesh_build.py`, so no gate table is needed.

## 9. File map

| Path                                        | Content                                     |
| ------------------------------------------- | ------------------------------------------- |
| `anki_tools/cli.py`                         | `vocab`, `_build_apkg`, and V0 to V4        |
| `anki_tools/languages/registry.py`          | the `vocab` and `tts` values of a language  |
| `anki_tools/apkg/package.py`                | `AnkiPackage.create`                        |
| `anki_tools/apkg/templates.py`              | template and CSS loaders                    |
| `vocab/english-<language>.csv`              | the word list of one language               |
| `templates/vocab/front.html`                | the front template                          |
| `templates/vocab/back.html`                 | the back template                           |
| `templates/shared/card.css`                 | the shared CSS of every deck                |
| `templates/shared/char_links.html`          | the Han character link script               |

## 10. Invariants

- The language values come from `_build_apkg`: the CSV path, the audio
  directory, the two names, the output file, and the target field name. Do not
  add a second copy of these values.
- The target field name equals the canonical language name in lower case. The
  audio file name comes from the target field value only.
- The deck title and the note model name are equal. A builder that creates the
  model sets both to the deck title.
- A card holds only the target, the transliteration, and the English. A review
  column stays in the CSV.
- A run writes no file when one row has no audio file. All rows or no APKG.
