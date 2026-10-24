# Anki Command Decks

This page gives the source deck and the target deck of each `anki` command.

See `docs/anki-input-sources.md` for the local input file and the upstream
source of each `anki fix` command.

## 1. Terms

- **Source deck**: the deck of the input APKG file that the command reads.
- **Target deck**: the deck of the output APKG file that the command writes.
- **Keep name**: the target deck has the name of the source deck.

A command that reads no APKG file has no source deck. A command that writes
no APKG file has no target deck.

The `--output` option sets the target file of a command that writes a deck.
The target name in the tables is the default name.

## 2. Commands that write a deck

| Command                         | Source deck                                    | Target file                          | Target deck name               |
| ------------------------------- | ---------------------------------------------- | ------------------------------------ | ------------------------------ |
| `anki fix zh <input>`           | the deck of `<input>`                          | `<input>_reordered.apkg`             | keep name                      |
| `anki fix fr <input>`           | the deck of `<input>`                          | `<input>_reordered.apkg`             | keep name                      |
| `anki fix fr <input> --select`  | the deck of `<input>`, plus the reserve pack   | `<input>_culled.apkg`                | `French 1000 Core Sentences`   |
| `anki fix fr <input> --core`    | the deck of `<input>`                          | `<input>_core.apkg`                  | `French Core Sentences`        |
| `anki fix yue [<input>]`        | the deck of `<input>`                          | `<input>_reordered.apkg`             | keep name                      |
| `anki fix es <input>`           | the deck of `<input>`                          | `<input>_reordered.apkg`             | keep name                      |
| `anki style apply <input>`      | the deck of `<input>`                          | `<input>_styled.apkg`                | keep name                      |
| `anki swadesh build <language>` | none, the data is `vocab/english-<lang>.csv`   | `apkg/swadesh_<lang>.apkg`           | `Swadesh <Language> <count>`   |
| `anki swadesh pack <language>`  | none, the data is `vocab/<lang>_sentences.csv` | `apkg/swadesh_<lang>_sentences.apkg` | `Swadesh <Language> Sentences` |

Notes:

- `anki fix fr` uses `apkg/6000_French_Sentences_w_Audio.apkg` when the
  argument is absent. The default applies to rank, `--select`, and `--core`.
- `anki fix yue` uses `apkg/LTL Cantonese Deck Level 1 - Short.apkg` when the
  argument is absent.
- `anki fix fr --select` sets the target name with `--name`. The default is
  `French 1000 Core Sentences`. The reserve pack is
  `apkg/swadesh_french_sentences.apkg`; `--no-reserve` removes it, and
  `--reserve-pack` selects another file.
- `anki fix fr --core` sets the target name with `--name`. The default is
  `French Core Sentences`.
- `anki swadesh build all` writes one target file for each build language:
  mandarin, spanish, greek, german, cantonese, arabic, and hindi.
- `anki swadesh pack` writes a sentence pack for each profile language:
  mandarin, french, spanish, greek, and cantonese.

## 3. Card count of the target deck

| Command                       | Cards in the target deck                                                           |
| ----------------------------- | ---------------------------------------------------------------------------------- |
| `anki fix zh`                 | the source cards, less a duplicate above the similarity threshold                  |
| `anki fix zh --max-sentences` | `--max-sentences` cards, rated and deduplicated                                    |
| `anki fix fr`                 | the source cards, less a duplicate above the similarity threshold                  |
| `anki fix fr --select`        | `--max-cards` cards, default 1000                                                  |
| `anki fix fr --core`          | `--core-max-cards` cards, plus `--tail-cards`                                      |
| `anki fix yue`                | the kept sentences of the selection                                                |
| `anki fix yue --no-select`    | the source cards, less a duplicate above the similarity threshold                  |
| `anki fix es`                 | the source cards, less a duplicate; `--max-sentences` also drops the unrated cells |
| `anki style apply`            | every card of the source deck                                                      |
| `anki swadesh build`          | one card for each row of the word CSV                                              |
| `anki swadesh pack`           | one card for each selected row of the sentence CSV                                 |

## 4. Deck names of the sentence packs

| Language  | Target file                             | Target deck name              |
| --------- | --------------------------------------- | ----------------------------- |
| mandarin  | `apkg/swadesh_mandarin_sentences.apkg`  | `Swadesh Mandarin Sentences`  |
| french    | `apkg/swadesh_french_sentences.apkg`    | `Swadesh French Sentences`    |
| spanish   | `apkg/swadesh_spanish_sentences.apkg`   | `Swadesh Spanish Sentences`   |
| greek     | `apkg/swadesh_greek_sentences.apkg`     | `Swadesh Greek Sentences`     |
| cantonese | `apkg/swadesh_cantonese_sentences.apkg` | `Swadesh Cantonese Sentences` |

## 5. Commands that write no deck

| Command                        | Reads                                                 | Writes                            |
| ------------------------------ | ----------------------------------------------------- | --------------------------------- |
| `anki freq yue`                | no APKG file, the Cifu and Apple Daily lists          | `data/cantonese_frequency.tsv`    |
| `anki inspect decks`           | the deck of an input APKG file                        | terminal text                     |
| `anki inspect models`          | the deck of an input APKG file                        | terminal text                     |
| `anki inspect fields`          | the deck of an input APKG file                        | terminal text                     |
| `anki inspect cards`           | the deck of an input APKG file                        | terminal text                     |
| `anki inspect templates`       | the deck of an input APKG file                        | terminal text                     |
| `anki inspect audio`           | the deck of an input APKG file                        | terminal text, optional audio dir |
| `anki inspect stats`           | the deck of an input APKG file                        | terminal text                     |
| `anki inspect sample`          | the deck of an input APKG file                        | terminal text                     |
| `anki preview`                 | the template and CSS files                            | `preview/cards.html`              |
| `anki fix fr --select --check` | the target deck and side files of an earlier cull     | gate table; exit code 1 on a fail |
| `anki fix fr --core --check`   | the target deck and side files of an earlier core run | gate table; exit code 1 on a fail |
| `anki swadesh fetch`           | no APKG file, it downloads the Tatoeba exports        | cache files                       |
| `anki swadesh sentences`       | no APKG file, the word CSV and the sentence cache     | `vocab/<lang>_sentences.csv`      |
| `anki swadesh audio`           | no APKG file, the sentence CSV                        | `audio/<lang>_sentences/`         |
| `anki swadesh audit`           | the target deck of an earlier pack, and its CSVs      | gate table; exit code 1 on a fail |

`anki swadesh audit <language>` opens `apkg/swadesh_<lang>_sentences.apkg` and
writes no deck. Run `anki swadesh pack <language>` first.

## 6. The write steps of a target deck

A fix command reads the source deck and saves the target deck. The source
file does not change.

`anki fix fr --select` and `anki fix fr --core` also write side files beside
the target deck: the card CSV, the coverage CSV, the manifest JSON, and for
core mode the ranking CSV.

`anki style apply` copies the source file to the target file and edits every
model of the copy. The deck name and the cards do not change.

A `swadesh` command has no source deck. It builds a new deck from CSV data and
audio files, and it writes the target file through a temporary file.
