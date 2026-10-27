# Test suite review

One review of `tests/`. It gives what was cut, why, and what is now uncovered.

## Numbers

| Item              | Before | Cut   | Now   |
| ----------------- | ------ | ----- | ----- |
| Test files        | 44     | 30    | 33    |
| Tests collected   | 553    | 379   | 406   |
| Lines in `tests/` | 9 296  | 6 432 | 7 073 |
| Run time          | 14 s   | 6 s   | 6 s   |

Two cuts happened. The first removed repetition and kept every claim. The
second removed the language verticals. The second cut loses coverage, and it
does not hide that.

## Cut 1: repetition

| Where                 | Change                                                                                                                                | Lines |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| `test_cli_fix.py`     | One route table per route, then one language per route.                                                                               | -380  |
| `test_french_cull.py` | Three tests built a full deck to test the shared writer. The output note type needs four such tests, so they stay. | -55   |
| four pack tests       | `== 207`, `== 212`, and `NOUN_ROWS = 84` were snapshots, not behavior. The numbers are gone. | -12   |
| `test_french_pack.py` | A SHA256 of `vocab/mandarin_sentences.csv` sat in the French file.                                                                    | -8    |

## Cut 2: the language verticals

| Deleted                       | Tests | What it held                                         |
| ----------------------------- | ----- | ---------------------------------------------------- |
| `test_spanish_pack.py`        | 17    | The Spanish word table and matcher                   |
| `test_spanish_cull.py`        | 8     | The Spanish frequency blend and clitic lemma         |
| `test_spanish_frequency.py`   | 9     | The Spanish frequency cache                          |
| `test_cantonese_pack.py`      | 19    | Jyutping, Han spans, the Cantonese matcher           |
| `test_cantonese_core.py`      | 8     | The Cantonese core adapter                           |
| `test_yue_deck_gates.py`      | 8     | The committed LTL deck: 647 cards, 223 top-500 words |
| `test_greek_pack.py`          | 19    | Final sigma, ano teleia, the shared foot row         |
| `test_greek_frames.py`        | 6     | The Greek frame data                                 |
| `test_mandarin_core.py`       | 11    | The Mandarin core adapter                            |
| `test_mandarin_frequency.py`  | 8     | The Mandarin frequency cache                         |
| `test_mandarin_fetch.py`      | 4     | The Tatoeba download                                 |
| `test_mandarin_audio.py`      | 15    | The shared audio module, through Mandarin data       |
| `test_mandarin_word_table.py` | 11    | The Mandarin word table loader                       |


Three of these files come back trimmed for the output note type. The next
section gives them.
The cut gave 379 tests, not 350. The restored critical tests and the note-type
tests give 399. The last 29 tests of the cut are the audit
gates `G2` to `G16` in `test_mandarin_audit.py`, the one end-to-end test of
`sentence_pack/pack.py` in `test_mandarin_deck.py`, and two small shared units.
They are not per-language copies, so cut 2 stops before them.

## Critical tests for the output note type

The output note type needs three of the deleted adapters. Their files are
back, trimmed to the new claims:

| File | Tests | Claim |
| --- | --- | --- |
| `test_cantonese_core.py` | 2 | The profile identities, and `Romanization` at position 4 |
| `test_mandarin_core.py` | 3 | The profile identities, the `{{Audio}}` token, and `Romanization` from `Sentence (Latin)` |
| `test_spanish_cull.py` | 1 | The five reserve field values in output order |

Four files that the cut keeps also hold tests for the note type:
`test_apkg_package.py` (the four writer methods), `test_core_deck.py` (the
output model and the core word), `test_cull_shared.py` (the cull note type and
the core word), and `test_french_cull.py` (the writer and the templates).

## Coverage that is now gone

These modules stay in the package and in the CLI. Only one narrow path of the
first three is read by a test:

| Module                                                      | Lines |
| ----------------------------------------------------------- | ----- |
| `anki_tools/languages/spanish/`                             | 2 323 |
| `anki_tools/languages/cantonese/`                           | 2 398 |
| `anki_tools/languages/greek/`                               | 1 289 |
| `anki_tools/languages/mandarin/{core,frequency,profile}.py` | 1 710 |

The output note type reads `spanish/cull.py` (the reserve fields),
`cantonese/core.py` (the profile and the extra field), and `mandarin/core.py`
(the profile and the templates). The rest of each module stays uncovered.

These commands still run the untested code:

| Command                                    | Languages                |
| ------------------------------------------ | ------------------------ |
| `anki fix`                                 | `es`, `yue`, `zh --core` (output note type only) |
| `anki swadesh`, `anki audit`, `anki vocab` | `es`, `el`, `yue`        |

A regression in one of them now reaches a user with no test to catch it.

`sentence_pack/pack.py` keeps its one end-to-end test in
`test_mandarin_deck.py`.

`sentence_pack/audio.py` lost the tests of its fetch, validate, and probe
path. These functions have no test now:

| Function                  | What it does                                                    |
| ------------------------- | --------------------------------------------------------------- |
| `validate_audio`          | Header and duration check                                       |
| `probe_audio`             | ffprobe duration. `test_french_cull.py` replaces it with a fake |
| `fetch_tts`               | gTTS download with retry                                        |
| `write_mp3s`, `tts_bytes` | Write and encode                                                |

The naming and source choice of the module keep coverage in
`test_french_pack.py`: `audio_hash`, `sentence_audio_name`,
`word_audio_name`, and `choose_source`.

## What to keep

| File                                                                                                | Why                                         |
| --------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| `test_apkg_package.py`                                                                              | The APKG writer is the artifact             |
| `test_import_graph.py`, `test_language_layout.py`                                                   | Cheap guards that catch a real break        |
| `test_core_deck.py`, `test_cull_shared.py`, `test_shared_coreflow.py`                               | One test per rule of the shared flow        |
| `test_mandarin_audit.py`                                                                            | The gates G2 to G16                         |
| `test_sentence_pack_guards.py`, `test_duplicates.py`, `test_char_links.py`, `test_ranking_cover.py` | One guard, one test                         |
| `test_french_pack.py`                                                                               | The reference language of the sentence pack |

## The lesson

The number of test _files_ and the number of test _lines_ both mislead. A file
of 34 tests held one route table four times. A file of 582 lines held 5 real
unit tests and 6 deck builds of shared code. But four pack files of 1 200 lines
held almost no repetition: each line tested a rule of one language.

Read the helpers and the fixtures first. The cut is there, not in the test
bodies.
