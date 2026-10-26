# anki_tools Package Layout

The package groups the modules by domain. Each file has one job. A language
lives in one directory only.

## 1. Top level

| Path                         | Content                                          |
| ---------------------------- | ------------------------------------------------ |
| `anki_tools/__init__.py`     | Public exports of the package                    |
| `anki_tools/cli.py`          | Cyclopts command tree and the `anki` entry point |
| `anki_tools/cache.py`        | Download a file into the cache                   |
| `anki_tools/preview.py`      | Build the browser preview of the card templates  |
| `anki_tools/preview_apkg.py` | Build the browser preview of a package's cards   |

`cli.py` also holds the deck field-name aliases that each `fix` command
passes to `extract_sentences`. A field name of one language stays in the
command, not in the shared module.
`__init__.py` loads a language helper on first use. Importing the package loads
no language package.

## 2. apkg

| Path                           | Content                                         |
| ------------------------------ | ----------------------------------------------- |
| `anki_tools/apkg/package.py`   | Read, write, and edit APKG files                |
| `anki_tools/apkg/templates.py` | Template and CSS loaders, and `apply_templates` |

## 3. ranking

Shared ranking code only. No language data.

| Path                              | Content                                               |
| --------------------------------- | ----------------------------------------------------- |
| `anki_tools/ranking/rank.py`      | Sentence data, frequency tables, shared score drivers |
| `anki_tools/ranking/latin.py`     | Shared Latin-script text helpers and word metrics     |
| `anki_tools/ranking/reorder.py`   | Reorder a deck from a ranking CSV                     |
| `anki_tools/ranking/cover.py`     | Generic greedy cover over items with changing gains   |
| `anki_tools/ranking/core.py`      | Core-word value, cost, passes, and order              |
| `anki_tools/ranking/pipeline.py`  | Shared extract, rank, write, and reorder stages       |
| `anki_tools/ranking/cullflow.py`  | Cull flow behind the `CullProfile` seam               |
| `anki_tools/ranking/cullgates.py` | Gates G1 to G15 of the cull deck                      |

`rank_word_based` and `rank_character_based` are the shared score drivers.
`jaccard_similarity`, `compute_similarity_penalties`, and the word metrics of
`latin.py` are the public shared helpers. A language package calls them.

`cover.greedy_cover` holds the selection loop. A language module gives the
gain, the heap key, the reject test, and the callbacks, so each language
keeps its own reason labels. `pipeline.run_ranked_deck` calls the ranker and
the optional reducer of a language, then writes the ranking CSV and reorders
the deck.

## 4. sentence_pack

The shared Swadesh sentence pipeline. The stage order and the guard order are
in `docs/swadesh-sentence-flow.md`.

| Path                                        | Content                                    |
| ------------------------------------------- | ------------------------------------------ |
| `anki_tools/sentence_pack/profile.py`       | Profile dataclass and the profile registry |
| `anki_tools/sentence_pack/tatoeba.py`       | S0 Tatoeba fetch cache                     |
| `anki_tools/sentence_pack/pool.py`          | S2 candidate pool and guard tables         |
| `anki_tools/sentence_pack/select.py`        | S3 greedy cover and duplicate guard        |
| `anki_tools/sentence_pack/audio.py`         | S4 audio cache                             |
| `anki_tools/sentence_pack/pack.py`          | S5 APKG build                              |
| `anki_tools/sentence_pack/audit.py`         | Gate runner and language-neutral gates     |
| `anki_tools/sentence_pack/guards.py`        | Shared sentence-shape and span helpers     |
| `anki_tools/sentence_pack/tokens.py`        | Shared token and romanization helpers      |
| `anki_tools/sentence_pack/english_sense.py` | English gloss sense check                  |

## 5. languages

One package per language. The package holds all the code of that language.

| Path                                                | Content                                            |
| --------------------------------------------------- | -------------------------------------------------- |
| `anki_tools/languages/registry.py`                  | `LANGUAGES`, `ALIASES`, and the CLI list           |
| `anki_tools/languages/cantonese/profile.py`         | Cantonese profile and word metrics                 |
| `anki_tools/languages/cantonese/frequency.py`       | Spoken and written frequency list and download     |
| `anki_tools/languages/cantonese/jyutping.py`        | Jyutping repair and Traditional convert            |
| `anki_tools/languages/cantonese/rank.py`            | Cantonese sentence ranking                         |
| `anki_tools/languages/mandarin/profile.py`          | Mandarin profile and word metrics                  |
| `anki_tools/languages/mandarin/pinyin.py`           | Pinyin repair for Chinese cards                    |
| `anki_tools/languages/mandarin/fix_steps.py`        | Pinyin and template steps after the reorder        |
| `anki_tools/languages/mandarin/characters.py`       | Han character helpers and similarity               |
| `anki_tools/languages/mandarin/rank.py`             | Mandarin sentence ranking                          |
| `anki_tools/languages/mandarin/stages.py`           | S1 to S3 caller of the shared pipeline             |
| `anki_tools/languages/french/profile.py`            | French profile and form matching                   |
| `anki_tools/languages/french/rank.py`               | French sentence ranking                            |
| `anki_tools/languages/french/cull.py`               | French cull profile, values, and bindings          |
| `anki_tools/languages/french/core.py`               | French core deck: profile, wrappers, and gates     |
| `anki_tools/languages/greek/profile.py`             | Greek profile and form matching                    |
| `anki_tools/languages/greek/rank.py`                | Greek sentence ranking                             |
| `anki_tools/languages/spanish/rank.py`              | Spanish sentence ranking                           |
| `anki_tools/languages/spanish/cull.py`              | Spanish cull profile, values, and bindings        |

`languages/mandarin/characters.py` holds the Han script helpers. Cantonese
imports them, because both languages write the Han script.

Each language package gives the profile names, so
`from anki_tools.languages import cantonese` gives the same names as a flat
language module.

The `spanish` package holds the ranking code and the cull profile. The
`anki swadesh vocab` command serves it from `vocab/english-<lang>.csv` and an
audio directory.

## 6. Import rules

- `languages/<lang>/` holds all the code of a language. No other directory
  holds language code.
- `sentence_pack` and `ranking` hold the shared stages and the shared scores.
  They import no language package.
- `apkg` holds the file model and the template resources. It imports no
  language package.
- `cache.py` holds the download helper that fills the XDG cache. It imports
  no language package.
- `cli.py` builds the command list from `languages.registry`. It is the only
  module that imports every package.
- One key for each language. `languages.registry.ALIASES` maps a short code
  (`yue`, `cmn`, `zh`, `el`) to the canonical name.

`tests/test_language_layout.py` enforces the rule for `ranking` and
`sentence_pack` in two ways. It fails when a shared module imports a language
package, and it fails when a shared module names a function, a class, or a
constant after a language. A script name such as `latin` is allowed.
