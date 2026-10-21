# anki_tools Package Layout

The package groups the modules by domain. Each file below has one job.

## 1. Top level

| Path                     | Content                                          |
| ------------------------ | ------------------------------------------------ |
| `anki_tools/__init__.py` | Public exports of the package                    |
| `anki_tools/cli.py`      | Cyclopts command tree and the `anki` entry point |

## 2. apkg

| Path                           | Content                                 |
| ------------------------------ | --------------------------------------- |
| `anki_tools/apkg/package.py`   | Read, write, and edit APKG files        |
| `anki_tools/apkg/templates.py` | Card template partials and the CSS path |

## 3. ranking

| Path                            | Content                                             |
| ------------------------------- | --------------------------------------------------- |
| `anki_tools/ranking/rank.py`    | Rank sentences by frequency, complexity, similarity |
| `anki_tools/ranking/reorder.py` | Reorder a deck from a ranking CSV                   |
| `anki_tools/ranking/cull_fr.py` | French 1000-sentence cull pipeline                  |

## 4. mandarin

These modules keep the Mandarin caller code for the shared stages.

| Path                             | Content                                         |
| -------------------------------- | ----------------------------------------------- |
| `anki_tools/mandarin/pinyin.py`  | Pinyin repair for Chinese cards                 |
| `anki_tools/mandarin/fetch.py`   | S0 Tatoeba fetch caller                         |
| `anki_tools/mandarin/swadesh.py` | S1 to S3 word table, pool, and selection caller |
| `anki_tools/mandarin/audio.py`   | S4 audio cache caller                           |
| `anki_tools/mandarin/deck.py`    | S5 APKG build caller                            |
| `anki_tools/mandarin/audit.py`   | Gates G1 to G18 for the Mandarin pack           |

## 5. cantonese

| Path                                | Content                                             |
| ----------------------------------- | --------------------------------------------------- |
| `anki_tools/cantonese/jyutping.py`  | Jyutping repair and Traditional conversion          |
| `anki_tools/cantonese/frequency.py` | Blended spoken and written frequency list           |
| `anki_tools/cantonese/select.py`    | Rules Y1 to Y8 and the Cantonese sentence selection |

## 6. sentence_pack

The shared Swadesh sentence pipeline. The stage order and the guard order are
in `docs/swadesh-sentence-flow.md`.

| Path                                      | Content                                |
| ----------------------------------------- | -------------------------------------- |
| `anki_tools/sentence_pack/pool.py`        | S2 candidate pool and guard tables     |
| `anki_tools/sentence_pack/select.py`      | S3 greedy cover and duplicate guard    |
| `anki_tools/sentence_pack/audio.py`       | S4 audio cache                         |
| `anki_tools/sentence_pack/pack.py`        | S5 APKG build                          |
| `anki_tools/sentence_pack/audit.py`       | Gate runner and language-neutral gates |
| `anki_tools/sentence_pack/guards.py`      | Shared span and occurrence helpers     |
| `anki_tools/sentence_pack/languages/*.py` | One profile per language               |

## 7. Import rules

- `sentence_pack` holds the shared pipeline. A language package holds the
  caller code and the language data.
- `apkg` holds the file model and the template resources. It does not import a
  language package.
- `ranking.rank` holds the shared ranking functions. It also holds the Chinese
  and Cantonese similarity helpers.
- `cli.py` is the only module that imports every package.
