# Test refactor plan

| Field   | Value                                                          |
| ------- | -------------------------------------------------------------- |
| Status  | Cull tests done; batches B1 to B3 running; B4 to B10 open      |
| Date    | 2026-11-01                                                     |
| Owner   | anki-tools                                                     |
| Related | `.agents/skills/refactor-tests/SKILL.md`                       |

## 1. Goal

Keep a small test suite that is stable, useful, readable, and easy to
maintain. The method comes from Matteo Vaccari, "Tests as data over tests as
code". The skill `refactor-tests` holds the procedure. This file holds the
findings and the plan for this repo. `docs/test-review.md` records the earlier
cuts.

## 2. Done

The cull tests were the first target (commit `89682ed`, paired with `c1de349`).

| Item             | Before | After |
| ---------------- | ------ | ----- |
| Test files       | 4      | 5     |
| Test functions   | 45     | 38    |
| Lines            | 1501   | 1145  |
| Internal mocks   | 1      | 0     |

- Shared builders now live in `tests/cull_fixture.py`.
- 32 deliberate breaks in production code: the suite caught all of them except
  one equivalent change.

## 3. Scope of the rest

The audit covered the 37 other test files. It changed no file.

| Item               | Now   | Plan removes    |
| ------------------ | ----- | --------------- |
| Lines              | 6 200 | 1 450 (23%)     |
| Test functions     | 331   | 115 (35%)       |
| Internal mocks     | 15    | 10 now, 6 later |

Fixture modules add about 120 lines, so the net saving is about 1 300 lines.

## 4. Largest gains

| Rank | File                                  | Lines | Tests | Main cause                                     |
| ---- | ------------------------------------- | ----- | ----- | ---------------------------------------------- |
| 1    | `test_core_deck.py`                   | -230  | -16   | Repeats other files; five one-off table tests  |
| 2    | `test_shared_coreflow.py`             | -170  | -16   | Nine card-filter tests with one body           |
| 3    | `test_mandarin_audit.py`              | -130  | -9    | 130 lines of setup; patches an argument it has |
| 4    | `test_assemble.py`                    | -115  | -1    | One test; its asserts move to `test_core_deck` |
| 5    | `test_card_templates.py`              | -110  | -13   | Ten tests only grep a CSS or HTML string       |
| 6    | `test_french_pack.py`                 | -70   | -3    | Duplicate shape-rule checks                    |
| 7    | `test_mandarin_deck.py`               | -65   | -1    | Duplicate table data; one internal patch       |
| 8    | `test_mandarin_sentence_selection.py` | -60   | -6    | Shape rules copied from other files            |
| 9    | `test_tokens_cache.py`                | -50   | -2    | Patches `tokenize`, not the edge `load_model`  |
| 10   | `test_swadesh_build.py`               | -45   | -4    | Patches four internal functions                |

The other 27 files each save 30 lines or fewer.

## 5. Problems found

Numbers refer to the problems in the skill (P1 to P10).

| Problem                   | Where it shows                                                                           |
| ------------------------- | ---------------------------------------------------------------------------------------- |
| P1 overlap                | The tier-F false-friend break fails both `test_core_deck` and `test_english_sense`       |
| P2 boilerplate            | `build_env` in `test_mandarin_audit` has about 130 lines for a 3-line assert             |
| P3 internal mocks         | `test_cli_fix` (6), `test_swadesh_build` (4), `test_tokens_cache` (2)                    |
| P4 pinned constants       | `cards == 19`, `sections == 5`, profile paths copied into tests, an exact call order     |
| P6 repeated cases         | `TestCardFilter` (9 tests), `TestBuildTargets` (5), three small tables as separate tests |
| P7 single-use helpers     | `_cards`, `_tokens` in `test_core_deck`; `_varint` in `test_output_shape`                |
| P8 data in code           | Mandarin `TABLE` and row data exist in two files                                         |
| P9 tests that cannot fail | `test_core_similarity`, the `kept_order_hash` assertion                                  |
| P10 slow or fragile       | `test_write_preview_writes_every_section`: 3 s and a data-file dependency                |

## 6. Shared fixtures to add

| Duplicate                                  | Copies | Move to                                  |
| ------------------------------------------ | ------ | ---------------------------------------- |
| `card`, `target`, `tiny_deck`              | 3      | `cull_fixture.py` (add `false_friends`)  |
| Core profile, `run_core`, `write_package`  | 2      | `cull_fixture.py`                        |
| MP3 stub bytes                             | 7      | `deck_fixture.MP3_STUB`                  |
| `write_mp3s`                               | 2      | `deck_fixture.py`                        |
| `build_anki21b_deck`                       | 3      | `deck_fixture.py`                        |
| Contract-deck fixture                      | 2      | `conftest.py`                            |
| spaCy fake                                 | 2      | new `token_fixture.py`                   |
| Mandarin table, rows, index writer         | 2      | new `mandarin_fixture.py`                |
| `_write_tsv`, Tatoeba pool builder         | 2      | new `tatoeba_fixture.py`                 |
| AST import reader                          | 2      | new `ast_fixture.py`                     |
| Shape-rule table (language, text, verdict) | 4      | one data table                           |

`build_ranking_deck` and `RANKING_NOTES` in `deck_fixture.py` are dead. Delete
them.

## 7. Batches

One worker run per batch. Run B1 first.

| Batch | Scope                                                                         | Risk           | Status  |
| ----- | ----------------------------------------------------------------------------- | -------------- | ------- |
| B1    | Fixtures only: `deck_fixture`, `conftest`; update four apkg test files        | low            | running |
| B2    | Pure-function files: `card_templates`, `char_links`, and five more            | low            | running |
| B3    | Small tables: `extract_sentences`, `duplicates`, and five more                | low            | running |
| B4    | `test_core_deck`; fold in `test_assemble`; fix `test_core_similarity`         | medium         | open    |
| B5    | Rule tables: `shared_coreflow`, `gate_id_map`, and two more                   | medium         | open    |
| B6    | Mandarin audit and deck, through `run_pack_audit(load_table=, probe=)`        | medium         | open    |
| B7    | Remove mocks: `tokens_cache`, `swadesh_build`, and two more                   | medium         | open    |
| B8    | `mandarin_sentence_selection`, `french_pack`, `tatoeba_fixture`               | medium         | open    |
| B9    | `test_cli_fix`                                                                | needs decision | done    |
| B10   | New tests for the gaps in section 8                                           | medium         | open    |

Check the per-file counts again before B4 and B5. Other people edit those
files.

### Decision for B9

Done. The owner accepted option A: `run_fix` takes `run_core`, `run_cull`,
`check_core`, and `check_cull` as keyword arguments, with the real functions as
defaults. `test_cli_fix.py` has no mocks and tables the routes for all four
languages. ADR 0049 records the seam.

## 8. Gaps

A surviving break is a behavior that no test checks. Twelve are real. Two are
equivalent changes and need no test.

| Area                  | No test for                                                                              |
| --------------------- | ---------------------------------------------------------------------------------------- |
| Mandarin audit gates  | G3 target, G6 `>=` boundary, G7 duration, G8 media, G13 fields, G16 file                |
| Core selection        | Duplicate-lemma guard, `outside_budget`, `lambda_repeat`, ubiquitous-share boundary      |
| Manifest              | The `kept_order_hash` value                                                              |
| Rank pipeline         | `extract_sentences`, similarity penalties, `structural_score`, `write_ranking_csv`       |
| Pack build            | Generic `build_pack`; French, Greek, Spanish, Cantonese flows                            |
| Core flow             | Mandarin and Cantonese `run_core`                                                        |
| CLI                   | `zh`, `yue`, `es` routes; the "no --select mode" error; `--words`; `--output` for core   |
| Audio                 | Validation failure, retry exhausted, unreadable ffprobe output                           |
| Swadesh vocab         | `--prune`; the missing-audio path                                                        |
| APKG                  | `delete_model` success path                                                              |

B10 adds one black-box test for each row. Start with the audit gates and the
core selection, because a mutation test already proves those gaps.

## 9. Rules for every batch

- Run `uv run pytest -q` before and after. The suite stays green.
- Make at least three deliberate breaks in production code. Restore each one.
  Add a test row for each break that survives.
- Stage only the files the batch changes. Others edit the tree in parallel.
- Keep temp files under a few MB. A fixture bug once wrote an 82 GB file.

## 10. Out of scope

`rank.py::jaccard_similarity` and `duplicates.jaccard` are production
duplicates. File a bead if you want to merge them.
