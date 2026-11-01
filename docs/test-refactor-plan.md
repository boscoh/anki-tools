# Test refactor plan

| Field   | Value                                                          |
| ------- | -------------------------------------------------------------- |
| Status  | All batches and the cut by value done; one seam open           |
| Date    | 2026-11-01                                                     |
| Owner   | anki-tools                                                     |
| Related | `~/.pi/agent/skills/refactor-tests/SKILL.md`                   |

## 1. Goal

Keep a small test suite that is stable, useful, readable, and easy to
maintain. The method comes from Matteo Vaccari, "Tests as data over tests as
code". The skill `refactor-tests` holds the procedure. This file holds the
findings and the plan for this repo. `docs/test-review.md` records the earlier
cuts.

### Big ideas

1. **Tests are data.** One table row is one case. A new case adds a row, not a
   function. The runner stays small. The cull tests now use seven tables.
2. **Test the public entry point.** A black-box test covers more with fewer
   tests. Private helpers, call order, and module layout stay free to change.
3. **Fake only at the edge.** The edge is network, TTS, ffprobe, and the spaCy
   model. Never patch repo code. If a test must replace repo code, add a seam:
   a keyword argument with the real function as default, and one ADR. `run_fix`
   is the first case (ADR 0050).
4. **Prove that a test can fail.** Break production code on purpose. Each kept
   test must fail for at least one break. A break that survives is a gap: add a
   row. The audits ran 72 breaks and found 14 survivors.
5. **Give setup one home.** Shared builders live in fixture modules. A helper
   used once goes inline. Setup longer than the input and the output is a
   smell.
6. **Delete what another test already proves.** One break once failed 13 tests
   in 9 files. Keep one test for each rule.
7. **Do not pin snapshots.** Row counts, constants, profile paths, file hashes,
   and call order record the code. They do not specify behavior.
8. **Measure the cut.** Report lines, tests, and mocks before and after. A
   cut batch must not add lines. Report gap lines apart from cut lines. B9 grew
   `test_cli_fix.py` from 176 to 258 lines, because it also filled gaps. A
   follow-up cut it to 183.
9. **Cut by value, not by feel.** Run many mutants. Record which test kills which.
   Delete a test only when another test kills the same mutants. Keep a test that
   alone kills a mutant. Then run the matrix again to prove that no mutant lost
   its killer.

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

Batches B1 to B3 (commits `f26cf9f`, `a169926`, `17a0502`) cut the pure-function
and small-table files. The counts cover only the files each batch touched.

| Batch | Files | Lines before | Lines after | Tests before | Tests after |
| ----- | ----- | ------------ | ----------- | ------------ | ----------- |
| B1    | 10    | 2462         | 2408        | 91           | 91          |
| B2    | 7     | 594          | 531         | 53           | 32          |
| B3    | 9     | 689          | 693         | 44           | 24          |

- B1 moved the shared deck builder, the MP3 writer, and the contract-deck fixture
  into `deck_fixture.py` and `conftest.py`. It deleted four dead or single-use
  helpers.
- B2 deleted 17 tests and tabled six groups. 11 deleted tests only grepped CSS or
  HTML strings.
- B3 rewrote `test_extract_sentences` onto the public `extract_sentences`, added
  `ast_fixture.py`, and removed the data-file dependency of `test_card_preview`.
  Lines rose by 4: the table rows cost more than the removed bodies saved.
- 13 deliberate breaks across the three batches. A test caught each one.

Batches B4 to B7 and the B9 follow-up came next (commits `ebd8d2b`, `65b571a`,
`4e7d347`, `baa575e`, `d60656d`). Each count covers the files the batch touched.
A fixture line count is separate, because the fixtures are shared.

| Batch  | Lines before | Lines after | Fixture lines | Tests before | Tests after | Mocks before | Mocks after |
| ------ | ------------ | ----------- | ------------- | ------------ | ----------- | ------------ | ----------- |
| B4     | 1093         | 871         | in the count  | 49           | 29          | 0            | 0           |
| B5     | 689          | 555         | +64           | 49           | 30          | 0            | 0           |
| B6     | 738          | 692         | +96           | 23           | 13          | 34           | 0           |
| B7     | 606          | 513         | +63           | 28           | 28          | 7            | 0           |
| B9 (2) | 258          | 183         | none          | 46           | 26          | 0            | 0           |

- B4 folded `test_assemble.py` into the `run_core` tests and tabled the core
  tests. `kept_order_hash` is now checked against the real sha256.
- B5 made one shape-rule table, `shape_fixture.py`, for the files that repeated it.
- B6 runs the Mandarin audit through `run_pack_audit(load_table=, probe=)`. It
  has no `monkeypatch` left. Two breaks in `pack.py` survived at first; one new
  test now catches both.
- B7 removed every patch of an internal function. Only `load_model`, gTTS, sleep,
  urlopen, and the `ROOT` constant stay patched.
- B9 follow-up: `test_cli_fix.py` went from 258 to 183 lines. It is still longer
  than the 176 lines of the start, because it now covers four languages.
- 44 deliberate breaks across these batches. A test caught every one, after the
  two B6 survivors got a test.
- Three worker commits swept in another worker's half-edited file. Commit
  `3e9e7bb` fails alone because it imports `shape_fixture.py` before it exists.
  Bisect from `65b571a` onward.

Batches B8 and B10 finished the plan (commits `03cc7ad`, `b2b774f`, `f9ff560`).

| Batch | Lines before | Lines after | Fixture lines | Tests before | Tests after | Mocks before | Mocks after |
| ----- | ------------ | ----------- | ------------- | ------------ | ----------- | ------------ | ----------- |
| B8    | 1339         | 1153        | +51           | 65           | 49          | 1            | 0           |
| B10   | 0            | 298         | 0             | 0            | 6 files     | 0            | 0           |

- B8 moved every shape-rule copy into `shape_fixture.py` and added
  `tatoeba_fixture.py`. It replaced the pinned review-file indexes with a check
  that the file is closed and agrees with the word table.
- B10 added one table or one end-to-end test for each gap: pack build, core
  flow for Mandarin and Cantonese, rank CSV, `delete_model`, preview, and card
  CSS. The B10 lines are gap lines, not cut lines.
- 15 more deliberate breaks. A test caught each one. Two B8 survivors got a
  table row.

### Result

The whole suite, from the first audit to the end:

| Item                    | Start | After B10 | After 2.1 | End   |
| ----------------------- | ----- | --------- | --------- | ----- |
| Test and fixture files  | 44    | 55        | 53        | 52    |
| Lines in `tests/`       | 7 800 | 7 403     | 6 747     | 6 523 |
| Test functions          | 374   | 270       | 225       | 213   |
| Tests collected         | 416   | 568       | 500       | 432   |
| `monkeypatch.setattr`   | 22    | 9         | 8         | 12    |
| Run time                | 6 s   | 8 s       | 7 s       | 6.4 s |

- **Fewer, smaller tests.** Functions fell by 43%, lines by 16%, and cases by 24%
  since the peak. The collected count is now above the start by 16 cases, because
  other people added tests, for example `test_core_recount.py`.
- **Mocks.** The count rose, because three edges are now faked: ToJyutping,
  `spacy.load`, and `subprocess.run`. Each fake replaces a real library call. No
  patch touches our own code except the two trivial wiring tests in
  `test_swadesh_build.py`.
- **Every cut test had a reason.** The cuts in section 2.1 kept all 507 mutants
  killed.

### 2.1 Cut by value

Commits `1559851`, `7326f5a`, `a9bdb54`, `9a39eb5`, `cb00fbf`, `199f0d6`. A
scratch copy of the repo held 822 mutants, made by a script and by hand. The run
recorded which test kills which mutant. A test whose kills another test also
makes was deleted.

| Result                                  | Value |
| --------------------------------------- | ----- |
| Mutants                                 | 822   |
| Killed                                  | 507   |
| Survived                                | 312   |
| Timeout (infinite loop)                 | 3     |
| Tests deleted                           | 41    |
| Lines removed                           | 634   |
| Mutants that lost their last killer     | 0     |

- All 41 deleted tests were redundant. No test was deleted for killing nothing.
  Each test that killed nothing had one hand-written mutant that only it killed.
- The script does not change strings, regexes, templates or return values. The
  first matrix marked such tests as dead by mistake. Add those mutators before
  the next run.
- Tests that guard a layout or data invariant stay, for example
  `test_import_graph`, `test_language_layout`, and the review-file check.
- The 312 survivors were not checked one by one. Some are equivalent changes.
  Section 8 lists the ones that matter.

### 2.2 Cut by library

Commits `026121e`, `184b862`, `081c586`, `b4ef4da`, `3c49cea`. The rule: test our
logic and our contracts. Do not test third-party libraries. Every test got a tier.

| Tier | Meaning                                                       | Action                      | Count |
| ---- | ------------------------------------------------------------- | --------------------------- | ----- |
| A    | Our decision logic, file contracts, routing, invariants       | Keep                        | 185   |
| B    | Our logic over a token, lemma, or audio source                | Keep, fake the library      | 17    |
| C    | Claims only what a library or a data row returns              | Cut                         | 20    |
| D    | Our curated data                                              | Keep cross-file checks only | 7     |

- Faked at the edge: `spacy.load`, jieba, ToJyutping, pypinyin, and `subprocess.run`
  for ffprobe. The fakes check our code, not the model data.
- Cut: 18 tests that pinned a lemma, a segmentation, a pinyin, a template string,
  a font, or a data-file row. Two more were replaced by pure or faked tests.
- 16 deliberate breaks. Four survived a cut at first, so the worker kept or
  rewrote the covering test.
- The suite went from 504 to 432 cases, 229 to 213 functions, and 7.9 s to 6.4 s.

Left on the real French model, because a fake model would write to the real
token cache `data/tokens_french.sqlite`:

| Test                                                    | Why it stays                         |
| ------------------------------------------------------- | ------------------------------------ |
| `test_the_gates_pass_on_a_tiny_cull`                    | Real French model                    |
| French register, highlight, form-key, noun-credited     | A break survived without each        |
| `test_the_sentence_pack_card_bolds_the_target` (French) | Profile token cache                  |

A seam on `_TOKEN_CACHE` would let a fake model run them. That is a production
change and needs an ADR.

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
| B1    | Fixtures only: `deck_fixture`, `conftest`; update four apkg test files        | low            | done    |
| B2    | Pure-function files: `card_templates`, `char_links`, and five more            | low            | done    |
| B3    | Small tables: `extract_sentences`, `duplicates`, and five more                | low            | done    |
| B4    | `test_core_deck`; fold in `test_assemble`; fix `test_core_similarity`         | medium         | done    |
| B5    | Rule tables: `shared_coreflow`, `gate_id_map`, and two more                   | medium         | done    |
| B6    | Mandarin audit and deck, through `run_pack_audit(load_table=, probe=)`        | medium         | done    |
| B7    | Remove mocks: `tokens_cache`, `swadesh_build`, and two more                   | medium         | done    |
| B8    | `mandarin_sentence_selection`, `french_pack`, `tatoeba_fixture`               | medium         | done    |
| B9    | `test_cli_fix`                                                                | needs decision | done    |
| B10   | New tests for the gaps in section 8                                           | medium         | done    |

Other people edit the same files. Check the per-file counts before you start a
new batch.

### Decision for B9

Done. The owner accepted option A: `run_fix` takes `run_core`, `run_cull`,
`check_core`, and `check_cull` as keyword arguments, with the real functions as
defaults. `test_cli_fix.py` has no mocks and tables the routes for all four
languages. ADR 0050 records the seam.

## 8. Gaps

The audit found 14 surviving breaks. Twelve were real gaps. Two were equivalent
changes and need no test. B4, B6, B7, B9 and B10 closed all twelve. One gap is
open:

| Area         | No test for                                                          | Needs                                                               |
| ------------ | -------------------------------------------------------------------- | ------------------------------------------------------------------- |
| Card preview | Preview build without the real `vocab/*.csv` files                   | A `vocab_dir` argument on `build_sections`, `vocab_card`, `write_preview` |

The `--prune` option no longer exists, so its gap is void. `build_mandarin_pack`
needs no test of its own: it wraps `build_pack`, which `test_mandarin_deck.py`
and `test_pack_build.py` run.

The cut by value found more surviving mutants. No test kills any sampled
mutant in these modules:

| Module                                   | Mutants, none killed |
| ---------------------------------------- | -------------------- |
| `ranking/reorder.py` (deleted, ADR 0058)  | 16 of 19             |
| `ranking/latin.py`                       | 5 of 5               |
| `cache.py`                               | 4 of 4               |
| Spanish `core.py`, `frequency.py`, `cull.py` | 4, 4 and 3 of the same |
| Greek rank module, Mandarin `frequency.py` and `pinyin.py` | all sampled |

Test the behavior of each module, or delete the module if nothing uses it. The
`reorder.py` row that matters most: no test checks that `delete_card` runs for a
removed card.

## 9. Rules for every batch

- Run `uv run pytest -q` before and after. The suite stays green.
- Make at least three deliberate breaks in production code. Restore each one.
  Add a test row for each break that survives.
- Stage only the files the batch changes. Others edit the tree in parallel.
- Keep temp files under a few MB. A fixture bug once wrote an 82 GB file.

## 10. Out of scope

The fix-route ADR is `0050-fix-route-takes-its-flows.md`. ADR 0049 is the
sense-exceptions ADR.

## 11. Next

| Step | Action                                                                                         |
| ---- | ---------------------------------------------------------------------------------------------- |
| 1    | Decide on the `vocab_dir` seam for the preview. It needs one short ADR.                        |
| 2    | Add string, regex and return-value mutators. Run the matrix on the cull, sentence-select and Mandarin files that were in flux. |
| 3    | Done in 2.2 for the French, Spanish, Mandarin, template and sense tests.                       |
| 4    | Turn the matrix script into a repo tool, `uv run python -m tools.mutants`. Run it before a release. |
| 5    | Decide for each module in the gap table (section 8) whether to test it or delete it.           |
| 6    | Decide on a `_TOKEN_CACHE` seam, so the French model tests can run on a fake model.            |
| 7    | Decide if a one-row guard for a swapped front and back template comes back.                    |
