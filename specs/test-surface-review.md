# Spec: Test-Surface Review

| Field    | Value                                                              |
| -------- | ------------------------------------------------------------------ |
| Status   | draft                                                              |
| Date     | 2026-11-04                                                         |
| Owner    | anki-tools                                                         |
| Method   | `improve-tests` skill: tests as data, deep-module vocabulary       |
| Related  | `docs/test-refactor-plan.md` (earlier cuts and mutation audits)    |
| Related  | `docs/adr/0050-fix-route-takes-its-flows.md` (the seam model)      |

## 1. Summary

The suite is already in good shape. This review finds narrow test-surface
opportunities. Each one makes a deep module testable through its interface, with
cases written as data.

| Item                | Value                                      |
| ------------------- | ------------------------------------------ |
| `uv run pytest -q`  | 432 passed, 0 failed                       |
| Pytest time         | 7.9 s (11.9 s wall with `uv`)              |
| Test files          | 43 `test_*.py` and 7 fixture files         |
| Test lines          | 6,524                                      |
| `parametrize` uses  | 61                                         |
| Commits             | 497                                        |

Method. A scout agent walked the tests and the code. The parent agent checked
candidates A, B, E and F against the code. Candidates C, D and G rest on the
scout report only. No mutation check ran in this review.

Internal patch targets are few. Only `swadesh_vocab.ROOT` and `tokens.load_model`
patch repo code. The other patches (`spacy.load`, `gtts`, `edge_tts`, `urlopen`,
`time.sleep`, `jyutping`) sit at the system edge.

## 2. Candidates

| Id | Candidate                                               | Strength        | Signs         | ADR conflict |
| -- | ------------------------------------------------------- | --------------- | ------------- | ------------ |
| A  | Run the core and cull gates on a real run output        | Strong          | P3 P5 P7 P2   | none         |
| B  | One profile table for the shared cull flow              | Worth exploring | P1 P7 P11     | none         |
| C  | Run the pack audit for every language                   | Worth exploring | P5 P1 P7 P11  | none         |
| D  | One case table for all language profiles                | Worth exploring | P5 P1 P12     | none         |
| E  | Add the two open seams: model cache and vocab directory | Worth exploring | P3 P10        | none         |
| F  | Give each test builder one home                         | Strong          | P7 P2         | none         |
| G  | Name tests for behavior and table the repeats           | Speculative     | P4 P6 P2      | none         |

### A. Run the core and cull gates on a real run output (Strong)

- Files: `tests/test_gates.py`, `tests/test_core_recount.py`,
  `tests/test_cull_shared.py`, `tests/test_cli_fix.py`; code in
  `anki_tools/ranking/coregates.py`, `cullgates.py`, `coreflow.py`, `cullflow.py`.
- Problem: No test runs `check_core` or `check_cull` on a real output.
  `test_cli_fix` only records the calls. Four hand-made reader fakes copy the
  `CoreFiles` and `CullFiles` interface, and each call needs `# type: ignore`:
  `test_gates.py::_CoverageFiles`, `::_CoreWordFiles`,
  `test_core_recount.py::_Files`, and the `SimpleNamespace` readers in
  `test_cull_shared.py`. Only `test_french_cull::test_the_gates_pass_on_a_tiny_cull`
  runs a real cull through the gate list, and only for French.
- Solution: Add one "run, then check" test for each flow. Add one table of
  corrupt-output rows, one row for each gate (K5, K6, K13, K23, K24 for core).
  Keep the pure gate tables. Build their reader from one shared fake in
  `tests/gate_fixture.py`.
- Seam: `coregates.check_core(output_path, profile)` and
  `cullgates.check_cull(output_path, profile)` on the output of
  `coreflow.run_core` and `cullflow.run_cull`.
- Benefits: locality (gate bugs concentrate in one table); leverage (one
  interface, five gates); four fakes become one.
- Harness change: production code none; test data one corrupt-output table; test
  runner one small helper.
- Decisions:
  - Surface: `coregates.run_gates(coregates.GATES, coregates.CoreFiles(path, profile))`
    on the `cull_fixture.run_core` output, plus one test of the `check_core` exit code.
  - Scope: core only. The cull gate tests only lose their fakes.
  - Fakes: `cull_fixture.gate_reader(profile, **loaders)` builds a `GateFileReader`.
    It replaces the four hand-made readers. No `gate_fixture.py` file.
  - Corrupt-output rows: gate id, side file, old text, new text, failed gate ids.
    Rows for K5, K13, K23, K24. K6 has no row, and the fixture holds one card.
  - Speed: `run_core` runs once for the module. Each row copies the run files.
  - Checks: a forced pass of K5, K13, K23, and K24 must fail its row.
- Result:
  - Tests added: 6 (one run-then-check, four corrupt-output rows, one `check_core`).
    The count goes from 432 to 438.
  - Tests changed: 7 (2 in `test_gates.py`, 2 in `test_core_recount.py`, 3 in
    `test_cull_shared.py`) use `gate_reader`. Seven `# type: ignore` lines are gone.
  - Time: the suite goes from about 7.3 s to about 9.6 s. K24 loads the French
    spaCy model, so each gate run costs about 0.4 s.
  - Gate ids: the K24 row also fails K13, because both read the coverage form.
    The K13 row edits a form to `cha`, a part of a word, so that only K13 fails.
  - Surviving mutants: the K6 limit set to 0.0 (no K6 row). The K13 tier test
    flipped to `!=` fails only the pure table, because the fixture tags each
    word as a verb.
- Follow-up (anki-5n5b):
  - Cull rows: `test_gates.py` has a module-scoped `run_cull` fixture and rows for R1
    (a card left in the pool), R2 (a lemma with 5 uses that is not exempt) and R5 (a kept
    card under the length window), plus one `check_cull` exit-code test and the pass case.
    The tiny French deck moved to `cull_fixture.cull_deck`; it now has 5 sentences.
  - R6 has no row. It reads only the output file, which a text edit cannot change.
  - K6: `test_k6_fails_when_the_run_keeps_two_cards_with_the_same_lemmas` runs the core
    with `similarity=1.1`, `per_word=2`, `ubiquitous_share=1.0` and a tokenizer that gives
    every card the same lemma set. Only K6 fails. The named helper is `_same_lemma_tokens`.
  - K13: `test_k13_tests_a_tier_f_target_by_its_lemma` runs the core with `core_words=1`
    (a new `run_core` argument) and the helper `_minou_tokens`, which tags chat as the
    noun lemma minou. The tier F target is not a form of the sentence, so the flip to
    `tier != "S"` fails the test.
  - Tests added: 7 (cull pass case, 3 rows, `check_cull`, K6, K13); 1 moved test removed
    from `test_french_cull.py`. The count goes from 438 to 444.
  - Mutants killed: R1, R2, R5 forced pass; K6 forced pass; K6 limit 1.01; K13 tier flip;
    R1 `left >= 0`; R2 without the exempt test; R2 `limit + 1`.
  - Surviving mutants: the K6 limit set to 0.0, because the K6 run fails with any limit
    that the equal lemma sets reach.
  - Time: the suite goes from 10.1 s to 10.8 s (wall 13.2 s to 12.8 s).

### B. One profile table for the shared cull flow (Worth exploring)

- Files: `tests/test_french_cull.py`, `test_cull_shared.py`, `test_core_deck.py`,
  `test_spanish_cull.py`; code in `anki_tools/ranking/cullflow.py` (1,354 lines).
- Problem: `test_french_cull.py::_write_cull_deck` repeats the body of
  `cullflow.run_cull` with four internal calls (`coreflow.note_values`,
  `assemble.output_fields`, `assemble.deck_templates`, `write_package`). Four
  tests check "one deck, one note type, title equals stem". Two tests run
  `UPDATE cards SET due = 42` and read the due back
  (`test_core_deck::test_the_kept_card_keeps_its_note_id_and_due`,
  `test_cull_shared::test_the_cull_keeps_the_due_of_a_kept_card`). `_tiny_deck`
  is defined in two files.
- Solution: Run every output check through `run_cull`, with one parametrized
  profile table for French and the fake profile. Run the reserve test through
  `run_cull` with `reserve_pack`. Delete `_write_cull_deck`. Keep the
  language-only tests: the register rule, the tokenizer lemma repair, and
  `load_or_build_frequency`.
- Seam: `cullflow.run_cull(request, profile)`.
- Benefits: one interface, N profiles; one four-call helper deleted; one test for
  each rule.
- Note: commit 89682ed cut the cull tests once, under `docs/test-refactor-plan.md`.
  The plan batches B1 to B3 cover other files and do not overlap this candidate.
- Harness change: production code none; test data one profile table; test runner
  one parametrized runner.
- Decisions (anki-y9g7):
  - Surface is `cullflow.run_cull(CullRequest, CullProfile)`. No new interface and no
    production change.
  - Step 1: delete `_write_cull_deck`, `_tiny_swadesh_deck`, `SWADESH_FIELDS` and
    `test_write_deck_inserts_the_reserve_card`. Fold their asserts into
    `test_the_cull_fills_the_core_word_and_the_gloss`: media set, model name equals
    `DeckIdentity(output).title`, note, card and media counts, and the reserve row.
  - Step 2: `test_a_second_profile_drives_every_output_value` becomes the table
    `test_a_profile_drives_every_output_value` over the fake, French and Spanish
    profiles. The English line, the licenses and the missing `reserve_pack` key move
    to `test_the_fake_profile_values_reach_the_output`.
  - Step 3: the French row uses `FRENCH_CULL_PROFILE` and `cull_deck`. The Spanish row
    uses `SPANISH_CULL_PROFILE` and the new `spanish_cull_deck`. Both rows use
    `no_reserve=True`, the real spaCy models, and `one_word_profile` for the faked
    frequency and sense files. The Spanish cull gates do not run.
  - No other test is deleted. `_tiny_deck` in `test_cull_shared.py` is not the same
    as `cull_fixture.tiny_deck`: it is a wrapper that builds a fixed three-row deck
    and calls `tiny_deck`. The two stay apart.
- Result:
  - Tests: 444 before, 447 after. Removed 1, added 4 (two table rows, one fake-only
    test, one French reserve-fields test).
  - Pytest time: 10.1 s before and 10.1 s after.
  - The French reserve-fields mutant survived the fold, because the folded test uses
    the fake `reserve_fields`. The new `test_french_cull` test kills it.
  - Mutants killed: Spanish `reserve_fields` swap (`test_spanish_cull`, not the new
    row, which uses `no_reserve=True`); `OUTPUT_FIELDS` order (all three rows);
    French `input_model_name` (French row); bolding removed (all three rows); deck
    name, manifest `input_model` (all three rows); gloss emptied and reserve media
    not copied (the folded test). No mutant survives. `git diff HEAD -- anki_tools/`
    is empty.

### C. Run the pack audit for every language (Worth exploring)

- Files: `tests/test_french_pack.py`, `test_mandarin_audit.py`,
  `test_pack_build.py`; code in `anki_tools/sentence_pack/audit.py` (735 lines).
- Problem: Only `test_mandarin_audit.py` calls `run_pack_audit`, through its own
  `build_env`. French builds a second `AuditInputs` fixture in `_audit_inputs` and
  calls single gates. `test_pack_build.py` runs `build_pack` for four languages on
  one card each, but never audits the output.
- Solution: Extend the `CARDS` table in `test_pack_build.py` with an audit step
  for each language: "audit passes except the word-table gate". Move `build_env`
  to `tests/pack_fixture.py`.
- Seam: `run_pack_audit(profile, ...)` on the output of `pack.build_pack`.
- Benefits: leverage (one table, five languages); the audit is tested as
  composed; one fixture, not two.
- Harness change: production code none; test data one extra column; test runner
  one extra step.

### D. One case table for all language profiles (Worth exploring)

- Files: `tests/test_french_pack.py`, `test_spanish_cull.py`,
  `test_cantonese_core.py`, `shape_fixture.py`; code in
  `anki_tools/languages/{french,spanish,greek,cantonese}/profile.py` (about 1,000
  lines each).
- Problem: Only French has a test file for its profile (13 tests).
  `test_spanish_cull.py` has 2 tests. Greek has one shape rule and one `CARDS`
  row. French-only tests such as `test_highlight_marks_the_matched_token`,
  `test_form_key_rule` and `test_a_noun_form_is_not_credited_by_its_verb` could
  run for every profile. The plan reports surviving mutants in the Spanish
  modules.
- Solution: Drive `highlight`, `hits`, `token_matches` and `prepare_text` from one
  case table, in `tests/language_cases.py` or a JSON file.
- Seam: the `LanguageProfile` methods.
- Benefits: a missing language shows as a gap in the table; a new language adds
  rows.
- Risk (P12): expected values for Greek and Cantonese need a trusted source, for
  example a native reader or a dictionary.
- Harness change: production code none; test data one case table; test runner one
  parametrized runner.
- Decisions (anki-neua):
  - Surface is the `LanguageProfile` object: `FRENCH.highlight`, `SPANISH.match_count`,
    `GREEK.hits`. Spanish uses `SPANISH` from `languages/spanish/profile.py`, not the
    `CoreProfile` of `shape_fixture`. No production change.
  - The table file is `tests/language_cases.py`, in the style of `shape_fixture.py`. It has one
    NamedTuple for each hook: `HighlightCase`, `MatchCase`, `HitCase`, `PrepareCase`,
    `UnitCase`. Each row has a last field `rule`, an id in words. `MatchCase` names its count
    field `matches`, because `count` clashes with `tuple.count`.
  - `tests/test_language_hooks.py` has one parametrized runner for each hook. The `unit_count`
    runner prepares the sentence first, as the profile docstring requires.
  - There is no `token_matches` test, because `match_count` is an alias of it. There is no
    meta-test that every language has a row.
  - The four French tests move into rows and are deleted. No other test is deleted.
  - Oracle rule (P12): each Spanish row comes from a rule in the hook docstrings. Greek,
    Cantonese and Mandarin get only mechanical rows: an exact word, an absent word, and one
    whole token in bold.
- Result:
  - Tests: 447 before, 503 after. Removed 12 test items (the four French tests), added 68 rows.
  - Pytest time: 10.2 s before and 10.4 s after.
  - Rows by language and hook:

    | Language  | highlight | match_count | hits | prepare_text | unit_count |
    | --------- | --------- | ----------- | ---- | ------------ | ---------- |
    | French    | 2         | 7           | 3    | 2            | 1          |
    | Spanish   | 10        | 10          | 7    | 4            | 4          |
    | Greek     | 2         | 2           | 2    | gap          | gap        |
    | Cantonese | 2         | 2           | 2    | gap          | gap        |
    | Mandarin  | 2         | 2           | 2    | gap          | gap        |

  - Breaks killed: Spanish `hits` `count == 1` flipped to `>= 1` (2 rows); Greek `hits`
    `count == 1` flipped to `!= 1` (the Greek exact-word row); Spanish `prepare_text` without
    NFC (the NFC row); Cantonese highlight `end` plus one (the Cantonese row and one pack-build
    test); Mandarin highlight span end set to the start of the last character (the Mandarin row
    and two Mandarin deck tests); French `hits` `>= 1` (the French double-match row, added
    after the break survived); French highlight without the `ne pas` rule (the `ne` row);
    Spanish `_reflexive_stem` returning an empty string (2 rows); Spanish `_token_keys`
    without the surface-only guard (3 rows).
  - Breaks that survive: Greek `hits` `>= 1`, because Greek has no row with two matches.
    Spanish `_reflexive_stem` `[:-3]`, because the carrier lemma of `acostar` gives the same
    key. Spanish `carrier_keys` without the early return for `SURFACE_ONLY` forms; the
    guard in `_token_keys` makes it equivalent for every row.
  - Rows left out because the output differed from the rule: none.
  - `git diff HEAD -- anki_tools/` is empty.
- Gaps (language and hook with no row):
  - Greek: no row for `prepare_text` or `unit_count`, and no row with a repeated word.
  - Cantonese: no row for `prepare_text` (Traditional conversion) or `unit_count`, and no
    row for the whole-token rule of `form_offsets` (`我` inside `我哋`).
  - Mandarin: no row for `prepare_text` (simplified conversion) or `unit_count`, and no row
    for a multi-character form.
  - Greek, Cantonese and Mandarin: no language-specific rule rows, because P12 needs a
    native reader for the expected values.
  - Spanish `unit_count`, `hits` and `highlight` have no row for a clitic or a contraction
    (`del`, `al`), because the Spanish docstrings state no rule for them.
- Open questions:
  - Who reads the Greek, Cantonese and Mandarin rules to give trusted rows?
  - Is the `SURFACE_ONLY` early return in `carrier_keys` dead code? The `_token_keys` guard
    covers it.

### E. Add the two open seams: model cache and vocab directory (Worth exploring)

- Files: `tests/token_fixture.py`, `test_swadesh_build.py`,
  `test_card_preview.py`; code in `anki_tools/tokens.py` (`_TOKEN_CACHE`,
  `load_model`), `swadesh_vocab.py` (`ROOT`), the preview module
  (`build_sections`, `vocab_card`, `write_preview`).
- Problem: Two internal patch targets remain. The preview cannot build without the
  real `vocab/*.csv` files. `docs/test-refactor-plan.md` section 11 lists both as
  open decisions (steps 1 and 6).
- Solution: Add each seam as a keyword argument with the real function as default,
  as ADR 0050 did for `run_fix`. Two adapters (production and fake) make each seam
  real. Write one ADR for each seam.
- Seam: `model` for the token cache; `vocab_dir` for the preview.
- Benefits: the last internal mocks go; the preview test runs offline.
- Harness change: production code one keyword argument for each seam; test data
  none; test runner a fake adapter.

### F. Give each test builder one home (Strong)

- Files: the same private name appears in two files each:
  - `_form`: `test_french_pack.py`, `test_mandarin_sentence_selection.py`
  - `_card`: `test_shared_coreflow.py`, `test_mandarin_audit.py`
  - `_profile`: `test_cull_shared.py`, `test_shared_coreflow.py`
  - `_tiny_deck`: `test_french_cull.py`, `test_cull_shared.py`
  - `mandarin_fixture.card` and `cull_fixture.card` share a name with different
    signatures.
- Problem: A reader cannot tell which `card` an import gives.
- Solution: Merge into the existing fixture modules, `tests/cull_fixture.py` and
  `tests/mandarin_fixture.py`. Rename for the concept. Delete the copies.
- Benefits: locality (one edit, one place); about eight copies deleted.
- Harness change: production code none; test data none; test runner fixture
  modules only.

### G. Name tests for behavior and table the repeats (Speculative)

- Files: `tests/test_core_deck.py` (475 lines, 18 tests),
  `test_mandarin_sentence_selection.py`, `test_french_pack.py`.
- Problem: Test names state a function pass: `test_the_seed_pass_*`,
  `test_the_completion_pass_*`, `test_key_order`, `test_card_cap`,
  `test_filter_counts`. `test_module_cleanup` holds four behaviors.
  `test_the_status_follows_the_count_the_quota_and_the_sense_row` has 24 lines for
  one assert. `test_core_deck.py` imports eight internal helpers (`card_cost`,
  `target_quotas`, `select`, `outside_lemmas`, `size_check`, `kept_order_hash`,
  `coverage_rows_of`, `drop_silent_cards`).
- Solution: Rename for the behavior and convert repeats to `parametrize`. Check
  that each private helper has a caller outside the tests. Keep `select` and
  `card_cost` as table tests. Read the `outside_lemmas`, `size_check` and
  `kept_order_hash` results from `run_core`.
- Seam: `coreflow.run_core` through `cull_fixture.run_core`;
  `ranking.core.select` for the pure selection rules.
- Harness change: production code none; test data tables; test runner none.

Decisions:

- Leave `tests/test_core_deck.py` as it is. Its names already state behavior and its
  helper tests stay.
- Do two jobs only: split two French tests and rename eight Mandarin tests.
- Each old assert survives in a split. The split adds no new assert.
- Rename in place with the style `test_<subject>_<verb>_<outcome>`. Keep the
  classes and bodies. Do not parametrize.
- Check each split with one break of the code under test per old assert. A
  rename needs no break.

Result:

- Dropped from the scope: `tests/test_core_deck.py`. Its names are already behavior
  names. Its helpers have production callers in `coreflow.py`.
- `test_module_cleanup` became a table of three `clean_term` rows, one
  `parse_module_terms` test and one `module_terms` count test.
- `test_audio_file_names_are_deterministic` became a hash test, a file name test
  and a table of two `choose_source` rows.
- Renamed in `tests/test_mandarin_sentence_selection.py`:
  `test_key_order`, `test_key_serialization`, `test_filter_counts`,
  `test_card_cap`, `test_per_word_limit`, `test_fallback_scaffold`,
  `test_scaffold_card_is_selected` and `test_pinyin_is_filled`.
- Test count: 503 before, 510 after. The French splits add 7 tests.
- Breaks, each restored with `git checkout`, and the test that failed:
  - `clean_term` keeps the note: the `clean_term` note row failed.
  - `parse_module_terms` starts at depth 1: the `parse_module_terms` test failed.
  - `module_terms` drops index 1: the count test failed.
  - `audio_hash` ignores the voice: the hash test failed.
  - The sentence name uses `audio_` for `tts_`: the file name test failed.
  - `choose_source` returns native for a non-commercial licence: the licence row failed.
  - `choose_source` returns native in `none` mode: the `none` row failed.

## 3. Not proposed

- The tests that read the shipped word table, the review file and
  `SENTENCES_PATH` (`test_french_pack`). They check production data. Re-check them
  when the CSV files change again.
- The case literals in `shape_fixture.py` and `mandarin_fixture.py`. They are
  already tables.

## 4. Top recommendation

A. It is the one flow where only a fake reader exercises the gates. It needs no
production change. The table form of A also serves F and B.

## 5. Next

| Step | Action                                                                      |
| ---- | --------------------------------------------------------------------------- |
| 1    | The user picks a candidate. Run the grilling loop on it.                    |
| 2    | Add new terms to `GLOSSARY.md`. Offer an ADR for each new seam (E).         |
| 3    | Hand off to the `refactor-tests` skill for the per-file rewrite.            |
| 4    | Make three deliberate breaks per batch (plan, section 9).                   |
