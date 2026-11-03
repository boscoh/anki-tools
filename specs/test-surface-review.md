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
- Note: `docs/test-refactor-plan.md` already cut the cull tests once. Check the
  cut-by-value matrix before a test is deleted.
- Harness change: production code none; test data one profile table; test runner
  one parametrized runner.

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
