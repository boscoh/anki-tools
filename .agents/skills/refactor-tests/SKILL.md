---
name: refactor-tests
description: Use when the user wants to shrink, clean up, or stabilise a test suite, remove redundant or brittle tests, convert repeated tests to data tables, or make tests easier to read and trust. Trigger on "refactor the tests", "too many tests", "tests break on every refactor", "tests as data", or "prune tests". Based on Matteo Vaccari, "Tests as data over tests as code".
---

# Refactor Tests

Reduce the test suite to a small set that is stable, useful, readable, and easy to maintain. Source: Matteo Vaccari, "Tests as data over tests as code" (martinfowler.com/articles/tests-as-data.html).

## Goal

A good test does three things:

- It predicts success in production: all tests pass means the feature works.
- It survives refactoring: a harmless internal change does not break it.
- It is easy to read: a reader sees the input, the expected output, and the name that states the intent.

## Problems to find

| Id  | Problem                                                                          |
|-----|----------------------------------------------------------------------------------|
| P1  | Large volume: many tests, many duplicates, gaps hard to see                      |
| P2  | Boilerplate hides the intent: setup code is longer than the input and the output |
| P3  | Coupling to the implementation: tests use mocks and stubs of internal functions  |
| P4  | Tests confirm the current code, bugs included, and do not specify behavior       |
| P5  | Isolated unit tests pass while the composed system fails                         |
| P6  | One-off scenarios repeat the same steps with small variations                    |
| P7  | Test helpers and DSL steps used once, or one helper per scenario                 |
| P8  | Test data lives in code, so an edit to it is hard to see in a diff               |
| P9  | The test cannot fail: no mutation of the code breaks it                          |
| P10 | Slow or fragile tests: external services, end-to-end paths for simple rules      |

## Techniques to apply

- Customer test: state behavior in the language of the user, not of the code.
- Black box: call the public entry point (a function, the CLI, a service layer). Do not name private helpers.
- Test table: one row per case, with name, input, and expected output. Use `pytest.mark.parametrize`.
- Data file: for rows with structured input, put cases in a CSV, JSON, or YAML file under `tests/data/` and write one small runner.
- Readable printer: render state as short text so a failure shows a clean diff.
- Side channel for setup: seed state through a small builder or fixture, not through the full flow.
- Fake at the edge only: replace external systems (network, TTS, file downloads). Never replace internal modules.
- Mutation check: change one operator or branch in production code. At least one test must fail with a clear message.
- Oracle: check expected values against a trusted source (documentation, a second implementation, a domain expert).
- Property tests: use for invariants (for example, "no input crashes"). They complement tables.

## Procedure

Do the steps in order. Do not skip the baseline.

1. **Baseline.** Run `uv run pytest -q`. Record the count and the run time. All tests must pass before you start. Count tests with `grep -c "^def test_\|^    def test_" tests/*.py`.
2. **Inventory.** List each test file, its test count, its line count, and its private helpers (`_name` functions). Mark the largest files first.
3. **Audit.** For each test, find which problem (P1 to P10) it has. Use these checks:
   - P2: does the test body have more setup lines than assert lines?
   - P3: does it use `monkeypatch`, `Mock`, or `patch` on a module of this repo?
   - P4: does the name describe a function ("test_foo_calls_bar") and not a behavior?
   - P5: is there no test that goes through the public entry point for this feature?
   - P6: do two or more tests have the same body with different literals?
   - P7: is a helper used by one test only?
   - P9: do mutation checks (step 6) fail to break any test?
4. **Decide.** Give each test one action:

   | Action  | When                                                                                     |
   |---------|------------------------------------------------------------------------------------------|
   | Keep    | It states a behavior, uses the public interface, and is short                            |
   | Delete  | It repeats another test, tests a private detail, tests only a mock call, or cannot fail  |
   | Merge   | Two tests differ only in the input or the expected value: make them rows in one table    |
   | Table   | Three or more tests share a body: convert to `parametrize`; move big tables to data file |
   | Rewrite | The behavior matters, but the test is coupled: call the public entry point instead       |
   | Promote | Unit tests pass but no test covers the whole flow: add one black-box test                |

5. **Refactor in small steps.** Change one file at a time. After each file, run `uv run pytest -q`. Keep the suite green. Use a git commit per file or per group.
   - Move shared helpers (`_card`, `_tiny_deck`, builders) to one module, such as `tests/deck_fixture.py` or `conftest.py`. Delete the duplicates.
   - Give each table row an `id` (the case name) so a failure names the case.
   - Keep the table columns to: name, input, expected. Do not put `if` or loops in the table runner.
   - Do not use a mock of a module in this repo. Fake only the network, the audio service, and the disk where needed.
6. **Validate.** For each behavior kept, make one deliberate break in production code. Run the tests. A test must fail with a clear message. Revert the break. Delete or fix tests that never fail.
7. **Protect.** Keep data files in a separate directory. Check `git diff --stat tests/data` in review: a change to expected values needs a stated reason. Do not weaken or disable a test to make a run pass.
8. **Report.** Run `uv run pytest -q` again. Write the report below.

## Rules for the final suite

- Name a test for the behavior ("a duplicate sentence is rejected"), not for the function.
- Each test or table row has one reason to fail.
- A test calls the public interface. A refactor that keeps behavior must not change a test.
- Do not add comments or docstrings that repeat the code. Use types on helpers.
- Use sphinx docstrings only where the repo already requires them for public helpers.
- Keep the test run fast. Do not call the network.
- Prefer fewer, wider tests over many narrow tests, but keep one failure easy to locate.
- Do not use emojis in reports or markdown. Align table columns.

## Checklist

- [ ] Baseline run passes and the count is recorded
- [ ] Every test has an action (keep, delete, merge, table, rewrite, promote)
- [ ] No test only checks that function A calls function B
- [ ] No mock of an internal module remains
- [ ] Tests with the same body are one table
- [ ] Table rows have names and no logic
- [ ] Helpers are defined once and each one is used more than once
- [ ] At least one black-box test covers each main flow
- [ ] A mutation check breaks at least one test per kept behavior
- [ ] The suite runs with `uv run pytest -q` and passes
- [ ] The report lists the before and after numbers

## Report format

```markdown
## Test refactor report

| Metric            | Before | After |
|-------------------|--------|-------|
| Test files        |        |       |
| Test cases        |        |       |
| Lines of test code|        |       |
| Mocked internals  |        |       |
| Run time (s)      |        |       |

### Actions

| Action  | Count | Examples (file::test) |
|---------|-------|-----------------------|
| Keep    |       |                       |
| Delete  |       |                       |
| Merge   |       |                       |
| Table   |       |                       |
| Rewrite |       |                       |
| Promote |       |                       |

### Coverage of behavior
- Behaviors with a black-box test:
- Behaviors with no test (gaps):

### Mutation checks
| Break made | Test that failed | Message clear? |
|------------|------------------|----------------|

### Follow-up
- Open items, with bead ids if filed
```

## Findings in this repo

Survey of `tests/` made before this skill was written (tests were not changed):

- Size: 44 test modules, about 7,800 lines, about 374 test functions. `uv run pytest -q` gives 416 passed in about 6 s. The suite is fast and green.
- P1: The largest files are `test_cull_shared.py` (792 lines), `test_core_deck.py` (721), `test_french_cull.py` (567), and `test_mandarin_sentence_selection.py` (505). Parallel language files (`test_french_cull.py`, `test_spanish_cull.py`, `test_cull_shared.py`) cover the same behavior and may overlap.
- P2 and P7: Many private helpers (about 90 `_name` functions). The same ones appear in more than one file: `_card`, `_target`, `_freq_row`, `_tiny_deck` in both `test_french_cull.py` and `test_cull_shared.py`. Setup is often longer than the assert.
- P3: 8 modules use `monkeypatch` of repo code. `test_mandarin_audit.py` has 32 uses, `test_cli_fix.py` 11, `test_swadesh_build.py` 7, `test_sentence_pack_audio.py` 6. Some patch internal functions (`load_word_table`, `probe_duration`) and not only the external edge.
- P6: Only 10 `parametrize` uses in the whole suite. Test names such as `test_the_pack_and_the_cull_agree_on_the_shape_rules` and the gate tests (`test_gate_*`) are candidates for tables.
- P8: Case data (`TABLE`, `CARDS`, `INDEX_FIELDS`) lives as Python literals inside test modules. No `tests/data/` directory exists, so a change to expected values is mixed with code changes.
- P5: Good sign: names state behavior ("a duplicate text is rejected"), and `deck_fixture.py` already shares one deck builder. Gap: it is not clear that each flow has one black-box test through the CLI or the flow entry point.
- P9: Not checked. No mutation check has been run. Run step 6 before trusting the count.
