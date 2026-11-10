# Input deck reading: one seam for every flow

| Field  | Value               |
| ------ | ------------------- |
| Status | Draft, for approval |
| Owner  | boscoh              |
| Date   | 2026-11-10          |

## Summary

Give every flow one way to read a sentence deck from an APKG file.

Delete the `load_input_deck` profile hook, because one implementation and zero test adapters do not justify a seam.

Put the deck readers in `ranking/deckreader.py`, because the cull reserve pack is a second deck shape.

**Biggest risk:** an output field name drifts between three copies and no gate sees it.

**The ask:** approve the four tasks, and confirm whether the reserve reader joins task 1.

## Decisions

| Id  | Decision                                                           | Reason                                 |
| --- | ------------------------------------------------------------------ | -------------------------------------- |
| D1  | Delete `load_input_deck` from both mode views                      | One implementation, zero test adapters |
| D2  | One `read_cards(path, profile)` returns cards and the silent count | Opens the package once, not twice      |
| D3  | Both flows print `Dropped, no audio file`                          | The cull discards a count it has       |
| D4  | Move the read cluster to `ranking/deckreader.py`                   | The reserve pack is a second shape     |
| D5  | Move the `OUTPUT_*` constants with the readers                     | A reverse import closes a cycle        |
| D6  | Build `assemble.OUTPUT_FIELDS` from those constants                | Stops the three-copy field drift       |
| D7  | Keep `note_values` and `output_values` in `coreflow`               | A move to `assemble` closes a cycle    |
| D8  | Keep `read_output` and `gate_files.read_notes` separate            | One resolves a note type, one does not |
| D9  | Amend ADR 0036, add one ADR for the reader seam                    | `AGENTS.md` asks for one ADR per seam  |

D1 needs confirmation. D4 needs confirmation, because it widens task 1.

### Rejected

- Give the cull the hook instead. One implementation would gain a second adapter, not a second behaviour.
- Merge `read_output` into `gate_files.py`. `read_notes` resolves nothing, so the merge adds a branch to a raw reader.
- Add a `DeckReader` class or a `Protocol`. Six module functions are the interface.
- Rename `sound_name` or `read_output` in the same change. They appear in six doc pages.

## Scope

In:

- The input read cluster in `ranking/coreflow.py`.
- The `load_input_deck` field in `ranking/scan.py` and `ranking/deckprofile.py`.
- The read call in `ranking/cullflow.py`.
- The five language binding functions.
- The import lines in the two gate modules and the pack audit.
- One ADR, plus an amendment to ADR 0036.

Out:

- `note_values`, `output_values`, and the manifest counts.
- The Swadesh pack producer stages S0 to S5.
- `gate_files.py` and `GateFileReader`.
- The card filter and the sense test.
- Any change to a written deck.

## Findings

The read path is a hook in the core and a direct call in the cull. Both modes state the same five values.

### Read path

| Fact                              | Value                           |
| --------------------------------- | ------------------------------- |
| Input read, core mode             | Hook, `scan.py:155`             |
| Input read, cull mode             | Direct call, `cullflow.py:1036` |
| Implementations of the read       | 1                               |
| Binding functions over it         | 5, no logic in any              |
| Bindings unused by their own mode | 2                               |
| Tests that replace the hook       | 0                               |

The hook exists to break a cycle: `coreflow` imports `scan`, so `scan` cannot call `coreflow.read_input_deck`.

### Cost

| Fact                                           | Value             |
| ---------------------------------------------- | ----------------- |
| APKG opens per input run                       | 2                 |
| APKG opens per output read                     | 2                 |
| Core tests that run flow logic with no package | 7 of 26           |
| Cull tests that run flow logic with no package | 0 of 29           |
| Package builders in `tests/cull_fixture.py`    | 4, about 78 lines |
| Copies of the five output field names          | 3                 |

Measured by reading the code:

- `read_input_deck` opens the package, then `drop_silent_cards` opens it again.
- `read_output_deck` opens the package, then `read_output` opens it again.

Assumption: the six verify commands use the flags of `anki fix` today.

Assumption: no language needs a second input deck shape today.

## Design

One reader, called by both flows. It opens the package once. It reads one card per note of the named sentence model. It drops the cards with no audio file. It returns the cards and the silent count.

The reader holds no selection and no ordering. It returns cards in input order with fresh indices.

The caller states the model name and the field names, as `<lang>` constants that the profile already holds.

Two deck shapes share the lower half. Both open the package, resolve the model by name, index the fields, and split the row. Each shape states its own field map and its own card record.

A boxed item is a record, not a class. No `Protocol`, no registry.

The input path now opens the package once. Keep that property: a reader that opens a package twice is a defect, not a style choice.

## Changes

| Area                     | Change                                            |
| ------------------------ | ------------------------------------------------- |
| `ranking/deckreader.py`  | New file: records, readers, constants             |
| `ranking/coreflow.py`    | The moved names leave; `note_values` stays        |
| `ranking/scan.py`        | Loses the path argument and the drop step         |
| `ranking/deckprofile.py` | Loses `load_input_deck`; gains `clean_sentence`   |
| `ranking/cullflow.py`    | Calls `read_cards`; prints the drop line          |
| `ranking/assemble.py`    | `OUTPUT_FIELDS` from the reader constants         |
| `ranking/coregates.py`   | Imports `read_output` from the reader             |
| `ranking/cullgates.py`   | Imports `read_output` from the reader             |
| `sentence_pack/audit.py` | Imports `read_output`; fixes the Sphinx reference |
| `languages/*/core.py`    | Delete the binding function                       |
| `languages/*/cull.py`    | Delete the binding function                       |
| `docs/adr/`              | Amend ADR 0036; add one ADR                       |
| `docs/`                  | Add a layout row; refresh the reader paths        |

The new file holds `InputCard`, `clean_english`, `sound_name`, `read_input_deck`, `read_output_deck`, `read_output`, the five `OUTPUT_*` constants, and `read_cards`.

## Tests

| Area        | Cases                                                              |
| ----------- | ------------------------------------------------------------------ |
| Reader      | One package, one open; cards in input order with fresh indices     |
| Reader      | A card with no audio file drops, and the count is returned         |
| Scan        | Runs on a card list, with no package on disk                       |
| Core flow   | Keeps the `Dropped, no audio file` line                            |
| Cull flow   | Prints the same line; the `"fake"` case no longer writes a package |
| Output read | One open; the note type and deck name still resolve                |
| Field names | `assemble.OUTPUT_FIELDS` equals the reader constants               |

## Safety and open questions

| Question                                   | Assumed default                                           |
| ------------------------------------------ | --------------------------------------------------------- |
| Does the reserve reader join task 1?       | Yes: same module, the per-target choice stays in the cull |
| Does the hook deletion land with the move? | Yes: separate waves, same day                             |
| Does any written deck change?              | No: no card, order, or manifest change                    |
| Does the pack producer need a reader?      | No: it reads TSV files with direct calls already          |

Owner of each question: boscoh.

The cull report gains one line. The report is printed, not written, so no committed artifact changes.

Re-run the four core gates and the two cull gates after wave 2.

## Task breakdown

| Id  | Wave | Outcome                                                            | Needs | Risk                       |
| --- | ---- | ------------------------------------------------------------------ | ----- | -------------------------- |
| T1  | 1    | Readers live in `deckreader.py`, no behaviour change               | none  | Low: a pure move           |
| T2  | 2    | The hook is gone, both flows call `read_cards`, both reports agree | T1    | Medium: touches both flows |
| T3  | 2    | ADR 0036 amended, one ADR added, docs refreshed                    | T1    | Low                        |
| T4  | 3    | The suite and the six gate runs pass                               | T2    | Low                        |

| Id  | Name                   | Tier     | Thinking | Worktree branch    | Commit |
| --- | ---------------------- | -------- | -------- | ------------------ | ------ |
| T1  | `inputreader-build-1`  | mid      | medium   | `inputreader/move` | yes    |
| T2  | `inputreader-build-2`  | mid      | medium   | `inputreader/seam` | yes    |
| T3  | `inputreader-build-3`  | fast     | low      | `inputreader/docs` | yes    |
| T4  | `inputreader-review-1` | frontier | high     | none               | no     |

| Id  | Verify command                           |
| --- | ---------------------------------------- |
| T1  | `uv run pytest -q`                       |
| T2  | `uv run pytest -q`                       |
| T3  | none, prose only                         |
| T4  | `uv run pytest -q` and the six gate runs |

Allowed files (table C, split for width):

- T1: `anki_tools/ranking/{deckreader,coreflow,coregates,cullgates,assemble,cullflow}.py`, `anki_tools/sentence_pack/audit.py`, `anki_tools/languages/*/{core,cull}.py`, `tests/test_deckreader.py`
- T2: `anki_tools/ranking/{coreflow,scan,deckprofile,cullflow}.py`, `anki_tools/languages/*/{core,cull}.py`, `tests/{test_core_deck,test_shared_coreflow}.py`, `tests/cull_fixture.py`
- T3: `docs/adr/`, `docs/anki-tools-layout.md`, `docs/{core-deck-flow,cull-deck-flow,gates,filters}.md`
- T4: none

Checks after each merge, in the parent checkout:

1. `uv run pytest -q`
2. `uv run anki fix fr --core --check`
3. `uv run anki fix es --core --check`
4. `uv run anki fix zh --core --check`
5. `uv run anki fix fr --select --check`
6. `uv run anki fix es --select --check`

Merge order per wave: the branch that changes the shared contract first, then the branch with the most files.

### Rollout

1. Land T1 alone. Confirm no output byte changes.
2. Land T2 and T3 in the same hour.
3. Run T4. If a gate fails, add a fix task to wave 3, do not amend a test.

## Appendix: current paths

| Name                    | File today                            |
| ----------------------- | ------------------------------------- |
| `InputCard`             | `ranking/coreflow.py`                 |
| `read_input_deck`       | `ranking/coreflow.py`                 |
| `note_values`           | `ranking/coreflow.py`                 |
| `read_output_deck`      | `ranking/coreflow.py`                 |
| `read_output`           | `ranking/coreflow.py`                 |
| `OUTPUT_*` constants    | `ranking/coreflow.py`                 |
| `load_input_deck` field | `ranking/scan.py`                     |
| The hook call           | `ranking/scan.py`                     |
| The cull direct call    | `ranking/cullflow.py`                 |
| The reserve read        | `ranking/cullflow.py`                 |
| The five bindings       | `languages/french/cull.py` and 4 more |

Find each one with `cymbal search` or `rg`. Line numbers move, so this table does not state them.
