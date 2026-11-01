# Gates

| Field   | Value                                                  |
| ------- | ------------------------------------------------------ |
| Status  | Consolidation done; pack P1 and one output reader open |
| Date    | 2026-11-01                                             |
| Owner   | anki-tools                                             |
| Related | `docs/filters.md`, ADR 0021, ADR 0027, ADR 0057        |

## 1. What a gate is

A gate reads a built file, reports one fault, and returns a pass or a fail. A
gate never rejects a sentence and never writes a file. A filter rejects a
sentence or a card before selection (`docs/filters.md`).

Each flow owns its gate prefix (ADR 0057): pack `P`, cull `R`, core `K`. Pack
`P5` and cull `R5` are different checks. The prefix names the flow.

| Flow            | Command                            | Gate ids                                        | Rows  |
| --------------- | ---------------------------------- | ----------------------------------------------- | ----- |
| Sentence pack   | `anki swadesh pack <lang>`         | P1 to P19, one list for each language           | 15-16 |
| Cull deck       | `anki fix <lang> <input> --select` | R1, R2, R5, R6                                  | 4     |
| Core deck       | `anki fix <lang> <input> --core`   | K5, K6, K13, K23, K24                           | 5     |
| Vocabulary deck | `anki swadesh vocab <lang>`        | none                                            | 0     |

The pack lists differ by language:

| Language        | Ids                                 | Rows |
| --------------- | ----------------------------------- | ---- |
| French, Spanish | P1 to P8, P10, P11, P13 to P16, P19 | 15   |
| Greek           | P1 to P8, P10, P11, P13 to P16, P18 | 15   |
| Mandarin        | P1 to P16                           | 16   |
| Cantonese       | P1 to P8, P10, P11, P13 to P18      | 16   |

Each flow doc gives the check of each pack, cull, and core row:
`docs/swadesh-sentence-flow.md`, `docs/cull-deck-flow.md`, and
`docs/core-deck-flow.md`.

## 2. What each gate checks

### Shared checks

More than one flow runs these checks. The cull and core rows call one function
in `ranking/gates.py` and pass their own values. The similarity limit is the
parameter.

| Check                  | Pack | Cull | Core | Plain meaning                                       |
| ---------------------- | ---- | ---- | ---- | --------------------------------------------------- |
| `gate_word_table`      | P1   | -    | -    | The word table is valid. Five copies exist (open).  |
| `gate_filter_recheck`  | P14  | R5   | K5   | Every kept card passes the filter rules.            |
| `gate_similarity`      | -    | R6   | K6   | No two cards are almost the same sentence.          |

### Cull-only and core-only checks

| Id  | Check                     | Plain meaning                                             |
| --- | ------------------------- | --------------------------------------------------------- |
| R1  | `gate_card_cap`           | The deck has the cap number of cards, or the pool ends.   |
| R2  | `gate_use_cap`            | No lemma appears in more cards than the cap.              |
| K13 | `gate_target_in_sentence` | Each card sentence holds its targets and its Core word.   |
| K23 | `gate_coverage_floor`     | The share of targets with a kept card reaches the floor.  |
| K24 | `gate_recount_forms`      | Each card holds its tier S forms, by substring or lemma.  |

The pack has more checks, for example P10 coverage. The cull has no coverage
gate (ADR 0045, ADR 0046). The core has one floor gate, K23 (ADR 0048). The
floor is `coverage_floor` in the language core profile: the current share
rounded down to 0.05. It is a ratchet, and a drop below it means the selection
or the input got worse. K24 is the independent recount (ADR 0052, ADR 0052
amendment): it imports no producer code and tests the form by a plain substring
match. A form absent from the text is a fault. A form present is not proof,
because `得到` is a substring of `听得到`. Mandarin and Cantonese use the plain
match. French and Spanish headwords inflect (`savoir`, `sais`), so
`CoreProfile.recount_model` names a spaCy model, and a form also holds when each
part of it is a lemma or a word of the text, or shares a stem of 4 characters
with one. Neither flow has a gate on the share of targets with one card or two
cards (ADR 0033).

### Retired ids

A removed row leaves a gap in the numbering. The ids are not reused.

| Flow | Retired ids                                                    | ADR                          |
| ---- | -------------------------------------------------------------- | ---------------------------- |
| Cull | R3, R4, R7 to R18                                              | 0033, 0035, 0039 to 0045     |
| Core | K1 to K4, K7 to K12, K14 to K22                               | 0038 to 0044, 0046           |
| Pack | sense, attribution, typography, license, and script gates      | 0039                         |

## 3. The gate contract

`anki_tools/gates.py` holds the contract. It sits below `ranking` and
`sentence_pack`, and it imports neither package.

| Part   | Name                                | Location                               |
| ------ | ----------------------------------- | -------------------------------------- |
| Gate   | `Gate(name, check)`                 | `anki_tools/gates.py`                  |
| Row    | `GateResult(name, passed, message)` | `anki_tools/gates.py`                  |
| Runner | `run_gates(gates, files)`           | `anki_tools/gates.py`                  |
| Reader | `GateFiles`, `GateFileReader`       | `anki_tools/gates.py`, `gate_files.py` |
| Report | `report(results, title)`            | `ranking/gates.py`                     |

A flow supplies a gate table, a file reader, and a title:

| Flow          | Gate table                              | Reader        | Title                   |
| ------------- | --------------------------------------- | ------------- | ----------------------- |
| Sentence pack | the language tuple plus `GENERIC_GATES` | `AuditInputs` | `Swadesh sentence pack` |
| Cull deck     | `GATES` in `ranking/cullgates.py`       | `CullFiles`   | `Cull deck`             |
| Core deck     | `GATES` in `ranking/coregates.py`       | `CoreFiles`   | `Core deck`             |

The pack declares `(name, check)` tuples, and a pack check reads
`(inputs, profile)`. `as_gates` in `sentence_pack/audit.py` wraps the pack table
and supplies the profile from the reader. Keep this form. The `check(files)`
form would touch about 60 pack check functions and gain nothing. Change it only
when a pack check must serve a second flow.

## 4. Deliberate differences

- The pack gates read input artifacts: the word table, the review CSV, and the
  sentence CSV. The cull and core gates read the output APKG, the output CSVs,
  and the manifest.
- Pack `P6` uses `profile.similarity` on the sentence text. Cull `R6` and core
  `K6` use the lemma Jaccard value on the full lemma set of a card.
- The pack manifest has a second shape. `sentence_pack/pack.py` writes it, and
  `P11` reads it. The rank manifest comes from `ranking/assemble.py`.
- The `sense` filter is the one test of the word sense (ADR 0039). No gate
  repeats it. Core `K13` stays: it tests that the form is in the sentence. It
  tests each credited target and the word in the `Core` field of the card
  (ADR 0049). A card with an empty `Core` field has no Core word to test.
- Cull `R1` checks the cap or the pool end. The pool ends when no passed card
  keeps the reason `max_cards`. The old core `K1` count check is retired
  (ADR 0043).
- Cull `R2` and `R6` read their limits from the producer constants, not from
  the manifest. Cull `R1` reads the cap from the manifest, because the user sets
  it with `--cards` (ADR 0046).
- Cull `R5` skips the reserve rows. They have their own length window
  (ADR 0046). It calls `filter_cards`, so it uses the same `unique` rule as the
  producer: the sentence key and the English key (ADR 0047).
- Mandarin `P12` checks the native audio authors, not every sentence author, so
  it holds its own function.

## 5. Decisions

The consolidation kept every gate id and every gate order. It changed the
function behind an id:

- One runner, one row type, and one report. The pack uses `as_gates`.
- One check where the check was a copy. The language copies of the attribution,
  typography, license, native audio, targets-in-table, and target-once checks
  are gone.
- The function name states the check. For example, `gate_hit` is now
  `gate_target_in_sentence`, and the cull `gate_card_count` is now
  `gate_card_cap`.
- A row that could not fail left the table. Its value is a report line or a
  manifest value:
  - core `K3` is Covered twice
  - core `K4` is the outside counts of the manifest
  - core `K15` is part of `K19`
  - core `K17` is Rejected names and Rejected loan words
  - pack `P20` is the manifest value `noun_gender_missing`
- A gate left the table when one build step wrote both sides of its comparison
  (ADR 0041), or when it only ran a filter again or tested a rule that no card
  can break (ADR 0039).
- A gate must be able to fail on a wrong analysis, not only on a broken writer.
  A status that the writer derives from a count cannot fail when the gate
  derives it again from the same count (ADR 0043). A gate that excludes the
  failure that its name states is a gate that cannot fail.

Rejected:

- One gate id namespace (ADR 0027). A report names the flow, and the renumber
  would touch about 200 references for no gain.
- One identical gate set for every flow. The pack selects sentences and has no
  card slot. The cull and core fill card slots and have no concept quota.

Deferred: one rule record schema and one manifest shape. Neither removes a
current fault, so both wait for a third consumer.

## 6. Open work

Both items are in `anki-ujvj`.

- Pack `P1`. Five language modules define `gate_word_table`. The French and
  Spanish `check_word_table` take no `csv_path`, so one shared function needs a
  profile hook for the path.
- One output reader. The pack `read_apkg` returns a different shape from the
  rank `read_output`. One reader lets the pack share the audio check. This is
  the highest-risk item.

## 7. Verification

Run the focused tests:

```bash
uv run pytest tests/test_gates.py tests/test_gate_id_map.py \
  tests/test_mandarin_audit.py tests/test_french_cull.py \
  tests/test_core_deck.py tests/test_shared_coreflow.py
```

Then print the gate table of each flow, and compare it with the table before
the change:

```bash
uv run anki swadesh audit french
uv run anki fix fr apkg/french_1000.apkg --select --check
uv run anki fix fr apkg/french_core.apkg --core --check
```

A changed pass or fail is a defect. A changed table length is a defect too,
unless the change removes a report-only row.

## 8. References

| Path                                     | Content                                               |
| ---------------------------------------- | ----------------------------------------------------- |
| `docs/filters.md`                        | the rule keys and the filter differences              |
| `docs/adr/0021-one-gate-file-reader.md`  | the module above both packages                        |
| `docs/adr/0027-gate-id-namespaces.md`    | the per-flow gate ids                                 |
| `anki_tools/gates.py`                    | the gate value, the row, the runner, and the reader   |
| `anki_tools/gate_files.py`               | the lazy file reader                                  |
| `anki_tools/ranking/gates.py`            | the report and the shared rank checks                 |
| `anki_tools/sentence_pack/audit.py`      | `AuditInputs`, `as_gates`, and the shared pack checks |
| `anki_tools/ranking/cullgates.py`        | the cull gates                                        |
| `anki_tools/ranking/coregates.py`        | the core gates                                        |
| `anki_tools/ranking/corerecount.py`      | core gate K24, no producer import                     |
| `anki_tools/languages/<lang>/profile.py` | the language gates                                    |
