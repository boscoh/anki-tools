# Gates

| Field   | Value                                                  |
| ------- | ------------------------------------------------------ |
| Status  | Consolidation done; pack G1 and one output reader open |
| Date    | 2026-11-01                                             |
| Owner   | anki-tools                                             |
| Related | `docs/filters.md`, ADR 0021, ADR 0027                  |

## 1. What a gate is

A gate reads a built file, reports one fault, and returns a pass or a fail. A
gate never rejects a sentence and never writes a file. A filter rejects a
sentence or a card before selection. `docs/filters.md` covers the filters.

Each flow owns its gate ids (ADR 0027). Pack `G5` is the one-time target check.
Cull `G5` is the filter recheck. A doc row, a report line, and an issue must
name the flow with the id.

One table shows the four flows at a glance:

| Flow            | Command                              | Gate table               | Filter table         |
| --------------- | ------------------------------------ | ------------------------ | -------------------- |
| Sentence pack   | `anki swadesh pack <lang>`           | G1 to G21, 18 to 20 rows | 14 rule ids          |
| Cull deck       | `anki fix <lang> <input> --select`   | G1 to G17                | 9 rule ids           |
| Core deck       | `anki fix <lang> <input> --core`     | K1 to K21, 18 rows       | 9 rule ids, C1 to C5 |
| Vocabulary deck | `anki swadesh vocab <lang>`          | none                     | none                 |

The filter column lives in `docs/filters.md`.

The reverse direction reads each check as a row and each flow as a column.

| Layer  | Check                     | Sentence pack                    | Cull deck                         | Core deck                         | Vocabulary deck |
| ------ | ------------------------- | -------------------------------- | --------------------------------- | --------------------------------- | --------------- |
| gate   | `gate_word_table`         | G1                               | -                                 | -                                 | -               |
| gate   | `gate_filter_recheck`     | G14                              | G5                                | K5                                | -               |
| gate   | `gate_similarity`         | -                                | G6                                | K6                                | -               |
| gate   | `gate_audio`              | -                                | G7                                | K7                                | -               |
| gate   | `gate_counts`             | -                                | G8                                | K8                                | -               |
| gate   | `gate_identity`           | -                                | G10                               | K9                                | -               |
| gate   | `gate_kept_order`         | -                                | G11                               | K10                               | -               |
| gate   | `gate_manifest_hashes`    | -                                | G13                               | K12                               | -               |
| gate   | `gate_templates`          | -                                | G15                               | K14                               | -               |
| gate   | `gate_core_word`          | -                                | G17                               | K19                               | -               |
| gate   | `gate_output_model`       | -                                | G16                               | K21                               | -               |
| gate   | `gate_attribution`        | G12 (Fr Es El Yue)               | -                                 | -                                 | -               |
| gate   | `gate_no_sense_mismatch`  | G21 (Fr Es Yue) G20 (El) G18 (zh) | -                                 | -                                 | -               |
| gate   | `gate_typography`         | G17 (Fr Es)                      | -                                 | -                                 | -               |
| gate   | `gate_no_blocked_license` | G18 (Fr Es)                      | -                                 | -                                 | -               |
| gate   | `gate_no_false_friend`    | G17 (zh) G20 (Yue)               | -                                 | -                                 | -               |
| gate   | `gate_script`             | G9 (Yue) G17 (El)                | -                                 | -                                 | -               |
| filter | `english_link`            | yes                              | -                                 | -                                 | -               |
| filter | `length_window`           | yes                              | yes                               | yes                               | -               |
| filter | `script`                  | yes                              | -                                 | -                                 | -               |
| filter | `repeat`                  | yes                              | -                                 | -                                 | -               |
| filter | `final_mark`              | yes                              | yes                               | yes                               | -               |
| filter | `one_final_mark`          | yes                              | yes                               | yes                               | -               |
| filter | `quote`                   | yes                              | yes                               | yes                               | -               |
| filter | `internal_mark`           | yes                              | yes                               | yes                               | -               |
| filter | `predicate`               | yes                              | -                                 | -                                 | -               |
| filter | `orphan`                  | yes                              | -                                 | -                                 | -               |
| filter | `english_words`           | yes                              | yes                               | yes                               | -               |
| filter | `unique`                  | yes                              | yes                               | yes                               | -               |
| filter | `unknown`                 | yes                              | -                                 | -                                 | -               |
| filter | `sense`                   | yes                              | -                                 | -                                 | -               |
| filter | `digit_marks`             | -                                | yes                               | yes                               | -               |
| filter | `propn`                   | -                                | yes                               | yes                               | -               |
| filter | C1 to C5                  | -                                | -                                 | yes                               | -               |

Language codes: Fr French, Es Spanish, El Greek, Yue Cantonese, zh Mandarin.
A filter id names one check in every flow that runs it. Each flow keeps its own
order. `docs/filters.md` §2 holds the meaning of each id and the order of each flow.

## 2. The gate contract

`anki_tools/gates.py` holds the contract. It sits below `ranking` and
`sentence_pack`, and it imports neither package.

| Part   | Name                                | Location                               | Flows            |
| ------ | ----------------------------------- | -------------------------------------- | ---------------- |
| Gate   | `Gate(name, check)`                 | `anki_tools/gates.py`                  | pack, cull, core |
| Row    | `GateResult(name, passed, message)` | `anki_tools/gates.py`                  | pack, cull, core |
| Runner | `run_gates(gates, files)`           | `anki_tools/gates.py`                  | pack, cull, core |
| Reader | `GateFiles`, `GateFileReader`       | `anki_tools/gates.py`, `gate_files.py` | pack, cull, core |
| Report | `report(results, title)`            | `ranking/gates.py`                     | pack, cull, core |

A flow supplies a gate table, a file reader, and a title.

| Flow            | Gate table                              | Reader        | Title                   |
| --------------- | --------------------------------------- | ------------- | ----------------------- |
| Sentence pack   | the language tuple plus `GENERIC_GATES` | `AuditInputs` | `Swadesh sentence pack` |
| Cull deck       | `GATES` in `ranking/cullgates.py`       | `CullFiles`   | `Cull deck`             |
| Core deck       | `GATES` in `ranking/coregates.py`       | `CoreFiles`   | `Core deck`             |
| Vocabulary deck | none                                    | none          | none                    |

The pack declares `(name, check)` tuples, and a pack check reads
`(inputs, profile)`. `as_gates` in `sentence_pack/audit.py` wraps the pack table
and supplies the profile from the reader.

Keep the pack form. The `check(files)` form would touch about 60 pack check
functions and gain nothing. Change it only when a pack check must serve a
second flow.

## 3. The gate tables

| Flow                           | Ids                                | Rows |
| ------------------------------ | ---------------------------------- | ---- |
| Sentence pack, French, Spanish | G1 to G8, G10 to G19, G21          | 19   |
| Sentence pack, Greek           | G1 to G8, G10 to G18, G20          | 18   |
| Sentence pack, Mandarin        | G1 to G18                          | 18   |
| Sentence pack, Cantonese       | G1 to G18, G20, G21                | 20   |
| Cull deck                      | G1 to G17                          | 17   |
| Core deck                      | K1, K2, K4 to K14, K16, K18 to K21 | 18   |
| Vocabulary deck                | none                               | 0    |

A removed row leaves a gap in the numbering. Each flow doc gives the check of
each row: `docs/swadesh-sentence-flow.md`, `docs/cull-deck-flow.md`,
`docs/core-deck-flow.md`.

## 4. Shared checks

The table lists the checks that more than one flow runs. A check outside the
table lives in one flow only, and the flow doc holds it.

| Family          | Function                  | Flows and ids                                                         |
| --------------- | ------------------------- | --------------------------------------------------------------------- |
| Word table      | `gate_word_table`         | pack G1, five copies, open                                            |
| Filter recheck  | `gate_filter_recheck`     | pack G14, cull G5, core K5, one name and three bodies                 |
| Similarity      | `gate_similarity`         | cull G6, core K6, one rule and two limits                             |
| Audio valid     | `gate_audio`              | cull G7, core K7                                                      |
| Media count     | `gate_counts`             | cull G8 `gate_note_count`, core K8                                    |
| Identity        | `gate_identity`           | cull G10, core K9                                                     |
| Order           | `gate_kept_order`         | cull G11, core K10 `gate_order`                                       |
| Manifest hashes | `gate_manifest_hashes`    | cull G13, core K12 `gate_hashes`                                      |
| Templates       | `gate_templates`          | cull G15 `gate_card_templates`, core K14                              |
| Core word       | `gate_core_word`          | cull G17, core K19                                                    |
| Output model    | `gate_output_model`       | cull G16, core K21                                                    |
| Attribution     | `gate_attribution`        | pack G12 of French, Spanish, Greek, Cantonese                         |
| English sense   | `gate_no_sense_mismatch`  | pack G21 of French, Spanish, Cantonese; G20 of Greek; G18 of Mandarin |
| Typography      | `gate_typography`         | pack G17 of French, Spanish                                           |
| Blocked license | `gate_no_blocked_license` | pack G18 of French, Spanish                                           |
| False friend    | `gate_no_false_friend`    | pack G17 of Mandarin, pack G20 of Cantonese                           |
| Script          | `gate_script`             | pack G9 of Cantonese, pack G17 of Greek                               |

The cull and core rows call the one function of `ranking/gates.py` and pass
their own values. The similarity limit, the manifest file keys, the CSS check,
and the kept rows are parameters.

The coverage family holds four checks that measure different things: pack G10,
cull G3, cull G4, and core K2. They stay separate.

## 5. Deliberate differences

- The pack gates read input artifacts: the word table, the review CSV, the
  sentence CSV. The cull and core gates read the output APKG, the output CSVs,
  and the manifest.
- Pack `G6` uses `profile.similarity` on the sentence text. Cull `G6` and core
  `K6` use the lemma Jaccard value.
- The pack manifest has a second shape. `sentence_pack/pack.py` writes it, and
  `G11` reads it. The rank manifest comes from `ranking/assemble.py`.
- Cull `G5` exempts `propn` for a reserve card. No pack rule tests a name, and
  `G14` needs those cards.
- Core `K1` checks a three-way count agreement. Cull `G1` checks the cap or the
  pool end. The two functions stay separate.
- Mandarin `G12` checks the native audio authors, not every sentence author, so
  it holds its own function.

## 6. Decisions

The consolidation kept every gate id and every gate order. It changed the
function behind an id:

- One runner, one row type, and one report. The pack uses `as_gates`.
- One check where the check was a copy. The language copies of the
  attribution, typography, license, native audio, targets-in-table, and
  target-once checks are gone.
- The function name states the check, for example `gate_hit` is
  `gate_target_in_sentence` and `gate_card_count` is `gate_card_cap`.
- A row that could not fail left the table. Its value is a report line or a
  manifest value: core `K3` is Covered twice, core `K15` is part of `K19`,
  core `K17` is Rejected names and Rejected loan words, and pack `G20` is the
  manifest value `noun_gender_missing`.

Rejected:

- One gate id namespace, ADR 0027. A report names the flow, and the renumber
  would touch about 200 references for no gain.
- One identical gate set for every flow. The pack selects sentences and has no
  card slot. The cull and core fill card slots and have no concept quota.

Deferred: one rule record schema and one manifest shape. Neither removes a
current fault, so both wait for a third consumer.

## 7. Open work

- Pack `G1`. Five language modules define `gate_word_table`. The French and
  Spanish `check_word_table` take no `csv_path`, so one shared function needs a
  profile hook for the path.
- One output reader. The pack `read_apkg` returns a different shape from the
  rank `read_output`. One reader lets pack `G13` use `gate_output_model`, and it
  lets the pack share the audio and template checks. This is the highest-risk
  item.

Both items are in `anki-ujvj`.

## 8. Verification

Run the focused tests:

```bash
uv run pytest tests/test_gates.py tests/test_gate_id_map.py \
  tests/test_mandarin_audit.py tests/test_french_cull.py \
  tests/test_core_deck.py tests/test_shared_coreflow.py
```

Then print the gate table of each flow, and compare it with the table before the
change:

```bash
uv run anki swadesh audit french
uv run anki fix fr apkg/french_1000.apkg --select --check
uv run anki fix fr apkg/french_core.apkg --core --check
```

A changed pass or fail is a defect. A changed table length is a defect too,
unless the change removes a report-only row.

## 9. References

| Path                                     | Content                                               |
| ---------------------------------------- | ----------------------------------------------------- |
| `docs/filters.md`                        | the rule keys and the filter differences              |
| `docs/adr/0021-one-gate-file-reader.md`  | the module above both packages                        |
| `docs/adr/0027-gate-id-namespaces.md`    | the per-flow gate ids                                 |
| `anki_tools/gates.py`                    | the gate value, the row, the runner, and the reader   |
| `anki_tools/gate_files.py`               | the lazy file reader                                  |
| `anki_tools/ranking/gates.py`            | the report and the shared rank checks                 |
| `anki_tools/sentence_pack/audit.py`      | `AuditInputs`, `as_gates`, and the shared pack checks |
| `anki_tools/ranking/cullgates.py`        | cull gates G1 to G17                                  |
| `anki_tools/ranking/coregates.py`        | core gates K1 to K21                                  |
| `anki_tools/languages/<lang>/profile.py` | the language gates                                    |
