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

| Flow            | Command                              | Gate table               | Filter table          |
| --------------- | ------------------------------------ | ------------------------ | --------------------- |
| Sentence pack   | `anki swadesh pack <lang>`           | G1 to G19, 15 to 16 rows | 7 rule ids            |
| Cull deck       | `anki fix <lang> <input> --select`   | G1 to G17, 10 rows       | 7 rule ids            |
| Core deck       | `anki fix <lang> <input> --core`     | K1 to K21, 13 rows       | 7 rule ids            |
| Vocabulary deck | `anki swadesh vocab <lang>`          | none                     | none                  |

The filter column lives in `docs/filters.md`.

The reverse direction reads each check as a row and each flow as a column.

| Layer  | Check                     | Sentence pack                    | Cull deck                         | Core deck                         | Vocabulary deck |
| ------ | ------------------------- | -------------------------------- | --------------------------------- | --------------------------------- | --------------- |
| gate   | `gate_word_table`         | G1                               | -                                 | -                                 | -               |
| gate   | `gate_filter_recheck`     | G14                              | G5                                | K5                                | -               |
| gate   | `gate_similarity`         | -                                | G6                                | K6                                | -               |
| gate   | `gate_counts`             | -                                | G8                                | K8                                | -               |
| gate   | `gate_manifest_hashes`    | -                                | G13                               | K12                               | -               |
| gate   | `gate_core_word`          | -                                | G17                               | K19                               | -               |
| gate   | `gate_output_model`       | -                                | G16                               | K21                               | -               |
| filter | `length_window`           | yes                              | yes                               | yes                               | -               |
| filter | `digit_marks`             | yes                              | yes                               | yes                               | -               |
| filter | `script`                  | yes                              | yes                               | yes                               | -               |
| filter | `repeat`                  | yes                              | yes                               | yes                               | -               |
| filter | `marks`                   | yes                              | yes                               | yes                               | -               |
| filter | `unique`                  | yes                              | yes                               | yes                               | -               |
| filter | `sense`                   | yes                              | yes                               | yes                               | -               |

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

| Flow                           | Ids                                             | Rows |
| ------------------------------ | ----------------------------------------------- | ---- |
| Sentence pack, French, Spanish | G1 to G8, G10, G11, G13 to G16, G19             | 15   |
| Sentence pack, Greek           | G1 to G8, G10, G11, G13 to G16, G18             | 15   |
| Sentence pack, Mandarin        | G1 to G16                                       | 16   |
| Sentence pack, Cantonese       | G1 to G8, G10, G11, G13 to G18                  | 16   |
| Cull deck                      | G1, G2, G5, G6, G8, G9, G12, G13, G16, G17      | 10   |
| Core deck                      | K1, K2, K5, K6, K8, K11 to K13, K16, K18 to K21 | 13   |
| Vocabulary deck                | none                                            | 0    |

A removed row leaves a gap in the numbering. These ids are retired and not
reused:

- Cull G3, G4, G7, G10, G11, G14, G15, and G18.
- Core K4, K7, K9, K10, K14, and K22.
- Pack gates for sense, attribution, typography, license, and script.

The ADRs are 0033, 0035, 0038, 0039, 0040, and 0041. Each flow doc gives the
check of each row: `docs/swadesh-sentence-flow.md`,
`docs/cull-deck-flow.md`, and `docs/core-deck-flow.md`.

Plain meaning of the cull gates:

| Id  | Gate                   | Plain meaning                                               |
| --- | ---------------------- | ----------------------------------------------------------- |
| G1  | `gate_card_cap`        | The deck has the cap number of cards, or the pool is empty. |
| G2  | `gate_use_cap`         | No lemma appears in too many cards.                         |
| G5  | `gate_filter_recheck`  | Every kept card passes the filter rules.                    |
| G6  | `gate_similarity`      | No two cards are almost the same sentence.                  |
| G8  | `gate_note_count`      | The deck, the CSV, and the manifest give the same count.    |
| G9  | `gate_media`           | The deck holds exactly the audio files of the kept cards.   |
| G12 | `gate_coverage_shape`  | The coverage CSV has one row for each target word.          |
| G13 | `gate_manifest_hashes` | The two CSV files are not stale.                            |
| G16 | `gate_output_model`    | One note type, the right fields, one card per note.         |
| G17 | `gate_core_word`       | Every note has a core word, a gloss, and bold text.         |

Plain meaning of the core gates:

| Id  | Gate                      | Plain meaning                                            |
| --- | ------------------------- | -------------------------------------------------------- |
| K1  | `gate_card_count`         | The CSV, the deck, and the manifest give the same count. |
| K2  | `gate_covered_targets`    | Every target that can have a card has a card.            |
| K5  | `gate_filter_recheck`     | Every kept card passes the filter rules.                 |
| K6  | `gate_similarity`         | No two cards are almost the same sentence.               |
| K8  | `gate_counts`             | The note, media, and manifest counts agree.              |
| K11 | `gate_coverage_agreement` | The coverage CSV agrees with the card CSV.               |
| K12 | `gate_hashes`             | The two CSV files are not stale.                         |
| K13 | `gate_target_in_sentence` | Each card sentence holds its target word.                |
| K16 | `gate_unchecked_sense`    | Few targets have an unchecked word sense.                |
| K18 | `gate_tail`               | Each tail card adds a new content lemma.                 |
| K19 | `gate_core_word`          | Every note has a core word, a gloss, and bold text.      |
| K20 | `gate_unranked`           | Few targets have no rank.                                |
| K21 | `gate_output_model`       | One note type, the right fields, one card per note.      |

## 4. Shared checks

The table lists the checks that more than one flow runs. A check outside the
table lives in one flow only, and the flow doc holds it.

| Family          | Function               | Flows and ids                                         |
| --------------- | ---------------------- | ----------------------------------------------------- |
| Word table      | `gate_word_table`      | pack G1, five copies, open                            |
| Filter recheck  | `gate_filter_recheck`  | pack G14, cull G5, core K5, one name and three bodies |
| Similarity      | `gate_similarity`      | cull G6, core K6, one rule and two limits             |
| Counts          | `gate_counts`          | cull G8 `gate_note_count`, core K8                    |
| Manifest hashes | `gate_manifest_hashes` | cull G13, core K12 `gate_hashes`                      |
| Core word       | `gate_core_word`       | cull G17, core K19                                    |
| Output model    | `gate_output_model`    | cull G16, core K21                                    |

The cull and core rows call the one function of `ranking/gates.py` and pass
their own values. The similarity limit, the manifest file keys, and the media
check are parameters.

The coverage family holds three checks that measure different things: pack G10,
cull G12, and core K2. They stay separate. The cull has no gate on the share of
targets with one card or two cards (ADR 0033).

## 5. Deliberate differences

- The pack gates read input artifacts: the word table, the review CSV, the
  sentence CSV. The cull and core gates read the output APKG, the output CSVs,
  and the manifest.
- Pack `G6` uses `profile.similarity` on the sentence text. Cull `G6` and core
  `K6` use the lemma Jaccard value on the full lemma set of a card.
- The pack manifest has a second shape. `sentence_pack/pack.py` writes it, and
  `G11` reads it. The rank manifest comes from `ranking/assemble.py`.
- The `sense` filter is the one test of the word sense in the pack, the cull, and
  the core (ADR 0039). No gate repeats it. Core `K13` and `K16` stay: `K13` tests
  that the form is in the sentence, and `K16` counts the unchecked targets.
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
  core `K17` is Rejected names and Rejected loan words, core `K4` is the outside
  counts of the manifest, and pack `G20` is the manifest value
  `noun_gender_missing`.
- A gate that only runs a filter again, or tests a text step or a legal rule that
  no card can break, left the table (ADR 0039). The `sense` filter is the one test
  of the word sense.

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
| `anki_tools/ranking/cullgates.py`        | cull gates G1 to G17, 10 rows                         |
| `anki_tools/ranking/coregates.py`        | core gates K1 to K21, 13 rows                         |
| `anki_tools/languages/<lang>/profile.py` | the language gates                                    |
