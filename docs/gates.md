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
sentence or a card before selection (`docs/filters.md`).

Each flow owns its gate ids (ADR 0027). Pack `G5` and cull `G5` are different
checks. A doc row, a report line, and an issue must name the flow with the id.

| Flow            | Command                            | Gate ids                                        | Rows  |
| --------------- | ---------------------------------- | ----------------------------------------------- | ----- |
| Sentence pack   | `anki swadesh pack <lang>`         | G1 to G19, one list for each language           | 15-16 |
| Cull deck       | `anki fix <lang> <input> --select` | G1, G2, G5, G6, G8, G9, G12, G13, G16, G17      | 10    |
| Core deck       | `anki fix <lang> <input> --core`   | K1, K2, K5, K6, K8, K11 to K13, K16, K18 to K21 | 13    |
| Vocabulary deck | `anki swadesh vocab <lang>`        | none                                            | 0     |

The pack lists differ by language:

| Language        | Ids                                 | Rows |
| --------------- | ----------------------------------- | ---- |
| French, Spanish | G1 to G8, G10, G11, G13 to G16, G19 | 15   |
| Greek           | G1 to G8, G10, G11, G13 to G16, G18 | 15   |
| Mandarin        | G1 to G16                           | 16   |
| Cantonese       | G1 to G8, G10, G11, G13 to G18      | 16   |

Each flow doc gives the check of each pack, cull, and core row:
`docs/swadesh-sentence-flow.md`, `docs/cull-deck-flow.md`, and
`docs/core-deck-flow.md`.

## 2. What each gate checks

### Shared checks

More than one flow runs these checks. The cull and core rows call one function
in `ranking/gates.py` and pass their own values. The similarity limit, the
manifest file keys, and the media check are the parameters.

| Check                  | Pack | Cull | Core | Plain meaning                                       |
| ---------------------- | ---- | ---- | ---- | --------------------------------------------------- |
| `gate_word_table`      | G1   | -    | -    | The word table is valid. Five copies exist (open).  |
| `gate_filter_recheck`  | G14  | G5   | K5   | Every kept card passes the filter rules.            |
| `gate_similarity`      | -    | G6   | K6   | No two cards are almost the same sentence.          |
| `gate_counts`          | -    | G8   | K8   | The deck, the CSV, and the manifest counts agree.   |
| `gate_manifest_hashes` | -    | G13  | K12  | The manifest hashes match the two CSV files.        |
| `gate_output_model`    | -    | G16  | K21  | One note type, the right fields, one card per note. |
| `gate_core_word`       | -    | G17  | K19  | Every note has a core word, a gloss, and bold text. |

Two names differ in the code. Cull G8 is `gate_note_count`, and core K12 is
`gate_hashes`. Both call the shared function. Core K8 also tests that the media
set equals the card audio. Cull G9 tests the media set alone.

### Cull-only and core-only checks

| Id  | Check                     | Plain meaning                                             |
| --- | ------------------------- | --------------------------------------------------------- |
| G1  | `gate_card_cap`           | The deck has the cap number of cards, or the pool ends.   |
| G2  | `gate_use_cap`            | No lemma appears in more cards than the cap.              |
| G9  | `gate_media`              | The deck holds exactly the audio files of the kept cards. |
| G12 | `gate_coverage_shape`     | The coverage CSV has one row for each target word.        |
| K1  | `gate_card_count`         | The CSV, the deck, and the manifest give the same count.  |
| K2  | `gate_covered_targets`    | Every target that can have a card has a card.             |
| K11 | `gate_coverage_agreement` | The coverage CSV agrees with the card CSV.                |
| K13 | `gate_target_in_sentence` | Each card sentence holds its target word.                 |
| K16 | `gate_unchecked_sense`    | The unchecked targets stay inside the limit.              |
| K18 | `gate_tail`               | Each tail card adds a new content lemma.                  |
| K20 | `gate_unranked`           | The unranked tier F targets stay inside the limit.        |

The pack has more checks, for example G10 coverage. The three coverage checks
(pack G10, cull G12, core K2) measure different things and stay separate. The
cull has no gate on the share of targets with one card or two cards (ADR 0033).

### Retired ids

A removed row leaves a gap in the numbering. The ids are not reused.

| Flow | Retired ids                                                | ADR                          |
| ---- | ---------------------------------------------------------- | ---------------------------- |
| Cull | G3, G4, G7, G10, G11, G14, G15, G18                        | 0033, 0035, 0039, 0040, 0041 |
| Core | K3, K4, K7, K9, K10, K14, K15, K17, K22                    | 0038, 0039, 0040, 0041       |
| Pack | sense, attribution, typography, license, and script gates  | 0039                         |

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
- Pack `G6` uses `profile.similarity` on the sentence text. Cull `G6` and core
  `K6` use the lemma Jaccard value on the full lemma set of a card.
- The pack manifest has a second shape. `sentence_pack/pack.py` writes it, and
  `G11` reads it. The rank manifest comes from `ranking/assemble.py`.
- The `sense` filter is the one test of the word sense (ADR 0039). No gate
  repeats it. Core `K13` and `K16` stay: `K13` tests that the form is in the
  sentence, and `K16` counts the unchecked targets.
- Core `K1` checks a three-way count agreement. Cull `G1` checks the cap or the
  pool end. The two functions stay separate.
- Mandarin `G12` checks the native audio authors, not every sentence author, so
  it holds its own function.

## 5. Decisions

The consolidation kept every gate id and every gate order. It changed the
function behind an id:

- One runner, one row type, and one report. The pack uses `as_gates`.
- One check where the check was a copy. The language copies of the attribution,
  typography, license, native audio, targets-in-table, and target-once checks
  are gone.
- The function name states the check. For example, `gate_hit` is now
  `gate_target_in_sentence`, and `gate_card_count` is now `gate_card_cap`.
- A row that could not fail left the table. Its value is a report line or a
  manifest value:
  - core `K3` is Covered twice
  - core `K4` is the outside counts of the manifest
  - core `K15` is part of `K19`
  - core `K17` is Rejected names and Rejected loan words
  - pack `G20` is the manifest value `noun_gender_missing`
- A gate left the table when one build step wrote both sides of its comparison
  (ADR 0041), or when it only ran a filter again or tested a rule that no card
  can break (ADR 0039).

Rejected:

- One gate id namespace (ADR 0027). A report names the flow, and the renumber
  would touch about 200 references for no gain.
- One identical gate set for every flow. The pack selects sentences and has no
  card slot. The cull and core fill card slots and have no concept quota.

Deferred: one rule record schema and one manifest shape. Neither removes a
current fault, so both wait for a third consumer.

## 6. Open work

Both items are in `anki-ujvj`.

- Pack `G1`. Five language modules define `gate_word_table`. The French and
  Spanish `check_word_table` take no `csv_path`, so one shared function needs a
  profile hook for the path.
- One output reader. The pack `read_apkg` returns a different shape from the
  rank `read_output`. One reader lets pack `G13` use `gate_output_model`, and it
  lets the pack share the audio check. This is the highest-risk item.

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
| `anki_tools/languages/<lang>/profile.py` | the language gates                                    |
