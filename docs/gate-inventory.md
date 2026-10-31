# Gate inventory and refactor

| Field   | Value                                                   |
| ------- | ------------------------------------------------------- |
| Status  | Proposed                                                |
| Date    | 2026-11-01                                              |
| Owner   | anki-tools                                              |
| Related | `docs/gate-filter-comparison.md`, `specs/gate-layer.md` |

## 1. Purpose

The four deck builders hold 59 gate table rows and about 77 gate functions.
About 21 functions are copies of another function. Some rows only report and
cannot fail. This document lists the gates by family and gives the
consolidation, rename, repurpose, and refactor plan.

The implementation plan for these actions is in `specs/gate-layer.md`.

The gate ids stay as they are. ADR 0027 records the per-flow namespaces. A
change below changes the function behind an id, not the id.

The gate runner and the row type are a separate change. See
`specs/gate-layer.md`. This document covers the gate content.

## 2. Scope

In scope:

- The gate functions of the pack, the cull, and the core.
- The language gate functions of French, Spanish, Greek, Cantonese, and
  Mandarin.
- The function names, the shared checks, and the report-only rows.

Out of scope:

- The gate ids and their order. ADR 0027.
- The gate runner and the row type. `specs/gate-layer.md`.
- The filter rules and their order.
- The deck-specific gate failures: `anki-0s3`, `anki-lb1s`, `anki-ann4`,
  `anki-tazk`.

## 3. Where the code lives

| Part                 | File                                      |
| -------------------- | ----------------------------------------- |
| Pack generic checks  | `anki_tools/sentence_pack/audit.py`       |
| Pack language checks | `anki_tools/languages/<lang>/profile.py`  |
| Mandarin pack checks | `anki_tools/languages/mandarin/stages.py` |
| Cull checks          | `anki_tools/ranking/cullgates.py`         |
| Core checks          | `anki_tools/ranking/coregates.py`         |
| Shared rank checks   | `anki_tools/ranking/gates.py`             |
| Gate reader          | `anki_tools/gate_files.py`                |
| Runner and row type  | `specs/gate-layer.md`                     |

## 4. Inventory by family

The table groups the gates by the property that they check. A shared family
can hold a different id in each flow.

| Family            | What the gate checks                       | Flows and ids                                           |
| ----------------- | ------------------------------------------ | ------------------------------------------------------- |
| Word table        | the reviewed table passes every S1 check   | pack G1                                                 |
| Unique forms      | no duplicate word form                     | pack G2                                                 |
| Targets in table  | every target is a table form               | pack G3, G19; Greek G19                                 |
| Unique Tatoeba id | no repeated source sentence                | pack G4                                                 |
| Target once       | the target appears one time                | pack G5                                                 |
| Similarity        | no near-duplicate pair                     | pack G6, cull G6, core K6                               |
| Sentence audio    | every card has a valid sentence MP3        | pack G7                                                 |
| Media count       | the audio file count equals the APKG media | pack G8                                                 |
| Word boundary     | the target matches a whole token           | pack G9 (FR, ES, EL)                                    |
| Coverage          | coverage reaches the gate share            | pack G10, cull G3, cull G4, core K2, core K3            |
| Manifest counts   | counts and hashes agree                    | pack G11, cull G8, cull G13, core K1, core K8, core K12 |
| Attribution       | the author section names every author      | pack G12                                                |
| Output model      | the APKG model, fields, and ordinals       | pack G13, cull G16, core K21                            |
| Filter recheck    | the kept cards pass the filters            | pack G14, cull G5, core K5                              |
| Review file       | no open row in the review CSV              | pack G15                                                |
| Word audio        | every target has a valid word MP3          | pack G16                                                |
| Language tail     | typography, script, gender, license, sense | pack G17 to G21                                         |
| Card cap          | the count is the cap or the pool end       | cull G1                                                 |
| Use cap           | no lemma exceeds the use cap               | cull G2                                                 |
| Audio valid       | every card has a valid MP3                 | cull G7, core K7                                        |
| Media set         | the media map equals the kept cards        | cull G9                                                 |
| Identity          | the deck title equals the manifest         | cull G10, core K9                                       |
| Order             | the kept order digest                      | cull G11, core K10                                      |
| Coverage rows     | the coverage CSV shape and counts          | cull G12, core K11                                      |
| Swadesh           | the reserve coverage                       | cull G14                                                |
| Templates         | the card templates and the CSS             | cull G15, core K14                                      |
| Core word         | the word, the gloss, and the bold          | cull G17, core K19                                      |
| Bold target       | the bold text is the target                | core K15                                                |
| Outside set       | the outside set and rare ratio             | core K4                                                 |
| Target hit        | the card holds its target form             | core K13                                                |
| Sense limit       | the unchecked targets stay below the limit | core K16                                                |
| Name report       | the rejected names and loan words          | core K17                                                |
| Tail              | every tail card brings a lemma             | core K18                                                |
| Unranked          | the unranked targets stay below the limit  | core K20                                                |

## 5. Consolidate

### 5.1 Language copies

The function is the same. Only the form attribute or the section name differs.

| Copies                          | One name                  | Note                                     | Risk |
| ------------------------------- | ------------------------- | ---------------------------------------- | ---- |
| pack G1 in FR, ES, EL, YUE, CMN | `gate_word_table`         | the profile supplies the CSV path        | low  |
| pack G12 in FR, ES, EL, YUE     | `gate_attribution`        | the profile supplies the section name    | low  |
| pack G9 in FR, ES, EL           | `gate_word_boundary`      | the same token match                     | low  |
| pack G17 in FR, ES              | `gate_typography`         | the same NFC and space check             | low  |
| pack G18 in FR, ES              | `gate_no_blocked_license` | the same license list                    | low  |
| pack G20 in FR, ES              | `gate_noun_gender`        | the manifest count differs today         | low  |
| pack G18 and Greek G18          | `gate_no_native_audio`    | the same row test                        | low  |
| pack G19 and Greek G19          | `gate_target_in_table`    | Cantonese already uses the generic check | low  |

Mandarin G12 is not in the list. It checks the native audio authors, not every
sentence author.

### 5.2 Rank pairs

The checks are the same intent. The code differs in the details.

| Copies             | One name               | Note                                           | Risk                               |
| ------------------ | ---------------------- | ---------------------------------------------- | ---------------------------------- |
| cull G6, core K6   | `gate_similarity`      | the limit comes from the params                | low                                |
| cull G13, core K12 | `gate_manifest_hashes` | the file keys are a parameter                  | low                                |
| cull G15, core K14 | `gate_templates`       | the CSS check is a parameter                   | low                                |
| cull G8, core K8   | `gate_counts`          | the core also checks the media set             | low                                |
| cull G11, core K10 | `gate_kept_order`      | the cull reads the CSV, the core the rank rows | low                                |
| cull G7, core K7   | `gate_audio`           | use the stronger cull check                    | medium: the core can start to fail |
| core K15 and K19   | `gate_core_word`       | K19 also checks the bold target                | low                                |

The card count is not a pair. Cull G1 checks the cap or the pool end. Core K1
checks a three-way agreement. Keep two functions.

## 6. Rename

The function name changes. The gate id and the check do not.

| Current                 | Proposed                  | Flow             |
| ----------------------- | ------------------------- | ---------------- |
| `gate_covered1`         | `gate_covered_targets`    | core K2          |
| `gate_covered2`         | `gate_coverage_twice`     | core K3          |
| `gate_hit`              | `gate_target_in_sentence` | core K13         |
| `gate_sense`            | `gate_unchecked_sense`    | core K16         |
| `gate_names`            | `gate_name_report`        | core K17         |
| `gate_card_count`       | `gate_card_cap`           | cull G1          |
| `gate_filter`           | `gate_filter_recheck`     | cull G5, core K5 |
| `gate_proper_sentences` | `gate_filter_recheck`     | pack G14         |
| `gate_target_coverage`  | `gate_coverage_once`      | cull G3          |
| `gate_double_coverage`  | `gate_coverage_twice`     | cull G4          |
| `gate_targets`          | `gate_target_in_table`    | Greek G19        |
| `gate_no_native`        | `gate_no_native_audio`    | Greek G18        |
| `gate_coverage`         | `gate_coverage_agreement` | core K11         |
| `gate_coverage_rows`    | `gate_coverage_shape`     | cull G12         |

Cull G12 and core K11 have close names today, but the checks differ. The new
names state the difference.

## 7. Repurpose

A gate table row must be able to fail. Four rows cannot fail today.

| Row      | Today                         | Change                                                                   |
| -------- | ----------------------------- | ------------------------------------------------------------------------ |
| core K3  | reports the two-card share    | give it `double_coverage_gate`, as cull G4 has, or move it to the report |
| core K17 | reports the rejected names    | move it to the report, or fail above a limit                             |
| pack G20 | reports the empty noun gender | move it to the report                                                    |
| core K15 | checks the bold target        | merge into K19                                                           |

A report-only value belongs in the report section, not in the gate table. This
is a behavior change at the report level only. No deck changes.

## 8. Refactor

1. **One common pack list.** The five language modules repeat G2 to G8, G10 to
   G11, and G13 to G16. Add `common_gates()` and let each language append G1,
   G9, G12, and G17 and above. This removes five copies of thirteen pairs.
2. **One output reader.** The pack `read_apkg` returns a different shape from
   the rank `read_output`. One reader lets pack G13 use `gate_output_model`,
   and it lets the pack share the audio and template checks. This is the
   highest-risk item.
3. **One `read_json` and one `read_csv`.** The pack and the rank each hold a
   copy.
4. **One kept-row helper and one audio-valid helper.**
5. **One gate table builder.** After `specs/gate-layer.md` lands, every flow
   declares `Gate` values. The builder can hold the common list.

## 9. Order

| Tier | Work                      | Behavior                          | Risk          |
| ---- | ------------------------- | --------------------------------- | ------------- |
| 1    | the eight language copies | none                              | low           |
| 2    | the seven rank pairs      | none, except the core audio check | low to medium |
| 3    | the renames               | none                              | low           |
| 4    | the report-only rows      | report and table length           | low           |
| 5    | the common pack list      | none                              | low           |
| 6    | the output reader         | pack G13 changes                  | high          |

Tier 1 and tier 2 remove about 21 function definitions. Tier 4 removes three
rows. The remaining rows keep their ids and their order.

## 10. Verification

After each tier, run the focused tests:

```bash
uv run pytest tests/test_gates.py tests/test_gate_id_map.py \
  tests/test_mandarin_audit.py tests/test_french_cull.py \
  tests/test_core_deck.py tests/test_shared_coreflow.py
```

Then run the gate table of each flow and compare it with the table before the
tier:

```bash
uv run anki swadesh audit french
uv run anki fix fr apkg/french_1000.apkg --select --check
uv run anki fix fr apkg/french_core.apkg --core --check
```

A changed pass or fail is a defect, except for the core audio check in tier 2
and the report-only rows in tier 4.

## 11. References

| Path                                              | Content                                    |
| ------------------------------------------------- | ------------------------------------------ |
| `docs/gate-filter-comparison.md`                  | the gate and filter comparison             |
| `specs/gate-layer.md`                             | one gate runner and one row type           |
| `docs/adr/0027-gate-id-namespaces.md`             | the per-flow gate ids                      |
| `anki_tools/sentence_pack/audit.py`               | the pack generic checks                    |
| `anki_tools/ranking/cullgates.py`                 | the cull checks                            |
| `anki_tools/ranking/coregates.py`                 | the core checks                            |
| `anki_tools/ranking/gates.py`                     | the shared rank checks                     |
| `anki_tools/languages/cantonese/profile.py`       | the Cantonese checks, a model for the rest |
