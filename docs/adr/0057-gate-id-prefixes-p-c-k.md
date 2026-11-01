# Give each flow its own gate id prefix: P, C, K

The pack and the cull both numbered their gates `G1`, `G2`, and so on. One id
named two checks. ADR 0027 set the rule "an id is valid only inside its owner
table" and kept the shared letter. A reader still had to carry the flow name,
and a bare `G5` in a report or an issue was not clear.

Status: accepted. It supersedes ADR 0027 for the gate ids. The rule ids `F` of
ADR 0027 are not changed.

## Decision

The prefix names the flow. Each flow keeps its numbers. Only the letter changes.

| Flow | Old prefix | New prefix | Owner table                |
| ---- | ---------- | ---------- | -------------------------- |
| Pack | `G`        | `P`        | `sentence_pack/audit.py`   |
| Cull | `G`        | `C`        | `ranking/cullgates.py`     |
| Core | `K`        | `K`        | `ranking/coregates.py`     |

The pack map. Each pack gate keeps its number:

| Old | New | Old | New | Old | New |
| --- | --- | --- | --- | --- | --- |
| G1  | P1  | G8  | P8  | G15 | P15 |
| G2  | P2  | G9  | P9  | G16 | P16 |
| G3  | P3  | G10 | P10 | G17 | P17 |
| G4  | P4  | G11 | P11 | G18 | P18 |
| G5  | P5  | G12 | P12 | G19 | P19 |
| G6  | P6  | G13 | P13 | G20 | P20 |
| G7  | P7  | G14 | P14 | G21 | P21 |

The cull map. Four gates stay in the table. The other ids are retired, and they
are not reused:

| Old | New | State                                  |
| --- | --- | -------------------------------------- |
| G1  | C1  | live, `gate_card_cap`                  |
| G2  | C2  | live, `gate_use_cap`                   |
| G5  | C5  | live, `gate_filter_recheck`            |
| G6  | C6  | live, `gate_similarity`                |
| G3  | C3  | retired, ADR 0033                      |
| G4  | C4  | retired, ADR 0033                      |
| G7 to G18 | C7 to C18 | retired, ADR 0035, 0039 to 0045 |

The core gates `K<N>` do not change. The retired core rules `C3` to `C5`
(ADR 0036) stay retired. Their ids are not reused.

The old ids stay in the text of old ADRs. Do not edit an old ADR. Read a `G`
id in an old ADR as a pack id or a cull id from the flow of that ADR, and use
the maps above.

## Consequences

A bare `P5`, `C5`, or `K5` names one check. A doc row no longer needs the flow
name to be clear.

The letter `C` now has three uses that do not share a table:

- the cull gate ids `C1`, `C2`, `C5`, and `C6`;
- the cull stage ids `C0` to `C5` in `docs/cull-deck-flow.md`, in
  `specs/french-1000-sentence-cull.md`, and in the docstring of `run_cull`;
- the retired core rule ids `C3` to `C5` (ADR 0036) and the core filter rule
  ids `C1` to `C5` in `specs/core-word-small-deck.md`.

A stage id and a gate id can look the same in one document. The cull flow
document says "stage" or "gate" before each such id. A follow-up issue renames
the cull stage ids.

A new gate joins one table, and its prefix shows the flow.

## Rejected

Keep `G` and add a flow word to each id. A report line and an issue would
still have a bare id that is not clear.

Renumber the gates. The numbers appear in 119 pack code references and in every
committed spec table. A letter change gives the same clarity with no change to
the numbers.
