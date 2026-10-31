# Give each flow its own gate and rule id namespace

Four gate tables and two rule tables use the same three letters. The numbers
repeat, so one id names two different checks:

| Id | Pack table | Cull table |
| --- | --- | --- |
| `G5` | the target occurs once in a sentence | the kept card passes F1 to F9 |
| `G14` | the card is a complete sentence | the merged deck covers the reserve |

The rule tables repeat in the same way. Pack rule `F8` is "at most one internal
mark" (`sentence_pack/pool.py`). Ranking rule `F8` is "no `PROPN` token"
(`ranking/cardfilter.py`). Pack rule `F4` is a repeated sentence. Ranking rule
`F4` is the final sentence mark.

The two tables started apart. The pack F table entered at 0b728e5 (2026-10-18)
with F1 to F12. The ranking F table entered at f5859eb (2026-10-20) with F1 to
F9. The pack G table entered at b7c3e87 (2026-10-18). The cull G table entered
at 3e13e76 (2026-10-20). Each table numbers from `G1`.

ADR 0006 records part of the map in `MARK_RULES` (`ranking/cullflow.py`). The
rest of the map has no record.

Status: accepted.

## Decision

An id is valid only inside its owner table. A doc row, a report message, and an
issue must name the flow with the id.

| Id | Owner table | Size |
| --- | --- | --- |
| pack F | `sentence_pack/pool.py` | F1 to F12, `unknown`, `sense` |
| rank F | `ranking/cardfilter.py` | F1 to F9 |
| pack G | `sentence_pack/audit.py` | G1 to G21 |
| cull G | `ranking/cullgates.py` | G1 to G17 |
| core K | `ranking/coregates.py` | K1 to K21 |
| core C | `ranking/coreflow.py` | C1 to C5 |

The pack and ranking rule keys map as follows:

| Pack key | Rank key | Rule |
| --- | --- | --- |
| F1 | F1 | the English field: a link in the pack, a 2 to 12 word window in the ranking table |
| F2 | F2 | the length window |
| F3 | F3 | no digit, bracket, link, or ellipsis |
| F5 | F4 | the final sentence mark |
| F6 | F5 | exactly one final mark |
| F7 | F6 | no quotation mark |
| F8 | F7 | at most one internal mark |
| F12 | F9 | the sentence text is unique |
| none | F8 | no `PROPN` token |
| F4 | none | the repeated sentence |
| F9 | none | the predicate tag |
| F10 | none | the orphan conjunction |
| F11 | none | the English word count |
| `unknown` | none | the unknown token budget |
| `sense` | none | the English gloss sense |

The map has three shapes. Pack `F5` to `F8` shift down one id in the ranking
table, and one dict, `MARK_RULES`, holds that shift. Pack `F12` is rank `F9`.
Pack `F1` to `F3` keep their id, and pack `F1` holds a different rule from rank
`F1`. Five pack rules have no rank key. Rank `F8` has no pack key: the pack has
no name rule (`git log -S PROPN` returns ranking files only).

The mapping is one dict, `MARK_RULES`. It is not a second table.

## Consequences

A reader must carry the flow name. The spec tables and the deck flow documents
do this already: they name the deck, not a bare id.

A new rule joins one table alone, and the id alone cannot show it. The map test
`tests/test_gate_id_map.py` fails until the new key is in the map or in the
pack-only or rank-only list. The test reads both tables, so no rule reaches one
table only.

## Rejected

One shared rule table for the pack and the ranking. The pack measures concept
coverage and the ranking fills card slots, so the two rule sets differ on
purpose. The pack has no card slot, and the ranking has no concept quota.

Renumber the cull gates above `G17`. `ranking/gates.py:5` says "The gate numbers
stay stable". A rename of eight ids would touch 119 pack code references, 74
`F8` references, and every committed spec table, and it would change the ids of
the fixed packages.
