# Delete the reorder module and the rank CSV

`ranking/reorder.py` read a rank CSV and did two jobs: it set the `due` value of
each card, and it deleted cards. The core flow was the only caller. It wrote
only rows with `kept` equal to 1, and it never asked for the unranked removal.
So the delete code never ran. The flow also wrote the deck twice, through a
temporary `.tmp.apkg`, to feed the module.

Status: accepted. It supersedes the `reorder_deck` parts of ADR 0011 and
ADR 0017.

## Decision

- Delete `anki_tools/ranking/reorder.py`, `reorder_deck`, and `load_ranking`.
- `DeckRun.card_order` replaces `rank_rows`. It lists the note ids of the kept
  cards in rank order.
- `assemble._write_ordered` writes the deck once, then sets `due` to the rank of
  each new card with `AnkiPackage.set_card_due`.
- The core flow no longer writes `<stem>.rank.csv`. No step read it.
- `write_ranking_csv` in `ranking/rank.py` stays for inspection.

## Consequences

- One deck write, no temporary package, and about 100 fewer lines.
- `test_the_run_writes_the_side_files_the_manifest_and_the_report` checks the
  `due` value of the core output.
- Specs written before this ADR still name `reorder_deck` and `.rank.csv`. They
  are records of the design at that time.
