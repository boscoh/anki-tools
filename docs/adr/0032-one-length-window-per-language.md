# One length window per language

The pack, the cull, and the core each had their own length window. The pack read
`min_units` and `max_units` from the language profile. The cull had `MIN_UNITS` and
`MAX_UNITS` constants, and each `core.py` file had its own pair. The three flows
counted units in three ways, and the `english_words` gate had one window for the pack
and another for the cull and the core.

Status: accepted. ADR 0034 supersedes the English window (`english_bounds`) and the
`--min-words` and `--max-words` flags.

## Decision

The language profile states the window once.

- `min_units` and `max_units` are the unit window of a card. A unit is a word, or a
  Han character for Mandarin and Cantonese. `profile.unit_count` counts it in every
  flow.
- `english_bounds` is the window of English words. Its default is
  `DEFAULT_ENGLISH_BOUNDS = (2, 12)`. All flows read it.
- `filter_reason` runs the shape guard first, then `english_words`. The pack, the
  cull, and the core share this order.
- `fallback_min_units` is the scaffold minimum. It stays pack-only.

The flags `--min-words` and `--max-words` replaced one end of the window for a run.
ADR 0034 removes them.

## Windows

| Language  | Pack, cull, and core | Pack scaffold minimum |
| --------- | -------------------- | --------------------- |
| French    | 4 to 10              | 3                     |
| Spanish   | 4 to 10              | 3                     |
| Greek     | 4 to 10              | 3                     |
| Mandarin  | 4 to 7               | `FALLBACK_MIN_HAN`    |
| Cantonese | 3 to 12              | `FALLBACK_MIN_HAN`    |

## Consequences

The cull and core windows change for five decks. The estimate comes from the stored
CSV files:

| Deck           | Old window | New window | Kept cards that now fail |
| -------------- | ---------- | ---------- | ------------------------ |
| French cull    | 3 to 12    | 4 to 10    | 304 of 925               |
| Spanish cull   | 3 to 12    | 4 to 10    | 57 of 750                |
| French core    | 5 to 9     | 4 to 10    | 15 of 273                |
| Mandarin core  | 2 to 8     | 4 to 7     | 44 of 147                |
| Cantonese core | 2 to 7     | 3 to 12    | 3 of 125                 |
| Spanish core   | 4 to 10    | 4 to 10    | 0 of 267                 |

The Mandarin window is much tighter than before.

A card that fails both a shape rule and `english_words` now has the shape rule id as
its reason.

## Open

The `apkg/*.csv` files are not rebuilt. They are stale: they hold the old rule ids and
the old windows. A rebuild changes the kept sets in the table above. Review the Mandarin
window before the rebuild.

## Rejected

A window for each flow. The pack and the cull then disagree on which sentence is valid,
and the pack audit and the cull gates cannot compare their counts.

A K5 gate that reads only the profile. A run with a flag then fails its own `--check`.
