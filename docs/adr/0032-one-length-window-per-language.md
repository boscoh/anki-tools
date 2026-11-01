# One length window per language

The pack, the cull, and the core each had their own length window. The pack read
`min_units` and `max_units` from the language profile. The cull had `MIN_UNITS` and
`MAX_UNITS` constants, and each `core.py` file had its own pair. The three flows
counted units in three ways, and the `english_words` gate had one window for the pack
and another for the cull and the core.

Status: accepted. ADR 0034 supersedes the English window (`english_bounds`) part.

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

A run can replace one end of the unit window. The flags are `--min-words` and
`--max-words`. They exist on `anki swadesh pack` and on `anki fix <lang>` in the
`--select` and `--core` modes. An empty flag keeps the profile value.
`CullRequest` and `CoreRequest` carry the values as `min_words` and `max_words`. Both
default to `None`.

Each run writes the window that it used to the manifest. Gate G5 (cull) and gate K5
(core) read the window from the stored params. K5 falls back to the profile for a
manifest without the params. A deck built with a flag therefore passes `--check`.

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

The Mandarin window is much tighter than before. The flags let a user keep the old
window for one run, for example `--min-words 2 --max-words 8`.

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
