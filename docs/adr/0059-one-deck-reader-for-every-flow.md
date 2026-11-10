# One deck reader for every flow

The core flow read the input deck through the `load_input_deck` profile hook,
and the cull flow called `read_input_deck` in `ranking/coreflow.py` with the
same five values. Each mode view stated those five values, so five language
binding functions existed over one implementation, and two of them served the
other mode. No test replaced the hook, so the seam had one implementation and
zero adapters. The two read paths also opened the input APKG twice: once to
read the cards, and once in `drop_silent_cards` to read the media map.

Status: accepted.

## Decision

- The read cluster lives in `anki_tools/ranking/deckreader.py`:
  `InputCard`, `clean_english`, `sound_name`, `read_input_deck`,
  `read_output_deck`, `read_output`, the five `OUTPUT_*` field constants,
  `drop_silent_cards`, and `read_cards`. The module imports the standard
  library and `anki_tools.apkg` only. `coreflow` re-exports the names.
- Delete the `load_input_deck` hook from `CoreProfile` and from
  `LanguageDeckProfile` and its `core()` view. One implementation and zero
  test adapters do not justify a seam.
- Both flows call `read_cards(path, profile)` with the profile values, and no
  profile supplies a read function.
- `read_cards` opens the package once, reads the cards, drops the silent ones,
  and returns the cards and the count of the dropped cards.
- `drop_silent_cards` takes the media names, not a path, so it cannot open the
  package a second time.
- `scan(profile, cards, spoken_weight)` takes a card list. `ScanResult` holds
  only the tokens, the ranks, and the names.
- `assemble.OUTPUT_FIELDS` is built from the `OUTPUT_*` constants, so the five
  field names have one copy.
- The cull report gains the line `Dropped, no audio file`. The report is
  printed, so no committed artifact changes.

## Consequences

One read call serves both flows, and the input package opens once per run. A
language supplies values only. A new deck shape adds a reader function to
`deckreader.py`, not a field to a profile. The cull now prints a count that it
discarded before.

## Rejected

Give the cull the hook instead. One implementation would gain a second adapter
and no second behaviour, and the seam would still hold no test.

Keep `drop_silent_cards` behind the hook. The hook returned cards, so the drop
step needed the path again, and the package opened twice.
