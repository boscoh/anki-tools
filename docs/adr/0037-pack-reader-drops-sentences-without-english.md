# Drop sentences without English in the pack reader, and retire english_link

The sentence pack had the first guard `english_link`: a Tatoeba sentence passes
when a linked English sentence has a text. A sentence that failed it was skipped
and was not counted as a failure.

Status: accepted.

## Decision

The pack no longer has the rule `english_link`. The key leaves `FILTER_KEYS` and
`FILTER_NAMES` in `sentence_pack/pool.py`.

- `build_candidate_pool` pairs each linked sentence with its English text
  (`pair_english`) before the guard loop. The loop reads only the paired
  sentences.
- The report keeps `total_linked`. It adds `english_paired`, the count of linked
  sentences that have an English text. The printed report shows both.
- The other guards keep their order: `length_window` to `sense`.

## Reason

A sentence with no English text is not a card of the pack. This is a fact about
the input, not a rule about the sentence. The old guard never rejected a
sentence in the count, so it was a reader step in the form of a guard. The silent
card drop of the core flow has the same shape (ADR 0036).

## Consequences

- The pack report has 7 filter counts, not 8. `counts["english_link"]` is now
  `report["english_paired"]`.
- The set of accepted sentences does not change.
- The cull and the core flows are not affected. Their rule F1 (2 to 12 English
  words) already rejects a card with an empty English text.

## Rejected

Keep `english_link` as a guard. It needs a place in every guard table, test, and
doc row, and it never counts a failure.
