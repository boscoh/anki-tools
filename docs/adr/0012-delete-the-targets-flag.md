# Give the cull one size knob and keep the reserve switch

The select mode of `anki fix fr` and `anki fix es` took three flags: `--cards`,
`--targets`, and `--reserve`. ADR 0009 kept all three. A review of how each
value reaches the output gave one removal:

- `--cards` sets `CullRequest.max_cards`, default 1000. The select and core
  modes read it. Rank mode reads no card cap, but the command accepted the flag
  without a message.
- `--targets` set `CullRequest.target_words`. That field defaults to 300 in
  `anki_tools/ranking/cullflow.py`, and the CLI default was
  `cullflow.DEFAULT_TARGET_WORDS`, the same 300. So the flag restated the
  request default, and no spec example, script, or test passed another value.
- `--reserve` sets `no_reserve=not reserve`. Spec section 16.8 measures two
  builds, the frequency-only cull and the cull with the Swadesh reserve, so this
  switch has two real values. ADR 0011 removed flags that restate one default
  and kept the flags that choose between measured builds.

`target_words` does change the output. It sets the exempt top targets, the
weights of the cover, and the coverage target list. The removal moves that
decision into the request, where the other measured presets already live.

Status: accepted

## Decision

Delete `--targets` from `fix fr` and `fix es`. Keep `--cards` and `--reserve`.

- The select mode passes no `target_words`, so the `CullRequest` default of 300
  applies. A different target count is a code edit of that field.
- The README states that `--cards` applies to `--select` and `--core` only.
  Rank mode reads no card cap.
- `docs/cull-deck-flow.md` names `target_words` as a request field, not a flag.
- The French cull spec drops the `--targets` row and gives `target_words` its
  place among the request fields. The `--reserve` row stays.
- The two tests that set `reserve=False` stay. No test used `targets`.

## Consequences

- A select build with the CLI defaults gives the same output: 300 targets, a
  cap of 1000 cards, and the reserve cards on.
- `anki fix fr --select --targets 400` fails. The run needs the request field.
- The select mode takes one size knob, `--cards`. A reader no longer sees two
  numbers that both claim to size the deck.
- `fix fr` and `fix es` take one parameter fewer. `--cards` and `--reserve`
  stay, so the two commands keep the same option list.
- ADR 0009 stays as the record of the option set of that date. This ADR removes
  one of its survivors.

## Alternatives

Keep `--targets`. It fails the test of ADR 0011: its CLI default equals the
request default, and no caller sets another value.

Fold `--targets` into `--cards`. The two values are different: the target list
holds 300 terms, and the card cap is 1000. One number cannot set both without
changing the measured build.

Delete `--reserve` too. The spec measured the frequency-only build, so the off
value is a real deck, not a restatement.

Add one generic override flag, for example `--option target_words=400`. ADR
0011 rejects such a flag: it opens every measured value again.
