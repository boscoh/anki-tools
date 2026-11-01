# Retire the audio gate G7 and K7

Cull gate G7 and core gate K7 shared one function, `gate_audio`. It failed a card
with no sound tag, a card whose audio name was not in the package, and a card
whose MP3 file had no duration. The duration test ran `ffprobe` on every kept
file of the output.

Status: accepted.

## Decision

The cull and the core no longer have the audio gate. The code removes
`gate_audio`, its two `Gate` entries, `probe_audio`, and the `audio` key of
`read_output`. The ids G7 and K7 are not reused (ADR 0027).

Both input readers drop a note with no sound tag. `drop_silent_cards` drops a card
whose audio file is not in the input media map. The cull flow now calls it, as
the core flow already did (ADR 0036).

## Reason

- A card with no sound tag cannot reach the output. `read_input_deck` and
  `read_output_deck` skip the note.
- A card with a missing file cannot reach the output. `drop_silent_cards`
  removes it before selection. The gates still catch a build fault that loses a
  file: cull G9 `gate_media` and core K8 `gate_counts` compare the media set with
  the card audio.
- The MP3 duration test is not needed. The `ffprobe` run costs time on every
  kept file of the output.

## Consequences

The cull `--check` runs 13 gates: G1, G2, G5, G6, G8 to G13, and G15 to G17. The
core `--check` runs 16 gates: K1, K2, K5, K6, K8 to K14, K16, and K18 to K21.

A silent or corrupt MP3 in the input deck now passes. The pack flow keeps its
own audio gates, G7 and G16, because it makes the audio files.

## Rejected

A duration probe of the input files at read time. It runs `ffprobe` on every
input card, and the cull keeps only a small part of them.
