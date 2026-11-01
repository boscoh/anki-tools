# Retire the sense, attribution, typography, license, and script gates

Status: accepted.

## Decision

These gates are gone. The ids are not reused (ADR 0027).

| Gate                      | Ids removed                                                        |
| ------------------------- | ------------------------------------------------------------------ |
| `gate_sense` (shared)     | cull G18, core K22                                                 |
| `gate_no_sense_mismatch`  | pack G21 (French, Spanish, Cantonese), G20 (Greek), G18 (Mandarin) |
| `gate_no_false_friend`    | pack G17 (Mandarin), G20 (Cantonese)                               |
| `gate_attribution`        | pack G12 (French, Spanish, Greek, Cantonese)                       |
| `gate_typography`         | pack G17 (French, Spanish)                                         |
| `gate_no_blocked_license` | pack G18 (French, Spanish)                                         |
| `gate_script`             | pack G9 (Cantonese), G17 (Greek)                                   |

The code also removes `sense_mismatches` and the tests of the removed gates. The
Mandarin attribution gate G12 stays. It checks the native audio authors, and it
is a different function.

## Reason

The `sense` filter is the one test of the word sense (ADR 0030). It runs in the
pack, the cull, and the core. The sense gates only ran the same test again on
the built file. The compound test is part of `sense`, so `gate_no_false_friend`
was a third copy.

The `script` and `digit_marks` filters already reject a Latin letter, a digit,
and a polytonic mark. Pack G14 runs them again on the built cards, so the
Cantonese and Greek script gates repeated G14.

The attribution, typography, and license gates test no filter. They tested a
text step or a legal rule that no card of the current packs can break. Every
French and Spanish card uses TTS audio, so no native recording exists. The
attribution file is a build output, and the pack writes it from the same rows.

## Consequences

- Pack gate lists: French and Spanish keep 15 rows, Greek 15 rows, Mandarin 16
  rows (G1 to G16), and Cantonese 16 rows.
- Cull `--check` runs G1, G2, G5 to G13, G15 to G17. Core `--check` runs K1, K2,
  K5 to K14, K16, and K18 to K21.
- A stale build can keep a card that fails `sense`. No gate finds it now. Run
  the build again.
- Core K13 and K16 stay. They test that the form is in the sentence and count
  the unchecked targets.
- The Greek letter and final sigma tests of `script_problem` still run on the
  word table. They do not run on a sentence.

## Rejected

Put the `sense` test in pack G14, cull G5, and core K5. This keeps a recheck but
adds a tier lookup to each recheck for no new fault class.
