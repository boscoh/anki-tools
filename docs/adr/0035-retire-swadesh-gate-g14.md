# Retire the Swadesh gate G14

Cull gate G14 failed when the deck covered fewer reserve concepts than the
Swadesh total minus the exceptions of the profile. The French cull gave 205 of
207, with 1 exception. The Spanish cull gave 200 of 207, with 4 exceptions.

Status: accepted.

## Decision

The cull no longer has gate G14. The code removes `gate_swadesh` and its `Gate`
entry. The id G14 is not reused. The ids G5 to G13 and G15 to G18 keep their
numbers (ADR 0027).

The manifest counts `swadesh_covered`, `swadesh_total`, and the exceptions stay.
The cull report still shows the Swadesh count.

## Reason

The gate needed a list of exceptions, and a person had to keep that list in step
with the data. A missing concept showed a gap in the input, not a fault in the
cull. In the French cull, the reserve card for `racine` and an input card had the
same sentence. The cull dropped the reserve card as a duplicate, and `max_cards`
then rejected the input copy. The deck lost the concept for a reason that no
gate rule could name.

## Consequences

The cull `--check` runs 15 gates: G1, G2, G5 to G13, and G15 to G18. A deck that
misses a reserve concept now passes. The reader sees the count in the report.

ADR 0033 says that G14 stays. This ADR replaces that part of ADR 0033.

## Rejected

A lower count or a longer exception list. Both only hide the gap.
