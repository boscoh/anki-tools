# Add core gate K24: a recount that imports no producer code

Core gates K5, K6, and K13 call the tokenizer, the lemma rules, and the sense
test of the build, and K23 reads the coverage the build wrote. A bug in one of
those passes the build and its gates alike. K24 reads the written files and the
standard library only, and it tests that each kept card text holds the form of
each tier S target and of the Core word, by a plain substring match.

Status: accepted

## Decision

- `ranking/corerecount.py` holds `gate_recount_forms`. A test reads its imports
  and fails when one is not `re`, `typing`, or `__future__`.
- A form absent from the text is a fault. A form present is not proof: `得到`
  is a substring of `听得到`. A pass means no form is missing, not that every
  hit is a correct one. The gate does not try to repair this over-match, because
  a repair needs the tokenizer.
- The gate tests tier S forms only. A tier F target is a lemma, and an
  inflected text does not hold it.
- `CoreProfile.exact_forms` is true for Mandarin and Cantonese. On French and
  Spanish the Swadesh headwords inflect (`savoir` in the table, `sais` in the
  card), so a substring test fails on 75 and 81 cards that the build placed
  correctly. K24 passes there and says `skipped`.

## Consequences

On the four core decks K24 passes on Mandarin (149 forms) and Cantonese (155
forms) and is skipped on French and Spanish. French and Spanish have no
independent recount until a check that needs no tokenizer exists, for example a
stem list that a person reviews. Gate K24 does not replace K13.

## Rejected

Run K24 on every language and accept the failures. A gate that always fails
hides the real faults.

Match the bold `<b>` text that the writer adds. The writer picks the bold with
the producer match rule, so the check would share the code under test.
