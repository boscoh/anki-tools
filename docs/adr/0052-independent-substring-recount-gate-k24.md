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

## Amendment 2026-11-01: French and Spanish

Bead anki-7j46. `CoreProfile.exact_forms` is gone. `CoreProfile.recount_model`
is a spaCy model name, empty for the plain substring test (Mandarin, Cantonese).
French (`fr_core_news_sm`) and Spanish (`es_core_news_sm`) set it, and K24 now
runs there.

- K24 lemmatizes the card text with the model, loaded once per gate run. It
  imports spaCy but no producer code. The import test allows `spacy`.
- A form holds when it is a substring of the text, or when each part of it
  (split on spaces and apostrophes, one-letter parts ignored) is a lemma or a
  word of the text, or shares a stem of 4 characters with one. The parts of
  `ne pas` need not touch, and `acostarse` holds on `acostar` and `acostarme`.
- The model and the build both use spaCy, so a model error is shared. The stem
  test is loose: `venir` holds on `venin`. The gate catches a form that the
  text lacks entirely, not a near form.
- Measured on the committed decks: French 277 kept cards, 382 forms, 2 flags;
  Spanish 273 kept cards, 356 forms, 0 flags. Before the part and stem rules the
  model alone gave 8 and 4 flags, all but two noise: `ne pas`, `s'étendre`,
  `quelques`, three Spanish reflexives, and `cantamos` (a model lemma error).
  The plain substring test flagged about 43 French and 75 Spanish cards.
- The 2 French flags are real: `fumée` (smoke, noun) is credited to cards that
  hold `fumer` (verb), notes 1485727927673 and 1485727931725. The decks are not
  rebuilt, so K24 fails on the French deck until the owner decides.

Greek has no core profile, so nothing changes there.

## Rejected

Run K24 on every language and accept the failures. A gate that always fails
hides the real faults.

Match the bold `<b>` text that the writer adds. The writer picks the bold with
the producer match rule, so the check would share the code under test.
