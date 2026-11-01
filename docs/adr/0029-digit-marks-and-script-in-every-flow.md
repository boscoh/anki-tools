# Run digit_marks and script in every flow

Each language `script_ok` hook mixed two tests: a digit test and an alphabet
test. The pack ran the hook as `script`. The cull and the core ran a separate
function, `script_rule`, as `digit_marks`. It rejected digits, brackets, links,
and ellipsis marks. The cull and the core had no alphabet test, and the pack had
no bracket, link, or ellipsis test.

Status: accepted.

## Decision

`digit_marks` is one guard in `sentence_pack/pool.py`. It has no language input.
It rejects any digit (`str.isdigit`), a bracket, a link mark, and an ellipsis
mark.

`script` keeps only the alphabet test. The five language `script_ok` hooks no
longer test a digit. French and Spanish accept only their own Latin letters.
Mandarin and Cantonese reject a Latin letter. Greek rejects a Latin letter and a
polytonic mark.

The shape guard table runs both guards after `length_window`. The pack, the cull,
and the core call `check_sentence_shape`, so the three flows share the order:
`length_window`, `digit_marks`, `script`, then the mark rules.

`LanguageDeckProfile` holds `script_ok`, a language fact. `CullProfile` and
`CoreProfile` receive it from the language profile. `CardShape` carries it to the
guard. The `Marks` protocol of ADR 0028 is now `ShapeProfile`, because it holds an
alphabet test and not only marks. `script_rule` and the mark constants of
`cardfilter.py` are gone.

## Consequences

The filter tables name 15 ids for the pack, 10 for the cull, and 15 for the core.

The cull and the core gain an alphabet test. Only the Cantonese core input holds
a Latin letter in a sentence. The names `Tom` and `LA` are examples.

The pack gains the bracket, link, and ellipsis test. No sentence in the current
sentence CSVs fails it, and the five pack audits give the same result as before.

The card counts before and after:

| Deck             | Input cards | Passed before | Passed after | Kept before | Kept after |
| ---------------- | ----------- | ------------- | ------------ | ----------- | ---------- |
| French core      | 5928        | 3338          | 3338         | 273         | 273        |
| Spanish core     | 9263        | 4719          | 4719         | 267         | 267        |
| Mandarin core    | 5009        | 2811          | 2811         | 147         | 147        |
| Cantonese core   | 999         | 923           | 865          | 126         | 125        |
| French cull      | 5928        | 4958          | 4958         | 924         | 924        |
| Spanish cull     | 9263        | 6421          | 6421         | 747         | 747        |

The Cantonese core deck lost three cards with a Latin name and gained two cards.
Gates K1 to K21 pass on the four rebuilt core decks. The French and Spanish cull
gates give the same failures as before (G17 for French; G3, G4, G14, and G17 for
Spanish).

## Open

The five sentence packs are not rebuilt. `data/tatoeba/` is not on this machine,
so the pool stage cannot run. A rebuild adds the `digit_marks` row to each pool
report and each manifest. `print_sentence_report` fails on an old report.

## Rejected

One guard that joins the digit test and the alphabet test. The French and Spanish
alphabet test accepts a set of letters, and the Chinese and Greek alphabet test
rejects a set of letters. Only the digit test has one rule for every language.
