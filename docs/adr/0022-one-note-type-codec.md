# One codec for the note type config

Three of the last sixty commits fixed a fault in the hand-rolled protobuf code
inside `AnkiPackage`. `ff86acf` read a tag as one byte, so a tag whose field
number is 16 or more broke. The code sat in a class of 56 methods and 1735
lines, and no test reached it without a package file.

Status: accepted

## Decision

`anki_tools/apkg/notetype.py` owns the note type config message. The interface
is six calls:

| Call | Contract |
| --- | --- |
| `parse(config) -> Config` | The kind, the sort field index, and the CSS |
| `with_css(config, css) -> bytes` | The CSS replaced in its wire place |
| `with_kind(config, kind) -> bytes` | The kind in its place, or at the end, and the CSS last |
| `clone(config, css, sort_field_idx) -> bytes` | Sort field index and CSS first, then the other fields |
| `parse_template(config) -> (qfmt, afmt)` | The two strings of a template config |
| `build_template(qfmt, afmt) -> bytes` | The template config |

`Config` is a frozen dataclass. The module holds the field-number constants
`FIELD_KIND`, `FIELD_SORT_IDX`, and `FIELD_CSS`, and the varint framing. It
imports no sqlite and no language package. `AnkiPackage` keeps the sqlite
statements and calls the module. There is no protobuf dependency.

The three writers keep their own wire order, because the packages that this
project writes depend on it. `with_css` replaces the CSS in place.
`with_kind` drops the CSS field and writes it last. `clone` writes the sort
field index and the CSS first and drops the source fields of those two numbers.
A field of an unknown wire type ends the message, as before.

## Consequences

A test gives raw bytes and opens no package,
`tests/test_apkg_notetype.py`. The `ff86acf` case, field 255 with a two-byte
tag, has its own test in that file for `clone` and for `with_kind`.

The three writers share one parse walk, `_read`, which gives each field its
slice of the input bytes. So a field that this module does not read passes
through unchanged.

## Rejected

A protobuf dependency. Three messages and twelve field numbers do not need one,
and `anki_tools/apkg/` imports no language package, so a dependency that needs
code generation is a larger seam.
