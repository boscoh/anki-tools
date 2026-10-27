"""Tests for the note type config codec.

The tests give raw protobuf bytes and open no package.
"""

from __future__ import annotations

from anki_tools.apkg import notetype

# Leading bytes of the note type config of "Spanish sentences cloze+" in
# apkg/Spanish 9000 sentences with native audio.apkg. The kind is 1, the sort
# field index 5, and the CSS starts at byte 7 with a length of 197.
CLOZE_CONFIG = bytes.fromhex("080110051ac5012e")

# Leading bytes of the note type config of "Mandarin (Chinese) Basic" in
# apkg/Mandarin (Chinese) (25-02-22).apkg. The kind field is absent and the CSS
# starts at byte 3.
STANDARD_CONFIG = bytes.fromhex("1af9022e63617264")

OTHER = b'{"tags":[],"vers":[]}'

#: A field 255 of seven bits of length. The tag needs two bytes.
FIELD_255 = bytes([0xFA, 0x0F, len(OTHER)]) + OTHER

CSS = b".card {}"

#: Kind 0, the CSS, then field 255.
CONFIG = bytes([0x08, 0x00, 0x1A, len(CSS)]) + CSS + FIELD_255


def test_the_parser_reads_the_kind_and_the_sort_index() -> None:
    """The field numbers come from the Anki proto message, not from the writer."""
    cloze = notetype.parse(CLOZE_CONFIG)
    standard = notetype.parse(STANDARD_CONFIG)

    assert cloze == notetype.Config(kind=1, sort_field_idx=5, css=".")
    assert standard.kind == 0
    assert standard.sort_field_idx is None
    assert standard.css == ".card"


def test_a_config_with_a_two_byte_tag_copies_whole() -> None:
    """Anki writes field 255, so a tag can hold more than one byte."""
    clone = notetype.clone(CONFIG, ".card { color: red }", 1)
    with_kind = notetype.with_kind(CONFIG, 2)

    assert clone.endswith(FIELD_255)
    assert FIELD_255 in with_kind
    assert notetype.parse(clone).css == ".card { color: red }"


def test_with_css_replaces_the_css_in_place() -> None:
    """The other fields keep their place, so the wire order does not change."""
    config = bytes([0x08, 0x01, 0x10, 0x02, 0x1A, len(CSS)]) + CSS + FIELD_255

    out = notetype.with_css(config, "new")

    assert out == bytes([0x08, 0x01, 0x10, 0x02, 0x1A, 0x03]) + b"new" + FIELD_255


def test_with_kind_replaces_the_kind_in_its_place() -> None:
    """The kind stays where the input holds it, and the CSS goes last."""
    out = notetype.with_kind(CONFIG, 2)

    assert out == bytes([0x08, 0x02]) + FIELD_255 + bytes([0x1A, len(CSS)]) + CSS


def test_with_kind_adds_a_kind_that_the_input_lacks() -> None:
    """A config with no kind field gets one at the end, before the CSS."""
    config = bytes([0x1A, len(CSS)]) + CSS + FIELD_255

    out = notetype.with_kind(config, 1)

    assert out == FIELD_255 + bytes([0x08, 0x01, 0x1A, len(CSS)]) + CSS


def test_clone_writes_the_sort_index_and_the_css_first() -> None:
    """The clone drops the source sort index and CSS, and it keeps the kind."""
    config = bytes([0x08, 0x01, 0x10, 0x05, 0x1A, len(CSS)]) + CSS + FIELD_255

    out = notetype.clone(config, "new", 3)

    assert out == (
        bytes([0x10, 0x03, 0x1A, 0x03])
        + b"new"
        + bytes([0x08, 0x01])
        + FIELD_255
    )


def test_the_template_config_round_trips() -> None:
    """The question format is field 1 and the answer format is field 2."""
    config = notetype.build_template("{{Front}}", "{{Back}}")

    assert config == bytes([0x0A, 9]) + b"{{Front}}" + bytes([0x12, 8]) + b"{{Back}}"
    assert notetype.parse_template(config) == ("{{Front}}", "{{Back}}")


def test_a_wire_type_other_than_0_or_2_ends_the_message() -> None:
    """A fixed64 field ends the read. with_css keeps it and the clone drops it."""
    config = bytes([0x08, 0x01, 0x19, 0x01])

    assert notetype.parse(config).kind == 1
    assert notetype.with_css(config, "new") == config
    assert notetype.clone(config, "new", 0) == (
        bytes([0x10, 0x00, 0x1A, 0x03]) + b"new" + bytes([0x08, 0x01])
    )
