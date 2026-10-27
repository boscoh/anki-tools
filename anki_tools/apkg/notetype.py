"""Read and write the note type config protobuf of an anki21b package.

Anki holds one note type config in the ``config`` column of the ``notetypes``
table. It is a protobuf message, ``anki.notetypes.Notetype.Config``. This
project reads field 1, the kind; field 2, the sort field index; and field 3,
the CSS. Every other field passes through as bytes, in the wire order of the
input. The ``anki2`` format holds the note type as JSON, so it has no such
message.

The ``templates`` table holds a second message. Field 1 is the question format
and field 2 the answer format.

A field with a wire type other than 0 or 2 ends the message. :func:`with_css`
keeps the bytes that follow it, and the other calls drop them.

The module holds the varint framing and no sqlite. :class:`AnkiPackage` keeps
the statements. ADR 0022 records the seam.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Kind of the note type: field 1 of the config message.
FIELD_KIND = 1

#: Zero-based index of the sort field: field 2 of the config message.
FIELD_SORT_IDX = 2

#: CSS of the note type: field 3 of the config message.
FIELD_CSS = 3


@dataclass(frozen=True)
class Config:
    """The note type config values that this project reads."""

    kind: int
    sort_field_idx: int | None
    css: str


def parse(config: bytes) -> Config:
    """Read the kind, the sort field index, and the CSS of a note type config.

    :param config: Raw protobuf config bytes.
    :returns: The values. The kind is 0 and the sort field index is None when
        the input holds no such field.
    """
    kind = 0
    sort_field_idx: int | None = None
    css = ""
    for field in _read(config)[0]:
        if field.number == FIELD_KIND and field.wire == 0:
            kind = field.value
        elif field.number == FIELD_SORT_IDX and field.wire == 0:
            sort_field_idx = field.value
        elif field.number == FIELD_CSS and field.wire == 2:
            css = _text(field.payload)
    return Config(kind, sort_field_idx, css)


def with_css(config: bytes, css: str) -> bytes:
    """Rebuild a note type config with a new CSS.

    The call keeps every other field in its place, so the wire order stays. A
    config with no CSS field gets none.

    :param config: Raw protobuf config bytes.
    :param css: New CSS string.
    :returns: Protobuf config bytes.
    """
    fields, tail = _read(config)
    parts = [
        _bytes_field(FIELD_CSS, css.encode("utf-8"))
        if field.number == FIELD_CSS and field.wire == 2
        else field.raw
        for field in fields
    ]
    return b"".join(parts) + tail


def with_kind(config: bytes, kind: int) -> bytes:
    """Rebuild a note type config with a new kind.

    The kind stays in its place when the input holds one, else it goes at the
    end. The CSS goes last, as bytes that this module writes.

    :param config: Raw protobuf config bytes.
    :param kind: Note type kind.
    :returns: Protobuf config bytes.
    """
    fields, _tail = _read(config)
    parts: list[bytes] = []
    found = False
    for field in fields:
        if field.number == FIELD_CSS and field.wire == 2:
            continue
        if field.number == FIELD_KIND:
            parts.append(_varint_field(FIELD_KIND, kind))
            found = True
        else:
            parts.append(field.raw)
    if not found:
        parts.append(_varint_field(FIELD_KIND, kind))
    css = parse(config).css
    if css:
        parts.append(_bytes_field(FIELD_CSS, css.encode("utf-8")))
    return b"".join(parts)


def clone(config: bytes, css: str, sort_field_idx: int) -> bytes:
    """Build the config of a cloned note type.

    The call writes the sort field index and the CSS first, then every other
    field of the input in its wire order. The kind of the input stays. A source
    value that points at another field of the input note type cannot hold.

    :param config: Config blob of the source note type.
    :param css: CSS of the new note type.
    :param sort_field_idx: Zero-based index of the sort field.
    :returns: Protobuf config bytes.
    """
    fields, _tail = _read(config)
    parts = [
        _varint_field(FIELD_SORT_IDX, sort_field_idx),
        _bytes_field(FIELD_CSS, css.encode("utf-8")),
    ]
    parts.extend(
        field.raw
        for field in fields
        if field.number not in (FIELD_SORT_IDX, FIELD_CSS)
    )
    return b"".join(parts)


def parse_template(config: bytes) -> tuple[str, str]:
    """Read the question format and the answer format of a template config.

    :param config: Raw protobuf config bytes.
    :returns: Question format and answer format. A format that the input does
        not hold is empty.
    """
    qfmt = ""
    afmt = ""
    for field in _read(config)[0]:
        if field.wire != 2:
            continue
        if field.number == 1:
            qfmt = _text(field.payload)
        elif field.number == 2:
            afmt = _text(field.payload)
    return qfmt, afmt


def build_template(qfmt: str, afmt: str) -> bytes:
    """Build the config of one template.

    :param qfmt: Question format.
    :param afmt: Answer format.
    :returns: Protobuf config bytes.
    """
    return _bytes_field(1, qfmt.encode("utf-8")) + _bytes_field(
        2, afmt.encode("utf-8")
    )


@dataclass(frozen=True)
class _Field:
    number: int
    wire: int
    raw: bytes
    value: int = 0
    payload: bytes = b""


def _read(data: bytes) -> tuple[list[_Field], bytes]:
    """Give the fields of a message and the bytes after the last field it reads."""
    fields: list[_Field] = []
    position = 0
    while position < len(data):
        start = position
        tag, position = _read_varint(data, position)
        number = tag >> 3
        wire = tag & 0x07
        if wire == 2:
            length, position = _read_varint(data, position)
            payload = data[position : position + length]
            position += length
            fields.append(
                _Field(number, wire, data[start:position], payload=payload)
            )
        elif wire == 0:
            value, position = _read_varint(data, position)
            fields.append(_Field(number, wire, data[start:position], value=value))
        else:
            return fields, data[start:]
    return fields, b""


def _read_varint(data: bytes, position: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while position < len(data):
        byte = data[position]
        position += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            break
        shift += 7
    return value, position


def _write_varint(value: int) -> bytes:
    out = bytearray()
    while value > 0x7F:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _varint_field(field: int, value: int) -> bytes:
    return _write_varint(field << 3) + _write_varint(value)


def _bytes_field(field: int, payload: bytes) -> bytes:
    return _write_varint(field << 3 | 2) + _write_varint(len(payload)) + payload


def _text(value: bytes) -> str:
    return value.decode("utf-8", errors="replace")
