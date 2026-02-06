#!/usr/bin/env python3
"""
Read, edit, and create Anki package (.apkg) files.

APKG Format Overview
--------------------
An .apkg file is a ZIP archive containing:
- collection.anki2 or collection.anki21 or collection.anki21b (SQLite database)
- media (JSON or protobuf mapping file IDs to filenames)
- 0, 1, 2, ... (media files named by numeric ID)

Database Formats
----------------
- collection.anki2: Legacy Anki 2.0 format (plain SQLite)
- collection.anki21: Anki 2.1 format (plain SQLite, same schema as anki2)
- collection.anki21b: Anki 2.1.50+ format (zstd-compressed SQLite, new schema)

Schema Differences (anki21b vs legacy)
--------------------------------------
Legacy (anki2/anki21):
    - Decks stored as JSON in col.decks column
    - Models stored as JSON in col.models column
    - Tables: col, notes, cards, revlog, graves

Anki21b:
    - Decks stored in separate 'decks' table (id, name, ...)
    - Models stored in 'notetypes' table with 'fields' table for field definitions
    - Tables: col, notes, cards, revlog, graves, decks, notetypes, fields,
              templates, deck_config, config, tags

Media File Format
-----------------
Legacy: JSON object mapping file ID strings to filenames
    {"0": "audio.mp3", "1": "image.png"}

Anki21b: zstd-compressed protobuf with repeated MediaEntry messages
    Entry order = file ID (0, 1, 2, ...)
    Each entry contains filename, file size, and SHA1 hash
"""

import json
import os
import re
import shutil
import sqlite3
import tempfile
import zipfile
from pathlib import Path

try:
    import zstandard
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False


class AnkiPackage:
    """
    Read, edit, and create Anki flashcard packages (.apkg files).

    Provides direct SQLite access to the underlying Anki database, supporting
    all three database formats: anki2 (legacy), anki21 (Anki 2.1), and anki21b
    (Anki 2.1.50+ with zstd compression).

    Use as a context manager to ensure proper cleanup::

        with AnkiPackage('deck.apkg') as pkg:
            cards = pkg.get_cards()
            pkg.update_note_field(note_id, 0, "new value")
            pkg.save('modified.apkg')

    For creating new packages from scratch, use the static :meth:`create` method.
    """

    def __init__(self, apkg_path: str | Path) -> None:
        """
        Initialize an AnkiPackage instance.

        :param apkg_path: Path to the .apkg file to open.

        .. note::
            Use as a context manager (``with`` statement) to ensure proper
            resource cleanup. The package is not opened until entering the
            context.
        """
        self.apkg_path = apkg_path
        self.temp_dir: str | None = None
        self.conn: sqlite3.Connection | None = None
        self._modified: bool = False
        self._db_format: str | None = None  # 'anki2', 'anki21', or 'anki21b'
        self._db_path: str | None = None

    def __enter__(self) -> "AnkiPackage":
        """Enter context manager, extract and open the APKG file."""
        self.temp_dir = tempfile.mkdtemp()

        with zipfile.ZipFile(self.apkg_path, "r") as zip_ref:
            zip_ref.extractall(self.temp_dir)

        db_path = self._select_and_prepare_database()
        self._db_path = db_path
        
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

        return self

    def _select_and_prepare_database(self) -> str:
        """
        Select the best database format and decompress if needed.

        :returns: Path to the database file to use.
        """
        db_path_21b = os.path.join(self.temp_dir, "collection.anki21b")
        db_path_21 = os.path.join(self.temp_dir, "collection.anki21")
        db_path_2 = os.path.join(self.temp_dir, "collection.anki2")

        # Prefer anki21b (newest format) if available and zstd is installed
        if os.path.exists(db_path_21b) and os.path.getsize(db_path_21b) > 0:
            if not HAS_ZSTD:
                raise ImportError(
                    "zstandard package required for anki21b format. "
                    "Install with: uv add zstandard"
                )
            decompressed_path = os.path.join(self.temp_dir, "collection_decompressed.db")
            self._decompress_zstd(db_path_21b, decompressed_path)
            self._db_format = "anki21b"
            return decompressed_path

        # Fall back to anki21 if it has more data than anki2
        if os.path.exists(db_path_21):
            size_21 = os.path.getsize(db_path_21)
            size_2 = os.path.getsize(db_path_2) if os.path.exists(db_path_2) else 0
            if size_21 > size_2:
                self._db_format = "anki21"
                return db_path_21

        # Default to anki2
        self._db_format = "anki2"
        return db_path_2

    def _decompress_zstd(self, src_path: str, dst_path: str) -> None:
        """
        Decompress a zstd-compressed file.

        :param src_path: Path to the compressed file.
        :param dst_path: Path to write the decompressed file.
        """
        with open(src_path, "rb") as f:
            dctx = zstandard.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                decompressed = reader.read()
        with open(dst_path, "wb") as f:
            f.write(decompressed)

    @property
    def db_format(self) -> str | None:
        """
        The database format detected in this package.

        :returns: One of ``'anki2'``, ``'anki21'``, ``'anki21b'``, or ``None``.
        """
        return self._db_format

    @property
    def _is_anki21(self) -> bool:
        """Backward compatibility property for Anki 2.1+ formats."""
        return self._db_format in ("anki21", "anki21b")

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager, close connection and clean up."""
        if self.conn:
            self.conn.close()

        # Clean up temp directory (unless save() was called)
        if self.temp_dir and not self._modified:
            shutil.rmtree(self.temp_dir)

    def get_decks(self) -> dict[str, dict]:
        """
        Get all decks in the collection.

        :returns: Dict mapping deck ID (string) to deck info dict with ``'name'`` key.

        .. note::
            Schema differs by format:
            - Legacy: JSON blob in ``col.decks``
            - Anki21b: Normalized ``decks`` table
        """
        cursor = self.conn.cursor()
        
        if self._db_format == "anki21b":
            cursor.execute("SELECT id, name FROM decks")
            decks = {}
            for row in cursor.fetchall():
                decks[str(row["id"])] = {"name": row["name"]}
            return decks
        
        cursor.execute("SELECT decks FROM col")
        decks_json = cursor.fetchone()[0]
        decks = json.loads(decks_json)
        return decks

    def _write_varint(self, value: int) -> bytes:
        """Write a varint to bytes."""
        result = b""
        while value > 127:
            result += bytes([(value & 0x7f) | 0x80])
            value >>= 7
        result += bytes([value])
        return result

    def _write_proto_string(self, field_num: int, value: str) -> bytes:
        """Write a protobuf length-delimited string field."""
        data = value.encode("utf-8")
        tag = (field_num << 3) | 2
        return bytes([tag]) + self._write_varint(len(data)) + data

    def _parse_template_config(self, config: bytes) -> dict[str, str]:
        """Parse protobuf config blob from templates table."""
        result = {"qfmt": "", "afmt": ""}
        pos = 0
        while pos < len(config):
            if pos >= len(config):
                break
            tag = config[pos]
            pos += 1
            field_num = tag >> 3
            wire_type = tag & 0x07
            
            if wire_type == 2:
                length, pos = self._read_varint(config, pos)
                value = config[pos:pos + length].decode("utf-8", errors="replace")
                pos += length
                if field_num == 1:
                    result["qfmt"] = value
                elif field_num == 2:
                    result["afmt"] = value
            elif wire_type == 0:
                _, pos = self._read_varint(config, pos)
            else:
                break
        
        return result

    def _build_template_config(self, qfmt: str, afmt: str) -> bytes:
        """Build protobuf config blob for templates table."""
        return self._write_proto_string(1, qfmt) + self._write_proto_string(2, afmt)

    def _parse_notetype_config(self, config: bytes) -> dict:
        """Parse protobuf config blob from notetypes table to extract CSS."""
        result = {"css": "", "other_fields": []}
        pos = 0
        while pos < len(config):
            start_pos = pos
            tag = config[pos]
            pos += 1
            field_num = tag >> 3
            wire_type = tag & 0x07
            
            if wire_type == 2:
                length, pos = self._read_varint(config, pos)
                if field_num == 3:
                    result["css"] = config[pos:pos + length].decode("utf-8", errors="replace")
                else:
                    result["other_fields"].append((start_pos, pos + length, field_num))
                pos += length
            elif wire_type == 0:
                _, end_pos = self._read_varint(config, pos)
                result["other_fields"].append((start_pos, end_pos, field_num))
                pos = end_pos
            else:
                break
        
        result["_original"] = config
        return result

    def _build_notetype_config(self, parsed: dict, new_css: str | None = None) -> bytes:
        """Rebuild notetype config with optionally updated CSS."""
        original = parsed["_original"]
        css = new_css if new_css is not None else parsed["css"]
        
        result = b""
        pos = 0
        
        while pos < len(original):
            tag = original[pos]
            field_num = tag >> 3
            wire_type = tag & 0x07
            start = pos
            pos += 1
            
            if wire_type == 2:
                length, pos = self._read_varint(original, pos)
                if field_num == 3:
                    result += self._write_proto_string(3, css)
                else:
                    result += original[start:pos + length]
                pos += length
            elif wire_type == 0:
                _, pos = self._read_varint(original, pos)
                result += original[start:pos]
            else:
                result += original[start:]
                break
        
        return result

    def get_models(self, include_templates: bool = False) -> dict[str, dict]:
        """
        Get all note models/templates (called "notetypes" in Anki UI).

        :param include_templates: If True, include template and CSS info for anki21b.
        :returns: Dict mapping model ID (string) to model info with ``'name'``
            and ``'flds'`` keys. The ``'flds'`` key contains a list of field
            dicts with ``'name'`` key.

        .. note::
            Schema differs by format:
            - Legacy: JSON blob in ``col.models``
            - Anki21b: ``notetypes`` + ``fields`` tables
        """
        cursor = self.conn.cursor()
        
        if self._db_format == "anki21b":
            cursor.execute("SELECT id, name FROM notetypes")
            models = {}
            for row in cursor.fetchall():
                model_id = str(row["id"])
                models[model_id] = {
                    "name": row["name"],
                    "flds": [],
                    "tmpls": [],
                    "css": "",
                }
            
            cursor.execute("SELECT ntid, name, ord FROM fields ORDER BY ntid, ord")
            for row in cursor.fetchall():
                model_id = str(row["ntid"])
                if model_id in models:
                    models[model_id]["flds"].append({"name": row["name"]})
            
            if include_templates:
                cursor.execute(
                    "SELECT ntid, name, ord, config FROM templates ORDER BY ntid, ord"
                )
                for row in cursor.fetchall():
                    model_id = str(row["ntid"])
                    if model_id in models:
                        config = self._parse_template_config(row["config"] or b"")
                        tmpl = {
                            "name": row["name"],
                            "ord": row["ord"],
                            "qfmt": config["qfmt"],
                            "afmt": config["afmt"],
                        }
                        models[model_id]["tmpls"].append(tmpl)
            
            return models
        
        cursor.execute("SELECT models FROM col")
        models_json = cursor.fetchone()[0]
        models = json.loads(models_json)
        return models

    def update_model_template(
        self,
        model_id: str | None = None,
        css: str | None = None,
        qfmt: str | None = None,
        afmt: str | None = None,
        template_index: int = 0,
    ) -> None:
        """
        Update CSS and/or card templates for a note type (model).

        :param model_id: Model ID to update. If None, updates the first model found.
        :param css: New CSS styling for the model.
        :param qfmt: New question (front) template.
        :param afmt: New answer (back) template.
        :param template_index: Which template to update (0 for first card type).
        :raises ValueError: If model not found.
        """
        cursor = self.conn.cursor()

        if self._db_format == "anki21b":
            if model_id is None:
                cursor.execute("SELECT id FROM notetypes LIMIT 1")
                row = cursor.fetchone()
                if not row:
                    raise ValueError("No models found in package")
                model_id = str(row["id"])

            if css is not None:
                cursor.execute(
                    "SELECT config FROM notetypes WHERE id = ?", (int(model_id),)
                )
                row = cursor.fetchone()
                if not row:
                    raise ValueError(f"Model {model_id} not found")
                
                parsed = self._parse_notetype_config(row["config"] or b"")
                new_config = self._build_notetype_config(parsed, css)
                cursor.execute(
                    "UPDATE notetypes SET config = ? WHERE id = ?",
                    (new_config, int(model_id)),
                )

            if qfmt is not None or afmt is not None:
                cursor.execute(
                    "SELECT config FROM templates WHERE ntid = ? AND ord = ?",
                    (int(model_id), template_index),
                )
                row = cursor.fetchone()
                if not row:
                    raise ValueError(
                        f"Template {template_index} not found for model {model_id}"
                    )

                current = self._parse_template_config(row["config"] or b"")
                new_qfmt = qfmt if qfmt is not None else current["qfmt"]
                new_afmt = afmt if afmt is not None else current["afmt"]
                new_config = self._build_template_config(new_qfmt, new_afmt)

                cursor.execute(
                    "UPDATE templates SET config = ? WHERE ntid = ? AND ord = ?",
                    (new_config, int(model_id), template_index),
                )
        else:
            cursor.execute("SELECT models FROM col")
            models_json = cursor.fetchone()[0]
            models = json.loads(models_json)

            if model_id is None:
                model_id = next(iter(models.keys()), None)
                if model_id is None:
                    raise ValueError("No models found in package")

            if model_id not in models:
                raise ValueError(f"Model {model_id} not found")

            model = models[model_id]

            if css is not None:
                model["css"] = css

            if qfmt is not None or afmt is not None:
                if template_index >= len(model.get("tmpls", [])):
                    raise ValueError(
                        f"Template {template_index} not found for model {model_id}"
                    )
                tmpl = model["tmpls"][template_index]
                if qfmt is not None:
                    tmpl["qfmt"] = qfmt
                if afmt is not None:
                    tmpl["afmt"] = afmt

            cursor.execute(
                "UPDATE col SET models = ? WHERE id = 1",
                (json.dumps(models),),
            )

        self.conn.commit()
        self._modified = True

    def get_notes(self) -> list[sqlite3.Row]:
        """
        Get all notes from the collection.

        :returns: List of note rows with ``id``, ``mid``, ``flds``, ``tags`` columns.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, mid, flds, tags FROM notes")
        return cursor.fetchall()

    def get_cards(self) -> list[sqlite3.Row]:
        """
        Get all cards with their note information.

        :returns: List of card rows with ``id``, ``nid``, ``did``, ``ord``,
            ``flds``, ``tags``, ``mid`` columns.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT cards.id, cards.nid, cards.did, cards.ord,
                   notes.flds, notes.tags, notes.mid
            FROM cards
            JOIN notes ON cards.nid = notes.id
        """)
        return cursor.fetchall()

    def parse_card(self, card: sqlite3.Row, models: dict, decks: dict) -> dict:
        """
        Parse a card into a readable format.

        :param card: Card row from :meth:`get_cards`.
        :param models: Models dict from :meth:`get_models`.
        :param decks: Decks dict from :meth:`get_decks`.
        :returns: Dict with ``card_id``, ``note_id``, ``deck``, ``model``,
            ``fields``, ``tags`` keys.
        """
        model_id = str(card["mid"])
        model = models.get(model_id, {})
        model_name = model.get("name", "Unknown")

        # Get field names from the model
        fields = model.get("flds", [])
        field_names = [f["name"] for f in fields]

        # Parse the field values (separated by \x1f)
        field_values = card["flds"].split("\x1f")

        # Create a dictionary of field name -> value
        card_data = dict(zip(field_names, field_values))

        # Get deck name
        deck_id = str(card["did"])
        deck = decks.get(deck_id, {})
        deck_name = deck.get("name", "Unknown")

        return {
            "card_id": card["id"],
            "note_id": card["nid"],
            "deck": deck_name,
            "model": model_name,
            "fields": card_data,
            "tags": card["tags"],
        }

    def get_media_mapping(self) -> dict[str, str]:
        """
        Get mapping of file IDs to filenames from the media file.

        :returns: Dict mapping numeric file IDs (as strings) to actual filenames.
            Example: ``{"0": "audio.mp3", "1": "image.png"}``.
            Returns empty dict if media file doesn't exist or can't be parsed.

        .. note::
            Media file formats differ:
            - Legacy (anki2/anki21): Plain JSON text
            - Anki21b: zstd-compressed protobuf

            The numeric file IDs correspond to files named ``0``, ``1``, ``2``, etc.
            in the APKG archive root.
        """
        media_path = os.path.join(self.temp_dir, "media")
        if not os.path.exists(media_path):
            return {}

        with open(media_path, "rb") as f:
            content = f.read()

        # Zstd magic: 0x28 0xB5 0x2F 0xFD (little-endian 0xFD2FB528)
        ZSTD_MAGIC = b'\x28\xb5\x2f\xfd'
        if content[:4] == ZSTD_MAGIC:
            if not HAS_ZSTD:
                return {}
            import io
            dctx = zstandard.ZstdDecompressor()
            # Use stream_reader because content size may not be in frame header
            with dctx.stream_reader(io.BytesIO(content)) as reader:
                content = reader.read()

        # Try JSON first (legacy format: {"0": "file.mp3", ...})
        try:
            return json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass

        # Try protobuf (anki21b: decompressed content is binary protobuf)
        try:
            return self._parse_media_protobuf(content)
        except Exception:
            pass

        return {}

    def _parse_media_protobuf(self, data: bytes) -> dict[str, str]:
        """
        Parse media mapping from protobuf format used in anki21b.

        :param data: Decompressed protobuf binary data.
        :returns: Dict mapping file ID strings to filenames.

        The file is zstd-compressed, then contains repeated protobuf messages.
        Entry order (0, 1, 2, ...) corresponds to media file IDs in the APKG.

        Protobuf wire format (each entry ~85 bytes)::

            0x0a <varint:msg_len>           # outer message, field 1, wire type 2
                0x0a <varint:str_len> <filename>  # field 1: filename (string)
                0x10 <varint:size>                # field 2: file size in bytes
                0x1a <20 bytes>                   # field 3: SHA1 hash (20 bytes)

        .. note::
            Field 2 values are NOT unique (multiple files can have same size),
            so we use entry order as the file ID, not field 2.
        """
        mapping = {}
        pos = 0
        entry_idx = 0
        
        while pos < len(data):
            if pos >= len(data):
                break
            
            tag = data[pos]
            pos += 1
            
            if tag != 0x0a:
                break
            
            msg_len, pos = self._read_varint(data, pos)
            if msg_len == 0:
                break
            
            msg_end = pos + msg_len
            filename = None
            
            while pos < msg_end:
                inner_tag = data[pos]
                pos += 1
                
                field_num = inner_tag >> 3
                wire_type = inner_tag & 0x07
                
                if field_num == 1 and wire_type == 2:
                    str_len, pos = self._read_varint(data, pos)
                    filename = data[pos:pos + str_len].decode("utf-8")
                    pos += str_len
                elif wire_type == 2:
                    skip_len, pos = self._read_varint(data, pos)
                    pos += skip_len
                elif wire_type == 0:
                    _, pos = self._read_varint(data, pos)
                else:
                    pos = msg_end
                    break
            
            if filename is not None:
                mapping[str(entry_idx)] = filename
            
            pos = msg_end
            entry_idx += 1
        
        return mapping

    def _read_varint(self, data: bytes, pos: int) -> tuple[int, int]:
        """
        Read a protobuf varint from data starting at pos.

        :param data: Binary data to read from.
        :param pos: Starting position in data.
        :returns: Tuple of (value, new_position).

        Varints use 7 bits per byte with MSB as continuation flag.
        Example: ``0x80 0x5a`` = ``(0x00 << 7) | 0x5a`` = ``0x2d00`` = 11520
        """
        result = 0
        shift = 0
        while pos < len(data):
            byte = data[pos]
            pos += 1
            result |= (byte & 0x7f) << shift
            if (byte & 0x80) == 0:
                break
            shift += 7
        return result, pos

    def extract_audio_files(
        self, output_dir: str | Path, audio_only: bool = True
    ) -> dict[str, str]:
        """
        Extract media files from .apkg to output directory.

        :param output_dir: Destination directory for extracted files.
        :param audio_only: If True, only extract audio files (.mp3, .wav, .ogg).
        :returns: Dict mapping filenames to extracted file paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        mapping = self.get_media_mapping()
        extracted = {}

        for file_id, filename in mapping.items():
            # Filter for audio files if requested
            if audio_only and not filename.endswith((".mp3", ".wav", ".ogg")):
                continue

            src = os.path.join(self.temp_dir, file_id)
            if not os.path.exists(src):
                continue

            dst = os.path.join(output_dir, filename)
            shutil.copy2(src, dst)
            extracted[filename] = dst

        return extracted

    def get_audio_for_card(self, card: sqlite3.Row, models: dict) -> list[str]:
        """
        Extract audio filenames referenced in a card's fields.

        :param card: Card row from database (with ``flds`` and ``mid`` fields).
        :param models: Models dict from :meth:`get_models`.
        :returns: List of audio filenames referenced in card (e.g., ``['hello.mp3']``).
        """
        audio_pattern = r"\[sound:(.*?)\]"

        # Get all field values
        field_values = card["flds"].split("\x1f")

        audio_files = []
        for field_value in field_values:
            matches = re.findall(audio_pattern, field_value)
            audio_files.extend(matches)

        return audio_files

    def get_audio_statistics(self) -> dict:
        """
        Get statistics about audio files in the package.

        :returns: Dict with keys ``total_media_files``, ``audio_files``,
            ``image_files``, ``audio_formats``.
        """
        mapping = self.get_media_mapping()
        audio_files = {
            k: v for k, v in mapping.items() if v.endswith((".mp3", ".wav", ".ogg"))
        }

        stats = {
            "total_media_files": len(mapping),
            "audio_files": len(audio_files),
            "image_files": len(mapping) - len(audio_files),
            "audio_formats": {},
        }

        # Count audio formats
        for filename in audio_files.values():
            ext = filename.split(".")[-1].lower()
            stats["audio_formats"][ext] = stats["audio_formats"].get(ext, 0) + 1

        return stats

    def add_media_file(
        self, file_path: str | Path, filename: str | None = None
    ) -> str:
        """
        Add a media file (audio/image) to the APKG package.

        :param file_path: Path to the file to add.
        :param filename: Filename to use in the package (defaults to basename).
        :returns: The filename that can be used in card fields (e.g., ``"audio.mp3"``).
        :raises FileNotFoundError: If file_path doesn't exist.

        :Example:

        >>> filename = pkg.add_media_file("/path/to/audio.mp3")
        >>> pkg.add_audio_to_card(note_id, 0, filename)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if filename is None:
            filename = os.path.basename(file_path)

        # Get current media mapping
        mapping = self.get_media_mapping()

        # Find next available ID
        existing_ids = [int(k) for k in mapping.keys() if k.isdigit()]
        next_id = max(existing_ids) + 1 if existing_ids else 0

        # Copy file to temp directory with the ID
        dest_path = os.path.join(self.temp_dir, str(next_id))
        shutil.copy2(file_path, dest_path)

        # Update media mapping
        mapping[str(next_id)] = filename
        media_path = os.path.join(self.temp_dir, "media")
        with open(media_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False)

        self._modified = True
        return filename

    def update_note_field(self, note_id: int, field_index: int, new_value: str) -> None:
        """
        Update a specific field in a note.

        :param note_id: The note ID.
        :param field_index: Zero-based index of the field to update.
        :param new_value: New field value.
        :raises ValueError: If note not found or field_index out of range.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT flds FROM notes WHERE id = ?", (note_id,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Note {note_id} not found")

        fields = row[0].split("\x1f")
        if field_index >= len(fields):
            raise ValueError(
                f"Field index {field_index} out of range (note has {len(fields)} fields)"
            )

        fields[field_index] = new_value
        updated_flds = "\x1f".join(fields)

        # Update mod time (Anki uses milliseconds since epoch)
        import time

        mod_time = int(time.time() * 1000)

        cursor.execute(
            """
            UPDATE notes 
            SET flds = ?, mod = ?
            WHERE id = ?
        """,
            (updated_flds, mod_time, note_id),
        )
        self.conn.commit()
        self._modified = True

    def move_card(self, card_id: int, new_deck_id: int) -> None:
        """
        Move a card to a different deck.

        :param card_id: The card ID.
        :param new_deck_id: The destination deck ID.
        """
        import time

        mod_time = int(time.time() * 1000)

        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE cards 
            SET did = ?, mod = ?
            WHERE id = ?
        """,
            (new_deck_id, mod_time, card_id),
        )
        self.conn.commit()
        self._modified = True

    def add_audio_to_card(
        self, note_id: int, field_index: int, audio_filename: str
    ) -> None:
        """
        Add an audio reference to a card field.

        :param note_id: The note ID.
        :param field_index: Zero-based index of the field to add audio to.
        :param audio_filename: Audio filename (e.g., ``"hello.mp3"``).
        :raises ValueError: If note not found or field_index out of range.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT flds FROM notes WHERE id = ?", (note_id,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Note {note_id} not found")

        fields = row[0].split("\x1f")
        if field_index >= len(fields):
            raise ValueError(f"Field index {field_index} out of range")

        # Add audio tag if not already present
        audio_tag = f"[sound:{audio_filename}]"
        if audio_tag not in fields[field_index]:
            fields[field_index] += " " + audio_tag

        updated_flds = "\x1f".join(fields)
        import time

        mod_time = int(time.time() * 1000)

        cursor.execute(
            """
            UPDATE notes 
            SET flds = ?, mod = ?
            WHERE id = ?
        """,
            (updated_flds, mod_time, note_id),
        )
        self.conn.commit()
        self._modified = True

    def create_deck(self, deck_name: str, deck_id: int | None = None) -> int:
        """
        Create a new deck in the collection.

        :param deck_name: Name of the new deck.
        :param deck_id: Optional deck ID (auto-generated if not provided).
        :returns: The deck ID of the newly created deck.
        :raises ValueError: If deck with given ID already exists.
        """
        import time

        current_time = int(time.time() * 1000)

        # Get existing decks
        cursor = self.conn.cursor()
        cursor.execute("SELECT decks FROM col")
        decks_json = cursor.fetchone()[0]
        decks = json.loads(decks_json)

        # Generate deck ID if not provided
        if deck_id is None:
            # Find the highest existing deck ID and add 1
            existing_ids = [int(did) for did in decks.keys() if did.isdigit()]
            deck_id = max(existing_ids) + 1 if existing_ids else 1

        deck_id_str = str(deck_id)

        # Check if deck already exists
        if deck_id_str in decks:
            raise ValueError(f"Deck with ID {deck_id} already exists")

        # Create deck structure (minimal required fields)
        new_deck = {
            "name": deck_name,
            "desc": "",
            "extendNew": 0,
            "extendRev": 0,
            "conf": 1,  # Default deck configuration
            "usn": -1,  # Unsynced
            "collapsed": False,
            "browserCollapsed": False,
            "newToday": [0, 0],  # [day, count]
            "revToday": [0, 0],
            "lrnToday": [0, 0],
            "timeToday": [0, 0],
            "dyn": 0,  # Not a dynamic deck
            "mod": current_time,
        }

        # Add to decks dict
        decks[deck_id_str] = new_deck

        # Update database
        cursor.execute(
            """
            UPDATE col 
            SET decks = ?
            WHERE id = 1
        """,
            (json.dumps(decks),),
        )
        self.conn.commit()
        self._modified = True

        return deck_id

    def copy_cards_to_deck(
        self,
        source_deck_id: int,
        target_deck_id: int,
        card_ids: list[int] | None = None,
    ) -> int:
        """
        Copy cards from one deck to another.

        :param source_deck_id: Source deck ID.
        :param target_deck_id: Target deck ID.
        :param card_ids: Optional list of specific card IDs to copy (all if None).
        :returns: Number of cards copied.
        """
        import time

        mod_time = int(time.time() * 1000)

        cursor = self.conn.cursor()

        # Build query
        if card_ids:
            placeholders = ",".join("?" * len(card_ids))
            query = f"""
                UPDATE cards 
                SET did = ?, mod = ?
                WHERE did = ? AND id IN ({placeholders})
            """
            params = [target_deck_id, mod_time, source_deck_id] + card_ids
        else:
            query = """
                UPDATE cards 
                SET did = ?, mod = ?
                WHERE did = ?
            """
            params = [target_deck_id, mod_time, source_deck_id]

        cursor.execute(query, params)
        count = cursor.rowcount
        self.conn.commit()
        self._modified = True

        return count

    def create_deck_from_cards(
        self, deck_name: str, card_ids: list[int]
    ) -> tuple[int, int]:
        """
        Create a new deck and move specified cards into it.

        :param deck_name: Name of the new deck.
        :param card_ids: List of card IDs to move to the new deck.
        :returns: Tuple of (new_deck_id, number_of_cards_moved).
        """
        # Create the new deck
        new_deck_id = self.create_deck(deck_name)

        # Get source deck ID from first card
        cursor = self.conn.cursor()
        cursor.execute("SELECT did FROM cards WHERE id = ?", (card_ids[0],))
        source_deck_id = cursor.fetchone()[0]

        # Move cards to new deck
        count = self.copy_cards_to_deck(source_deck_id, new_deck_id, card_ids)

        return new_deck_id, count

    def get_audio_files_for_note(self, note_id: int) -> set[str]:
        """
        Get all audio files referenced by a note.

        :param note_id: The note ID.
        :returns: Set of audio filenames referenced in the note.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT flds FROM notes WHERE id = ?", (note_id,))
        row = cursor.fetchone()
        if not row:
            return set()

        audio_pattern = r"\[sound:(.*?)\]"
        fields = row[0].split("\x1f")
        audio_files = set()
        for field_value in fields:
            matches = re.findall(audio_pattern, field_value)
            audio_files.update(matches)

        return audio_files

    def is_audio_file_used(self, audio_filename: str) -> bool:
        """
        Check if an audio file is used by any note.

        :param audio_filename: The audio filename to check.
        :returns: True if the audio file is referenced by any note.
        """
        cursor = self.conn.cursor()
        audio_pattern = f"%[sound:{audio_filename}]%"
        cursor.execute("SELECT COUNT(*) FROM notes WHERE flds LIKE ?", (audio_pattern,))
        count = cursor.fetchone()[0]
        return count > 0

    def delete_media_file(self, audio_filename: str) -> bool:
        """
        Delete a media file from the APKG package.

        :param audio_filename: The audio filename to delete.
        :returns: True if file was deleted, False if not found.
        """
        mapping = self.get_media_mapping()

        # Find the file ID for this filename
        file_id = None
        for fid, fname in mapping.items():
            if fname == audio_filename:
                file_id = fid
                break

        if file_id is None:
            return False

        # Remove from mapping
        del mapping[file_id]

        # Delete the actual file
        file_path = os.path.join(self.temp_dir, file_id)
        if os.path.exists(file_path):
            os.remove(file_path)

        # Update media mapping
        media_path = os.path.join(self.temp_dir, "media")
        with open(media_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False)

        self._modified = True
        return True

    def delete_card(self, card_id: int, cleanup_audio: bool = True) -> dict:
        """
        Delete a card and optionally clean up unused audio files.

        :param card_id: The card ID to delete.
        :param cleanup_audio: If True, delete audio files that are no longer used.
        :returns: Dict with keys ``card_deleted``, ``note_deleted``,
            ``audio_files_deleted``.
        :raises ValueError: If card not found.
        """
        cursor = self.conn.cursor()

        # Get the note ID for this card
        cursor.execute("SELECT nid FROM cards WHERE id = ?", (card_id,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Card {card_id} not found")

        note_id = row[0]

        # Get audio files used by this note before deletion
        audio_files = self.get_audio_files_for_note(note_id) if cleanup_audio else set()

        # Delete the card
        cursor.execute("DELETE FROM cards WHERE id = ?", (card_id,))
        cards_deleted = cursor.rowcount

        # Check if note has any remaining cards
        cursor.execute("SELECT COUNT(*) FROM cards WHERE nid = ?", (note_id,))
        remaining_cards = cursor.fetchone()[0]

        note_deleted = False
        if remaining_cards == 0:
            # Delete the note if no cards remain
            cursor.execute("DELETE FROM notes WHERE id = ?", (note_id,))
            note_deleted = cursor.rowcount > 0

        self.conn.commit()
        self._modified = True

        # Clean up unused audio files
        audio_files_deleted = []
        if cleanup_audio and note_deleted:
            for audio_filename in audio_files:
                if not self.is_audio_file_used(audio_filename):
                    if self.delete_media_file(audio_filename):
                        audio_files_deleted.append(audio_filename)

        return {
            "card_deleted": cards_deleted > 0,
            "note_deleted": note_deleted,
            "audio_files_deleted": audio_files_deleted,
        }

    def delete_cards(self, card_ids: list[int], cleanup_audio: bool = True) -> dict:
        """
        Delete multiple cards and optionally clean up unused audio files.

        :param card_ids: List of card IDs to delete.
        :param cleanup_audio: If True, delete audio files that are no longer used.
        :returns: Dict with keys ``cards_deleted``, ``notes_deleted``,
            ``audio_files_deleted``.
        """
        results = {"cards_deleted": 0, "notes_deleted": 0, "audio_files_deleted": []}

        # Track notes that will be deleted (no remaining cards)
        notes_to_check = set()

        # Delete all cards first
        cursor = self.conn.cursor()
        for card_id in card_ids:
            cursor.execute("SELECT nid FROM cards WHERE id = ?", (card_id,))
            row = cursor.fetchone()
            if row:
                note_id = row[0]
                cursor.execute("DELETE FROM cards WHERE id = ?", (card_id,))
                if cursor.rowcount > 0:
                    results["cards_deleted"] += 1
                    # Check if note has remaining cards
                    cursor.execute(
                        "SELECT COUNT(*) FROM cards WHERE nid = ?", (note_id,)
                    )
                    if cursor.fetchone()[0] == 0:
                        notes_to_check.add(note_id)

        # Delete notes with no remaining cards
        for note_id in notes_to_check:
            cursor.execute("DELETE FROM notes WHERE id = ?", (note_id,))
            if cursor.rowcount > 0:
                results["notes_deleted"] += 1

        self.conn.commit()
        self._modified = True

        # Clean up unused audio files after all deletions
        if cleanup_audio:
            mapping = self.get_media_mapping()
            for audio_filename in list(mapping.values()):
                if not self.is_audio_file_used(audio_filename):
                    if self.delete_media_file(audio_filename):
                        results["audio_files_deleted"].append(audio_filename)

        return results

    def delete_note(self, note_id: int, cleanup_audio: bool = True) -> dict:
        """
        Delete a note and all its associated cards, optionally cleaning up unused audio.

        :param note_id: The note ID to delete.
        :param cleanup_audio: If True, delete audio files that are no longer used.
        :returns: Dict with keys ``note_deleted``, ``cards_deleted``,
            ``audio_files_deleted``.
        """
        cursor = self.conn.cursor()

        # Get audio files used by this note before deletion
        audio_files = self.get_audio_files_for_note(note_id) if cleanup_audio else set()

        # Delete all cards associated with this note
        cursor.execute("DELETE FROM cards WHERE nid = ?", (note_id,))
        cards_deleted = cursor.rowcount

        # Delete the note
        cursor.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        note_deleted = cursor.rowcount > 0

        self.conn.commit()
        self._modified = True

        # Clean up unused audio files
        audio_files_deleted = []
        if cleanup_audio and note_deleted:
            for audio_filename in audio_files:
                if not self.is_audio_file_used(audio_filename):
                    if self.delete_media_file(audio_filename):
                        audio_files_deleted.append(audio_filename)

        return {
            "note_deleted": note_deleted,
            "cards_deleted": cards_deleted,
            "audio_files_deleted": audio_files_deleted,
        }

    def save(self, output_path: str | Path | None = None) -> None:
        """
        Save changes back to an APKG file.

        :param output_path: Path to save the updated APKG file.
            If None, overwrites the original file.

        .. note::
            For anki21b format, the database is recompressed with zstd.
        """
        if not self._modified:
            return

        if output_path is None:
            output_path = self.apkg_path

        # Ensure database is closed before repackaging
        if self.conn:
            self.conn.close()
            self.conn = None

        # For anki21b, recompress the database
        if self._db_format == "anki21b":
            self._recompress_anki21b()

        # Repackage the APKG file
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zip_ref:
            for root, dirs, files in os.walk(self.temp_dir):
                for file in files:
                    # Skip the decompressed database file
                    if file == "collection_decompressed.db":
                        continue
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, self.temp_dir)
                    zip_ref.write(file_path, arcname)

        # Reopen connection if needed
        self.conn = sqlite3.connect(self._db_path)
        self.conn.row_factory = sqlite3.Row

        print(f"Saved changes to {output_path}")

    def _recompress_anki21b(self) -> None:
        """Recompress the decompressed database back to anki21b format."""
        decompressed_path = os.path.join(self.temp_dir, "collection_decompressed.db")
        anki21b_path = os.path.join(self.temp_dir, "collection.anki21b")
        
        with open(decompressed_path, "rb") as f:
            data = f.read()
        
        cctx = zstandard.ZstdCompressor()
        compressed = cctx.compress(data)
        
        with open(anki21b_path, "wb") as f:
            f.write(compressed)

    @staticmethod
    def create(
        output_path: str | Path,
        deck_name: str,
        fields: list[str],
        cards: list[dict[str, str]],
        media_files: list[str] | None = None,
        model_name: str = "Basic",
        question_format: str | None = None,
        answer_format: str | None = None,
    ) -> None:
        """
        Create a new APKG file from scratch.

        Uses genanki to generate a complete Anki package with a custom note model,
        deck, cards, and optional media files.

        :param output_path: Path where the .apkg file will be created.
        :param deck_name: Display name of the deck in Anki.
        :param fields: Ordered list of field names for the note model. The first
            field is used as the question by default.
        :param cards: List of card data dictionaries. Each dict should have keys
            matching the field names. Missing keys default to empty string.
            Audio references use Anki's ``[sound:filename.mp3]`` format.
        :param media_files: Absolute paths to media files (audio, images) to
            include in the package. Files are copied into the APKG.
        :param model_name: Name of the note type/model shown in Anki.
        :param question_format: Anki template for the question (front) side.
            Uses Anki's ``{{field_name}}`` syntax. Defaults to showing the
            first field.
        :param answer_format: Anki template for the answer (back) side.
            Defaults to ``{{FrontSide}}<hr>`` followed by remaining fields.

        :Example:

        >>> AnkiPackage.create(
        ...     "vocab.apkg",
        ...     "My Vocab",
        ...     fields=["audio", "word", "meaning"],
        ...     cards=[
        ...         {"audio": "[sound:hello.mp3]", "word": "hello", "meaning": "greeting"},
        ...         {"audio": "[sound:bye.mp3]", "word": "goodbye", "meaning": "farewell"},
        ...     ],
        ...     media_files=["audio/hello.mp3", "audio/bye.mp3"],
        ...     question_format="{{audio}}",
        ...     answer_format="{{audio}}<hr>{{word}}<br>{{meaning}}",
        ... )
        """
        import random
        import genanki

        model_id = random.randrange(1 << 30, 1 << 31)
        deck_id = random.randrange(1 << 30, 1 << 31)

        if question_format is None:
            question_format = "{{" + fields[0] + "}}"

        if answer_format is None:
            parts = ["{{FrontSide}}", "<hr id=answer>"]
            for field in fields[1:]:
                parts.append("{{" + field + "}}")
            answer_format = "\n".join(parts)

        model = genanki.Model(
            model_id,
            model_name,
            fields=[{"name": f} for f in fields],
            templates=[
                {
                    "name": "Card 1",
                    "qfmt": question_format,
                    "afmt": answer_format,
                }
            ],
        )

        deck = genanki.Deck(deck_id, deck_name)

        for card in cards:
            field_values = [card.get(f, "") for f in fields]
            note = genanki.Note(model=model, fields=field_values)
            deck.add_note(note)

        package = genanki.Package(deck)
        if media_files:
            package.media_files = media_files
        package.write_to_file(str(output_path))
