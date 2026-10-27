# Put the media map in one module with format dispatch

Anki rejected the anki21b packages of this project with `500: Unknown frame
descriptor`. The collection frame was fixed first. The media map stayed JSON:
`AnkiPackage.add_media_file` and `delete_media_file` wrote JSON for every
format. Anki reads the anki21b media map as a zstd frame of a protobuf message,
so the first byte of the JSON gave the error.

Two more faults came from the same code. The writer ignored the key contract:
the anki21b protobuf has no key, and the entry order is the file id. And the
writer rewrote the whole map for each file, so the cull wrote the map thousands
of times, once for each deleted media file and once for each reserve file.

Status: accepted

## Decision

`anki_tools/apkg/media.py` owns the media map. The interface is four calls:

| Call | Contract |
| --- | --- |
| `read(package) -> dict[str, str]` | Archive file name to media file name |
| `write(package, mapping)` | Write the map in the format of the package |
| `add(package, path, filename) -> str` | Copy the file under a free archive name |
| `prune(package, used)` | Delete the unused files and write the map one time |

`delete(package, filename)` is a fifth call for the one-file delete that
`delete_card` uses.

`anki2` and `anki21` hold JSON. `anki21b` holds a zstd frame of a protobuf
`MediaEntries` message. The anki21b writer renumbers the kept files to `0..n-1`
in the archive and writes one entry per file in that order. Each entry holds
field 1, the file name; field 2, the file size; field 3, the 20-byte SHA1. The
archive holds no other media file. A note field references the media file name,
never the archive name.

The module holds the open map in `package.media_map`, and `AnkiPackage.save`
writes it. So `add` and `delete` do not write the map, and a run that adds many
files writes it one time.

`coreflow.write_deck` now calls `delete_cards(..., cleanup_audio=False)` and one
`prune_media(package.used_media())` after the kept notes and the reserve cards
are written. `AnkiPackage.get_media_mapping`, `add_media_file`,
`delete_media_file`, and the new `prune_media` and `used_media` methods stay as
thin wrappers, so the callers and the tests keep their contract.

## Consequences

The anki21b media map is a zstd protobuf that the reader of this project reads
back. A round trip test holds both formats in place.

The cull writes the map one time, not once per deleted or reserve file. The
prune reads the used names in one pass over the notes, not one SQL query for
each media file.

The anki2 JSON output does not change. The format of the output follows the
format of the input, so an `anki21b` package in gives an `anki21b` package out.

## Not decided here

The media map of a real Anki export can hold fields that this module ignores on
read and does not write: the modification time. Anki recomputes that field when
it imports the package.
