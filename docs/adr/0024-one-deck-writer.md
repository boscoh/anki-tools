# Write every output package through one deck writer

The three build routes state the note type shape twice.

The fix routes build an `OutputShape` and call `shape.write_package`, which
opens the input collection, deletes the dropped cards, clones the input note
type, and saves. The two Swadesh routes call `AnkiPackage.create` with ten
keywords (`pack.py:543`, `cli.py:780`), which builds a package through genanki.

The shape facts are the same set in each route: deck name, note type name,
fields, templates, CSS, sort field index, deck description, tags, and media. The
card values cross the seam in two forms: `write_package` takes a positional list
per note id, and `create` takes a mapping per card. Media crosses it in two
forms as well: a list of source paths, and a source path plus an archive name.

Status: accepted. Extends ADR 0018 from the fix routes to the three build
routes. The naming rule of ADR 0015 stays with the callers for now.

## Decision

`anki_tools/apkg/deck_writer.py` owns the write. One entry function:

```python
write_deck(job: DeckWrite) -> RunFiles
```

`DeckWrite` holds:

- `output_path`
- `input_path: Path | None` — None writes a new package
- `shape: NoteTypeShape` — deck name, note type name, fields, template pair,
  CSS, sort field index, deck description, and the input note type name that the
  edit adapter needs
- `cards: Sequence[Card]` — values keyed by field name, plus tags
- `media: Sequence[MediaFile]` — source path and archive name
- `sides: Sequence[SideTable]` — zero or more CSV tables
- `manifest: Mapping | None` — the payload without `files`; the writer adds the
  hashes
- `manifest_files: Mapping[str, Path]` — the files to hash into the payload

`RunFiles` holds `output_path`, `manifest_path`, and `sides: Mapping[str, Path]`
keyed by suffix, for example `"core.csv"`.

`write_sides(job)` is the second function: it writes the side CSVs and the
manifest. `write_deck` calls it, and so does a caller that writes the package in
two steps: the core mode that reorders the deck writes the package to a
temporary file, reorders it, then writes the sides and the manifest beside the
final file.

The writer prints nothing. `print_run(lines, files)` prints the report lines
and the written paths, and the caller decides when to call it.

Two adapters sit behind the seam:

- new package: `AnkiPackage.create`, genanki, one template pair
- edit package: `shape.write_package`, which clones and strips the input
  collection

The writer states no name. The caller gives the output path and the manifest
path.

`assemble.write_run` becomes the fix adapter over the writer: it maps `DeckRun`
and `DeckProfile` to one `DeckWrite`. The vocab build leaves `cli.py` and becomes
`anki_tools/swadesh_vocab.py`, the vocab adapter. `pack.build_pack` becomes the
pack adapter.

## Consequences

One caller of `AnkiPackage.create` and one caller of `shape.write_package`. One
card record and one media record, so the two package writers cannot disagree
about them.

The interface is three functions over one record. The implementation holds the
two package writers, the media copy, the side tables, and the manifest. Three
routes and their tests cross the seam.

`RunFiles.card_csv` becomes `RunFiles.sides["core.csv"]`. The two call sites are
`mandarin/core.py:298` and `cantonese/core.py:259`.

`tests/test_output_shape.py` covers both adapters. `tests/test_assemble.py`
narrows to the fix adapter: the side tables, the manifest values, and the report
lines.

## Rejected

Two functions, `write_new` and `write_edited`. The adapter choice then leaks to
three call sites, and one of them must pick the adapter from an option.

Keeping the positional card list. The new-package adapter has no note ids, and
the edit adapter can map names to positions in one place.

The writer names the artifacts. A rename would then edit one module, but the
five Swadesh profiles still hold four name values each, which ADR 0015 already
forbids. Move the naming into the writer when those values are gone.

A `report` field on `DeckWrite` so the writer prints. The core mode prints its
paths after the reorder, so the writer would print a path that it is about to
replace. The caller decides when to print.
