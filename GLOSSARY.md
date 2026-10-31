# Anki Tools

Read, edit, and build Anki packages for language learning. The package ranks and
selects sentences, then writes decks. This is one context, so the glossary lives
at the repository root.

## Package model

**Package (APKG)**:
One Anki export file. It holds decks, cards, fields, and media.
_Avoid_: deck file, apkg file

**Deck**:
A named group of cards that Anki reviews together.
_Avoid_: collection, table

**Card**:
The review unit. One template renders one card side.
_Avoid_: flashcard, item

**Field**:
One named value of a card, for example the target word or the audio name.
_Avoid_: column, attribute

**Template**:
The card layout and the CSS that style it.
_Avoid_: skin, theme

**Media map**:
The archive entry `media` that gives one archive file name for each media file
name. An `anki2` package holds JSON. An `anki21b` package holds a zstd frame of a
protobuf message, and the entry order is the file id. The code lives in
`anki_tools/apkg/media.py`.
_Avoid_: media index, media listing

**Note type config**:
The protobuf message in the `config` column of the `notetypes` table of an
`anki21b` package. It holds the kind, the sort field index, and the CSS. The
templates table holds one message for each template. The code lives in
`anki_tools/apkg/notetype.py`.
_Avoid_: notetype blob, model config

## Decks

**Vocabulary deck**:
The deck with one card per Swadesh concept, its form, and its audio. The
`anki swadesh vocab` command builds it from an `english-<lang>.csv` file and an
audio directory, for example `apkg/swadesh_vocab_mandarin.apkg`. The code lives in
`anki_tools/swadesh_vocab.py`.
_Avoid_: word deck, word list deck

**Core deck**:
The small deck that holds one card per Swadesh target, then the tier F words.
Each card shows one example sentence of a large sentence deck. The
`anki fix <lang> --core` command writes it.
_Avoid_: starter deck, base deck, small deck
## Deck output

**Output stem**:
The file stem of the written package. The deck title, the note type name, and
the side-file stems all equal it. The code lives in
`anki_tools/ranking/identity.py`.
_Avoid_: deck name, model name, title

**Output identity**:
The one name that a written package and its side files share. ADR 0015 records
the rule.
_Avoid_: deck identity, artifact name

**Manifest**:
The JSON side file of a written package. It records the input file, the deck
title and the note type name, the options, the counts, and the `sha256` of the
input and of each side file. The gates read the counts and the hashes from it,
and compare `source.deck` and `source.model` with the package. The core and cull
routes write it in `anki_tools/ranking/assemble.py`; the sentence pack writes a
second shape in `anki_tools/sentence_pack/pack.py`. ADR 0018 records the fix
shape.
_Avoid_: build record, sidecar, report

**Output package shape**:
The invariant of a written package: one deck, one standard note type with the
stated fields, and one card per note. The code lives in
`anki_tools/apkg/shape.py`.
_Avoid_: deck layout, package template

**Deck writer**:
The one module that writes an output package: the note type shape, the card
values, the media, the side tables, the manifest, and the report of one run.
Two adapters sit behind it: write a new package, and edit an input package. The
code lives in `anki_tools/apkg/deck_writer.py`. ADR 0024 records the interface.
_Avoid_: builder, exporter, packager

**Card template assembly**:
The one module that turns the front text, the back text, the CSS layers, the
field names, a character-link switch, and the blocks to strip into the template
pair and the CSS of one note type. The code lives in
`anki_tools/apkg/templates.py`. ADR 0023 records the interface.
_Avoid_: template builder, card layout

**Deck assembler**:
The fix-run adapter over the deck writer. It states the output shape from the
deck profile interface, and the side tables, the manifest values, and the
report lines of one fix run. The core selection and the cull selection are its
two adapters. The code lives in `anki_tools/ranking/assemble.py`. ADR 0018
records the interface.
_Avoid_: builder, exporter

**Deck profile interface**:
The eight members of a mode view that the deck assembler reads: the input
model name, the three input field names, the romanization source, the
templates, the highlight wrapper, and the card CSS. The core mode view and
the cull mode view both satisfy it. The code lives in
`anki_tools/ranking/assemble.py`.
_Avoid_: profile protocol, writer contract

**Gate table**:
The runner, the report, and the shared gates of the core and cull modes. Each
mode supplies a loader table to one lazy file reader. The runner, the report,
and the shared gates live in `anki_tools/ranking/gates.py`. The reader lives in
`anki_tools/gate_files.py`.
_Avoid_: audit list, check list

**Language deck profile**:
The language facts of one deck family, with a core mode view and a cull mode
view over them. The code lives in `anki_tools/ranking/deckprofile.py`.
_Avoid_: deck config, deck settings

**Mode view**:
The `CoreProfile` or the `CullProfile` that a language deck profile gives for
one flow. It holds the language facts and the values of that mode.
_Avoid_: mode profile, sub profile

**Fix run**:
The shared route behind the four `anki fix` commands: the path check, the
`--check` guard, the mode branch, the request build, and the rank report. The
code lives in `anki_tools/ranking/fixrun.py`.
_Avoid_: dispatcher, router

## Sentence pack

**Sentence pack**:
The shared pipeline that turns Tatoeba exports and a reviewed word table into one
APKG. The code lives in `anki_tools/sentence_pack/`.
_Avoid_: deck builder, generator

**Stage**:
One numbered step of the sentence pack flow, `S0` to `S5`. The command of a stage
reads the output of the stage before it.
_Avoid_: phase, step

**Candidate pool**:
The sentences that pass every guard, before the selection (S2).
_Avoid_: corpus, list

**Guard**:
A test that rejects a sentence early, for example the sentence shape or the
missing English link. The guards run in one fixed order.
_Avoid_: filter, check

**Gate**:
An audit check on the built files, for example `G4`. A gate reports a fault in the
output. It does not reject a sentence.
_Avoid_: guard, test

**Frame**:
A manual sentence with Tatoeba id 0. It fills a concept that has no good Tatoeba
sentence.
_Avoid_: fallback, stub

**Tier**:
The class of a selected sentence: `core` or `scaffold`. A `core` sentence
covers the target within the window. A `scaffold` sentence comes from the wider
window and covers the target as a fallback. The tier is not the core deck.
_Avoid_: level, grade

**Unknown word**:
A token in a sentence that the frequency table does not list.
_Avoid_: out-of-vocabulary word, OOV

## Ranking and selection

**Target**:
The word or form that a card teaches. It is the `target` field of the CSV.
_Avoid_: keyword, headword

**Concept**:
A numbered meaning of the Swadesh list, for example index 4 is "we".
_Avoid_: entry, item

**Form**:
The written shape of a concept in one language.
_Avoid_: spelling, word

**Variant**:
One accepted form of a target. The `primary` variant is the default.
_Avoid_: alternative, option

**Profile**:
The data object of one language. It holds the frequencies, the stoplists, and the
score weights.
_Avoid_: config, settings

**Rank key**:
The sortable string that orders the candidates of one target.
_Avoid_: score, priority

**Cover**:
The loop that picks the fewest items and gains the most targets. The code lives in
`ranking/cover.py`.
_Avoid_: set cover, solver

**Cull**:
The flow that filters and selects the cards of a large sentence deck down to a
small deck, and writes the cull CSV, the coverage CSV, and the manifest. The code
lives in `ranking/cullflow.py`.
_Avoid_: filter, prune

**Cull profile**:
The frozen `CullProfile` that holds every value and hook of one language for the
cull, for example the model fields, the templates, the reserve pack, the content
parts of speech, the shortest lemma, the sentence marks, and the file paths. A
language supplies one profile and no flow code.
_Avoid_: cull config, cull settings

**Ranking CSV**:
The table with one row for each target and its candidates. The reorder step reads
it and writes the deck.
_Avoid_: report, export

## Languages

**Language code**:
The short key of a language, for example `yue` or `fr`. `languages.registry.ALIASES`
maps an alias to the canonical name.
_Avoid_: locale, language tag

**Language package**:
The directory `anki_tools/languages/<lang>/`. It holds all the code of one
language.
_Avoid_: language module, plugin
