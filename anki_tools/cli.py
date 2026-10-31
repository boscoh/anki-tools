"""
CLI tools for Anki package (.apkg) files.

Commands:
    inspect - Diagnostic tools to inspect .apkg files
    swadesh - Build Swadesh decks and run their gates
    fix - Fix and optimize decks (zh, fr, yue)
    style - Apply CSS and card templates to any deck
    preview - Build an HTML preview of the templates or of a package
"""

import importlib.util
import os
import re
import shutil
import webbrowser
from pathlib import Path
from typing import Annotated, Callable, Literal

import cyclopts
from cyclopts import Parameter

from anki_tools import swadesh_vocab
from anki_tools.apkg.package import AnkiPackage
from anki_tools.preview import write_preview
from anki_tools.preview_apkg import write_apkg_preview
from anki_tools.sentence_pack import audio as pack_audio
from anki_tools.sentence_pack import pack as sentence_pack
from anki_tools.sentence_pack import pool as sentence_pool
from anki_tools.sentence_pack import select as sentence_select
from anki_tools.sentence_pack import tatoeba as sentence_tatoeba
from anki_tools.sentence_pack.audit import print_audit as print_gate_table
from anki_tools.sentence_pack.profile import LanguageProfile
from anki_tools.languages import (
    LANGUAGES,
    profile_languages,
    resolve,
    vocab_languages,
)
from anki_tools.languages.registry import FixCommand
from anki_tools.ranking import fixrun
from anki_tools.apkg.templates import (
    DEFAULT_BACK_PATH,
    DEFAULT_CSS_PATH,
    DEFAULT_FRONT_PATH,
    apply_templates,
)

app = cyclopts.App(help="CLI tools for Anki package (.apkg) files")


# =============================================================================
# Inspect commands - diagnostic tools for .apkg files
# =============================================================================

inspect_app = cyclopts.App(name="inspect", help="Inspect Anki package (.apkg) files")
app.command(inspect_app)


@inspect_app.command
def cards(
    apkg_path: Path,
    *,
    limit: int = 10,
    field: str | None = None,
    search: str | None = None,
):
    """List cards in the deck.

    :param apkg_path: Path to .apkg file.
    :param limit: Maximum cards to show (0 for all).
    :param field: Only show this field name.
    :param search: Filter cards containing this text.
    """
    with AnkiPackage(str(apkg_path)) as pkg:
        cards = pkg.get_cards()
        models = pkg.get_models()
        decks = pkg.get_decks()

        print(f"Total cards: {len(cards)}\n")

        shown = 0
        for i, card in enumerate(cards):
            parsed = pkg.parse_card(card, models, decks)
            fields = parsed["fields"]

            if search:
                text = " ".join(str(v) for v in fields.values())
                if search.lower() not in text.lower():
                    continue

            print(f"Card {i}:")
            print(f"  Deck: {parsed['deck']}")
            print(f"  Type: {parsed['model']}")

            if field:
                if field in fields:
                    value = re.sub(r"<[^>]+>", "", str(fields[field]))
                    print(f"  {field}: {value[:200]}")
            else:
                for name, value in fields.items():
                    value = re.sub(r"<[^>]+>", "", str(value))
                    print(f"  {name}: {value[:100]}")

            if parsed["tags"]:
                print(f"  Tags: {parsed['tags']}")
            print()

            shown += 1
            if limit and shown >= limit:
                remaining = len(cards) - shown
                if remaining > 0:
                    print(f"... and {remaining} more cards")
                break


@inspect_app.command
def decks(apkg_path: Path):
    """List all decks in the package.

    :param apkg_path: Path to .apkg file.
    """
    with AnkiPackage(str(apkg_path)) as pkg:
        decks = pkg.get_decks()
        cards = pkg.get_cards()

        print(f"Decks ({len(decks)}):\n")

        for deck_id, deck_info in decks.items():
            name = deck_info.get("name", "Unknown")
            card_count = len([c for c in cards if str(c["did"]) == deck_id])
            print(f"  {name}")
            print(f"    ID: {deck_id}")
            print(f"    Cards: {card_count}")
            print()


@inspect_app.command
def models(apkg_path: Path, *, verbose: bool = False):
    """List note types (models) and their fields.

    :param apkg_path: Path to .apkg file.
    :param verbose: If True, show all model details.
    """
    with AnkiPackage(str(apkg_path)) as pkg:
        models = pkg.get_models()

        print(f"Note Types ({len(models)}):\n")

        for model_id, model_info in models.items():
            name = model_info.get("name", "Unknown")
            fields = [f["name"] for f in model_info.get("flds", [])]
            templates = [t["name"] for t in model_info.get("tmpls", [])]

            print(f"  {name}")
            print(f"    ID: {model_id}")
            print(f"    Fields: {', '.join(fields)}")
            print(f"    Templates: {', '.join(templates)}")

            if verbose:
                print(f"    CSS length: {len(model_info.get('css', ''))}")
                for tmpl in model_info.get("tmpls", []):
                    print(f"    Template '{tmpl['name']}':")
                    print(f"      Front: {len(tmpl.get('qfmt', ''))} chars")
                    print(f"      Back: {len(tmpl.get('afmt', ''))} chars")
            print()


@inspect_app.command
def templates(
    apkg_path: Path,
    *,
    model_name: str | None = None,
):
    """Show card templates (question and answer) for each note type.

    :param apkg_path: Path to .apkg file.
    :param model_name: Filter by model name (substring match).
    """
    with AnkiPackage(str(apkg_path)) as pkg:
        models = pkg.get_models(include_templates=True)

        for model_id, model_info in models.items():
            name = model_info.get("name", "Unknown")
            if model_name and model_name.lower() not in name.lower():
                continue

            try:
                width = os.get_terminal_size().columns
            except OSError:
                width = 80
            sep = "=" * width
            rule = "-" * width

            print()
            print(sep)
            print(f"#  {name} (ID: {model_id})")
            print(sep)
            for tmpl in model_info.get("tmpls", []):
                print()
                print(f"##  Template: {tmpl.get('name', 'Card 1')}")
                print(rule)
                print("###  Question (front)")
                print(rule)
                print(tmpl.get("qfmt", "(none)"))
                print()
                print(rule)
                print("###  Answer (back)")
                print(rule)
                print(tmpl.get("afmt", "(none)"))
                print()
            print()


@inspect_app.command
def fields(apkg_path: Path, model_name: str | None = None):
    """List all field names across models.

    :param apkg_path: Path to .apkg file.
    :param model_name: Filter by model name.
    """
    with AnkiPackage(str(apkg_path)) as pkg:
        models = pkg.get_models()

        for _model_id, model_info in models.items():
            name = model_info.get("name", "Unknown")

            if model_name and model_name.lower() not in name.lower():
                continue

            fields = model_info.get("flds", [])

            print(f"{name}:")
            for i, field in enumerate(fields):
                print(f"  {i}: {field['name']}")
            print()


@inspect_app.command
def audio(apkg_path: Path, *, extract: Path | None = None):
    """Show audio/media statistics and optionally extract files.

    :param apkg_path: Path to .apkg file.
    :param extract: Directory to extract audio files to.
    """
    with AnkiPackage(str(apkg_path)) as pkg:
        stats = pkg.get_audio_statistics()

        print("Media Statistics:")
        print(f"  Total files: {stats['total_media_files']}")
        print(f"  Audio files: {stats['audio_files']}")
        print(f"  Image files: {stats['image_files']}")
        print(f"  Audio formats: {stats['audio_formats']}")
        print()

        if extract:
            extracted = pkg.extract_audio_files(str(extract))
            print(f"Extracted {len(extracted)} audio files to: {extract}")
            for filename in list(extracted.keys())[:10]:
                print(f"  {filename}")
            if len(extracted) > 10:
                print(f"  ... and {len(extracted) - 10} more")


@inspect_app.command
def stats(apkg_path: Path):
    """Show overall statistics for the package.

    :param apkg_path: Path to .apkg file.
    """
    with AnkiPackage(str(apkg_path)) as pkg:
        cards = pkg.get_cards()
        decks = pkg.get_decks()
        models = pkg.get_models()
        audio_stats = pkg.get_audio_statistics()

        print(f"File: {apkg_path.name}")
        print(f"Size: {apkg_path.stat().st_size / 1024:.1f} KB")
        print()
        print("Counts:")
        print(f"  Cards: {len(cards)}")
        print(f"  Decks: {len(decks)}")
        print(f"  Note Types: {len(models)}")
        print(f"  Media Files: {audio_stats['total_media_files']}")
        print(f"  Audio Files: {audio_stats['audio_files']}")
        print()

        print("Decks:")
        for deck_id, deck_info in decks.items():
            count = len([c for c in cards if str(c["did"]) == deck_id])
            print(f"  {deck_info.get('name', 'Unknown')}: {count} cards")
        print()

        print("Note Types:")
        for model_id, model_info in models.items():
            count = len([c for c in cards if str(c["mid"]) == model_id])
            fields = [f["name"] for f in model_info.get("flds", [])]
            print(f"  {model_info.get('name', 'Unknown')}: {count} cards")
            print(f"    Fields: {', '.join(fields)}")


@inspect_app.command
def sample(apkg_path: Path, *, count: int = 5):
    """Show sample field values from cards.

    :param apkg_path: Path to .apkg file.
    :param count: Number of sample values per field.
    """
    with AnkiPackage(str(apkg_path)) as pkg:
        cards = pkg.get_cards()
        models = pkg.get_models()
        decks = pkg.get_decks()

        for model_id, model_info in models.items():
            model_name = model_info.get("name", "Unknown")
            field_names = [f["name"] for f in model_info.get("flds", [])]

            model_cards = [c for c in cards if str(c["mid"]) == model_id]
            if not model_cards:
                continue

            print(f"{model_name} ({len(model_cards)} cards):")
            print("-" * 40)

            for field_name in field_names:
                print(f"\n  {field_name}:")
                samples = []
                for card in model_cards[: count * 2]:
                    parsed = pkg.parse_card(card, models, decks)
                    value = parsed["fields"].get(field_name, "")
                    value = re.sub(r"<[^>]+>", "", str(value)).strip()
                    if value and value not in samples:
                        samples.append(value[:80])
                    if len(samples) >= count:
                        break

                for sample in samples:
                    print(f"    - {sample}")
            print()


# =============================================================================
# Style - apply CSS and card templates (any deck)
# =============================================================================


style_app = cyclopts.App(
    name="style", help="Apply CSS and card templates to an Anki deck"
)
app.command(style_app)


@style_app.command
def apply(
    apkg_path: Path,
    *,
    output: Path | None = None,
    css: Path | None = None,
    front: Path | None = None,
    back: Path | None = None,
):
    """Apply CSS and card templates to a deck.

    The CSS starts with the shared sheet `templates/shared/card.css` and then
    the deck `card.css`, so a deck rule overrides a shared rule. The templates
    and the deck `card.css` default to the repository `templates/default/`
    directory.
    Auto-maps mismatched field names to closest deck fields.

    :param apkg_path: Input .apkg file.
    :param output: Output .apkg file (default: input_styled.apkg).
    :param css: Deck CSS file (default: `card.css` in `templates/default/`).
    :param front: Front template file (default: `front.html` in
        `templates/default/`).
    :param back: Back template file (default: `back.html` in
        `templates/default/`).
    """
    apkg_path = Path(apkg_path)
    if output is None:
        output = apkg_path.parent / f"{apkg_path.stem}_styled.apkg"
    output = Path(output)

    if apkg_path.resolve() != output.resolve():
        try:
            shutil.copy(apkg_path, output)
        except OSError as exc:
            raise RuntimeError(f"Cannot copy {apkg_path} to {output}.") from exc

    _n, applied = apply_templates(
        output,
        css_path=css or DEFAULT_CSS_PATH,
        front_path=front or DEFAULT_FRONT_PATH,
        back_path=back or DEFAULT_BACK_PATH,
        verbose=True,
    )
    if not applied:
        print(
            f"\nNothing to apply. Check {DEFAULT_CSS_PATH}, "
            f"{DEFAULT_FRONT_PATH}, and {DEFAULT_BACK_PATH} first."
        )
        return
    print(f"\nSaved: {output}")


# =============================================================================
# Fix commands - optimize decks by language
# =============================================================================

fix_app = cyclopts.App(name="fix", help="Fix and optimize Anki decks")
app.command(fix_app)


_PARAMS_DOC = """
:param apkg_path: Input .apkg file.
:param output: Output .apkg file (default: from the language and mode).
:param core: Build the core deck.
:param words: Core mode: tier F words after the Swadesh set.
:param check: Run the gates on the output and do not run the mode again.
"""


def _declare_core_fix(
    route: fixrun.FixLanguage, declaration: FixCommand
) -> Callable[..., None]:
    """Declare the ``fix`` command of a language without a cull mode.

    :param route: Values of the shared fix route.
    :param declaration: Declared ``fix`` command of the language.
    :returns: Command with the shared option set.
    """
    def command(
        apkg_path: Path = declaration.apkg_path,
        *,
        output: Path | None = None,
        core: bool = False,
        cards: Annotated[
            int | None, Parameter(help="Core mode: card cap (300).")
        ] = None,
        words: int = declaration.words,
        check: bool = False,
    ) -> None:
        fixrun.run_fix(
            route,
            apkg_path=apkg_path,
            output=output,
            core=core,
            cards=cards,
            words=words,
            check=check,
        )

    command.__doc__ = declaration.description + _PARAMS_DOC
    return command


def _declare_cull_fix(
    route: fixrun.FixLanguage, declaration: FixCommand
) -> Callable[..., None]:
    """Declare the ``fix`` command of a language with a cull mode.

    :param route: Values of the shared fix route.
    :param declaration: Declared ``fix`` command of the language.
    :returns: Command with the shared option set and the select options.
    """
    def command(
        apkg_path: Path = declaration.apkg_path,
        *,
        output: Path | None = None,
        select: Annotated[bool, Parameter(help=declaration.select_help)] = False,
        core: bool = False,
        cards: Annotated[
            int | None,
            Parameter(help="Select and core mode: card cap (select 1000, core 300)."),
        ] = None,
        reserve: Annotated[
            bool, Parameter(help="Select mode: add the Swadesh reserve cards.")
        ] = True,
        words: int = declaration.words,
        check: bool = False,
    ) -> None:
        fixrun.run_fix(
            route,
            apkg_path=apkg_path,
            output=output,
            core=core,
            select=select,
            cards=cards,
            reserve=reserve,
            words=words,
            check=check,
        )

    command.__doc__ = declaration.description + _PARAMS_DOC
    return command


for _language in LANGUAGES.values():
    _declaration = _language.fix
    if _declaration is None:
        continue
    _declare = _declare_cull_fix if _declaration.cull else _declare_core_fix
    _route = fixrun.FixLanguage(
        name=_language.name,
        label=_language.label,
        core=_declaration.core,
        cull=_declaration.cull,
    )
    _command = _declare(_route, _declaration)
    globals()[_declaration.code] = _command
    fix_app.command(_command, name=_language.name, alias=_declaration.code)

# =============================================================================
# Swadesh commands - build decks and run their gates
# =============================================================================


VOCAB_LANGUAGES = vocab_languages()
SENTENCE_PACK_LANGUAGE_NAMES = ", ".join(
    language.name for language in profile_languages()
)

swadesh_app = cyclopts.App(
    name="swadesh", help="Build Swadesh decks and run their gates"
)
app.command(swadesh_app)


@swadesh_app.command
def vocab(
    language: Annotated[
        str, Parameter(help=f"One of {', '.join(VOCAB_LANGUAGES)}, or all.")
    ],
    prune: bool = True,
):
    """Build the Swadesh vocabulary deck for a language.

    :param language: One of the supported languages, or all.
    :param prune: Delete audio files that the current list does not use.
    """
    lang = resolve(language)
    if lang == "all":
        for name in VOCAB_LANGUAGES:
            swadesh_vocab.build_vocab_deck(name, prune=prune)
        return

    if lang not in VOCAB_LANGUAGES:
        print(f"Unknown language: {language}. Supported: {', '.join(VOCAB_LANGUAGES)}")
        return
    swadesh_vocab.build_vocab_deck(lang, prune=prune)


def _fetch_language(profile: LanguageProfile, refresh: bool) -> None:
    """Download the Tatoeba exports of one language and English (stage S0).

    :param profile: Profile of the language. Its code selects the export.
    :param refresh: Ignore the cache and download every file again.
    """
    files = sentence_tatoeba.language_files(
        profile.tatoeba_lang, native_audio=profile.native_audio
    )
    files.update(sentence_tatoeba.language_files("eng", native_audio=False))
    results = sentence_tatoeba.fetch_tatoeba(files, refresh=refresh)
    for name, info in results.items():
        print(f"{name}.tsv: {info['rows']} rows ({info['source']})")


@swadesh_app.command
def pack(
    language: Annotated[str, Parameter(help=f"One of {SENTENCE_PACK_LANGUAGE_NAMES}.")],
    *,
    refresh: bool = False,
    stage: Literal["fetch", "sentences", "audio", "all"] = "all",
    per_word: int = 2,
    max_cards: int = 450,
    max_unknown: int = 2,
    min_words: int | None = None,
    max_words: int | None = None,
    native_audio: str = "none",
    voice: str = "",
    slow_audio: bool = False,
    pack_license: str = "public",
    tts: str = "",
) -> None:
    """Build one Swadesh sentence pack from the Tatoeba exports (S0 to S5).

    The command runs the stages in order and stops after the ``--stage``
    value. The default value runs every stage and writes the APKG.

    :param language: Language name, for example mandarin.
    :param refresh: Ignore the Tatoeba cache and download every file again.
    :param stage: Last stage to run: fetch, sentences, audio, or all.
    :param per_word: Card limit per word form.
    :param max_cards: Hard card cap.
    :param max_unknown: Unknown token budget per sentence.
    :param min_words: Minimum word count of a card. The profile default
        applies when empty.
    :param max_words: Maximum word count of a card. The profile default
        applies when empty.
    :param native_audio: Tatoeba tier: none, auto, or only.
    :param voice: TTS voice name. The profile default applies when empty.
    :param slow_audio: Generate the slow variant.
    :param pack_license: public or private.
    :param tts: TTS engine, gtts or edge. The installed engine applies when
        empty.
    """
    entry = LANGUAGES.get(resolve(language))
    if entry is None or entry.module is None:
        print(
            f"Unsupported language: {language}. "
            f"Supported: {SENTENCE_PACK_LANGUAGE_NAMES}."
        )
        return
    _fetch_language(entry.profile, refresh=refresh)
    if stage == "fetch":
        return
    _language_sentences(
        entry.module,
        entry.profile,
        per_word=per_word,
        max_cards=max_cards,
        max_unknown=max_unknown,
        min_words=min_words,
        max_words=max_words,
    )
    if stage == "sentences":
        return
    _language_audio(
        entry.module,
        entry.profile,
        native_audio=native_audio,
        voice=voice,
        slow_audio=slow_audio,
        pack_license=pack_license,
        tts=tts,
    )
    if stage == "audio":
        return
    _language_pack(entry.module, entry.profile, pack_license=pack_license)


@swadesh_app.command
def audit(
    language: Annotated[str, Parameter(help=f"One of {SENTENCE_PACK_LANGUAGE_NAMES}.")],
) -> None:
    """Run the quality gates of a Swadesh sentence pack.

    :param language: Language name, for example mandarin.
    """
    lang = resolve(language)
    entry = LANGUAGES.get(lang)
    if entry is None or entry.audit is None:
        print(
            f"Unsupported language: {language}. "
            f"Supported: {SENTENCE_PACK_LANGUAGE_NAMES}."
        )
        return
    code = print_gate_table(entry.audit())
    if code:
        raise SystemExit(code)


def _language_sentences(
    module,
    profile,
    per_word: int,
    max_cards: int,
    max_unknown: int,
    min_words: int | None,
    max_words: int | None,
) -> None:
    """Build the candidate pool and the card CSV files of one sentence pack.

    :param module: Language module of the pack.
    :param profile: Language profile of the pack.
    :param per_word: Card limit per word form.
    :param max_cards: Hard card cap.
    :param max_unknown: Unknown token budget per sentence.
    :param min_words: Minimum word count of a core card.
    :param max_words: Maximum word count of a card.
    """
    table = module.load_word_table()
    failures = module.check_word_table(table)
    if failures:
        for failure in failures:
            print(f"Error: {failure}")
        raise SystemExit(1)
    known = module.known_words(table)
    pool, pool_report = sentence_pool.build_candidate_pool(
        table,
        profile,
        min_units=profile.min_units if min_words is None else min_words,
        max_units=profile.max_units if max_words is None else max_words,
        max_unknown=max_unknown,
        fallback=True,
        known_words=known,
        make_candidate=module.make_candidate,
        frames_path=sentence_pool.default_frames_path(profile),
    )
    cards, selection = sentence_select.select_sentences(
        pool,
        table,
        profile,
        per_word=per_word,
        max_cards=max_cards,
        finalize=profile.finalize_candidate,
    )
    report = sentence_pool.build_pack_report(pool_report, selection)
    print(f"wrote {module.write_sentence_csv(cards)}")
    print(f"wrote {module.write_sentence_review_csv(pool, table, per_word=per_word)}")
    print(f"wrote {sentence_pool.write_pool_report(report, module.POOL_REPORT_PATH)}")
    sentence_pool.print_sentence_report(report)


def _language_audio(
    module,
    profile,
    native_audio: str,
    voice: str,
    slow_audio: bool,
    pack_license: str,
    tts: str,
) -> None:
    """Generate and validate the audio files of one sentence pack.

    :param module: Language module of the pack.
    :param profile: Language profile of the pack.
    :param native_audio: Tatoeba tier: none, auto, or only.
    :param voice: TTS voice name.
    :param slow_audio: Generate the slow variant.
    :param pack_license: public or private.
    :param tts: TTS engine, gtts or edge.
    """
    if tts and tts not in ("gtts", "edge"):
        print(f"Unsupported tts engine: {tts}. Use gtts or edge.")
        return
    rows = pack_audio.load_sentence_rows(module.SENTENCES_PATH)
    engine = "gtts"
    if tts:
        engine = tts
        if tts == "edge" and importlib.util.find_spec("edge_tts") is None:
            print("edge-tts is not installed. Use --tts gtts.")
            return
    try:
        result = pack_audio.generate_audio(
            rows,
            profile,
            voice=voice or profile.voice,
            native_audio=native_audio,
            slow_audio=slow_audio,
            pack_license=pack_license,
            engine=engine,
            csv_path=module.SENTENCES_PATH,
        )
    except ValueError as error:
        print(f"Error: {error}")
        raise SystemExit(1) from error
    print(
        f"engine {result['engine']} voice {result['voice']}: "
        f"{result['tts']} tts, {result['native']} native, "
        f"{result['words']} word, {result['skipped']} cached, "
        f"{result['failed']} failed, {result['duration_s']:.1f} s"
    )
    for failure in result["failures"]:
        print(f"failed: {failure['filename']} {failure['error']}")
    print(f"wrote {result['index_path']}")
    if result["csv_path"]:
        print(f"wrote {result['csv_path']}")


def _language_pack(module, profile, pack_license: str) -> None:
    """Build the APKG file, the manifest, and the attribution of one pack.

    :param module: Language module of the pack.
    :param profile: Language profile of the pack.
    :param pack_license: public or private.
    """
    result = sentence_pack.build_pack(
        profile,
        module.load_word_table,
        pack_license=pack_license,
    )
    counts = result["manifest"]["counts"]
    print(
        f"{counts['cards']} cards, {counts['scaffold']} scaffold, "
        f"{counts['tts_audio']} tts, {counts['native_audio']} native, "
        f"{result['media_count']} media files, {result['skipped']} skipped"
    )
    print(f"wrote {result['output']}")
    print(f"wrote {result['manifest_path']}")


# =============================================================================
# Preview - build the browser preview of the card templates
# =============================================================================


@app.command
def preview(
    output: Path = Path("preview/cards.html"),
    *,
    apkg: Path | None = None,
    limit: int = 50,
    deck: str | None = None,
    search: str | None = None,
    media: bool = True,
    media_dir: Path | None = None,
    open_browser: bool = False,
):
    """Build a browser preview of the card templates or of a package.

    Without ``--apkg`` the command reads the repository templates. With
    ``--apkg`` the command reads the note fields, the card templates, and the
    media files of that package.

    :param output: HTML file to write (default: ``preview/cards.html``).
    :param apkg: Package to preview. Defaults to the repository templates.
    :param limit: Maximum card count of the package preview. Zero shows all.
    :param deck: Only show package cards of a deck with this text.
    :param search: Only show package cards with a field that holds this text.
    :param media: Extract media and link it in the page.
    :param media_dir: Directory for the extracted media files.
    :param open_browser: Open the written file in the default browser.
    """
    if apkg is None:
        cards, sections = write_preview(output)
        print(f"wrote {output} with {cards} cards in {sections} sections")
    else:
        result = write_apkg_preview(
            apkg,
            output,
            limit=limit,
            deck=deck,
            search=search,
            extract_media=media,
            media_dir=media_dir,
        )
        print(
            f"wrote {result.output_path} with {result.cards} of "
            f"{result.total} cards in {result.sections} sections and "
            f"{result.media} media files"
        )
    if open_browser:
        webbrowser.open(Path(output).resolve().as_uri())


def main() -> None:
    """Main entry point. Invokes the cyclopts app."""
    app()


if __name__ == "__main__":
    main()
