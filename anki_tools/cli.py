"""
CLI tools for Anki package (.apkg) files.

Commands:
    inspect - Diagnostic tools to inspect .apkg files
    swadesh - Build Swadesh vocabulary APKG files with audio
    fix - Fix and optimize decks (zh, fr, yue)
    style - Apply CSS and card templates to any deck
"""

import csv
import os
import re
import shutil
import time
from pathlib import Path
from textwrap import dedent
from typing import Literal

import cyclopts
from gtts import gTTS

from anki_tools.jyutping import fix_jyutping
from anki_tools.mandarin_audio import (
    DEFAULT_VOICE as MANDARIN_DEFAULT_VOICE,
    generate_audio,
    load_sentence_rows,
)
from anki_tools.mandarin_audit import print_audit, run_audit
from anki_tools.mandarin_deck import build_pack
from anki_tools.mandarin_fetch import fetch_tatoeba
from anki_tools.mandarin_swadesh import (
    build_candidate_pool,
    build_pack_report,
    check_word_table,
    load_word_table,
    print_sentence_report,
    select_sentences,
    write_pool_report,
    write_sentence_csv,
    write_sentence_review_csv,
)
from anki_tools.package import AnkiPackage
from anki_tools.pinyin import fix_pinyin
from anki_tools.rank import (
    Sentence,
    YUE_FREQUENCY_ATTRIBUTION,
    char_similarity_zh_same_len,
    extract_sentences,
    rank_sentences_es,
    rank_sentences_fr,
    rank_sentences_yue,
    rank_sentences_zh,
    reduce_sentences_stratified,
    word_similarity_es,
    write_ranking_csv,
)


from anki_tools.reorder import reorder_deck
from anki_tools.yue_frequency import (
    CIFU_ATTRIBUTION,
    FREQUENCY_PATH,
    build_yue_tokenizer,
    check_frequency,
    download_cifu,
    load_yue_frequency,
    rank_change_rows,
    read_cifu,
    read_overrides,
    write_freq_change_csv,
    write_frequency_tsv,
)
from anki_tools.yue_select import (
    SelectionOptions,
    print_selection_summary,
    select_yue_sentences,
    write_coverage_csv,
    write_select_csv,
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

def _extract_template_fields(template: str) -> set[str]:
    """Extract field names from Anki template ({{FieldName}} syntax).

    :param template: Template string (qfmt or afmt).
    :returns: Set of field names referenced.
    """
    matches = re.findall(r"\{\{([^#/}]+)\}\}", template)
    builtins = {"FrontSide", "Tags", "Type", "Deck", "Subdeck", "Card"}
    return {m.strip() for m in matches if m.strip() not in builtins}


def _find_best_match(field: str, deck_fields: set[str]) -> str | None:
    """Find the best matching deck field for a template field.

    :param field: Template field name.
    :param deck_fields: Set of deck field names.
    :returns: Best matching deck field name or None.
    """
    field_lower = field.lower()

    for deck_field in deck_fields:
        if deck_field.lower() == field_lower:
            return deck_field

    field_words = set(re.split(r"[^a-zA-Z]+", field_lower))
    best_match = None
    best_score = 0

    for deck_field in deck_fields:
        deck_words = set(re.split(r"[^a-zA-Z]+", deck_field.lower()))
        common = field_words & deck_words
        if common and len(common) > best_score:
            best_score = len(common)
            best_match = deck_field

    return best_match


def _apply_field_mapping(template: str, mapping: dict[str, str]) -> str:
    """Replace field references in template using mapping.

    :param template: Template string with {{FieldName}} refs.
    :param mapping: Dict mapping old field names to new.
    :returns: Template with refs replaced.
    """
    result = template
    for old_name, new_name in mapping.items():
        result = result.replace(f"{{{{{old_name}}}}}", f"{{{{{new_name}}}}}")
    return result


def _apply_style(
    target_apkg: Path,
    *,
    css_path: Path = Path("templates/card.css"),
    front_path: Path = Path("templates/front.html"),
    back_path: Path = Path("templates/back.html"),
    verbose: bool = True,
) -> tuple[int, list[str]]:
    """Load CSS/templates, map fields to deck, update target package. Returns (models_updated, applied_names)."""
    with AnkiPackage(str(target_apkg)) as pkg:
        models = pkg.get_models(include_templates=True)
        all_fields = set()
        for model in models.values():
            all_fields.update(f["name"] for f in model.get("flds", []))
        if verbose:
            print("Deck fields:")
            for _mid, model in models.items():
                print(f"  {', '.join(f['name'] for f in model['flds'])}")

    css_content = None
    if css_path.exists():
        css_content = css_path.read_text(encoding="utf-8")
        if verbose:
            print(f"\nLoaded: {css_path}")
    elif verbose:
        print(f"\nNo {css_path} found, skipping CSS")

    qfmt = None
    if front_path.exists():
        qfmt = front_path.read_text(encoding="utf-8")
        if verbose:
            print(f"Loaded: {front_path}")
        missing = _extract_template_fields(qfmt) - all_fields
        if missing:
            mapping = {f: _find_best_match(f, all_fields) for f in missing}
            mapping = {k: v for k, v in mapping.items() if v}
            if mapping and verbose:
                print(f"  Auto-mapping: {mapping}")
            qfmt = _apply_field_mapping(qfmt, mapping)
    elif verbose:
        print(f"No {front_path} found, skipping front template")

    afmt = None
    if back_path.exists():
        afmt = back_path.read_text(encoding="utf-8")
        if verbose:
            print(f"Loaded: {back_path}")
        missing = _extract_template_fields(afmt) - all_fields
        if missing:
            mapping = {f: _find_best_match(f, all_fields) for f in missing}
            mapping = {k: v for k, v in mapping.items() if v}
            if mapping and verbose:
                print(f"  Auto-mapping: {mapping}")
            afmt = _apply_field_mapping(afmt, mapping)
    elif verbose:
        print(f"No {back_path} found, skipping back template")

    if not css_content and not qfmt and not afmt:
        return (0, [])

    with AnkiPackage(str(target_apkg)) as pkg:
        models = pkg.get_models()
        for model_id in models:
            pkg.update_model_template(
                model_id=model_id, css=css_content, qfmt=qfmt, afmt=afmt
            )
        pkg.save(target_apkg)

    applied = []
    if css_content:
        applied.append("CSS")
    if qfmt:
        applied.append("front")
    if afmt:
        applied.append("back")
    if verbose:
        print(f"Updated {len(models)} model(s)")
    return (len(models), applied)


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

    Looks for templates/card.css, templates/front.html, templates/back.html by default.
    Auto-maps mismatched field names to closest deck fields.

    :param apkg_path: Input .apkg file.
    :param output: Output .apkg file (default: input_styled.apkg).
    :param css: CSS file (default: templates/card.css).
    :param front: Front template file (default: templates/front.html).
    :param back: Back template file (default: templates/back.html).
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

    _n, applied = _apply_style(
        output,
        css_path=css or Path("templates/card.css"),
        front_path=front or Path("templates/front.html"),
        back_path=back or Path("templates/back.html"),
        verbose=True,
    )
    if not applied:
        print("\nNothing to apply. Create templates/card.css, templates/front.html, or templates/back.html first.")
        return
    print(f"\nSaved: {output}")


# =============================================================================
# Freq commands - build word frequency lists
# =============================================================================

freq_app = cyclopts.App(name="freq", help="Build word frequency lists")
app.command(freq_app)


@freq_app.command(name="yue")
def freq_yue(
    *,
    output: Path = FREQUENCY_PATH,
    spoken_weight: float = 0.7,
    refresh: bool = False,
    check: bool = False,
):
    """Build the Cantonese word frequency list from spoken and written counts.

    Blends the Cifu spoken counts with the Cifu and Apple Daily written counts,
    folds variant characters, and applies the override file. Downloads the two
    source files into ~/.cache/anki-tools when they are absent.

    :param output: Output TSV file, sorted by rank.
    :param spoken_weight: Weight of the spoken term in the blend (0 to 1).
    :param refresh: Download the source files again.
    :param check: Run gates G1, G3, and G4 and exit with code 1 on a failure.
    """
    data = load_yue_frequency(spoken_weight=spoken_weight, refresh=refresh)
    write_frequency_tsv(data, output)
    print(f"Wrote {len(data.rows)} words to {output}")
    print(f"  {CIFU_ATTRIBUTION}")
    print(f"  {YUE_FREQUENCY_ATTRIBUTION}")
    if not check:
        return
    failed = False
    for gate, ok, message in check_frequency(data):
        print(f"{gate} {'PASS' if ok else 'FAIL'}: {message}")
        failed = failed or not ok
    if failed:
        raise SystemExit(1)


# =============================================================================
# Fix commands - optimize decks by language
# =============================================================================

fix_app = cyclopts.App(name="fix", help="Fix and optimize Anki decks")
app.command(fix_app)


@fix_app.command
def zh(
    apkg_path: Path,
    *,
    output: Path | None = None,
    model_id: int | None = None,
    keep_filtered: bool = False,
    no_pinyin_fix: bool = False,
    no_style: bool = False,
    verbose: bool = False,
    max_sentences: int | None = None,
    similarity_threshold: float = 0.67,
    min_chars: int = 0,
):
    """Fix Chinese deck: rank, reorder, fix pinyin, and apply styling.

    Ranks sentences by frequency and complexity, reorders the deck,
    fixes pinyin formatting, and applies card styling.
    Optionally reduce to max_sentences using stratified complexity sampling
    with character-based similarity deduplication.

    :param apkg_path: Input .apkg file.
    :param output: Output .apkg file (default: input_reordered.apkg).
    :param model_id: Filter by specific model ID.
    :param keep_filtered: If True, keep filtered cards (names, invalid).
    :param no_pinyin_fix: If True, skip pinyin correction.
    :param no_style: If True, skip card styling (CSS/templates).
    :param verbose: If True, show detailed output.
    :param max_sentences: Optional max sentence count (e.g., 3000).
    :param similarity_threshold: Jaccard threshold for deduplication (default 0.67).
        Uses unique-char-set-size matching so only same-unique-length sentences are
        compared. For 3-unique-char sentences Jaccard jumps from 0.5 directly to 1.0
        (no values between), so 0.67 catches only true anagrams like '你要我'/'我要你'.
        0.5 would also remove valid pairs that share 2 of 4 unique chars ('我在这'/'你在这').
    :param min_chars: Skip dedup for sentences with fewer raw chars than this (default 0 = off).
    """
    if output is None:
        output = apkg_path.parent / f"{apkg_path.stem}_reordered.apkg"

    ranking_csv = apkg_path.parent / f"{apkg_path.stem}.rank.csv"

    print("=" * 50)
    print("Processing Chinese Anki Deck")
    print("=" * 50)
    print(f"Input:   {apkg_path}")
    print(f"Output:  {output}")
    print(f"Ranking: {ranking_csv}")
    if max_sentences:
        print(f"Target:  {max_sentences} sentences (stratified reduction)")
    print()

    print("Step 1: Ranking sentences...")
    sentences = extract_sentences(str(apkg_path), model_id)

    if not sentences:
        print("No sentences found!")
        return

    ranked = rank_sentences_zh(sentences)

    if max_sentences and len(ranked) > max_sentences:
        print(f"\nStep 2: Reducing to {max_sentences} sentences...")
        ranked = reduce_sentences_stratified(
            ranked,
            target_count=max_sentences,
            similarity_threshold=similarity_threshold,
            similarity_fn=char_similarity_zh_same_len,
            min_chars=min_chars,
        )

    write_ranking_csv(ranked, str(ranking_csv))

    print("\nStep 3: Reordering deck...")
    stats = reorder_deck(
        str(apkg_path),
        str(output),
        str(ranking_csv),
        remove_filtered=not keep_filtered,
        remove_unranked=bool(max_sentences),
    )

    if not no_pinyin_fix:
        print("\nStep 4: Fixing pinyin...")
        fix_pinyin(
            output,
            output=output,
            csv_output=Path("/dev/null"),
            verbose=verbose,
        )

    styles_applied = 0
    if not no_style:
        print("\nStep 5: Applying card styles...")
        styles_applied, applied = _apply_style(
            Path(output),
            css_path=Path("templates/card.css"),
            front_path=Path("templates/front.html"),
            back_path=Path("templates/back.html"),
            verbose=False,
        )
        if applied:
            print(f"  Applied: {', '.join(applied)} to {styles_applied} model(s)")

    print()
    print("=" * 50)
    print("DONE!")
    print("=" * 50)
    print(f"  Sentences ranked:    {len(ranked)}")
    print(f"  Cards reordered:     {stats['cards_reordered']}")
    print(f"  Cards removed:       {stats['cards_removed']}")
    print(f"  Styles applied:      {styles_applied} model(s)")
    print(f"  Output:              {output}")
    print(f"  Ranking CSV:         {ranking_csv}")


DEFAULT_FRENCH_APKG = Path("apkg/6000_French_Sentences_w_Audio.apkg")


@fix_app.command
def fr(
    apkg_path: Path = DEFAULT_FRENCH_APKG,
    *,
    output: Path | None = None,
    model_id: int | None = None,
    text_field: str = "French",
    keep_filtered: bool = False,
):
    """Fix French deck: rank and reorder by complexity, frequency, and uniqueness.

    Uses word-based scoring. Ranks sentences and reorders the deck optimally.

    :param apkg_path: Input .apkg file.
    :param output: Output .apkg file (default: input_reordered.apkg).
    :param model_id: Filter by specific model ID.
    :param text_field: Preferred sentence field (default: French); inferred from deck if missing.
    :param keep_filtered: If True, keep filtered cards instead of removing.
    """
    apkg_path = Path(apkg_path)
    if not apkg_path.exists():
        print(f"Error: File not found: {apkg_path}")
        return

    if output is None:
        output = apkg_path.parent / f"{apkg_path.stem}_reordered.apkg"

    ranking_csv = apkg_path.parent / f"{apkg_path.stem}.rank.csv"

    print("=" * 50)
    print("Processing French Anki Deck")
    print("=" * 50)
    print(f"Input:   {apkg_path}")
    print(f"Output:  {output}")
    print(f"Ranking: {ranking_csv}")
    print()

    print("Step 1: Ranking sentences...")
    sentences = extract_sentences(str(apkg_path), model_id, text_field=text_field)
    if not sentences:
        print("No sentences found!")
        return

    ranked = rank_sentences_fr(sentences)
    write_ranking_csv(ranked, str(ranking_csv))

    print("\nStep 2: Reordering deck...")
    stats = reorder_deck(
        str(apkg_path),
        str(output),
        str(ranking_csv),
        remove_filtered=not keep_filtered,
    )

    print()
    print("=" * 50)
    print("DONE!")
    print("=" * 50)
    print(f"  Sentences ranked:    {len(ranked)}")
    print(f"  Cards reordered:     {stats['cards_reordered']}")
    print(f"  Cards removed:       {stats['cards_removed']}")
    print(f"  Output:              {output}")
    print(f"  Ranking CSV:         {ranking_csv}")


DEFAULT_CANTONESE_APKG = Path("apkg/LTL Cantonese Deck Level 1 - Short.apkg")


@fix_app.command
def yue(
    apkg_path: Path = DEFAULT_CANTONESE_APKG,
    *,
    output: Path | None = None,
    model_id: int | None = None,
    keep_filtered: bool = False,
    max_sentences: int | None = None,
    per_word: int = 2,
    max_uses: int = 8,
    top_words: int = 1500,
    order: Literal["coverage", "score"] = "coverage",
    no_select: bool = False,
    min_han: int = 3,
    max_han: int = 12,
    max_unknown: int = 2,
    known_top: int = 2000,
    similarity_threshold: float = 0.67,
    no_final_mark: bool = False,
    freq: Literal["blend", "spoken", "written"] = "blend",
    spoken_weight: float = 0.7,
    freq_report: bool = False,
):
    """Fix Cantonese deck: convert to traditional, fix jyutping, select, rank, and reorder.

    Converts simplified to traditional Chinese, generates accurate jyutping,
    and ranks by spoken and written frequency and complexity. Then filters
    fragments and hard sentences, and keeps the fewest sentences that show the
    most frequent spoken words, easiest first. Writes .rank.csv, .select.csv,
    and .coverage.csv next to the input.

    :param apkg_path: Input .apkg file.
    :param output: Output .apkg file (default: input_reordered.apkg).
    :param model_id: Filter by specific model ID.
    :param keep_filtered: With --no-select, keep high-similarity cards.
    :param max_sentences: Stop the selection at this count (default: stop at zero gain).
    :param per_word: Cards wanted for each target word.
    :param max_uses: Most kept sentences for one word outside the top 100 (0 is no limit).
    :param top_words: Size of the target set.
    :param order: coverage (easy to hard) or score (order of the ranker).
    :param no_select: Only rank and reorder, as before.
    :param min_han: Minimum Han characters in a sentence.
    :param max_han: Maximum Han characters in a sentence.
    :param max_unknown: Unknown words allowed in a sentence.
    :param known_top: Size of the known word set.
    :param similarity_threshold: Character Jaccard threshold for duplicates.
    :param no_final_mark: Keep sentences with no final mark.
    :param freq: Frequency term: blend, spoken, or written.
    :param spoken_weight: Weight of the spoken term in the blend (0 to 1).
    :param freq_report: Write the sentences that the frequency change moves.
    """
    apkg_path = Path(apkg_path)
    if not apkg_path.exists():
        print(f"Error: File not found: {apkg_path}")
        return

    if output is None:
        output = apkg_path.parent / f"{apkg_path.stem}_reordered.apkg"

    ranking_csv = apkg_path.parent / f"{apkg_path.stem}.rank.csv"
    select_csv = apkg_path.parent / f"{apkg_path.stem}.select.csv"
    coverage_csv = apkg_path.parent / f"{apkg_path.stem}.coverage.csv"

    print("=" * 50)
    print("Processing Cantonese Anki Deck")
    print("=" * 50)
    print(f"Input:   {apkg_path}")
    print(f"Output:  {output}")
    print(f"Ranking: {ranking_csv}")
    print()

    print("Step 1: Ranking sentences...")
    sentences = extract_sentences(str(apkg_path), model_id)

    if not sentences:
        print("No sentences found!")
        return

    extracted = list(sentences)
    ranked = rank_sentences_yue(sentences, freq=freq, spoken_weight=spoken_weight)
    if freq_report and freq != "written":
        comparison = rank_sentences_yue(
            [
                Sentence(note_id=s.note_id, text=s.text, original_order=s.original_order)
                for s in sentences
            ],
            freq="written",
        )
        rows = rank_change_rows(comparison, ranked)
        change_csv = apkg_path.parent / f"{apkg_path.stem}.freq_change.csv"
        write_freq_change_csv(rows, change_csv)
        print(f"  {len(rows)} sentences move by more than 100 places: {change_csv}")

    if no_select:
        write_ranking_csv(ranked, str(ranking_csv))
        print("\nStep 2: Reordering deck...")
        stats = reorder_deck(
            str(apkg_path),
            str(output),
            str(ranking_csv),
            remove_filtered=not keep_filtered,
        )
    else:
        options = SelectionOptions(
            max_sentences=max_sentences,
            per_word=per_word,
            max_uses=max_uses,
            top_words=top_words,
            order=order,
            min_han=min_han,
            max_han=max_han,
            max_unknown=max_unknown,
            known_top=known_top,
            similarity_threshold=similarity_threshold,
            final_mark=not no_final_mark,
        )
        print("\nStep 2: Selecting sentences...")
        frequency = load_yue_frequency(freq=freq, spoken_weight=spoken_weight)
        tokenizer = build_yue_tokenizer(
            read_cifu(download_cifu()), extra_words=read_overrides()
        )
        ranked_ids = {id(sentence) for sentence in ranked}
        dropped = [s for s in extracted if id(s) not in ranked_ids]
        report = select_yue_sentences(ranked, dropped, frequency, tokenizer, options)
        write_ranking_csv(report.ordered, str(ranking_csv))
        write_select_csv(report, select_csv)
        write_coverage_csv(report, coverage_csv)
        print_selection_summary(report, options)

        print("\nStep 3: Reordering deck...")
        stats = reorder_deck(
            str(apkg_path),
            str(output),
            str(ranking_csv),
            remove_filtered=False,
            remove_unranked=True,
        )

    print("\nFixing jyutping...")
    fix_jyutping(
        output,
        output=output,
        verbose=False,
    )

    print()
    print("=" * 50)
    print("DONE!")
    print("=" * 50)
    print(f"  Sentences ranked:    {len(ranked)}")
    print(f"  Cards reordered:     {stats['cards_reordered']}")
    print(f"  Cards removed:       {stats['cards_removed']}")
    print(f"  Output:              {output}")
    print(f"  Ranking CSV:         {ranking_csv}")
    if not no_select:
        print(f"  Select CSV:          {select_csv}")
        print(f"  Coverage CSV:        {coverage_csv}")


@fix_app.command
def es(
    apkg_path: Path,
    *,
    output: Path | None = None,
    model_id: int | None = None,
    text_field: str = "Spanish",
    keep_filtered: bool = False,
    max_sentences: int | None = None,
    similarity_threshold: float = 0.5,
):
    """Fix Spanish deck: rank, deduplicate, and reorder by complexity and frequency.

    Optionally reduce to max_sentences using stratified complexity sampling.

    :param apkg_path: Input .apkg file.
    :param output: Output .apkg file (default: input_reordered.apkg).
    :param model_id: Filter by specific model ID.
    :param text_field: Preferred sentence field (default: Spanish).
    :param keep_filtered: If True, keep filtered cards instead of removing.
    :param max_sentences: Optional max sentence count (e.g., 3000).
    :param similarity_threshold: Jaccard threshold for deduplication (default 0.5).
    """
    apkg_path = Path(apkg_path)
    if not apkg_path.exists():
        print(f"Error: File not found: {apkg_path}")
        return

    if output is None:
        output = apkg_path.parent / f"{apkg_path.stem}_reordered.apkg"

    ranking_csv = apkg_path.parent / f"{apkg_path.stem}.rank.csv"

    print("=" * 50)
    print("Processing Spanish Anki Deck")
    print("=" * 50)
    print(f"Input:   {apkg_path}")
    print(f"Output:  {output}")
    print(f"Ranking: {ranking_csv}")
    if max_sentences:
        print(f"Target:  {max_sentences} sentences (stratified reduction)")
    print()

    print("Step 1: Ranking sentences...")
    sentences = extract_sentences(str(apkg_path), model_id, text_field=text_field)
    if not sentences:
        print("No sentences found!")
        return

    print(f"Extracted {len(sentences)} sentences")

    ranked = rank_sentences_es(sentences)

    # Step 2: Reduce sentences if max_sentences specified
    if max_sentences and len(ranked) > max_sentences:
        print(f"\nStep 2: Reducing to {max_sentences} sentences...")
        ranked = reduce_sentences_stratified(
            ranked,
            target_count=max_sentences,
            similarity_threshold=similarity_threshold,
            similarity_fn=word_similarity_es,
        )

    write_ranking_csv(ranked, str(ranking_csv))

    print("\nStep 3: Reordering deck...")
    stats = reorder_deck(
        str(apkg_path),
        str(output),
        str(ranking_csv),
        remove_filtered=not keep_filtered,
        remove_unranked=bool(max_sentences),  # Remove unranked cards when reducing
    )

    print()
    print("=" * 50)
    print("DONE!")
    print("=" * 50)
    print(f"  Sentences ranked:    {len(ranked)}")
    print(f"  Cards reordered:     {stats['cards_reordered']}")
    print(f"  Cards removed:       {stats['cards_removed']}")
    print(f"  Output:              {output}")
    print(f"  Ranking CSV:         {ranking_csv}")


# =============================================================================
# Swadesh commands - build vocabulary APKG files with audio
# =============================================================================

def _clean_text_for_audio(text: str) -> str:
    """Remove parenthetical notes and variants from text for audio processing.

    :param text: Raw text (e.g. from CSV).
    :returns: Cleaned text for TTS.
    """
    text = re.sub(r"\s*\([^)]*\)", "", text)
    return text.split("/")[0].strip()


def _parse_vocab_entries(config: dict) -> list[dict]:
    """Parse vocabulary CSV file to extract entries.

    :param config: Config dict with vocab_file and audio_field_name.
    :returns: List of entry dicts with audio_filename set.
    """
    csv_path = Path(config["vocab_file"])
    if not csv_path.is_absolute():
        csv_path = Path(__file__).parent.parent / config["vocab_file"]
    entries = []
    try:
        handle = open(csv_path, "r", encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Cannot read the vocabulary file {csv_path}.") from exc
    with handle:
        reader = csv.DictReader(handle)
        for row in reader:
            entries.append({k: v.strip() if v else v for k, v in row.items()})

    for entry in entries:
        text = entry[config["audio_field_name"]]
        cleaned_text = _clean_text_for_audio(text).replace(" ", "_")
        entry["audio_filename"] = f"{cleaned_text}.mp3"

    return entries


def _prune_outdated_audio(audio_dir: Path, keep: set[str]) -> list[str]:
    """Delete audio files that no list entry refers to.

    :param audio_dir: Directory with generated audio files.
    :param keep: File names used by the current list entries.
    :returns: File names of the deleted audio files.
    """
    if not audio_dir.is_dir():
        return []

    removed = []
    for path in sorted(audio_dir.iterdir()):
        if not path.is_file() or path.name in keep:
            continue
        path.unlink()
        removed.append(path.name)

    return removed


def _generate_audio_files(entries: list[dict], config: dict) -> None:
    """Generate audio files for entries using TTS if they don't exist.

    :param entries: List of entry dicts from _parse_vocab_entries.
    :param config: Config dict with language and audio_dir.
    """
    language = config.get("language")
    if not language:
        return

    GTTS_LANG = {
        "arabic": "ar",
        "cantonese": "yue",
        "german": "de",
        "greek": "el",
        "hindi": "hi",
        "mandarin": "zh-cn",
        "spanish": "es",
    }

    lang = GTTS_LANG.get(language.lower())
    if not lang:
        print(f"Unknown language for gTTS: {language}")
        return

    audio_dir = Path(config["audio_dir"])
    if not audio_dir.is_absolute():
        audio_dir = Path(__file__).parent.parent / config["audio_dir"]
    audio_dir.mkdir(parents=True, exist_ok=True)

    delay_seconds = 0.5
    total_entries = len(entries)
    existing_count = 0
    generated_count = 0

    for i, entry in enumerate(entries, 1):
        audio_path = audio_dir / entry["audio_filename"]

        if audio_path.exists():
            existing_count += 1
            continue

        try:
            text = entry[config["audio_field_name"]]
            text = _clean_text_for_audio(text)

            tts = gTTS(text=text, lang=lang, slow=False)
            tts.save(str(audio_path))
            generated_count += 1
            print(f"[{i}/{total_entries}] Generated: {entry['audio_filename']}")

            if delay_seconds > 0 and i < total_entries:
                time.sleep(delay_seconds)
        except Exception as e:
            print(f"Error generating {entry['audio_filename']}: {e}")
            if delay_seconds > 0:
                time.sleep(delay_seconds)

    if existing_count > 0 or generated_count > 0:
        print(
            f"\nAudio generation complete: {generated_count} generated, "
            f"{existing_count} already existed"
        )


def _content_field_order(csv_fields: list[str], audio_field_name: str) -> list[str]:
    """Order fields: target language, transliteration (if any), english, then rest.

    :param csv_fields: All CSV column names.
    :param audio_field_name: Name of the audio/target language field.
    :returns: Ordered list of field names for card content.
    """
    target = next(
        (f for f in csv_fields if f.lower() == audio_field_name.lower()),
        audio_field_name,
    )
    rest = [f for f in csv_fields if f != target]
    result = [target]
    for name in ("transliteration", "english"):
        found = next((f for f in rest if f.lower() == name), None)
        if found:
            result.append(found)
    for f in rest:
        if f not in result:
            result.append(f)
    return result


def _build_apkg(language: str, prune: bool = True) -> None:
    """Build APKG file with vocabulary and audio for the given language.

    :param language: Language name (e.g. mandarin, spanish).
    :param prune: Delete audio files that the current list does not use.
    """
    lang_lower = language.lower()
    lang_title = language.title()
    print(f"Building {lang_title} vocabulary APKG...")

    config = {
        "language": language,
        "vocab_file": f"vocab/english-{lang_lower}.csv",
        "audio_dir": f"audio/audio_{lang_lower}",
        "deck_name": f"Swadesh {lang_title}",
        "model_name": f"Swadesh {lang_title}",
        "output_file": f"swadesh_{lang_lower}.apkg",
        "audio_field_name": lang_lower,
    }

    entries = _parse_vocab_entries(config)
    config["deck_name"] = f"{config['deck_name']} {len(entries)}"

    _generate_audio_files(entries, config)

    audio_dir = Path(config["audio_dir"])
    if not audio_dir.is_absolute():
        audio_dir = Path(__file__).parent.parent / config["audio_dir"]
    if not audio_dir.exists():
        print(f"Error: Audio directory not found: {audio_dir}")
        return

    keep = {entry["audio_filename"] for entry in entries}
    if prune:
        removed = _prune_outdated_audio(audio_dir, keep)
        if removed:
            print(f"Removed {len(removed)} outdated audio files")

    # Load CSS from templates/card.css if it exists
    css_path = Path(__file__).parent.parent / "templates" / "card.css"
    css_content = None
    if css_path.exists():
        css_content = css_path.read_text(encoding="utf-8")

    csv_fields = (
        [k for k in entries[0].keys() if k != "audio_filename"] if entries else []
    )
    content_order = _content_field_order(csv_fields, config["audio_field_name"])
    fields = content_order + ["audio"]

    char_click_script = dedent(r"""
    <script>
    function expand(elemId, url) {
        var elem = document.getElementById(elemId);
        if (elem) {
        var isCJK = ch => /[\u4e00-\u9fff]/.test(ch);
        elem.innerHTML = elem.innerHTML
            .split("")
            .filter(ch => !/\s/.test(ch))
            .map(ch => isCJK(ch) ? `<a href="${url}${ch}">${ch}</a>` : ch)
            .join("");
        }
    }
    expand('chars', `https://cantonese.org/search.php?q=`)
    </script>
    """)

    cjk_fields = {"characters", "hanzi", "cantonese", "mandarin", "chinese"}
    script_15em_fields = {"arabic", "hindi"}
    answer_parts = ["{{audio}}", "<hr id=answer>"]
    for field in content_order:
        if field.lower() in cjk_fields:
            answer_parts.append(
                f"<div id='chars' style='font-size: 3rem'>{{{{{field}}}}}</div>"
            )
        elif field.lower() in script_15em_fields:
            answer_parts.append(f"<div style='font-size: 1.5em'>{{{{{field}}}}}</div>")
        else:
            answer_parts.append(f"<div style='font-size: 2rem'>{{{{{field}}}}}</div>")
    answer_parts.append(char_click_script)
    answer_format = "\n".join(answer_parts)

    cards = []
    media_files = []
    missing: list[str] = []

    for entry in entries:
        audio_path = audio_dir / entry["audio_filename"]

        if not audio_path.exists():
            missing.append(entry.get("english", entry["audio_filename"]))
            print(f"Warning: Audio file not found: {audio_path}")
            continue

        card = {"audio": f"[sound:{entry['audio_filename']}]"}
        for field in content_order:
            card[field] = entry.get(field, "")
        cards.append(card)

        media_files.append(str(audio_path.absolute()))

    if missing:
        print(f"Error: {len(missing)} forms have no audio: " + ", ".join(missing))
        return

    apkg_dir = Path("apkg")
    apkg_dir.mkdir(exist_ok=True)

    output_path = apkg_dir / config["output_file"]

    print(f"Creating APKG: {output_path}")
    print(f"  Deck: {config['deck_name']}")
    print(f"  Cards: {len(cards)}")
    print(f"  Audio files: {len(media_files)}")
    if css_content:
        print("  CSS: templates/card.css")

    tmp_path = output_path.with_name(output_path.name + ".tmp")
    AnkiPackage.create(
        tmp_path,
        config["deck_name"],
        fields=fields,
        cards=cards,
        media_files=media_files,
        model_name=config["model_name"],
        question_format="{{audio}}",
        answer_format=answer_format,
        css=css_content,
        sort_field_index=0,
    )
    tmp_path.replace(output_path)

    print(f"\nSuccessfully created: {output_path}")


SWADESH_LANGUAGES = (
    "mandarin",
    "spanish",
    "greek",
    "german",
    "cantonese",
    "arabic",
    "hindi",
)

swadesh_app = cyclopts.App(
    name="swadesh", help="Build Swadesh vocabulary APKG files with audio"
)
app.command(swadesh_app)


@swadesh_app.command
def build(language: str, prune: bool = True):
    """Build Swadesh vocabulary APKG for a language.

    :param language: One of the supported languages, or all.
    :param prune: Delete audio files that the current list does not use.
    """
    lang = language.lower().strip()
    if lang == "all":
        for name in SWADESH_LANGUAGES:
            _build_apkg(name, prune=prune)
        return

    if lang not in SWADESH_LANGUAGES:
        print(
            f"Unknown language: {language}. Supported: {', '.join(SWADESH_LANGUAGES)}"
        )
        return
    _build_apkg(lang, prune=prune)


@swadesh_app.command
def fetch(*, refresh: bool = False) -> None:
    """Fetch the Tatoeba exports for the Mandarin sentence pack.

    :param refresh: Ignore the cache and download every file again.
    """
    results = fetch_tatoeba(refresh=refresh)
    for name, info in results.items():
        print(f"{name}.tsv: {info['rows']} rows ({info['source']})")

@swadesh_app.command
def sentences(
    language: str,
    per_word: int = 2,
    max_cards: int = 450,
    max_unknown: int = 2,
    min_han: int = 4,
    max_han: int = 10,
) -> None:
    """Build the Mandarin sentence candidates and write the card CSV files.

    :param language: Language name. Only mandarin is available.
    :param per_word: Card limit per word form.
    :param max_cards: Hard card cap.
    :param max_unknown: Unknown token budget per sentence.
    :param min_han: Minimum Han character count for a core card.
    :param max_han: Maximum Han character count.
    """
    lang = language.lower().strip()
    if lang != "mandarin":
        print(f"Unsupported language: {language}. Only mandarin is available.")
        return

    table = load_word_table()
    failures = check_word_table(table)
    if failures:
        for failure in failures:
            print(f"Error: {failure}")
        return

    pool, pool_report = build_candidate_pool(
        table,
        min_han=min_han,
        max_han=max_han,
        max_unknown=max_unknown,
        fallback=True,
    )
    cards, selection = select_sentences(
        pool, table, per_word=per_word, max_cards=max_cards
    )
    report = build_pack_report(pool_report, selection)

    sentence_path = write_sentence_csv(cards)
    review_path = write_sentence_review_csv(pool, table, per_word=per_word)
    report_path = write_pool_report(report)

    print_sentence_report(report)
    print(f"wrote {sentence_path}")
    print(f"wrote {review_path}")
    print(f"wrote {report_path}")


@swadesh_app.command(name="audio")
def sentence_audio(
    language: str,
    *,
    native_audio: str = "none",
    voice: str = MANDARIN_DEFAULT_VOICE,
    slow_audio: bool = False,
    pack_license: str = "public",
) -> None:
    """Generate and validate the audio files for the Mandarin sentence pack.

    :param language: Language name. Only mandarin is available.
    :param native_audio: Tatoeba tier: none, auto, or only.
    :param voice: edge-tts voice name.
    :param slow_audio: Generate the slow variant.
    :param pack_license: public or private.
    """
    lang = language.lower().strip()
    if lang != "mandarin":
        print(f"Unsupported language: {language}. Only mandarin is available.")
        return

    rows = load_sentence_rows()
    result = generate_audio(
        rows,
        voice=voice,
        native_audio=native_audio,
        slow_audio=slow_audio,
        pack_license=pack_license,
    )
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


@swadesh_app.command
def pack(
    language: str,
    *,
    pack_license: str = "public",
) -> None:
    """Build the Mandarin Swadesh sentence pack.

    :param language: Language name. Only mandarin is available.
    :param pack_license: public or private.
    """
    lang = language.lower().strip()
    if lang != "mandarin":
        print(f"Unsupported language: {language}. Only mandarin is available.")
        return

    result = build_pack(pack_license=pack_license)
    counts = result["manifest"]["counts"]
    print(
        f"{counts['cards']} cards, {counts['scaffold']} scaffold, "
        f"{counts['tts_audio']} tts, {counts['native_audio']} native, "
        f"{result['media_count']} media files, {result['skipped']} skipped"
    )
    print(f"wrote {result['output']}")
    print(f"wrote {result['manifest_path']}")
    print(f"wrote {result['attribution_path']}")


@swadesh_app.command
def audit(language: str) -> None:
    """Run the quality gates G1 to G16 on the Mandarin sentence pack.

    :param language: Language name. Only mandarin is available.
    """
    lang = language.lower().strip()
    if lang != "mandarin":
        print(f"Unsupported language: {language}. Only mandarin is available.")
        return

    code = print_audit(run_audit())
    if code:
        raise SystemExit(code)


@swadesh_app.command
def list_languages():
    """List supported Swadesh languages."""
    print("Supported languages:", ", ".join(SWADESH_LANGUAGES))


def main() -> None:
    """Main entry point. Invokes the cyclopts app."""
    app()


if __name__ == "__main__":
    main()
