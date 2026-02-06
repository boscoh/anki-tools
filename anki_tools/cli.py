"""
CLI tools for Anki package (.apkg) files.

Commands:
    inspect - Diagnostic tools to inspect .apkg files
    swadesh - Build Swadesh vocabulary APKG files with audio
    process - Process Chinese Anki decks (rank, reorder, fix pinyin)
"""

import csv
import re
import tempfile
import time
from pathlib import Path

import cyclopts
from gtts import gTTS

from anki_tools.package import AnkiPackage
from anki_tools.rank import (
    Sentence,
    extract_sentences,
    load_frequency_data,
    rank_sentences,
    export_csv,
)
from anki_tools.reorder import reorder_deck, load_ranking
from anki_tools.pinyin import fix_pinyin, check_pinyin

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
    """
    List cards in the deck.

    Args:
        apkg_path: Path to .apkg file
        limit: Maximum cards to show (0 for all)
        field: Only show this field name
        search: Filter cards containing this text
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
    """
    List all decks in the package.

    Args:
        apkg_path: Path to .apkg file
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
    """
    List note types (models) and their fields.

    Args:
        apkg_path: Path to .apkg file
        verbose: Show all model details
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
def fields(apkg_path: Path, model_name: str | None = None):
    """
    List all field names across models.

    Args:
        apkg_path: Path to .apkg file
        model_name: Filter by model name
    """
    with AnkiPackage(str(apkg_path)) as pkg:
        models = pkg.get_models()

        for model_id, model_info in models.items():
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
    """
    Show audio/media statistics and optionally extract files.

    Args:
        apkg_path: Path to .apkg file
        extract: Directory to extract audio files to
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
    """
    Show overall statistics for the package.

    Args:
        apkg_path: Path to .apkg file
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
    """
    Show sample field values from cards.

    Args:
        apkg_path: Path to .apkg file
        count: Number of sample values per field
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
# Swadesh commands - build vocabulary APKG files with audio
# =============================================================================

swadesh_app = cyclopts.App(name="swadesh", help="Build Swadesh vocabulary APKG files with audio")
app.command(swadesh_app)


def _clean_text_for_audio(text: str) -> str:
    """Remove parenthetical notes and variants from text for audio processing."""
    text = re.sub(r"\s*\([^)]*\)", "", text)
    return text.split("/")[0].strip()


def _parse_csv_entries(csv_file: str):
    """Parse CSV file and return entries as dictionaries."""
    entries = []
    csv_path = Path(csv_file)
    if not csv_path.is_absolute():
        csv_path = Path(__file__).parent.parent / csv_file
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned_row = {k: v.strip() if v else v for k, v in row.items()}
            entries.append(cleaned_row)
    return entries


def _parse_vocab_entries(config):
    """Parse vocabulary CSV file to extract entries."""
    entries = _parse_csv_entries(config["vocab_file"])

    for entry in entries:
        text = entry[config["audio_field_name"]]
        cleaned_text = _clean_text_for_audio(text).replace(" ", "_")
        entry["audio_filename"] = f"{cleaned_text}.mp3"

    return entries


def _generate_audio_files(entries, config):
    """Generate audio files for entries using TTS if they don't exist."""
    audio_dir = Path(config["audio_dir"])
    if not audio_dir.is_absolute():
        audio_dir = Path(__file__).parent.parent / audio_dir
    audio_dir.mkdir(parents=True, exist_ok=True)

    generate_audio = config.get("generate_audio")
    if not generate_audio:
        return

    delay_seconds = generate_audio.get("delay_seconds", 0)
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
            lang = generate_audio["lang"]
            slow = generate_audio.get("slow", False)

            tts = gTTS(text=text, lang=lang, slow=slow)
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


def _get_csv_fields(entries):
    """Get all CSV column names except the generated audio_filename."""
    if not entries:
        return []
    return [key for key in entries[0].keys() if key != "audio_filename"]


def _build_apkg(config):
    """Build APKG file with vocabulary and audio based on config."""
    entries = _parse_vocab_entries(config)

    _generate_audio_files(entries, config)

    audio_dir = Path(config["audio_dir"])
    if not audio_dir.is_absolute():
        audio_dir = Path(__file__).parent.parent / audio_dir
    if not audio_dir.exists():
        print(f"Error: Audio directory not found: {audio_dir}")
        return

    csv_fields = _get_csv_fields(entries)
    fields = ["audio"] + csv_fields

    answer_parts = ["{{audio}}", "<hr id=answer>"]
    for field in csv_fields[1:] + [csv_fields[0]]:
        answer_parts.append(f"<div style='font-size: 2rem'>{{{{{field}}}}}</div>")
    answer_format = "\n".join(answer_parts)

    cards = []
    media_files = []

    for entry in entries:
        audio_path = audio_dir / entry["audio_filename"]

        if not audio_path.exists():
            print(f"Warning: Audio file not found: {audio_path}")
            continue

        card = {"audio": f"[sound:{entry['audio_filename']}]"}
        for field in csv_fields:
            card[field] = entry.get(field, "")
        cards.append(card)

        media_files.append(str(audio_path.absolute()))

    if not media_files:
        print("Error: No audio files found. Check audio files exist.")
        return

    apkg_dir = Path("apkg")
    apkg_dir.mkdir(exist_ok=True)

    output_path = apkg_dir / config["output_file"]

    print(f"Creating APKG: {output_path}")
    print(f"  Deck: {config['deck_name']}")
    print(f"  Cards: {len(cards)}")
    print(f"  Audio files: {len(media_files)}")

    AnkiPackage.create(
        output_path,
        config["deck_name"],
        fields=fields,
        cards=cards,
        media_files=media_files,
        model_name=config["model_name"],
        question_format="{{audio}}",
        answer_format=answer_format,
    )

    print(f"\nSuccessfully created: {output_path}")


@swadesh_app.command
def mandarin():
    """Build Mandarin vocabulary APKG."""
    print("Building Mandarin vocabulary APKG...")
    config = {
        "vocab_file": "vocab/english-mandarin.csv",
        "audio_dir": "vocab/audio_mandarin",
        "deck_name": "Mandarin Swadesh 207",
        "model_name": "Mandarin Audio",
        "output_file": "mandarin_swadesh.apkg",
        "audio_field_name": "characters",
        "generate_audio": {
            "lang": "zh-CN",
            "slow": False,
            "delay_seconds": 0.5,
        },
    }
    _build_apkg(config)


@swadesh_app.command
def spanish():
    """Build Spanish vocabulary APKG."""
    print("Building Spanish vocabulary APKG...")
    config = {
        "vocab_file": "vocab/english-spanish.csv",
        "audio_dir": "vocab/audio_spanish",
        "deck_name": "Spanish Swadesh 207",
        "model_name": "Spanish Audio",
        "output_file": "spanish_swadesh.apkg",
        "audio_field_name": "spanish",
        "generate_audio": {
            "lang": "es",
            "slow": False,
            "delay_seconds": 0.5,
        },
    }
    _build_apkg(config)


@swadesh_app.command
def greek():
    """Build Greek vocabulary APKG."""
    print("Building Greek vocabulary APKG...")
    config = {
        "vocab_file": "vocab/english-greek.csv",
        "audio_dir": "vocab/audio_greek",
        "deck_name": "Greek Swadesh 207",
        "model_name": "Greek Audio",
        "output_file": "greek_swadesh.apkg",
        "audio_field_name": "greek",
        "generate_audio": {
            "lang": "el",
            "slow": False,
            "delay_seconds": 0.5,
        },
    }
    _build_apkg(config)


@swadesh_app.command
def german():
    """Build German vocabulary APKG."""
    print("Building German vocabulary APKG...")
    config = {
        "vocab_file": "vocab/english-german.csv",
        "audio_dir": "vocab/audio_german",
        "deck_name": "German Swadesh 207",
        "model_name": "German Audio",
        "output_file": "german_swadesh.apkg",
        "audio_field_name": "german",
        "generate_audio": {
            "lang": "de",
            "slow": False,
            "delay_seconds": 0.5,
        },
    }
    _build_apkg(config)


@swadesh_app.command
def cantonese():
    """Build Cantonese vocabulary APKG."""
    print("Building Cantonese vocabulary APKG...")
    config = {
        "vocab_file": "vocab/english-cantonese.csv",
        "audio_dir": "vocab/audio_cantonese",
        "deck_name": "Cantonese Swadesh 207",
        "model_name": "Cantonese Audio",
        "output_file": "cantonese_swadesh.apkg",
        "audio_field_name": "cantonese",
        "generate_audio": {
            "lang": "yue",
            "slow": False,
            "delay_seconds": 0.5,
        },
    }
    _build_apkg(config)


# =============================================================================
# Process commands - Chinese deck processing pipeline
# =============================================================================

process_app = cyclopts.App(name="process", help="Process Chinese Anki decks")
app.command(process_app)


@process_app.command
def rank(
    apkg_path: Path,
    *,
    output: Path | None = None,
    model_id: int | None = None,
    vocab_dir: str = "vocab",
):
    """
    Rank sentences by frequency and complexity.

    Outputs a CSV with sentences ranked for optimal learning order.

    Args:
        apkg_path: Input .apkg file
        output: Output CSV file (default: ranked_sentences.csv)
        model_id: Filter by specific model ID
        vocab_dir: Directory containing frequency data
    """
    if output is None:
        output = Path("ranked_sentences.csv")

    print(f"Ranking sentences from: {apkg_path}")

    sentences = extract_sentences(str(apkg_path), model_id)
    if not sentences:
        print("No sentences found!")
        return

    freq_data = load_frequency_data(vocab_dir)
    ranked = rank_sentences(sentences, freq_data)
    export_csv(ranked, str(output))

    print(f"\nRanked {len(ranked)} sentences -> {output}")


@process_app.command
def reorder(
    apkg_path: Path,
    ranking_csv: Path,
    *,
    output: Path | None = None,
    keep_filtered: bool = False,
):
    """
    Reorder deck based on ranking CSV.

    Updates card order and optionally removes filtered cards (names, invalid).

    Args:
        apkg_path: Input .apkg file
        ranking_csv: Ranking CSV from 'rank' command
        output: Output .apkg file (default: input_reordered.apkg)
        keep_filtered: Keep filtered cards instead of removing
    """
    if output is None:
        output = apkg_path.parent / f"{apkg_path.stem}_reordered.apkg"

    print(f"Reordering deck: {apkg_path}")
    print(f"Using ranking: {ranking_csv}")

    stats = reorder_deck(
        str(apkg_path),
        str(output),
        str(ranking_csv),
        remove_filtered=not keep_filtered,
    )

    print(f"\nReordered {stats['cards_reordered']} cards")
    print(f"Removed {stats['cards_removed']} filtered cards")
    print(f"Output: {output}")


@process_app.command
def fix(
    apkg_path: Path,
    *,
    output: Path | None = None,
    verbose: bool = False,
):
    """
    Fix pinyin formatting in deck.

    Corrects: lowercase, syllable separation, tone marks, common errors.

    Args:
        apkg_path: Input .apkg file
        output: Output .apkg file (default: input_fixed.apkg)
        verbose: Show all corrections
    """
    if output is None:
        output = apkg_path.parent / f"{apkg_path.stem}_fixed.apkg"

    print(f"Fixing pinyin in: {apkg_path}")

    fix_pinyin(
        apkg_path,
        output=output,
        csv_output=None,
        verbose=verbose,
    )

    print(f"\nOutput: {output}")


def _export_summary_csv(
    ranked: list[Sentence],
    remove_ranks: set[int],
    output_path: str,
    fix_pinyin_enabled: bool = True,
) -> int:
    """Export summary CSV with all changes."""
    pinyin_corrections = 0
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank",
            "sentence",
            "pinyin",
            "pinyin_corrected",
            "pinyin_reason",
            "english",
            "original_order",
            "complexity",
            "frequency",
            "similarity_penalty",
            "similar_to",
            "final_score",
            "removed",
        ])
        
        for i, sent in enumerate(ranked, 1):
            removed = "yes" if i in remove_ranks else ""
            
            pinyin_corrected = ""
            pinyin_reason = ""
            if fix_pinyin_enabled and not removed:
                correction, reason = check_pinyin(sent.chinese, sent.pinyin)
                if correction:
                    pinyin_corrected = correction
                    pinyin_reason = reason
                    pinyin_corrections += 1
            
            writer.writerow([
                i,
                sent.chinese,
                sent.pinyin,
                pinyin_corrected,
                pinyin_reason,
                sent.english,
                sent.original_order,
                f"{sent.complexity_score:.1f}",
                f"{sent.frequency_score:.1f}",
                f"{sent.similarity_penalty:.1f}",
                sent.similar_to if sent.similar_to else "",
                f"{sent.final_score:.2f}",
                removed,
            ])
    
    return pinyin_corrections


@process_app.command(name="all")
def process_all(
    apkg_path: Path,
    *,
    output: Path | None = None,
    summary_csv: Path | None = None,
    model_id: int | None = None,
    keep_filtered: bool = False,
    no_pinyin_fix: bool = False,
    vocab_dir: str = "vocab",
    verbose: bool = False,
):
    """
    Run complete pipeline: rank + reorder + fix pinyin.

    This is the main command for processing a Chinese Anki deck.

    Args:
        apkg_path: Input .apkg file
        output: Output .apkg file (default: input_processed.apkg)
        summary_csv: Output summary CSV (default: input_summary.csv)
        model_id: Filter by specific model ID
        keep_filtered: Keep filtered cards (names, invalid)
        no_pinyin_fix: Skip pinyin correction
        vocab_dir: Directory with frequency data
        verbose: Show detailed output
    """
    if output is None:
        output = apkg_path.parent / f"{apkg_path.stem}_processed.apkg"
    if summary_csv is None:
        summary_csv = apkg_path.parent / f"{apkg_path.stem}_summary.csv"

    print("=" * 50)
    print("Processing Chinese Anki Deck")
    print("=" * 50)
    print(f"Input:   {apkg_path}")
    print(f"Output:  {output}")
    print(f"Summary: {summary_csv}")
    print()

    print("Step 1: Ranking sentences...")
    sentences = extract_sentences(str(apkg_path), model_id)

    if not sentences:
        print("No sentences found!")
        return

    freq_data = load_frequency_data(vocab_dir)
    ranked = rank_sentences(sentences, freq_data)

    ranking_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ).name
    export_csv(ranked, ranking_file)

    _, remove_ranks = load_ranking(ranking_file)

    print("\nStep 2: Reordering deck...")
    stats = reorder_deck(
        str(apkg_path),
        str(output),
        ranking_file,
        remove_filtered=not keep_filtered,
    )

    if not no_pinyin_fix:
        print("\nStep 3: Fixing pinyin...")
        fix_pinyin(
            output,
            output=output,
            csv_output=Path("/dev/null"),
            verbose=verbose,
        )

    print("\nStep 4: Exporting summary...")
    pinyin_corrections = _export_summary_csv(
        ranked,
        remove_ranks if not keep_filtered else set(),
        str(summary_csv),
        fix_pinyin_enabled=not no_pinyin_fix,
    )

    print()
    print("=" * 50)
    print("DONE!")
    print("=" * 50)
    print(f"  Sentences ranked:    {len(ranked)}")
    print(f"  Cards reordered:     {stats['cards_reordered']}")
    print(f"  Cards removed:       {stats['cards_removed']}")
    print(f"  Pinyin corrections:  {pinyin_corrections}")
    print(f"  Output:  {output}")
    print(f"  Summary: {summary_csv}")


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
