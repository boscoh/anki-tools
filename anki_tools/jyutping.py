"""Fix jyutping in Cantonese Anki decks: convert to traditional and regenerate."""

import re
from pathlib import Path

import opencc
import ToJyutping

from anki_tools.package import AnkiPackage


def simplified_to_traditional(text: str) -> str:
    """Convert simplified Chinese to traditional Chinese.

    :param text: Text with simplified Chinese characters.
    :returns: Text with traditional Chinese characters.
    """
    try:
        converter = opencc.OpenCC('s2t')
        return converter.convert(text)
    except Exception:
        return text


def get_jyutping(text: str) -> str:
    """Generate jyutping romanization for Cantonese text.

    :param text: Cantonese text (traditional Chinese).
    :returns: Space-separated jyutping with tone numbers.
    """
    try:
        clean_text = re.sub(r'[！？。，、：；"""（）]', "", text)
        if not clean_text:
            return ""
        result = ToJyutping.get_jyutping_text(clean_text)
        return result
    except Exception:
        return ""


def fix_jyutping(
    apkg_path: Path,
    *,
    output: Path | None = None,
    verbose: bool = False,
):
    """Fix jyutping in a Cantonese Anki deck.

    Converts text to traditional Chinese and regenerates jyutping romanization.

    :param apkg_path: Input .apkg file to process.
    :param output: Output .apkg file (default: overwrites input).
    :param verbose: If True, print all corrections.
    """
    apkg_path = Path(apkg_path)
    if not apkg_path.exists():
        raise FileNotFoundError(f"File not found: {apkg_path}")

    if output is None:
        output = apkg_path

    with AnkiPackage(str(apkg_path)) as pkg:
        cards = pkg.get_cards()
        models = pkg.get_models()
        decks = pkg.get_decks()

        print(f"Reading: {apkg_path}")
        print(f"Found {len(cards)} cards")

        corrections = {}

        for i, card in enumerate(cards):
            parsed = pkg.parse_card(card, models, decks)
            fields = parsed.get("fields", {})

            cantonese = (
                fields.get("Front")
                or fields.get("Cantonese")
                or fields.get("Traditional")
                or fields.get("Chinese")
                or ""
            )
            jyutping_val = (
                fields.get("Jyutping")
                or fields.get("jyutping")
                or fields.get("Romanization")
                or ""
            )

            if not cantonese and not jyutping_val:
                field_values = list(fields.values())
                cantonese = field_values[0] if len(field_values) > 0 else ""

            cantonese = re.sub(r"<[^>]+>", "", str(cantonese)).strip()
            jyutping_val = re.sub(r"<[^>]+>", "", str(jyutping_val)).strip()

            traditional = simplified_to_traditional(cantonese)
            new_jyutping = get_jyutping(traditional)

            if new_jyutping and new_jyutping != jyutping_val:
                corrections[i] = {
                    "card": card,
                    "cantonese": cantonese,
                    "traditional": traditional,
                    "old_jyutping": jyutping_val,
                    "new_jyutping": new_jyutping,
                }

        if corrections:
            print(f"\nApplying {len(corrections)} corrections...")
            updated = 0

            for i, corr in corrections.items():
                card = corr["card"]
                nid = card["nid"]

                model_id = str(card["mid"])
                model = models.get(model_id, {})
                field_names = [f["name"] for f in model.get("flds", [])]

                jyutping_idx = None
                for idx, name in enumerate(field_names):
                    if name.lower() in ["jyutping", "romanization"]:
                        jyutping_idx = idx
                        break

                if jyutping_idx is None:
                    continue

                pkg.update_note_field(nid, jyutping_idx, corr["new_jyutping"])
                updated += 1

            pkg.save(str(output))
            print(f"Updated {updated} notes")
            if output != apkg_path:
                print(f"Saved to: {output}")

        print(f"Found {len(corrections)} cards needing correction")

        if verbose and corrections:
            print("\nCorrections applied:")
            for i, corr in corrections.items():
                print(
                    f"  {i}: {corr['traditional']:<15} '{corr['old_jyutping']}' -> '{corr['new_jyutping']}'"
                )
