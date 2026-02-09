"""Fix pinyin in Chinese Anki decks: lowercase, syllable-separated, correct tones."""

import csv
import re
from pathlib import Path

from pypinyin import Style, pinyin

from anki_tools.package import AnkiPackage

POLYPHONIC_SKIP = {
    "音樂",
    "銀行",
    "乾淨",
    "睡覺",
    "和",
    "長",
    "行",
    "樂",
    "發",
    "數",
    "教",
    "覺",
    "著",
    "了",
    "地",
    "的",
}

NEUTRAL_TONE_PATTERNS = [
    "明白",
    "喜欢",
    "这里",
    "那里",
    "哪里",
    "回来",
    "起来",
    "出来",
    "进来",
    "先生",
    "漂亮",
    "便宜",
    "清楚",
    "告诉",
    "知道",
    "认识",
    "觉得",
    "意思",
    "东西",
    "时候",
    "朋友",
    "客气",
    "舒服",
    "麻烦",
    "厉害",
    "暖和",
    "小姐",
    "謝謝",
    "記得",
    "爸爸",
    "媽媽",
    "姐姐",
    "妹妹",
    "哥哥",
    "弟弟",
    "太太",
    "奶奶",
    "爷爷",
    "姑姑",
    "叔叔",
    "阿姨",
]

PREFERRED_PRONUNCIATION = {
    "shuí": "shéi",
}


def get_pypinyin(hanzi: str) -> str:
    """Get pinyin from pypinyin library (syllable per character).

    :param hanzi: Chinese text.
    :returns: Space-separated pinyin with tone marks.
    """
    clean = re.sub(r"[（(].*?[）)]", "", hanzi)
    clean = re.sub(r'[！？。，、：；""' "]", "", clean)
    clean = clean.split("/")[0].strip()
    if not clean:
        return ""
    result = pinyin(clean, style=Style.TONE)
    return " ".join([p[0] for p in result])


def normalize_pinyin(p: str) -> str:
    """Normalize pinyin: lowercase, remove punctuation, single spaces.

    :param p: Raw pinyin string.
    :returns: Normalized pinyin.
    """
    p = p.lower().strip()
    p = re.sub(r"[^a-zāáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ\s]", "", p)
    p = re.sub(r"\s+", " ", p).strip()
    return p


def normalize_no_space(p: str) -> str:
    """Normalize pinyin without spaces for syllable comparison.

    :param p: Raw pinyin string.
    :returns: Pinyin with spaces removed.
    """
    p = normalize_pinyin(p)
    return re.sub(r"\s+", "", p)


def remove_tones(p: str) -> str:
    """Remove tone marks for base comparison.

    :param p: Pinyin with tone marks.
    :returns: Pinyin with tone marks replaced by base vowels.
    """
    p = re.sub(r"[āáǎà]", "a", p.lower())
    p = re.sub(r"[ēéěè]", "e", p)
    p = re.sub(r"[īíǐì]", "i", p)
    p = re.sub(r"[ōóǒò]", "o", p)
    p = re.sub(r"[ūúǔù]", "u", p)
    p = re.sub(r"[ǖǘǚǜ]", "v", p)
    return p


def is_neutral_tone_word(hanzi: str) -> bool:
    """Check if word has acceptable neutral tone variation.

    :param hanzi: Chinese text.
    :returns: True if word matches a known neutral-tone pattern.
    """
    for pattern in NEUTRAL_TONE_PATTERNS:
        if pattern in hanzi:
            return True
    return False


def format_pinyin(csv_pinyin: str) -> str:
    """Format pinyin: lowercase, remove punctuation.

    :param csv_pinyin: Raw pinyin from CSV.
    :returns: Formatted pinyin.
    """
    p = csv_pinyin.lower().strip()
    p = re.sub(r"[^a-zāáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ\s]", "", p)
    p = re.sub(r"\s+", " ", p).strip()
    return p


def needs_formatting(csv_pinyin: str) -> bool:
    """Check if pinyin needs lowercase or punctuation removal.

    :param csv_pinyin: Pinyin string to check.
    :returns: True if formatting is needed.
    """
    csv_letters = re.sub(
        r"[^a-zA-ZāáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜĀÁǍÀĒÉĚÈĪÍǏÌŌÓǑÒŪÚǓÙǕǗǙǛ\s]", "", csv_pinyin
    )
    return csv_letters != csv_letters.lower()


def check_pinyin(hanzi: str, csv_pinyin: str) -> tuple[str, str]:
    """Check pinyin and return (correction, reason) if needed.

    :param hanzi: Chinese text.
    :param csv_pinyin: Pinyin from card/CSV.
    :returns: Tuple of (corrected_pinyin, reason). Empty strings if no change.
    """
    if not hanzi or not csv_pinyin:
        return "", ""

    if "-->" in csv_pinyin:
        correct = get_pypinyin(hanzi)
        return correct, "corrupted: regenerated from pypinyin"

    fixed = csv_pinyin.lower()
    for old, new in PREFERRED_PRONUNCIATION.items():
        if old in fixed:
            fixed = fixed.replace(old, new)
            formatted = format_pinyin(fixed)
            return formatted, f"preferred pronunciation: {old} -> {new}"

    if re.search(r"[jqxln][īíǐì]u", csv_pinyin.lower()):
        fixed = csv_pinyin
        for consonant in ["j", "q", "x", "n", "l"]:
            fixed = re.sub(f"{consonant}īu", f"{consonant}iū", fixed)
            fixed = re.sub(f"{consonant}íu", f"{consonant}iú", fixed)
            fixed = re.sub(f"{consonant}ǐu", f"{consonant}iǔ", fixed)
            fixed = re.sub(f"{consonant}ìu", f"{consonant}iù", fixed)
        if fixed != csv_pinyin:
            return format_pinyin(fixed), "iu typo: tone mark should be on u not i"

    expected = get_pypinyin(hanzi)
    csv_norm = normalize_no_space(csv_pinyin)
    exp_norm = normalize_no_space(expected)
    csv_base = remove_tones(csv_norm)
    exp_base = remove_tones(exp_norm)

    hanzi_clean = re.sub(r'[！？。，、：；""' "（） ]", "", hanzi)
    hanzi_chars = len([c for c in hanzi_clean if "\u4e00" <= c <= "\u9fff"])

    csv_clean = re.sub(
        r"[^a-zA-ZāáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜĀÁǍÀĒÉĚÈĪÍǏÌŌÓǑÒŪÚǓÙǕǗǙǛ\s]", "", csv_pinyin
    )
    words = csv_clean.split()
    csv_syllables = 0
    for word in words:
        csv_syllables += len(
            re.findall(
                r"[aeiouāáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜAEIOUĀÁǍÀĒÉĚÈĪÍǏÌŌÓǑÒŪÚǓÙǕǗǙǛ]+",
                word,
                re.I,
            )
        )

    if hanzi_chars >= 2 and csv_syllables > 0:
        if csv_syllables < hanzi_chars:
            return (
                expected,
                f"missing syllables: {csv_syllables} for {hanzi_chars} chars",
            )

    if "儿" in hanzi:
        if needs_formatting(csv_pinyin):
            return format_pinyin(csv_pinyin), "needs formatting"
        return "", ""

    skip_tone_check = (
        any(char in hanzi for char in POLYPHONIC_SKIP)
        or "不" in hanzi
        or "一" in hanzi
        or is_neutral_tone_word(hanzi)
        or hanzi.endswith(("阿", "呀", "嘛", "吧", "呢", "啊"))
        or "得" in hanzi
    )

    if skip_tone_check:
        formatted = format_pinyin(csv_pinyin)
        csv_words = formatted.split()
        if hanzi_chars >= 2 and len(csv_words) < hanzi_chars:
            return (
                expected,
                f"needs syllable separation: {len(csv_words)} words for {hanzi_chars} chars",
            )
        if needs_formatting(csv_pinyin):
            return formatted, "needs formatting"
        return "", ""

    if csv_base != exp_base and len(csv_base) > 2:
        if "number" in hanzi.lower():
            if needs_formatting(csv_pinyin):
                return format_pinyin(csv_pinyin), "needs formatting"
            return "", ""
        if len(set(csv_base) ^ set(exp_base)) > 3:
            return expected, f"pinyin mismatch: expected {expected}"

    formatted = format_pinyin(csv_pinyin)
    csv_words = formatted.split()
    if hanzi_chars >= 2 and len(csv_words) < hanzi_chars and csv_norm == exp_norm:
        return (
            expected,
            f"needs syllable separation: {len(csv_words)} words for {hanzi_chars} chars",
        )
    if needs_formatting(csv_pinyin):
        return formatted, "needs formatting"

    return "", ""


def fix_pinyin(
    apkg_path: Path,
    *,
    output: Path | None = None,
    csv_output: Path | None = None,
    verbose: bool = False,
):
    """Fix pinyin in a Chinese Anki deck.

    Converts pinyin to lowercase, separates syllables, and fixes common errors.

    :param apkg_path: Input .apkg file to process.
    :param output: Output .apkg file (default: input_fixed.apkg).
    :param csv_output: Output CSV report (default: input_cards.csv).
    :param verbose: If True, print all corrections.
    :raises FileNotFoundError: If apkg_path does not exist.
    """
    apkg_path = Path(apkg_path)
    if not apkg_path.exists():
        raise FileNotFoundError(f"File not found: {apkg_path}")

    if output is None:
        output = apkg_path.parent / f"{apkg_path.stem}_fixed.apkg"
    if csv_output is None:
        csv_output = apkg_path.parent / f"{apkg_path.stem}_cards.csv"

    with AnkiPackage(str(apkg_path)) as pkg:
        cards = pkg.get_cards()
        models = pkg.get_models()
        decks = pkg.get_decks()

        print(f"Reading: {apkg_path}")
        print(f"Found {len(cards)} cards")

        rows = []
        corrections = {}

        for i, card in enumerate(cards):
            parsed = pkg.parse_card(card, models, decks)
            fields = parsed.get("fields", {})

            chinese = (
                fields.get("Chinese")
                or fields.get("Hanzi")
                or fields.get("中文")
                or fields.get("Sentence")
                or ""
            )
            pinyin_val = (
                fields.get("Pinyin")
                or fields.get("拼音")
                or fields.get("Sentence (Latin)")
                or ""
            )
            meaning = (
                fields.get("Meaning")
                or fields.get("English")
                or fields.get("Sentence (Translation)")
                or ""
            )

            if not chinese and not pinyin_val:
                field_values = list(fields.values())
                chinese = field_values[0] if len(field_values) > 0 else ""
                pinyin_val = field_values[1] if len(field_values) > 1 else ""

            chinese = re.sub(r"<[^>]+>", "", str(chinese)).strip()
            pinyin_val = re.sub(r"<[^>]+>", "", str(pinyin_val)).strip()
            meaning = re.sub(r"<[^>]+>", "", str(meaning)).strip()

            correction, reason = check_pinyin(chinese, pinyin_val)
            if correction:
                corrections[i] = {
                    "card": card,
                    "hanzi": chinese,
                    "old_pinyin": pinyin_val,
                    "new_pinyin": correction,
                    "reason": reason,
                }

            rows.append(
                {
                    "index": i,
                    "hanzi": chinese,
                    "pinyin": pinyin_val,
                    "meaning": meaning,
                    "correction": correction,
                    "reason": reason,
                }
            )

        if corrections:
            print(f"\nApplying {len(corrections)} corrections...")
            updated = 0

            for i, corr in corrections.items():
                card = corr["card"]
                nid = card["nid"]

                model_id = str(card["mid"])
                model = models.get(model_id, {})
                field_names = [f["name"] for f in model.get("flds", [])]

                pinyin_idx = None
                for idx, name in enumerate(field_names):
                    if name.lower() in ["pinyin", "拼音", "sentence (latin)"]:
                        pinyin_idx = idx
                        break

                if pinyin_idx is None:
                    continue

                pkg.update_note_field(nid, pinyin_idx, corr["new_pinyin"])
                updated += 1

            pkg.save(str(output))
            print(f"Updated {updated} notes")
            print(f"Saved to: {output}")

    if csv_output and str(csv_output) != "/dev/null":
        with open(csv_output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "index",
                    "hanzi",
                    "pinyin",
                    "meaning",
                    "correction",
                    "reason",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nWrote {len(rows)} cards to {csv_output}")

    print(f"Found {len(corrections)} cards needing correction")

    if verbose and corrections:
        print("\nCorrections applied:")
        for i, corr in corrections.items():
            print(
                f"  {i}: {corr['hanzi']:<15} '{corr['old_pinyin']}' -> '{corr['new_pinyin']}'"
            )
            print(f"       Reason: {corr['reason']}")
