"""Cantonese word frequency list that blends spoken and written counts.

The spoken term comes from the Cifu lexicon (Lai and Winterstein 2020). The
written term is the mean of the Cifu written count and the Apple Daily count.
A variant map folds common spoken characters into the news-standard form, and
an override file fixes words that the corpora split or miss.

See ``specs/yue-sentence-selection.md`` sections 5 and 6.
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Protocol

import jieba

from anki_tools.ranking.rank import (
    FrequencyData,
    Sentence,
    _read_word_count_tsv,
    download_cached_file,
    download_yue_frequency,
    frequency_data_from_ranks,
    yue_frequency_path,
)

CIFU_URL = "https://raw.githubusercontent.com/gwinterstein/Cifu/master/Lexicon/Cifu-v1.txt"
CIFU_ATTRIBUTION = (
    "Cantonese spoken frequency: Cifu, Lai and Winterstein 2020 (gwinterstein/Cifu)"
)
VOCAB_DIR = Path(__file__).resolve().parent.parent.parent / "vocab"
VARIANTS_PATH = VOCAB_DIR / "yue_variants.tsv"
OVERRIDES_PATH = VOCAB_DIR / "yue_frequency_overrides.tsv"
FREQUENCY_PATH = Path("data/yue_frequency.tsv")

MIN_SPOKEN_COUNT = 5
SPOKEN_DICT_WEIGHT = 40
WRITTEN_DICT_WEIGHT = 2
FREQ_SOURCES = ("blend", "spoken", "written")

_HAN = re.compile(r"[\u4e00-\u9fff]")
_COLUMNS = (
    "word",
    "jyutping",
    "spoken_count",
    "spoken_pm",
    "written_pm",
    "blend_pm",
    "rank",
    "source",
)


@dataclass(frozen=True)
class CifuEntry:
    """One Cifu word.

    :param jyutping: Jyutping reading.
    :param spoken: Count in the spoken adult genre.
    :param written: Count in the written genre.
    """

    jyutping: str
    spoken: int
    written: int


@dataclass
class FreqRow:
    """One row of the frequency list.

    :param word: Word in the news-standard form.
    :param jyutping: Cifu reading, empty when the word is not in Cifu.
    :param spoken_count: Cifu spoken adult count.
    :param spoken_pm: Spoken count per million tokens.
    :param written_pm: Mean of the Cifu and Apple Daily written values.
    :param blend_pm: Value of the selected term (the blend by default).
    :param rank: Rank by ``blend_pm``, 1 is most frequent.
    :param source: ``cifu+ad``, ``cifu``, ``ad``, or ``override``.
    """

    word: str
    jyutping: str
    spoken_count: int
    spoken_pm: float
    written_pm: float
    blend_pm: float
    rank: int = 0
    source: str = ""


@dataclass
class YueFrequency:
    """A frequency list and its variant map.

    :param rows: Rows sorted by rank.
    :param variants: Variant character form to the standard form.
    """

    rows: list[FreqRow]
    variants: dict[str, str] = field(default_factory=dict)

    @cached_property
    def ranks(self) -> dict[str, int]:
        """Word to rank, with each variant mapped to its standard form."""
        ranks = {row.word: row.rank for row in self.rows}
        for variant, standard in self.variants.items():
            if standard in ranks:
                ranks.setdefault(variant, ranks[standard])
        return ranks


def cifu_path() -> Path:
    """Return the cache path of the Cifu lexicon.

    Set the environment variable ``CIFU_PATH`` to use another file.

    :returns: Path to the cached Cifu file.
    """
    override = os.environ.get("CIFU_PATH")
    if override:
        return Path(override)
    return yue_frequency_path().with_name("cifu-v1.tsv")


def download_cifu(path: Path | None = None, *, force: bool = False) -> Path:
    """Download the Cifu lexicon into the cache.

    :param path: Cache destination. Defaults to :func:`cifu_path`.
    :param force: Download again when the cached file already exists.
    :returns: Path to the cached file.
    """
    dest = Path(path) if path else cifu_path()
    try:
        return download_cached_file(CIFU_URL, dest, force=force)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Cannot download {CIFU_URL}. Set CIFU_PATH to a local copy of the file."
        ) from exc


def _count(value: str | None) -> int:
    text = (value or "").strip()
    return int(text) if text.isdigit() else 0


def read_cifu(path: Path) -> dict[str, CifuEntry]:
    """Read the Cifu lexicon.

    Empty count cells read as 0. A word that occurs twice gets the summed counts.

    :param path: Cifu TSV file with a header line.
    :returns: Word to :class:`CifuEntry`.
    """
    try:
        handle = open(path, encoding="utf-8", newline="")
    except OSError as exc:
        raise RuntimeError(
            f"Cannot read the Cifu file {path}. Set CIFU_PATH to a valid file."
        ) from exc
    entries: dict[str, CifuEntry] = {}
    with handle:
        reader = csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            word = (row.get("Word") or "").strip()
            if not word:
                continue
            spoken = _count(row.get("SpokenAdult"))
            written = _count(row.get("Written"))
            old = entries.get(word)
            if old:
                spoken += old.spoken
                written += old.written
            entries[word] = CifuEntry((row.get("JyutPing") or "").strip(), spoken, written)
    if not entries:
        raise ValueError(f"No words found in {path}")
    return entries


def _read_table(path: Path, columns: int) -> list[list[str]]:
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        parts = line.split("\t")
        if number == 0 or not line.strip() or len(parts) < columns:
            continue
        rows.append([part.strip() for part in parts])
    return rows


def read_variants(path: Path = VARIANTS_PATH) -> dict[str, str]:
    """Read the variant map.

    :param path: TSV file with columns variant and standard.
    :returns: Variant to standard form.
    """
    return {row[0]: row[1] for row in _read_table(path, 2)}


def read_overrides(path: Path = OVERRIDES_PATH) -> dict[str, str]:
    """Read the frequency overrides.

    :param path: TSV file with columns word, rank_like, and reason.
    :returns: Word to the word that gives its frequency.
    """
    return {row[0]: row[1] for row in _read_table(path, 2)}


def build_yue_frequency(
    cifu: dict[str, CifuEntry],
    apple: dict[str, int],
    *,
    variants: dict[str, str] | None = None,
    overrides: dict[str, str] | None = None,
    freq: str = "blend",
    spoken_weight: float = 0.7,
) -> YueFrequency:
    """Blend the Cifu and Apple Daily counts into one ranked list.

    ``blend_pm`` is ``w * spoken_pm + (1 - w) * written_pm`` when the spoken
    count is at least 5, else ``written_pm``. The ``spoken`` source uses
    ``spoken_pm`` alone, and the ``written`` source uses ``written_pm`` alone.

    :param cifu: Cifu entries.
    :param apple: Apple Daily word counts.
    :param variants: Variant to standard form. Counts of a variant add to the standard form.
    :param overrides: Word to the word that gives its value.
    :param freq: ``blend``, ``spoken``, or ``written``.
    :param spoken_weight: Value of ``w``.
    :returns: Rows sorted by rank.
    """
    if freq not in FREQ_SOURCES:
        raise ValueError(f"freq must be one of {FREQ_SOURCES}, got {freq!r}")
    variants = variants or {}
    overrides = overrides or {}
    spoken_total = sum(e.spoken for e in cifu.values()) or 1
    written_total = sum(e.written for e in cifu.values()) or 1
    apple_total = sum(apple.values()) or 1

    spoken: dict[str, int] = {}
    cifu_written: dict[str, int] = {}
    apple_count: dict[str, int] = {}
    jyutping: dict[str, str] = {}
    in_cifu: set[str] = set()
    for word, entry in cifu.items():
        key = variants.get(word, word)
        spoken[key] = spoken.get(key, 0) + entry.spoken
        cifu_written[key] = cifu_written.get(key, 0) + entry.written
        in_cifu.add(key)
        if entry.jyutping and (key == word or key not in jyutping):
            jyutping[key] = entry.jyutping
    for word, count in apple.items():
        key = variants.get(word, word)
        apple_count[key] = apple_count.get(key, 0) + count

    rows: dict[str, FreqRow] = {}
    for word in set(spoken) | set(apple_count):
        if not _HAN.search(word):
            continue
        spoken_pm = 1e6 * spoken.get(word, 0) / spoken_total
        parts = []
        if cifu_written.get(word, 0) > 0:
            parts.append(1e6 * cifu_written[word] / written_total)
        if apple_count.get(word, 0) > 0:
            parts.append(1e6 * apple_count[word] / apple_total)
        written_pm = sum(parts) / len(parts) if parts else 0.0
        has_spoken = spoken.get(word, 0) >= MIN_SPOKEN_COUNT
        if freq == "spoken":
            value = spoken_pm if has_spoken else 0.0
        elif freq == "written" or not has_spoken:
            value = written_pm
        else:
            value = spoken_weight * spoken_pm + (1 - spoken_weight) * written_pm
        in_apple = apple_count.get(word, 0) > 0
        if word in in_cifu and in_apple:
            source = "cifu+ad"
        else:
            source = "cifu" if word in in_cifu else "ad"
        rows[word] = FreqRow(
            word, jyutping.get(word, ""), spoken.get(word, 0), spoken_pm, written_pm, value,
            source=source,
        )

    override_words: set[str] = set()
    for word, like in overrides.items():
        if like not in rows:
            continue
        value = rows[like].blend_pm
        if word in rows:
            rows[word].blend_pm = value
            rows[word].source = "override"
        else:
            rows[word] = FreqRow(word, "", 0, 0.0, 0.0, value, source="override")
        override_words.add(word)

    ordered = sorted(
        rows.values(),
        key=lambda row: (-row.blend_pm, -row.written_pm, row.word in override_words, row.word),
    )
    for rank, row in enumerate(ordered, 1):
        row.rank = rank
    return YueFrequency(ordered, dict(variants))


def load_yue_frequency(
    *,
    freq: str = "blend",
    spoken_weight: float = 0.7,
    refresh: bool = False,
) -> YueFrequency:
    """Load the sources, downloading missing files, and build the list.

    :param freq: ``blend``, ``spoken``, or ``written``.
    :param spoken_weight: Value of ``w``.
    :param refresh: Download both source files again.
    :returns: The ranked list.
    """
    cifu = read_cifu(download_cifu(force=refresh))
    apple = _read_word_count_tsv(download_yue_frequency(force=refresh))
    return build_yue_frequency(
        cifu,
        apple,
        variants=read_variants(),
        overrides=read_overrides(),
        freq=freq,
        spoken_weight=spoken_weight,
    )


def to_frequency_data(data: YueFrequency, max_words: int) -> FrequencyData:
    """Convert the list to the rank tables of the sentence ranker.

    :param data: The ranked list.
    :param max_words: Number of most frequent words to keep.
    :returns: Filled :class:`FrequencyData`.
    """
    ranks = {row.word: row.rank for row in data.rows if row.rank <= max_words}
    for variant, standard in data.variants.items():
        if standard in ranks:
            ranks.setdefault(variant, ranks[standard])
    return frequency_data_from_ranks(ranks)


def write_frequency_tsv(data: YueFrequency, path: Path) -> None:
    """Write the list as a TSV file sorted by rank.

    :param data: The ranked list.
    :param path: Output file.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(path, "w", encoding="utf-8", newline="")
    except OSError as exc:
        raise RuntimeError(f"Cannot write the frequency list {path}.") from exc
    with handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(_COLUMNS)
        for row in data.rows:
            writer.writerow([
                row.word,
                row.jyutping,
                row.spoken_count,
                f"{row.spoken_pm:.1f}",
                f"{row.written_pm:.1f}",
                f"{row.blend_pm:.1f}",
                row.rank,
                row.source,
            ])


def check_frequency(data: YueFrequency) -> list[tuple[str, bool, str]]:
    """Run gates G1, G3, and G4 of the spec on the list.

    :param data: The ranked list.
    :returns: A list of gate name, pass flag, and message.
    """
    ranks = data.ranks
    results = []
    contiguous = [row.rank for row in data.rows] == list(range(1, len(data.rows) + 1))
    results.append((
        "G1",
        len(data.rows) >= 40_000 and contiguous,
        f"{len(data.rows)} rows, ranks contiguous: {contiguous}",
    ))
    for gate, words, limit in (
        ("G3", ("係", "嘅", "唔", "我", "你"), 20),
        ("G4", ("你好", "早晨", "唔該"), 3000),
    ):
        found = {word: ranks.get(word) for word in words}
        ok = all(rank is not None and rank <= limit for rank in found.values())
        results.append((gate, ok, f"ranks within {limit}: {found}"))
    return results


FREQ_CHANGE_COLUMNS: tuple[str, ...] = ("sentence", "rank_before", "rank_after", "move")


def rank_change_rows(
    before: list[Sentence], after: list[Sentence], threshold: int = 100
) -> list[dict[str, Any]]:
    """List sentences whose rank moves by more than a threshold.

    :param before: Ranking with the first frequency term, best first.
    :param after: Ranking with the second frequency term, best first.
    :param threshold: Smallest move in places to report.
    :returns: Rows sorted by the size of the move, largest first.
    """
    old = {s.note_id: i for i, s in enumerate(before, 1)}
    rows: list[dict[str, Any]] = []
    for new_rank, sentence in enumerate(after, 1):
        old_rank = old.get(sentence.note_id)
        if old_rank is None or abs(new_rank - old_rank) <= threshold:
            continue
        rows.append({
            "sentence": sentence.text,
            "rank_before": old_rank,
            "rank_after": new_rank,
            "move": old_rank - new_rank,
        })
    return sorted(rows, key=lambda row: -abs(row["move"]))


def write_freq_change_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write the sentences that the new frequency term moves.

    :param rows: Rows from :func:`rank_change_rows`.
    :param path: Output file.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(path, "w", encoding="utf-8", newline="")
    except OSError as exc:
        raise RuntimeError(f"Cannot write {path}.") from exc
    with handle:
        writer = csv.DictWriter(handle, fieldnames=FREQ_CHANGE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class Tokenizer(Protocol):
    """Minimal word tokenizer: a ``lcut`` method returns the tokens of a text."""

    def lcut(self, text: str) -> list[str]: ...


def build_yue_tokenizer(
    cifu: dict[str, CifuEntry],
    *,
    extra_words: dict[str, str] | None = None,
    dict_path: Path | None = None,
) -> jieba.Tokenizer:
    """Build a jieba tokenizer for Cantonese from the Cifu words.

    The global jieba dictionary stays unchanged. Each Cifu word has the weight
    ``spoken_count * 40 + written_count * 2``. Override words get the weight of
    the top word.

    :param cifu: Cifu entries.
    :param extra_words: Override words (word to the word that gives its value).
    :param dict_path: Where to write the dictionary file. Defaults to the cache folder.
    :returns: An initialized tokenizer.
    """
    path = dict_path or cifu_path().with_name("cifu-jieba-dict.txt")
    weights = {
        word: entry.spoken * SPOKEN_DICT_WEIGHT + entry.written * WRITTEN_DICT_WEIGHT
        for word, entry in cifu.items()
        if _HAN.search(word)
    }
    lines = [f"{word} {weight}" for word, weight in weights.items() if weight > 0]
    text = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")
    tokenizer = jieba.Tokenizer(str(path))
    tokenizer.initialize()
    top = max(weights.values(), default=1)
    for word in extra_words or {}:
        tokenizer.add_word(word, freq=top)
    return tokenizer


def han_tokens(tokenizer: Tokenizer, text: str) -> list[str]:
    """Split text into tokens that contain a Han character.

    :param tokenizer: Tokenizer from :func:`build_yue_tokenizer`.
    :param text: Sentence text.
    :returns: Tokens, in order.
    """
    return [token for token in tokenizer.lcut(text) if _HAN.search(token)]


def spoken_token_coverage(
    tokenizer: Tokenizer, texts: list[str], cifu: dict[str, CifuEntry]
) -> float:
    """Share of Han tokens that have a spoken count in Cifu (gate G2).

    :param tokenizer: Cantonese tokenizer.
    :param texts: Sentence texts.
    :param cifu: Cifu entries.
    :returns: A value between 0 and 1.
    """
    tokens = [token for text in texts for token in han_tokens(tokenizer, text)]
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in cifu and cifu[t].spoken > 0) / len(tokens)
