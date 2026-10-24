"""Build ``data/spanish_frequency.tsv`` from the FrequencyWords ``es_full`` list.

Run: ``uv run python scripts/build_spanish_frequency.py``.

The source file is downloaded into the cache when it is missing, its sha256 is
checked, the surface forms are lemmatized with spaCy ``es_core_news_sm``, and
the counts are aggregated to the lemma for the rank. The top 50,000 rows are
kept.
"""

from __future__ import annotations

import csv
import hashlib
import unicodedata
from collections import Counter
from pathlib import Path

from anki_tools.cache import download_cached_file
from anki_tools.languages.spanish.frequency import (
    ES_FULL_SHA256,
    ES_FULL_URL,
    FREQUENCY_PATH,
    LICENSE,
    MAX_ROWS,
    SOURCE_NAME,
    TOKENIZER,
    _COLUMNS,
)

SOURCE_PATH = Path("data/es_full.txt")


def _read_counts(path: Path) -> dict[str, int]:
    """Read the ``word count`` pairs of the source file.

    :param path: FrequencyWords ``es_full`` file.
    :returns: Word to summed occurrence count.
    """
    counts: Counter[str] = Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 2:
                continue
            counts[parts[0]] += int(parts[-1])
    return dict(counts)


def _sha256(path: Path) -> str:
    """Give the sha256 digest of a file.

    :param path: File to hash.
    :returns: Hex digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lemmatize(words: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Lemmatize the surface forms with the Spanish spaCy model.

    :param words: Surface forms.
    :returns: Word to lemma and word to part of speech.
    """
    import spacy

    nlp = spacy.load(TOKENIZER, exclude=["parser", "ner"])
    lemmas: dict[str, str] = {}
    poss: dict[str, str] = {}
    for word, doc in zip(words, nlp.pipe(words, batch_size=512)):
        tokens = [token for token in doc if token.text.strip()]
        if len(tokens) == 1:
            token = tokens[0]
            lemmas[word] = unicodedata.normalize("NFC", token.lemma_.lower())
            poss[word] = token.pos_
        else:
            joined = " ".join(
                unicodedata.normalize("NFC", token.lemma_.lower()) for token in tokens
            )
            lemmas[word] = joined or word.lower()
            poss[word] = ""
    return lemmas, poss


def build_spanish_frequency(source_path: Path, *, words: int | None = None) -> None:
    """Build and write the Spanish frequency cache.

    :param source_path: FrequencyWords ``es_full`` file.
    :param words: Cap on the lemmatized words, for a fast test run only.
    """
    counts = _read_counts(source_path)
    ordered = sorted(counts, key=lambda word: (-counts[word], word))
    if words is not None:
        ordered = ordered[:words]
    lemmas, poss = _lemmatize(ordered)

    lemma_counts: Counter[str] = Counter()
    for word in ordered:
        lemma_counts[lemmas[word]] += counts[word]

    ranked = sorted(lemma_counts, key=lambda lemma: (-lemma_counts[lemma], lemma))
    ranks = {lemma: rank for rank, lemma in enumerate(ranked, start=1)}

    total = sum(counts.values())
    rows = sorted(
        ordered,
        key=lambda word: (ranks[lemmas[word]], -counts[word], word),
    )[:MAX_ROWS]

    FREQUENCY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FREQUENCY_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(_COLUMNS)
        for word in rows:
            writer.writerow(
                [
                    word,
                    lemmas[word],
                    counts[word],
                    f"{counts[word] * 1_000_000 / total:.6f}",
                    ranks[lemmas[word]],
                    poss[word],
                    SOURCE_NAME,
                    LICENSE,
                ]
            )


def main() -> None:
    """Download the source, check it, and build the cache."""
    source = download_cached_file(ES_FULL_URL, SOURCE_PATH)
    digest = _sha256(source)
    if digest != ES_FULL_SHA256:
        raise RuntimeError(
            f"sha256 mismatch for {ES_FULL_URL}: got {digest}, "
            f"expected {ES_FULL_SHA256}"
        )
    build_spanish_frequency(source)


if __name__ == "__main__":
    main()
