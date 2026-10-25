"""Spanish word frequency cache from the FrequencyWords ``es_full`` list.

The cache has one row per surface form. The lemma column comes from the spaCy
``es_core_news_sm`` model. The rank is the rank of the lemma after the counts
of its surface forms are summed. Only the top 50,000 rows are kept.
The last two columns hold the lemma count of the Tatoeba spoken corpus and the
``wordfreq`` occurrences per million of the lemma. The cull flow reads them.

See ``specs/core-deck-all-languages.md`` sections 8.1, 8.2, and 11 step S3.
"""

from __future__ import annotations

import csv
import hashlib
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from wordfreq import zipf_frequency

from anki_tools.cache import download_cached_file
from anki_tools.ranking import cullflow

FREQUENCY_PATH = Path("data/spanish_frequency.tsv")
ES_SOURCE_PATH = Path("data/es_full.txt")
TATOEBA_PATH = Path("data/tatoeba/spa_sentences.tsv")
TATOEBA_CACHE_PATH = Path("data/spanish_tatoeba_lemmas.tsv")

ES_FULL_URL = (
    "https://raw.githubusercontent.com/hermitdave/FrequencyWords/"
    "525f9b560de45753a5ea01069454e72e9aa541c6/content/2018/es/es_full.txt"
)
ES_FULL_SHA256 = "b4794c425f3500c21a0ff9367175e36795a978c928a36670e9ad6ae206fb7728"
SOURCE_VERSION = "525f9b560de45753a5ea01069454e72e9aa541c6"
SOURCE_NAME = "FrequencyWords es_full"
LICENSE = "CC BY-SA 4.0"
TOKENIZER = "es_core_news_sm"
WORDFREQ_LANG = "es"
MAX_ROWS = 50_000

_COLUMNS = (
    "word",
    "lemma",
    "count",
    "pm",
    "rank",
    "pos",
    "source",
    "license",
    "tatoeba_count",
    "written_pm",
)


@dataclass(frozen=True)
class SpanishFrequencyRow:
    """One row of the cached Spanish frequency list.

    :param word: Surface form, as it appears in the deck.
    :param lemma: Lemma of the form from spaCy.
    :param count: Token count of the surface form.
    :param pm: Count per million tokens.
    :param rank: 1-based rank of the lemma.
    :param pos: spaCy part of speech of the surface form.
    :param source: Short source name.
    :param license: SPDX license identifier.
    :param tatoeba_count: Count of the lemma in the Tatoeba spoken corpus.
    :param written_pm: ``wordfreq`` occurrences per million of the lemma.
    """

    word: str
    lemma: str
    count: int
    pm: float
    rank: int
    pos: str
    source: str
    license: str
    tatoeba_count: int
    written_pm: float


def load_frequency_es(
    path: Path = FREQUENCY_PATH,
) -> list[SpanishFrequencyRow]:
    """Read the cached Spanish frequency list.

    :param path: TSV file with the columns ``word``, ``lemma``, ``count``,
        ``pm``, ``rank``, ``pos``, ``source``, ``license``, ``tatoeba_count``,
        and ``written_pm``.
    :returns: Rows in file order.
    """
    rows: list[SpanishFrequencyRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle, delimiter="\t"):
            rows.append(
                SpanishFrequencyRow(
                    word=record["word"],
                    lemma=record["lemma"],
                    count=int(record["count"]),
                    pm=float(record["pm"]),
                    rank=int(record["rank"]),
                    pos=record["pos"],
                    source=record["source"],
                    license=record["license"],
                    tatoeba_count=int(record["tatoeba_count"]),
                    written_pm=float(record["written_pm"]),
                )
            )
    return rows


def frequency_ranks_es(
    rows: Sequence[SpanishFrequencyRow],
) -> dict[str, int]:
    """Map each lemma to its 1-based rank.

    :param rows: Frequency rows.
    :returns: Lemma to rank. The most frequent lemma has rank 1.
    """
    return {row.lemma: row.rank for row in rows}


def _read_lemma_counts(path: Path) -> Counter[str]:
    """Read the lemma counts of the Tatoeba cache.

    :param path: Cache TSV with the columns ``lemma`` and ``count``.
    :returns: Lemma to token count.
    """
    counts: Counter[str] = Counter()
    with path.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle, delimiter="\t"):
            counts[record["lemma"]] = int(record["count"])
    return counts


def _write_lemma_counts(counts: Counter[str], path: Path) -> None:
    """Write the lemma counts in lemma order, so the cache is deterministic.

    :param counts: Lemma to token count.
    :param path: Cache TSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["lemma", "count"])
        for lemma in sorted(counts):
            writer.writerow([lemma, counts[lemma]])


def _corpus_texts(path: Path) -> Iterator[str]:
    """Read the sentence text of the Tatoeba Spanish file.

    :param path: Tatoeba sentence TSV, columns ``id``, ``lang``, and ``text``.
    :returns: One sentence text per row.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        for record in csv.reader(handle, delimiter="\t"):
            if len(record) >= 3:
                yield record[2]


def tatoeba_lemma_counts(
    tatoeba_path: Path = TATOEBA_PATH,
    cache_path: Path = TATOEBA_CACHE_PATH,
) -> Counter[str]:
    """Count the lemma of every Tatoeba token, with a resumable cache.

    One spaCy pass over the corpus writes the cache, so a later call reads the
    counts without a second pass.

    :param tatoeba_path: Tatoeba Spanish sentence TSV.
    :param cache_path: Cache file of the counts.
    :raises RuntimeError: The corpus and the cache are both absent.
    :returns: Lemma to token count.
    """
    if cache_path.exists():
        return _read_lemma_counts(cache_path)
    if not tatoeba_path.exists():
        raise RuntimeError(
            f"The Tatoeba Spanish corpus {tatoeba_path} is absent. "
            "Run `anki swadesh fetch spanish` to fetch it."
        )
    counts: Counter[str] = Counter()
    for tokens in cullflow.tokenize_texts(
        _corpus_texts(tatoeba_path), model=TOKENIZER
    ):
        counts.update(lemma for lemma, _pos in tokens)
    _write_lemma_counts(counts, cache_path)
    return counts


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
                unicodedata.normalize("NFC", token.lemma_.lower())
                for token in tokens
            )
            lemmas[word] = joined or word.lower()
            poss[word] = ""
    return lemmas, poss


def build_frequency_es(
    source_path: Path = ES_SOURCE_PATH,
    *,
    words: int | None = None,
    tatoeba_path: Path = TATOEBA_PATH,
    cache_path: Path = TATOEBA_CACHE_PATH,
) -> list[SpanishFrequencyRow]:
    """Build the frequency rows from the FrequencyWords ``es_full`` list.

    The source file is downloaded into ``source_path`` when it is absent. One
    spaCy pass over the Tatoeba corpus gives the spoken count of each lemma,
    and ``wordfreq`` gives the written occurrences per million.

    :param source_path: FrequencyWords ``es_full`` file.
    :param words: Cap on the surface forms, for a fast test run only.
    :param tatoeba_path: Tatoeba Spanish sentence TSV.
    :param cache_path: Cache file of the Tatoeba lemma counts.
    :raises RuntimeError: The source digest differs from
        :data:`ES_FULL_SHA256`.
    :returns: Rows in rank order, capped at :data:`MAX_ROWS`.
    """
    if not source_path.exists():
        download_cached_file(ES_FULL_URL, source_path)
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if digest != ES_FULL_SHA256:
        raise RuntimeError(
            f"sha256 mismatch for {ES_FULL_URL}: got {digest}, "
            f"expected {ES_FULL_SHA256}"
        )
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
    spoken = tatoeba_lemma_counts(tatoeba_path, cache_path)
    order = sorted(ordered, key=lambda word: (ranks[lemmas[word]], -counts[word], word))
    return [
        SpanishFrequencyRow(
            word=word,
            lemma=lemmas[word],
            count=counts[word],
            pm=counts[word] * 1_000_000 / total,
            rank=ranks[lemmas[word]],
            pos=poss[word],
            source=SOURCE_NAME,
            license=LICENSE,
            tatoeba_count=spoken.get(lemmas[word], 0),
            written_pm=10 ** (zipf_frequency(lemmas[word], WORDFREQ_LANG) - 3),
        )
        for word in order[:MAX_ROWS]
    ]


def write_frequency_tsv(
    rows: Iterable[SpanishFrequencyRow],
    path: Path,
) -> None:
    """Write the frequency rows as a TSV file in the given order.

    :param rows: Frequency rows.
    :param path: Output file.
    :raises RuntimeError: The file cannot be written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("w", encoding="utf-8", newline="")
    except OSError as exc:
        raise RuntimeError(f"Cannot write the frequency list {path}.") from exc
    with handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(_COLUMNS)
        for row in rows:
            writer.writerow(
                [
                    row.word,
                    row.lemma,
                    row.count,
                    f"{row.pm:.6f}",
                    row.rank,
                    row.pos,
                    row.source,
                    row.license,
                    row.tatoeba_count,
                    f"{row.written_pm:.6f}",
                ]
            )
__all__ = [
    "ES_FULL_SHA256",
    "ES_FULL_URL",
    "ES_SOURCE_PATH",
    "FREQUENCY_PATH",
    "LICENSE",
    "MAX_ROWS",
    "SOURCE_NAME",
    "SOURCE_VERSION",
    "TATOEBA_CACHE_PATH",
    "TATOEBA_PATH",
    "TOKENIZER",
    "WORDFREQ_LANG",
    "SpanishFrequencyRow",
    "build_frequency_es",
    "frequency_ranks_es",
    "load_frequency_es",
    "tatoeba_lemma_counts",
    "write_frequency_tsv",
]
