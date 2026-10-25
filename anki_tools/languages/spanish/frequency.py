"""Spanish word frequency cache from the FrequencyWords ``es_full`` list.

The cache has one row per surface form. The lemma column comes from the spaCy
``es_core_news_sm`` model. The rank is the rank of the lemma after the counts
of its surface forms are summed. Only the top 50,000 rows are kept.

The table also carries a spoken term from the Tatoeba sentence corpus and a
written term from ``wordfreq``, so a cull can blend the two. The columns map
to the French cull table: French ``word`` is the lemma here, and
``tatoeba_count``, ``tatoeba_pm``, ``written_pm``, and ``zipf`` have the same
meaning.

See ``specs/core-deck-all-languages.md`` sections 8.1, 8.2, and 11 step S3.
"""

from __future__ import annotations

import csv
import hashlib
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from anki_tools.cache import download_cached_file
from anki_tools.languages.spanish.profile import prepare_text, spanish_tokens
from anki_tools.ranking import cullflow
from anki_tools.ranking.corelist import CONTENT_POS

FREQUENCY_PATH = Path("data/spanish_frequency.tsv")
ES_SOURCE_PATH = Path("data/es_full.txt")
TATOEBA_PATH = Path("data/tatoeba/spa_sentences.tsv")

ES_FULL_URL = (
    "https://raw.githubusercontent.com/hermitdave/FrequencyWords/"
    "525f9b560de45753a5ea01069454e72e9aa541c6/content/2018/es/es_full.txt"
)
ES_FULL_SHA256 = "b4794c425f3500c21a0ff9367175e36795a978c928a36670e9ad6ae206fb7728"
SOURCE_VERSION = "525f9b560de45753a5ea01069454e72e9aa541c6"
SOURCE_NAME = "FrequencyWords es_full"
LICENSE = "CC BY-SA 4.0"
TOKENIZER = "es_core_news_sm"
MAX_ROWS = 50_000

# ``hacerlo`` gives the lemma ``hacer él``. The cull keys on the verb.
CLITIC_LEMMAS = frozenset(
    {
        "yo",
        "tú",
        "él",
        "ella",
        "nosotros",
        "nosotras",
        "vosotros",
        "vosotras",
        "ellos",
        "ellas",
        "se",
        "me",
        "te",
        "nos",
        "os",
        "le",
        "les",
        "lo",
        "la",
        "los",
        "las",
    }
)


def base_lemma(lemma: str) -> str:
    """Drop the clitic pronoun that the Spanish model appends to a verb lemma.

    :param lemma: Lemma from the Spanish spaCy model.
    :returns: The lemma without its clitic pronoun.
    """
    head, _, tail = lemma.partition(" ")
    return head if tail in CLITIC_LEMMAS else lemma

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
    "tatoeba_pm",
    "written_pm",
    "zipf",
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
    :param tatoeba_count: Occurrence count of the lemma in the Tatoeba corpus.
    :param tatoeba_pm: Occurrences per million content tokens of Tatoeba.
    :param written_pm: Occurrences per million of the ``wordfreq`` list.
    :param zipf: ``wordfreq`` Zipf value of the lemma.
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
    tatoeba_pm: float
    written_pm: float
    zipf: float


def load_frequency_es(
    path: Path = FREQUENCY_PATH,
) -> list[SpanishFrequencyRow]:
    """Read the cached Spanish frequency list.

    :param path: TSV file with the columns of :data:`_COLUMNS`.
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
                    tatoeba_pm=float(record["tatoeba_pm"]),
                    written_pm=float(record["written_pm"]),
                    zipf=float(record["zipf"]),
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
                unicodedata.normalize("NFC", token.lemma_.lower()) for token in tokens
            )
            lemmas[word] = joined or word.lower()
            poss[word] = ""
    return lemmas, poss


def _count_tatoeba(tatoeba_path: Path) -> tuple[Counter[str], int]:
    """Count the lemmas of the Tatoeba Spanish sentences.

    :param tatoeba_path: Tatoeba sentence TSV, columns ``id``, ``lang``, and
        ``text``.
    :returns: Lemma counts, and the count of content tokens.
    """
    counts: Counter[str] = Counter()
    content_tokens = 0
    with tatoeba_path.open(newline="", encoding="utf-8") as handle:
        for record in csv.reader(handle, delimiter="\t"):
            if len(record) < 3:
                continue
            for token in spanish_tokens(prepare_text(record[2])):
                if token.punctuation:
                    continue
                counts[unicodedata.normalize("NFC", token.lemma.lower())] += 1
                if token.pos in CONTENT_POS:
                    content_tokens += 1
    return counts, content_tokens


def build_frequency_es(
    source_path: Path = ES_SOURCE_PATH,
    *,
    words: int | None = None,
    tatoeba_path: Path = TATOEBA_PATH,
) -> list[SpanishFrequencyRow]:
    """Build the frequency rows from the FrequencyWords ``es_full`` list.

    The source file is downloaded into ``source_path`` when it is absent.

    :param source_path: FrequencyWords ``es_full`` file.
    :param words: Cap on the surface forms, for a fast test run only.
    :param tatoeba_path: Tatoeba sentence TSV of the Spanish corpus.
    :raises RuntimeError: The source digest differs from
        :data:`ES_FULL_SHA256`.
    :returns: Rows in rank order, capped at :data:`MAX_ROWS`.
    """
    from wordfreq import zipf_frequency

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
    tatoeba_counts, content_tokens = _count_tatoeba(tatoeba_path)
    scores: dict[str, tuple[int, float, float, float]] = {}
    for lemma in set(lemmas.values()):
        zipf = zipf_frequency(lemma, "es")
        tatoeba_count = tatoeba_counts.get(lemma, 0)
        tatoeba_pm = tatoeba_count / content_tokens * 1e6 if content_tokens else 0.0
        scores[lemma] = (tatoeba_count, tatoeba_pm, 10 ** (zipf - 3), zipf)
    order = sorted(ordered, key=lambda word: (ranks[lemmas[word]], -counts[word], word))
    rows: list[SpanishFrequencyRow] = []
    for word in order[:MAX_ROWS]:
        lemma = lemmas[word]
        tatoeba_count, tatoeba_pm, written_pm, zipf = scores[lemma]
        rows.append(
            SpanishFrequencyRow(
                word=word,
                lemma=lemma,
                count=counts[word],
                pm=counts[word] * 1_000_000 / total,
                rank=ranks[lemma],
                pos=poss[word],
                source=SOURCE_NAME,
                license=LICENSE,
                tatoeba_count=tatoeba_count,
                tatoeba_pm=tatoeba_pm,
                written_pm=written_pm,
                zipf=zipf,
            )
        )
    return rows


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
                    f"{row.tatoeba_pm:.6f}",
                    f"{row.written_pm:.6f}",
                    f"{row.zipf:.3f}",
                ]
            )


def cull_frequency_rows(
    rows: Sequence[SpanishFrequencyRow],
) -> list[cullflow.FrequencyRow]:
    """Map the Spanish rows to the shared cull rows, one row per base lemma.

    The cache holds one row per surface form, and the cull reads one row per
    lemma. The rows of one base lemma merge, because the tokenizer drops the
    clitic pronoun that the model appends to a verb lemma. The spoken terms add,
    because the rows share the content-token denominator. The written terms take
    their largest value, because a clitic form is absent from ``wordfreq``.

    :param rows: Spanish frequency rows in rank order.
    :returns: One cull row per base lemma, in the given order.
    """
    groups: dict[str, list[SpanishFrequencyRow]] = {}
    for row in rows:
        groups.setdefault(base_lemma(row.lemma), []).append(row)
    return [
        cullflow.FrequencyRow(
            word=lemma,
            tatoeba_count=sum(row.tatoeba_count for row in group),
            tatoeba_pm=sum(row.tatoeba_pm for row in group),
            written_pm=max(row.written_pm for row in group),
            zipf=max(row.zipf for row in group),
        )
        for lemma, group in groups.items()
    ]


__all__ = [
    "CLITIC_LEMMAS",
    "ES_FULL_SHA256",
    "ES_FULL_URL",
    "ES_SOURCE_PATH",
    "FREQUENCY_PATH",
    "LICENSE",
    "MAX_ROWS",
    "SOURCE_NAME",
    "SOURCE_VERSION",
    "TOKENIZER",
    "TATOEBA_PATH",
    "SpanishFrequencyRow",
    "base_lemma",
    "build_frequency_es",
    "cull_frequency_rows",
    "frequency_ranks_es",
    "load_frequency_es",
    "write_frequency_tsv",
]
