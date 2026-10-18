"""Quality gates G1 to G16 for the Mandarin Swadesh sentence pack.

:func:`run_audit` reads the reviewed word table, the sentence CSV, the audio
index, the APKG file, the manifest, and ``ATTRIBUTION.md``. It returns one row
per gate of spec section 13. :func:`print_audit` writes the gate table and
gives the process exit code.

The module only reads files. It never writes the pack or the audio cache.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from anki_tools.mandarin_audio import (
    AUDIO_INDEX_NAME,
    MAX_DURATION_S,
    MIN_DURATION_S,
    probe_duration,
)
from anki_tools.mandarin_deck import (
    FIELDS,
    MODEL_NAME,
    WORD_PREFIX,
    audio_author_map,
    load_audio_index,
)
from anki_tools.mandarin_swadesh import (
    FILTER_NAMES,
    FINAL_MARKS,
    QUOTE_MARKS,
    SIMILARITY_LIMIT,
    WordForm,
    check_sentence_shape,
    check_word_table,
    count_form_occurrences,
    has_orphan_conjunction,
    has_predicate,
    load_word_table,
)
from anki_tools.package import AnkiPackage
from anki_tools.rank import (
    char_similarity_zh,
    get_chinese_chars,
    is_repeated_zh_sentence,
)

OUTPUT_PATH = Path("apkg/swadesh_mandarin_sentences.apkg")
SENTENCES_PATH = Path("vocab/mandarin_sentences.csv")
WORD_CSV_PATH = Path("vocab/english-mandarin.csv")
REVIEW_PATH = Path("vocab/swadesh_review.csv")
AUDIO_DIR = Path("audio/mandarin_sentences")
MANIFEST_PATH = Path("apkg/swadesh_mandarin_sentences.manifest.json")
ATTRIBUTION_PATH = Path("ATTRIBUTION.md")

CORE_COVERAGE_MIN = 0.80
CORE_MIN_HAN = 4
SCAFFOLD_MIN_HAN = 3
MAX_HAN = 10
OPEN_STATUS = "open"
LATIN_OR_DIGIT = re.compile(r"[A-Za-z0-9０-９]")
FAILURE_SAMPLE = 3
DETAIL_WIDTH = 110

GateRow = tuple[str, bool, str]


class _Inputs:
    """Lazy reader for the files that the gates share.

    A read error is cached and re-raised for each gate that needs the file, so
    one missing input marks only the dependent gates as failed.
    """

    def __init__(
        self,
        apkg_path: Path,
        sentences_csv: Path,
        word_csv: Path,
        review_csv: Path,
        audio_dir: Path,
        manifest_path: Path,
        attribution_path: Path,
    ) -> None:
        """Store the paths and start an empty cache.

        :param apkg_path: APKG file of the pack.
        :param sentences_csv: Sentence CSV with one row per card.
        :param word_csv: Reviewed Swadesh word CSV.
        :param review_csv: Word review CSV.
        :param audio_dir: Directory with the MP3 files and ``index.csv``.
        :param manifest_path: Manifest JSON file.
        :param attribution_path: Attribution Markdown file.
        """
        self.apkg_path = Path(apkg_path)
        self.sentences_csv = Path(sentences_csv)
        self.word_csv = Path(word_csv)
        self.review_csv = Path(review_csv)
        self.audio_dir = Path(audio_dir)
        self.manifest_path = Path(manifest_path)
        self.attribution_path = Path(attribution_path)
        self._cache: dict[str, tuple[bool, Any]] = {}

    def _load(self, name: str, loader: Callable[[], Any]) -> Any:
        if name not in self._cache:
            try:
                self._cache[name] = (True, loader())
            except Exception as error:  # noqa: BLE001 - cached for every gate
                self._cache[name] = (False, error)
        ok, value = self._cache[name]
        if not ok:
            raise value
        return value

    def table(self) -> list[WordForm]:
        """Read the word table.

        :returns: One :class:`WordForm` per reviewed form.
        """
        return self._load("table", lambda: load_word_table(csv_path=self.word_csv))

    def rows(self) -> list[dict[str, str]]:
        """Read the sentence CSV rows.

        :returns: One mapping per card row.
        """
        return self._load("rows", lambda: _read_rows(self.sentences_csv))

    def index(self):
        """Read the audio index.

        :returns: The sentence and word file maps of the audio cache.
        """
        return self._load(
            "index", lambda: load_audio_index(self.audio_dir / AUDIO_INDEX_NAME)
        )

    def manifest(self) -> dict[str, Any]:
        """Read the manifest JSON file.

        :returns: The manifest mapping.
        """
        return self._load("manifest", lambda: _read_json(self.manifest_path))

    def apkg(self) -> dict[str, Any]:
        """Open the APKG file and summarize it.

        :returns: Note count, models, and media file names.
        """
        return self._load("apkg", lambda: _read_apkg(self.apkg_path))


def _read_rows(path: Path) -> list[dict[str, str]]:
    try:
        with Path(path).open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError as error:
        raise FileNotFoundError(f"cannot read {path}: {error}") from error


def _read_json(path: Path) -> dict[str, Any]:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as error:
        raise FileNotFoundError(f"cannot read {path}: {error}") from error
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"bad JSON in {path}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"expected an object in {path}")
    return data


def _read_apkg(path: Path) -> dict[str, Any]:
    with AnkiPackage(path) as package:
        notes = package.get_notes()
        models = package.get_models()
        media = set(package.get_media_mapping().values())
    return {"notes": len(notes), "models": models, "media": media}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return f"sha256:{digest}"


def _sample(items: Iterable[str], limit: int = FAILURE_SAMPLE) -> str:
    """Join a bounded number of items for a failure message.

    :param items: Values to show.
    :param limit: Maximum values before a count suffix.
    :returns: Comma-separated values with an ``+N more`` suffix.
    """
    values = list(items)
    shown = ", ".join(values[:limit])
    if len(values) > limit:
        shown += f" (+{len(values) - limit} more)"
    return shown


def _audio_id(row: dict[str, str]) -> int:
    try:
        return int(str(row.get("audio_id", "")).strip() or 0)
    except ValueError:
        return 0


def _core_problem(sentence: str) -> str | None:
    key = check_sentence_shape(sentence, CORE_MIN_HAN, MAX_HAN)
    if key is not None:
        return f"{key} {FILTER_NAMES[key]}"
    if not has_predicate(sentence):
        return "F9 predicate tag"
    if has_orphan_conjunction(sentence):
        return "F10 orphan conjunction"
    return None


def _scaffold_problem(sentence: str) -> str | None:
    han_len = len(get_chinese_chars(sentence))
    if not SCAFFOLD_MIN_HAN <= han_len <= MAX_HAN:
        return f"F2 length {han_len} outside {SCAFFOLD_MIN_HAN} to {MAX_HAN}"
    if LATIN_OR_DIGIT.search(sentence):
        return "F3 Latin letter or digit"
    if is_repeated_zh_sentence(sentence):
        return "F4 repeated pattern"
    if not sentence or sentence[-1] not in FINAL_MARKS:
        return "F5 final mark"
    if any(char in QUOTE_MARKS for char in sentence):
        return "F7 quotation mark"
    if not has_predicate(sentence):
        return "F9 predicate tag"
    if has_orphan_conjunction(sentence):
        return "F10 orphan conjunction"
    return None


def _gate_g1(inputs: _Inputs) -> tuple[bool, str]:
    table = inputs.table()
    failures = check_word_table(
        table, csv_path=inputs.word_csv, review_path=inputs.review_csv
    )
    concepts = len({form.swadesh_index for form in table})
    if failures:
        return False, _sample(failures)
    return True, f"{concepts} concepts, {len(table)} forms"


def _gate_g2(inputs: _Inputs) -> tuple[bool, str]:
    counts = Counter(form.mandarin for form in inputs.table())
    duplicates = sorted(form for form, count in counts.items() if count > 1)
    if duplicates:
        return False, f"{len(duplicates)} duplicate form(s): {_sample(duplicates)}"
    return True, f"{len(counts)} unique forms"


def _gate_g3(inputs: _Inputs) -> tuple[bool, str]:
    forms = {form.mandarin for form in inputs.table()}
    rows = inputs.rows()
    missing = sorted({row["target"] for row in rows if row["target"] not in forms})
    if missing:
        return False, (
            f"{len(missing)} card target(s) not in {inputs.word_csv}: "
            f"{_sample(missing)}"
        )
    return True, f"{len(rows)} card targets found in the word table"


def _gate_g4(inputs: _Inputs) -> tuple[bool, str]:
    counts = Counter(row.get("tatoeba_id", "").strip() for row in inputs.rows())
    duplicates = sorted(value for value, count in counts.items() if count > 1)
    if duplicates:
        return (
            False,
            f"{len(duplicates)} duplicate Tatoeba id(s): {_sample(duplicates)}",
        )
    return True, f"{len(counts)} unique Tatoeba ids"


def _gate_g5(inputs: _Inputs) -> tuple[bool, str]:
    rows = inputs.rows()
    bad = [
        (
            row["target"],
            row["sentence"],
            count_form_occurrences(row["sentence"], row["target"]),
        )
        for row in rows
    ]
    bad = [item for item in bad if item[2] != 1]
    if bad:
        details = [
            f"{target} x{count} in {sentence}" for target, sentence, count in bad
        ]
        return (
            False,
            f"{len(bad)} card(s) do not hold the target once: {_sample(details)}",
        )
    return True, f"{len(rows)} cards hold the target once"


def _gate_g6(inputs: _Inputs) -> tuple[bool, str]:
    sentences = [row.get("sentence", "") for row in inputs.rows()]
    offenders: list[str] = []
    for first in range(len(sentences)):
        for second in range(first + 1, len(sentences)):
            score = char_similarity_zh(sentences[first], sentences[second])
            if score >= SIMILARITY_LIMIT:
                offenders.append(
                    f"{sentences[first]} / {sentences[second]} = {score:.2f}"
                )
    if offenders:
        return False, (
            f"{len(offenders)} card pair(s) reach the {SIMILARITY_LIMIT:.2f} limit: "
            f"{_sample(offenders)}"
        )
    return True, f"{len(sentences)} cards stay below {SIMILARITY_LIMIT:.2f}"


def _gate_g7(inputs: _Inputs) -> tuple[bool, str]:
    index = inputs.index()
    rows = inputs.rows()
    missing: list[str] = []
    invalid: list[str] = []
    for row in rows:
        name = index.sentences.get(row["sentence"])
        if not name:
            missing.append(f"{row['sentence']} has no index row")
            continue
        path = inputs.audio_dir / name
        if not path.is_file():
            missing.append(f"{name} for {row['sentence']}")
            continue
        try:
            duration = probe_duration(path)
        except RuntimeError:
            invalid.append(f"{name} is not readable")
            continue
        if not MIN_DURATION_S <= duration <= MAX_DURATION_S:
            invalid.append(f"{name} is {duration:.2f} s")
    if missing or invalid:
        return False, (
            f"{len(missing) + len(invalid)} sentence file(s) invalid "
            f"(missing: {_sample(missing) or 'none'}; "
            f"duration: {_sample(invalid) or 'none'})"
        )
    return True, (
        f"{len(rows)} sentence files, {MIN_DURATION_S:.1f} to {MAX_DURATION_S:.1f} s"
    )


def _expected_media(inputs: _Inputs) -> set[str]:
    index = inputs.index()
    names: set[str] = set()
    for row in inputs.rows():
        if row.get("audio_missing", "").strip().lower() == "true":
            continue
        sentence_name = index.sentences.get(row["sentence"])
        word_name = index.words.get(row["target"])
        if sentence_name:
            names.add(sentence_name)
        if word_name:
            names.add(word_name)
    return names


def _gate_g8(inputs: _Inputs) -> tuple[bool, str]:
    expected = _expected_media(inputs)
    media = inputs.apkg()["media"]
    if len(media) != len(expected):
        return False, (
            f"APKG holds {len(media)} media files, cards use {len(expected)}"
        )
    missing = sorted(expected - media)
    extra = sorted(media - expected)
    if missing or extra:
        return False, (
            f"media names differ (missing: {_sample(missing) or 'none'}; "
            f"unused: {_sample(extra) or 'none'})"
        )
    return True, f"{len(media)} media files for the card audio"


def _gate_g9(inputs: _Inputs) -> tuple[bool, str]:
    rows = inputs.rows()
    bad = []
    for row in rows:
        han_len = len(get_chinese_chars(row["sentence"]))
        syllables = len(row.get("pinyin", "").split())
        if han_len != syllables:
            bad.append(f"{row['sentence']}: {han_len} Han, {syllables} syllables")
    if bad:
        return False, f"{len(bad)} card(s) mismatch pinyin: {_sample(bad)}"
    return True, f"{len(rows)} cards match pinyin syllables"


def _gate_g10(inputs: _Inputs) -> tuple[bool, str]:
    index_of = {form.mandarin: form.swadesh_index for form in inputs.table()}
    concepts = set(index_of.values())
    covered = {
        index_of[row["target"]] for row in inputs.rows() if row["target"] in index_of
    }
    ratio = len(covered) / len(concepts) if concepts else 0.0
    message = f"core coverage {len(covered)}/{len(concepts)} = {ratio:.3f}"
    if ratio < CORE_COVERAGE_MIN:
        return False, f"{message} is below the {CORE_COVERAGE_MIN:.2f} minimum"
    return True, message


def _gate_g11(inputs: _Inputs) -> tuple[bool, str]:
    rows = inputs.rows()
    counts = inputs.manifest().get("counts", {})
    expected = {
        "words": len(inputs.table()),
        "cards": len(rows),
        "scaffold": sum(1 for row in rows if row.get("tier") == "scaffold"),
        "native_audio": sum(1 for row in rows if row.get("audio_source") == "native"),
        "tts_audio": sum(1 for row in rows if row.get("audio_source") == "tts"),
    }
    problems = [
        f"counts.{key}: manifest {counts.get(key)} != {value}"
        for key, value in expected.items()
        if counts.get(key) != value
    ]
    files = inputs.manifest().get("files", {})
    for key, path in (
        ("source_csv", inputs.word_csv),
        ("sentences_csv", inputs.sentences_csv),
    ):
        actual = _sha256(path)
        if files.get(key) != actual:
            problems.append(f"files.{key}: manifest {files.get(key)} != {actual}")
    if problems:
        return False, _sample(problems, 4)
    return True, f"{len(expected)} counts and 2 file hashes match"


def _gate_g12(inputs: _Inputs) -> tuple[bool, str]:
    path = inputs.attribution_path
    if not path.is_file():
        return False, f"attribution file missing: {path}"
    text = path.read_text(encoding="utf-8")
    native = [row for row in inputs.rows() if row.get("audio_source") == "native"]
    if not native:
        return True, "no native audio; the attribution file exists"
    authors = audio_author_map()
    names = {
        authors.get(_audio_id(row)) or row.get("tatoeba_author", "").strip()
        for row in native
    }
    missing = sorted(name for name in names if name and name not in text)
    if missing:
        return False, (
            f"{len(missing)} native audio author(s) not in {path}: {_sample(missing)}"
        )
    return True, f"{len(names)} native audio author(s) named in {path}"


def _gate_g13(inputs: _Inputs) -> tuple[bool, str]:
    info = inputs.apkg()
    rows = inputs.rows()
    problems: list[str] = []
    if info["notes"] != len(rows):
        problems.append(f"{info['notes']} notes but {len(rows)} cards")
    models = info["models"]
    model = next((item for item in models.values() if item["name"] == MODEL_NAME), None)
    if model is None:
        names = sorted(item.get("name", "") for item in models.values())
        problems.append(f"model {MODEL_NAME!r} not found in {names}")
    else:
        fields = [field["name"] for field in model["flds"]]
        if fields != FIELDS:
            problems.append(f"model fields {fields} != {FIELDS}")
    if problems:
        return False, _sample(problems)
    return True, f"{info['notes']} notes with model {MODEL_NAME!r}"


def _gate_g14(inputs: _Inputs) -> tuple[bool, str]:
    rows = inputs.rows()
    failures: list[str] = []
    for row in rows:
        sentence = row.get("sentence", "")
        if row.get("tier") == "scaffold":
            problem = _scaffold_problem(sentence)
        else:
            problem = _core_problem(sentence)
        if problem:
            failures.append(f"{sentence}: {problem}")
    if failures:
        return False, (
            f"{len(failures)} card(s) fail the proper-sentence rules: "
            f"{_sample(failures)}"
        )
    return True, f"{len(rows)} cards pass the proper-sentence rules"


def _gate_g15(inputs: _Inputs) -> tuple[bool, str]:
    if not inputs.review_csv.is_file():
        return False, f"review file missing: {inputs.review_csv}"
    rows = _read_rows(inputs.review_csv)
    open_rows = [
        row.get("index", "")
        for row in rows
        if row.get("status", "").strip().lower() == OPEN_STATUS
    ]
    if open_rows:
        return False, (f"{len(open_rows)} open review row(s): {_sample(open_rows)}")
    return True, f"{len(rows)} reviewed rows, none open"


def _gate_g16(inputs: _Inputs) -> tuple[bool, str]:
    index = inputs.index()
    rows = inputs.rows()
    bad: list[str] = []
    for row in rows:
        target = row["target"]
        name = index.words.get(target)
        if not name or not name.startswith(WORD_PREFIX):
            bad.append(f"{target} has no word file in the index")
            continue
        if not (inputs.audio_dir / name).is_file():
            bad.append(f"{name} for {target} is missing")
    if bad:
        return False, f"{len(bad)} card(s) lack word audio: {_sample(bad)}"
    return True, f"{len(rows)} cards have a word file"


_GATES: tuple[tuple[str, Callable[[_Inputs], tuple[bool, str]]], ...] = (
    ("G1", _gate_g1),
    ("G2", _gate_g2),
    ("G3", _gate_g3),
    ("G4", _gate_g4),
    ("G5", _gate_g5),
    ("G6", _gate_g6),
    ("G7", _gate_g7),
    ("G8", _gate_g8),
    ("G9", _gate_g9),
    ("G10", _gate_g10),
    ("G11", _gate_g11),
    ("G12", _gate_g12),
    ("G13", _gate_g13),
    ("G14", _gate_g14),
    ("G15", _gate_g15),
    ("G16", _gate_g16),
)


def run_audit(
    apkg_path: Path = OUTPUT_PATH,
    sentences_csv: Path = SENTENCES_PATH,
    word_csv: Path = WORD_CSV_PATH,
    review_csv: Path = REVIEW_PATH,
    audio_dir: Path = AUDIO_DIR,
    manifest_path: Path = MANIFEST_PATH,
    attribution_path: Path = ATTRIBUTION_PATH,
) -> list[GateRow]:
    """Run the quality gates G1 to G16.

    :param apkg_path: APKG file of the pack.
    :param sentences_csv: Sentence CSV with one row per card.
    :param word_csv: Reviewed Swadesh word CSV.
    :param review_csv: Word review CSV.
    :param audio_dir: Directory with the MP3 files and ``index.csv``.
    :param manifest_path: Manifest JSON file.
    :param attribution_path: Attribution Markdown file.
    :returns: One ``(gate, passed, message)`` row per gate, G1 first.
    """
    inputs = _Inputs(
        apkg_path=apkg_path,
        sentences_csv=sentences_csv,
        word_csv=word_csv,
        review_csv=review_csv,
        audio_dir=audio_dir,
        manifest_path=manifest_path,
        attribution_path=attribution_path,
    )
    results: list[GateRow] = []
    for gate, check in _GATES:
        try:
            passed, message = check(inputs)
        except Exception as error:  # noqa: BLE001 - a gate reports read failures
            passed, message = False, f"{type(error).__name__}: {error}"
        results.append((gate, passed, message))
    return results


def print_audit(results: Sequence[GateRow]) -> int:
    """Print the gate table.

    :param results: Rows from :func:`run_audit`.
    :returns: 0 when every gate passes, else 1.
    """
    print(f"{'Gate':<4}  {'Result':<6}  Detail")
    print(f"{'-' * 4}  {'-' * 6}  {'-' * DETAIL_WIDTH}")
    for gate, passed, message in results:
        detail = (
            message
            if len(message) <= DETAIL_WIDTH
            else message[: DETAIL_WIDTH - 3] + "..."
        )
        print(f"{gate:<4}  {'PASS' if passed else 'FAIL':<6}  {detail}")
    failed = [gate for gate, passed, _ in results if not passed]
    print()
    if failed:
        print(f"{len(failed)} of {len(results)} gates failed: {', '.join(failed)}")
        return 1
    print(f"all {len(results)} gates passed")
    return 0
