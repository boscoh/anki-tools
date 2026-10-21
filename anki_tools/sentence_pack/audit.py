"""Quality gates of a Swadesh sentence pack.

:func:`run_gates` reads the reviewed word table, the sentence CSV, the audio
index, the APKG file, the manifest, and the attribution file. It returns one row
per gate. :func:`print_audit` writes the gate table and gives the process exit
code. The module only reads files.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from anki_tools.apkg.package import AnkiPackage
from anki_tools.sentence_pack.audio import (
    AUDIO_INDEX_NAME,
    MAX_DURATION_S,
    MIN_DURATION_S,
    probe_duration,
)
from anki_tools.sentence_pack.english_sense import sense_mismatches
from anki_tools.sentence_pack.pack import WORD_PREFIX, load_audio_index
from anki_tools.sentence_pack.guards import check_sentence_shape
from anki_tools.sentence_pack.pool import (
    SCAFFOLD_SHAPE_KEYS,
    shape_failure,
)
from anki_tools.sentence_pack.profile import (
    LanguageProfile,
    WordRow,
    duplicate_limit,
)

OPEN_STATUS = "open"
LATIN_OR_DIGIT = re.compile(r"[A-Za-z0-9０-９]")
FAILURE_SAMPLE = 3
DETAIL_WIDTH = 110

GateRow = tuple[str, bool, str]
GateCheck = Callable[["AuditInputs", LanguageProfile], tuple[bool, str]]


class AuditInputs:
    """Lazy reader for the files that the gates share.

    A read error is cached and re-raised for each gate that needs the file, so
    one missing input marks only the dependent gates as failed.
    """

    def __init__(
        self,
        profile: LanguageProfile,
        *,
        apkg_path: Path,
        sentences_csv: Path,
        word_csv: Path,
        review_csv: Path,
        audio_dir: Path,
        manifest_path: Path,
        attribution_path: Path,
        load_table: Callable[..., Sequence[WordRow]],
        probe: Callable[[Path], float] = probe_duration,
    ) -> None:
        """Store the paths and start an empty cache.

        :param profile: Language profile.
        :param apkg_path: APKG file of the pack.
        :param sentences_csv: Sentence CSV with one row per card.
        :param word_csv: Reviewed Swadesh word CSV.
        :param review_csv: Word review CSV.
        :param audio_dir: Directory with the MP3 files and ``index.csv``.
        :param manifest_path: Manifest JSON file.
        :param attribution_path: Attribution Markdown file.
        :param load_table: Callable that reads the reviewed word table.
        :param probe: Function that reads the duration of a file.
        """
        self.profile = profile
        self.apkg_path = Path(apkg_path)
        self.sentences_csv = Path(sentences_csv)
        self.word_csv = Path(word_csv)
        self.review_csv = Path(review_csv)
        self.audio_dir = Path(audio_dir)
        self.manifest_path = Path(manifest_path)
        self.attribution_path = Path(attribution_path)
        self.load_table = load_table
        self.probe = probe
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

    def table(self) -> list[Any]:
        """Read the word table.

        :returns: One word form per reviewed form.
        """
        return self._load(
            "table", lambda: list(self.load_table(csv_path=self.word_csv))
        )

    def rows(self) -> list[dict[str, str]]:
        """Read the sentence CSV rows.

        :returns: One mapping per card row.
        """
        return self._load("rows", lambda: read_rows(self.sentences_csv))

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
        return self._load("manifest", lambda: read_json(self.manifest_path))

    def apkg(self) -> dict[str, Any]:
        """Open the APKG file and summarize it.

        :returns: Note count, models, and media file names.
        """
        return self._load("apkg", lambda: read_apkg(self.apkg_path))


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into row mappings.

    :param path: CSV file.
    :raises FileNotFoundError: The file cannot be read.
    :returns: One mapping per data row.
    """
    try:
        with Path(path).open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError as error:
        raise FileNotFoundError(f"cannot read {path}: {error}") from error


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object file.

    :param path: JSON file.
    :raises FileNotFoundError: The file cannot be read.
    :raises ValueError: The file is not a JSON object.
    :returns: The mapping.
    """
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


def read_apkg(path: Path) -> dict[str, Any]:
    """Open an APKG file and summarize it.

    :param path: APKG file.
    :returns: Note count, models, and media file names.
    """
    with AnkiPackage(path) as package:
        notes = package.get_notes()
        models = package.get_models()
        media = set(package.get_media_mapping().values())
    return {"notes": len(notes), "models": models, "media": media}


def sha256_value(path: Path) -> str:
    """Build the manifest hash value of a file.

    :param path: File to hash.
    :returns: ``sha256:`` followed by the hex digest.
    """
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return f"sha256:{digest}"


def sample(items: Iterable[str], limit: int = FAILURE_SAMPLE) -> str:
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


def coverage_ratio(inputs: AuditInputs) -> tuple[int, int, float]:
    """Compute the core coverage of a pack.

    :param inputs: Lazy file reader of the audit.
    :returns: Covered concept count, total concept count, and the ratio.
    """
    positions = inputs.profile.form_positions(inputs.table())
    concepts = {index for index, _ in positions.values()}
    covered = {
        positions[row["target"]][0]
        for row in inputs.rows()
        if row["target"] in positions
    }
    ratio = len(covered) / len(concepts) if concepts else 0.0
    return len(covered), len(concepts), ratio


def core_problem(sentence: str, profile: LanguageProfile) -> str | None:
    """Give the first proper-sentence failure of a core card.

    :param sentence: Card sentence.
    :param profile: Language profile.
    :returns: Failure text, or None when the sentence passes.
    """
    key = check_sentence_shape(
        sentence, profile, profile.min_units, profile.max_units
    )
    if key is not None:
        return f"{key} failed"
    if not profile.has_predicate(sentence):
        return "F9 predicate tag"
    if profile.has_orphan(sentence):
        return "F10 orphan conjunction"
    return None


def scaffold_problem(sentence: str, profile: LanguageProfile) -> str | None:
    """Give the first proper-sentence failure of a scaffold card.

    :param sentence: Card sentence.
    :param profile: Language profile.
    :returns: Failure text, or None when the sentence passes.
    """
    key = shape_failure(
        sentence,
        profile,
        profile.fallback_min_units,
        profile.max_units,
        keys=SCAFFOLD_SHAPE_KEYS,
    )
    if key == "F2":
        return (
            f"F2 length {profile.unit_count(sentence)} outside "
            f"{profile.fallback_min_units} to {profile.max_units}"
        )
    if key is not None:
        return f"{key} failed"
    if not profile.has_predicate(sentence):
        return "F9 predicate tag"
    if profile.has_orphan(sentence):
        return "F10 orphan conjunction"
    return None


def gate_unique_forms(
    inputs: AuditInputs, profile: LanguageProfile
) -> tuple[bool, str]:
    """Gate G2: no duplicate word form.

    :param inputs: Lazy file reader of the audit.
    :param profile: Language profile.
    :returns: Pass flag and detail text.
    """
    counts = Counter(
        profile.form_text(form)
        for form in inputs.table()
        if form.variant != "shared"
    )
    duplicates = sorted(form for form, count in counts.items() if count > 1)
    if duplicates:
        return False, f"{len(duplicates)} duplicate form(s): {sample(duplicates)}"
    return True, f"{len(counts)} unique forms"


def gate_targets_in_table(
    inputs: AuditInputs, profile: LanguageProfile
) -> tuple[bool, str]:
    """Gate G3: every card target exists in the reviewed CSV.

    :param inputs: Lazy file reader of the audit.
    :param profile: Language profile.
    :returns: Pass flag and detail text.
    """
    forms = {profile.form_text(form) for form in inputs.table()}
    rows = inputs.rows()
    missing = sorted({row["target"] for row in rows if row["target"] not in forms})
    if missing:
        return False, (
            f"{len(missing)} card target(s) not in {inputs.word_csv}: "
            f"{sample(missing)}"
        )
    return True, f"{len(rows)} card targets found in the word table"


def gate_unique_tatoeba_ids(
    inputs: AuditInputs, profile: LanguageProfile  # noqa: ARG001 - gate signature
) -> tuple[bool, str]:
    """Gate G4: every card has a unique Tatoeba id.

    :param inputs: Lazy file reader of the audit.
    :param profile: Language profile.
    :returns: Pass flag and detail text.
    """
    counts = Counter(row.get("tatoeba_id", "").strip() for row in inputs.rows())
    duplicates = sorted(value for value, count in counts.items() if count > 1)
    if duplicates:
        return False, (
            f"{len(duplicates)} duplicate Tatoeba id(s): {sample(duplicates)}"
        )
    return True, f"{len(counts)} unique Tatoeba ids"


def gate_target_once(
    inputs: AuditInputs, profile: LanguageProfile
) -> tuple[bool, str]:
    """Gate G5: the target appears one time in each sentence.

    :param inputs: Lazy file reader of the audit.
    :param profile: Language profile.
    :returns: Pass flag and detail text.
    """
    rows = inputs.rows()
    bad = [
        (row["target"], row["sentence"], profile.match_count(row["sentence"], row["target"]))
        for row in rows
    ]
    bad = [item for item in bad if item[2] != 1]
    if bad:
        details = [
            f"{target} x{count} in {sentence}" for target, sentence, count in bad
        ]
        return False, (
            f"{len(bad)} card(s) do not hold the target once: {sample(details)}"
        )
    return True, (
        f"{len(rows)} cards hold the target once"
        if rows
        else "no cards"
    )


def gate_similarity(
    inputs: AuditInputs, profile: LanguageProfile
) -> tuple[bool, str]:
    """Gate G6: no card pair reaches the Jaccard limit.

    :param inputs: Lazy file reader of the audit.
    :param profile: Language profile.
    :returns: Pass flag and detail text.
    """
    limit = duplicate_limit(profile)
    sentences = [row.get("sentence", "") for row in inputs.rows()]
    offenders: list[str] = []
    for first in range(len(sentences)):
        for second in range(first + 1, len(sentences)):
            score = profile.similarity(sentences[first], sentences[second])
            if score >= limit:
                offenders.append(
                    f"{sentences[first]} / {sentences[second]} = {score:.2f}"
                )
    if offenders:
        return False, (
            f"{len(offenders)} card pair(s) reach the {limit:.2f} limit: "
            f"{sample(offenders)}"
        )
    return True, f"{len(sentences)} cards stay below {limit:.2f}"


def gate_sentence_audio(
    inputs: AuditInputs, profile: LanguageProfile  # noqa: ARG001 - gate signature
) -> tuple[bool, str]:
    """Gate G7: every card has a valid sentence file.

    :param inputs: Lazy file reader of the audit.
    :param profile: Language profile.
    :returns: Pass flag and detail text.
    """
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
            duration = inputs.probe(path)
        except RuntimeError:
            invalid.append(f"{name} is not readable")
            continue
        if not MIN_DURATION_S <= duration <= MAX_DURATION_S:
            invalid.append(f"{name} is {duration:.2f} s")
    if missing or invalid:
        return False, (
            f"{len(missing) + len(invalid)} sentence file(s) invalid "
            f"(missing: {sample(missing) or 'none'}; "
            f"duration: {sample(invalid) or 'none'})"
        )
    return True, (
        f"{len(rows)} sentence files, {MIN_DURATION_S:.1f} to {MAX_DURATION_S:.1f} s"
    )


def expected_media(inputs: AuditInputs) -> set[str]:
    """Collect the media file names that the card rows use.

    :param inputs: Lazy file reader of the audit.
    :returns: File names of the sentence and word audio of every card.
    """
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


def gate_media_count(
    inputs: AuditInputs, profile: LanguageProfile  # noqa: ARG001 - gate signature
) -> tuple[bool, str]:
    """Gate G8: the audio file count equals the media count of the APKG.

    :param inputs: Lazy file reader of the audit.
    :param profile: Language profile.
    :returns: Pass flag and detail text.
    """
    expected = expected_media(inputs)
    media = inputs.apkg()["media"]
    if len(media) != len(expected):
        return False, (
            f"APKG holds {len(media)} media files, cards use {len(expected)}"
        )
    missing = sorted(expected - media)
    extra = sorted(media - expected)
    if missing or extra:
        return False, (
            f"media names differ (missing: {sample(missing) or 'none'}; "
            f"unused: {sample(extra) or 'none'})"
        )
    return True, f"{len(media)} media files for the card audio"


def gate_coverage(
    inputs: AuditInputs, profile: LanguageProfile
) -> tuple[bool, str]:
    """Gate G10: core coverage reaches the minimum.

    :param inputs: Lazy file reader of the audit.
    :param profile: Language profile.
    :returns: Pass flag and detail text.
    """
    covered, concepts, ratio = coverage_ratio(inputs)
    message = f"core coverage {covered}/{concepts} = {ratio:.3f}"
    if ratio < profile.coverage_min:
        return False, f"{message} is below the {profile.coverage_min:.2f} minimum"
    return True, message


def gate_manifest_counts(
    inputs: AuditInputs, profile: LanguageProfile  # noqa: ARG001 - gate signature
) -> tuple[bool, str]:
    """Gate G11: manifest counts and file hashes match the pack.

    :param inputs: Lazy file reader of the audit.
    :param profile: Language profile.
    :returns: Pass flag and detail text.
    """
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
        actual = sha256_value(path)
        if files.get(key) != actual:
            problems.append(f"files.{key}: manifest {files.get(key)} != {actual}")
    if problems:
        return False, sample(problems, 4)
    return True, f"{len(expected)} counts and 2 file hashes match"


def gate_apkg(
    inputs: AuditInputs, profile: LanguageProfile
) -> tuple[bool, str]:
    """Gate G13: the APKG opens and lists the expected notes.

    :param inputs: Lazy file reader of the audit.
    :param profile: Language profile.
    :returns: Pass flag and detail text.
    """
    info = inputs.apkg()
    rows = inputs.rows()
    problems: list[str] = []
    if info["notes"] != len(rows):
        problems.append(f"{info['notes']} notes but {len(rows)} cards")
    models = info["models"]
    model = next(
        (item for item in models.values() if item["name"] == profile.model_name), None
    )
    if model is None:
        names = sorted(item.get("name", "") for item in models.values())
        problems.append(f"model {profile.model_name!r} not found in {names}")
    else:
        fields = [field["name"] for field in model["flds"]]
        if fields != list(profile.fields):
            problems.append(f"model fields {fields} != {list(profile.fields)}")
    if problems:
        return False, sample(problems)
    return True, f"{info['notes']} notes with model {profile.model_name!r}"


def gate_proper_sentences(
    inputs: AuditInputs, profile: LanguageProfile
) -> tuple[bool, str]:
    """Gate G14: core and scaffold cards pass the proper-sentence rules.

    :param inputs: Lazy file reader of the audit.
    :param profile: Language profile.
    :returns: Pass flag and detail text.
    """
    rows = inputs.rows()
    failures: list[str] = []
    for row in rows:
        sentence = row.get("sentence", "")
        if row.get("tier") == "scaffold":
            problem = scaffold_problem(sentence, profile)
        else:
            problem = core_problem(sentence, profile)
        if problem:
            failures.append(f"{sentence}: {problem}")
    if failures:
        return False, (
            f"{len(failures)} card(s) fail the proper-sentence rules: "
            f"{sample(failures)}"
        )
    return True, f"{len(rows)} cards pass the proper-sentence rules"


def gate_review_file(
    inputs: AuditInputs, profile: LanguageProfile  # noqa: ARG001 - gate signature
) -> tuple[bool, str]:
    """Gate G15: the word review file exists and has no open row.

    :param inputs: Lazy file reader of the audit.
    :param profile: Language profile.
    :returns: Pass flag and detail text.
    """
    if not inputs.review_csv.is_file():
        return False, f"review file missing: {inputs.review_csv}"
    rows = read_rows(inputs.review_csv)
    open_rows = [
        row.get("index", "")
        for row in rows
        if row.get("status", "").strip().lower() == OPEN_STATUS
    ]
    if open_rows:
        return False, f"{len(open_rows)} open review row(s): {sample(open_rows)}"
    return True, f"{len(rows)} reviewed rows, none open"


def gate_word_audio(
    inputs: AuditInputs, profile: LanguageProfile  # noqa: ARG001 - gate signature
) -> tuple[bool, str]:
    """Gate G16: every card target has a valid word file.

    :param inputs: Lazy file reader of the audit.
    :param profile: Language profile.
    :returns: Pass flag and detail text.
    """
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
        return False, f"{len(bad)} card(s) lack word audio: {sample(bad)}"
    return True, f"{len(rows)} cards have a word file"


def gate_no_native_audio(
    inputs: AuditInputs, profile: LanguageProfile  # noqa: ARG001 - gate signature
) -> tuple[bool, str]:
    """Gate G18: no card uses the native audio source.

    :param inputs: Lazy file reader of the audit.
    :param profile: Language profile.
    :returns: Pass flag and detail text.
    """
    native = [
        row["sentence"] for row in inputs.rows() if row.get("audio_source") == "native"
    ]
    if native:
        return False, f"{len(native)} card(s) use native audio: {sample(native)}"
    return True, "no card uses native audio"


def gate_target_in_table(
    inputs: AuditInputs, profile: LanguageProfile
) -> tuple[bool, str]:
    """Gate G19: every target is a table form with a card of its own.

    :param inputs: Lazy file reader of the audit.
    :param profile: Language profile.
    :returns: Pass flag and detail text.
    """
    table = inputs.table()
    forms = {profile.form_text(form) for form in table}
    rows_by_form: dict[str, list[str]] = {}
    for form in table:
        rows_by_form.setdefault(profile.form_text(form), []).append(form.variant)
    shared = {
        name for name, variants in rows_by_form.items() if set(variants) == {"shared"}
    }
    rows = inputs.rows()
    problems: list[str] = []
    for target in sorted({row["target"] for row in rows}):
        if target not in forms:
            problems.append(f"{target} is not in the word table")
        elif target in shared:
            problems.append(f"{target} is a shared form")
    if problems:
        return False, f"{len(problems)} target(s) invalid: {sample(problems)}"
    return True, f"{len(rows)} card targets are table forms with a card"


def gate_no_false_friend(
    inputs: AuditInputs, profile: LanguageProfile
) -> tuple[bool, str]:
    """Gate: no card uses its target in a different-sense compound.

    :param inputs: Lazy file reader of the audit.
    :param profile: Language profile.
    :returns: Pass flag and detail text.
    """
    if profile.false_friend_spans is None:
        return True, "the language has no false-friend guard"
    rows = inputs.rows()
    bad: list[str] = []
    for row in rows:
        sentence = row.get("sentence", "")
        target = row.get("target", "")
        if profile.false_friend_spans(sentence, target):
            bad.append(f"{target} in {sentence}")
    if bad:
        return False, (
            f"{len(bad)} card(s) use the target in a different sense: {sample(bad)}"
        )
    return True, f"{len(rows)} cards use each target in its Swadesh sense"


def gate_no_sense_mismatch(
    inputs: AuditInputs, profile: LanguageProfile
) -> tuple[bool, str]:
    """Gate: no card English holds a different sense.

    :param inputs: Lazy file reader of the audit.
    :param profile: Language profile.
    :returns: Pass flag and detail text.
    """
    rows = inputs.rows()
    gloss_by_form = {
        profile.form_text(form): form.english for form in inputs.table()
    }
    bad = sense_mismatches(rows, gloss_by_form, profile.sense_filter)
    if bad:
        return False, (
            f"{len(bad)} card(s) use the target in a different sense: {sample(bad)}"
        )
    return True, f"{len(rows)} cards use each target in its Swadesh sense"


GENERIC_GATES: tuple[tuple[str, GateCheck], ...] = (
    ("G2", gate_unique_forms),
    ("G3", gate_targets_in_table),
    ("G4", gate_unique_tatoeba_ids),
    ("G5", gate_target_once),
    ("G6", gate_similarity),
    ("G7", gate_sentence_audio),
    ("G8", gate_media_count),
    ("G10", gate_coverage),
    ("G11", gate_manifest_counts),
    ("G13", gate_apkg),
    ("G14", gate_proper_sentences),
    ("G15", gate_review_file),
    ("G16", gate_word_audio),
    ("G18", gate_no_native_audio),
    ("G19", gate_target_in_table),
)


def run_gates(
    gates: Sequence[tuple[str, GateCheck]],
    inputs: AuditInputs,
    profile: LanguageProfile,
) -> list[GateRow]:
    """Run a gate list and collect one row per gate.

    :param gates: Ordered gate names and checks.
    :param inputs: Lazy file reader of the audit.
    :param profile: Language profile.
    :returns: One ``(gate, passed, message)`` row per gate.
    """
    results: list[GateRow] = []
    for gate, check in gates:
        try:
            passed, message = check(inputs, profile)
        except Exception as error:  # noqa: BLE001 - a gate reports read failures
            passed, message = False, f"{type(error).__name__}: {error}"
        results.append((gate, passed, message))
    return results


def print_audit(results: Sequence[GateRow]) -> int:
    """Print the gate table.

    :param results: Rows from :func:`run_gates`.
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
