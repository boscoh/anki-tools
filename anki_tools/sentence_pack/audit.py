"""Quality gates of a Swadesh sentence pack.

:func:`as_gates` wraps the pack gate table as :class:`anki_tools.gates.Gate`
values, so the pack runs :func:`anki_tools.gates.run_gates`. The checks read the
reviewed word table, the sentence CSV, the audio index, the APKG file, the
manifest, and the attribution file. :func:`print_audit` writes the gate table
and gives the process exit code. :func:`run_pack_audit` binds a profile and a
gate list to the default paths of the profile. The module only reads files.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from anki_tools.apkg.deck_writer import sha256_value
from anki_tools.gate_files import GateFileReader
from anki_tools.gates import Gate, GateResult, run_gates
from anki_tools.ranking.coreflow import read_output
from anki_tools.ranking.identity import DeckIdentity, default_output
from anki_tools.sentence_pack.audio import (
    AUDIO_INDEX_NAME,
    MAX_DURATION_S,
    MIN_DURATION_S,
    probe_duration,
)
from anki_tools.sentence_pack.pack import (
    ATTRIBUTION_PATH,
    WORD_PREFIX,
    load_audio_index,
)
from anki_tools.sentence_pack.guards import check_sentence_shape
from anki_tools.sentence_pack.pool import shape_failure
from anki_tools.sentence_pack.profile import (
    LanguageProfile,
    WordRow,
    duplicate_limit,
)

OPEN_STATUS = "open"
FAILURE_SAMPLE = 3
DETAIL_WIDTH = 110

GateCheck = Callable[["AuditInputs", LanguageProfile], tuple[bool, str]]


class AuditInputs(GateFileReader):
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
        super().__init__(
            {
                "table": lambda: list(self.load_table(csv_path=self.word_csv)),
                "rows": lambda: read_rows(self.sentences_csv),
                "index": lambda: load_audio_index(
                    self.audio_dir / AUDIO_INDEX_NAME
                ),
                "manifest": lambda: read_json(self.manifest_path),
                "output": lambda: read_output(self.apkg_path),
            }
        )

    def table(self) -> list[Any]:
        """Read the word table.

        :returns: One word form per reviewed form.
        """
        return self._load("table")

    def index(self):
        """Read the audio index.

        :returns: The sentence and word file maps of the audio cache.
        """
        return self._load("index")

    def output(self) -> dict[str, Any]:
        """Open the APKG file and summarize it.

        :returns: The mapping of :func:`anki_tools.ranking.coreflow.read_output`.
        """
        return self._load("output")


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
    """Give the first complete-sentence failure of a core card.

    :param sentence: Card sentence.
    :param profile: Language profile.
    :returns: Failure text, or None when the sentence passes.
    """
    key = check_sentence_shape(
        sentence, profile, profile.min_units, profile.max_units
    )
    if key is not None:
        return f"{key} failed"
    return None


def scaffold_problem(sentence: str, profile: LanguageProfile) -> str | None:
    """Give the first complete-sentence failure of a scaffold card.

    :param sentence: Card sentence.
    :param profile: Language profile.
    :returns: Failure text, or None when the sentence passes.
    """
    key = shape_failure(
        sentence,
        profile,
        profile.fallback_min_units,
        profile.max_units,
        scaffold=True,
    )
    if key == "length_window":
        return (
            f"length_window length {profile.unit_count(sentence)} outside "
            f"{profile.fallback_min_units} to {profile.max_units}"
        )
    if key is not None:
        return f"{key} failed"
    return None


def gate_word_table(
    inputs: AuditInputs, profile: LanguageProfile
) -> tuple[bool, str]:
    """Gate P1: the reviewed word table passes every S1 check.

    :param inputs: Lazy file reader of the audit.
    :param profile: Language profile.
    :returns: Pass flag and detail text.
    """
    table = inputs.table()
    failures = profile.check_word_table(
        table, inputs.word_csv, inputs.review_csv
    )
    concepts = len({form.swadesh_index for form in table})
    if failures:
        return False, sample(failures)
    return True, f"{concepts} concepts, {len(table)} forms"


def gate_unique_forms(
    inputs: AuditInputs, profile: LanguageProfile
) -> tuple[bool, str]:
    """Gate P2: no duplicate word form.

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
    """Gate P3: every card target is a table form that is not shared.

    A shared form has no card of its own, so a target on it is a fault.

    :param inputs: Lazy file reader of the audit.
    :param profile: Language profile.
    :returns: Pass flag and detail text.
    """
    table = inputs.table()
    forms = {profile.form_text(form) for form in table}
    variants: dict[str, list[str]] = {}
    for form in table:
        variants.setdefault(profile.form_text(form), []).append(form.variant)
    shared = {name for name, kinds in variants.items() if set(kinds) == {"shared"}}
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


def gate_unique_tatoeba_ids(
    inputs: AuditInputs, profile: LanguageProfile  # noqa: ARG001 - gate signature
) -> tuple[bool, str]:
    """Gate P4: every card has a unique Tatoeba id.

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
    """Gate P5: the target appears one time in each sentence.

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
    """Gate P6: no card pair reaches the Jaccard limit.

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
    """Gate P7: every card has a valid sentence file.

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
    """Gate P8: the audio file count equals the media count of the APKG.

    :param inputs: Lazy file reader of the audit.
    :param profile: Language profile.
    :returns: Pass flag and detail text.
    """
    expected = expected_media(inputs)
    media = inputs.output()["media"]
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
    """Gate P10: core coverage reaches the minimum.

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
    """Gate P11: manifest counts and file hashes match the pack.

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


def gate_output_model(
    inputs: AuditInputs, profile: LanguageProfile
) -> tuple[bool, str]:
    """Gate P13: the APKG holds the expected notes and note type.

    The check reads the one output reader, which needs a single note type.

    :param inputs: Lazy file reader of the audit.
    :param profile: Language profile.
    :returns: Pass flag and detail text.
    """
    info = inputs.output()
    rows = inputs.rows()
    problems: list[str] = []
    if info["notes"] != len(rows):
        problems.append(f"{info['notes']} notes but {len(rows)} cards")
    model_name = DeckIdentity(inputs.apkg_path).model_name
    if model_name not in info["note_models"]:
        names = sorted(info["note_models"])
        problems.append(f"model {model_name!r} not found in {names}")
    else:
        fields = info["fields"]
        if fields != list(profile.fields):
            problems.append(f"model fields {fields} != {list(profile.fields)}")
    if problems:
        return False, sample(problems)
    return True, f"{info['notes']} notes with model {model_name!r}"


def gate_filter_recheck(
    inputs: AuditInputs, profile: LanguageProfile
) -> tuple[bool, str]:
    """Gate P14: core and scaffold cards pass the complete-sentence rules.

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
            f"{len(failures)} card(s) fail the complete-sentence rules: "
            f"{sample(failures)}"
        )
    return True, f"{len(rows)} cards pass the complete-sentence rules"


def gate_review_file(
    inputs: AuditInputs, profile: LanguageProfile  # noqa: ARG001 - gate signature
) -> tuple[bool, str]:
    """Gate P15: the word review file exists and has no open row.

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
    """Gate P16: every card target has a valid word file.

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
    """Gate P18: no card uses the native audio source.

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


GENERIC_GATES: tuple[tuple[str, GateCheck], ...] = (
    ("P2", gate_unique_forms),
    ("P3", gate_targets_in_table),
    ("P4", gate_unique_tatoeba_ids),
    ("P5", gate_target_once),
    ("P6", gate_similarity),
    ("P7", gate_sentence_audio),
    ("P8", gate_media_count),
    ("P10", gate_coverage),
    ("P11", gate_manifest_counts),
    ("P13", gate_output_model),
    ("P14", gate_filter_recheck),
    ("P15", gate_review_file),
    ("P16", gate_word_audio),
    ("P18", gate_no_native_audio),
)


def as_gates(gates: Sequence[tuple[str, GateCheck]]) -> tuple[Gate, ...]:
    """Wrap the pack gate table as gate values.

    :param gates: Ordered gate names and pack checks.
    :returns: One gate per name, with the profile of the reader.
    """
    return tuple(
        Gate(name, lambda files, check=check: check(files, files.profile))
        for name, check in gates
    )


def run_pack_audit(
    profile: LanguageProfile,
    gates: Sequence[tuple[str, GateCheck]],
    *,
    apkg_path: Path | None = None,
    sentences_csv: Path | None = None,
    word_csv: Path | None = None,
    review_csv: Path | None = None,
    audio_dir: Path | None = None,
    manifest_path: Path | None = None,
    attribution_path: Path = ATTRIBUTION_PATH,
    load_table: Callable[..., Sequence[WordRow]],
    probe: Callable[[Path], float] = probe_duration,
) -> list[GateResult]:
    """Run the gates of one pack on the default paths of the pack.

    :param profile: Language profile of the pack.
    :param gates: Ordered gate names and checks.
    :param apkg_path: APKG file of the pack, default the pack default output.
    :param sentences_csv: Sentence CSV, default the profile file.
    :param word_csv: Reviewed word CSV, default the profile file.
    :param review_csv: Word review CSV, default the profile file.
    :param audio_dir: Audio directory, default the profile directory.
    :param manifest_path: Manifest JSON, default the side file of the APKG.
    :param attribution_path: Attribution Markdown file.
    :param load_table: Callable that reads the reviewed word table.
    :param probe: Function that reads the duration of a file.
    :returns: One result per gate.
    """
    identity = DeckIdentity(
        apkg_path if apkg_path is not None else default_output(profile.name, "pack")
    )
    inputs = AuditInputs(
        profile,
        apkg_path=identity.output_path,
        sentences_csv=sentences_csv or profile.sentence_csv,
        word_csv=word_csv or profile.word_csv,
        review_csv=review_csv or profile.word_review_csv,
        audio_dir=audio_dir or profile.audio_dir,
        manifest_path=manifest_path or identity.side("manifest.json"),
        attribution_path=attribution_path,
        load_table=load_table,
        probe=probe,
    )
    return run_gates(as_gates(gates), inputs)


def print_audit(results: Sequence[GateResult], title: str = "") -> int:
    """Print the gate table.

    :param results: Rows from :func:`run_gates`.
    :param title: Flow name of the table, printed above it. ADR 0027 records
        why an id alone does not name the flow.
    :returns: 0 when every gate passes, else 1.
    """
    if title:
        print(title)
        print()
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
