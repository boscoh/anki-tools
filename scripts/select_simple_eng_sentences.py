"""CLI: ``select`` (Tatoeba TSV → English list, default 4–10 words linear),
``translate-tibetan`` (English lines → Tibetan via LLM), and
``simple-anki`` (English gTTS + EN/BO TSV → .apkg).
"""

from __future__ import annotations

import asyncio
import bz2
import gzip
import hashlib
import json
import os
from collections import deque
import re
import sys
import textwrap
import time
import traceback
from typing import Any

import json_repair
from cyclopts import App
from dotenv import load_dotenv
from path import Path
from rich.console import Console
from rich.pretty import Pretty

from anki_tools.apkg.package import AnkiPackage
from anki_tools.ranking.rank import latin_text_dedup_key, word_similarity_fr

_ROOT = Path(__file__).absolute().parent.parent
_ARCHIVE = _ROOT / "archive" / "tibetan"

load_dotenv(_ROOT / ".env")

app = App(
    help=(
        "Commands: select — English lines from Tatoeba (default 4–10 words, linear); "
        "translate-tibetan — LLM English→Tibetan; "
        "simple-anki — English gTTS + Anki .apkg."
    ),
)

ERR_CONSOLE = Console(stderr=True)


def _log_translate(message: str, *, style: str = "cyan", quiet: bool = False) -> None:
    if quiet:
        return
    ERR_CONSOLE.print(f"[bold cyan]translate-tibetan[/] {message}", style=style)


def _log_translate_usage(
    usage: dict[str, Any] | Any,
    *,
    quiet: bool = False,
) -> None:
    if quiet:
        return
    ERR_CONSOLE.print("[dim]usage[/dim]")
    ERR_CONSOLE.print(Pretty(usage, indent_guides=True, expand_all=False))


TIBETAN_SYSTEM_PROMPT = textwrap.dedent("""
    You translate English into Tibetan (བོད་ཡིག), using correct Tibetan Unicode
    script. Preserve meaning; use natural, standard written Tibetan. Do not add
    explanations, romanization-only output, or English. Return valid JSON only,
    no markdown fences or other text.
""").strip()

TIBETAN_USER_PROMPT_TEMPLATE = textwrap.dedent("""
    Translate each English line into Tibetan. Return a single JSON object with
    one key "items", a list of objects in the same order as the input, each:
    {{"english": "<source line>", "tibetan": "<Tibetan translation>"}}

    English lines:
    {numbered_lines}
""").strip()

TIBETAN_TRANSLATION_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "submit_translations",
            "description": "Submit English to Tibetan translations in source order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "english": {"type": "string"},
                                "tibetan": {"type": "string"},
                            },
                            "required": ["english", "tibetan"],
                        },
                    }
                },
                "required": ["items"],
            },
        },
    }
]


def _strip_bom(s: str) -> str:
    if s.startswith("\ufeff"):
        return s[1:]
    return s


def _json_object_candidates(raw: str) -> list[str]:
    text = _strip_bom(raw.strip())
    if not text:
        return []
    out: list[str] = []
    seen: set[str] = set()

    def add(s: str) -> None:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)

    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        add(m.group(1))
    loose = re.sub(r"^```(?:json)?\s*", "", text)
    loose = re.sub(r"\s*```\s*$", "", loose)
    add(loose)
    add(text)
    return out


def _decode_json_object_attempts(fragment: str) -> dict | None:
    try:
        v = json.loads(fragment)
        return v if isinstance(v, dict) else None
    except json.JSONDecodeError:
        pass
    brace = fragment.find("{")
    if brace != -1:
        try:
            v, _ = json.JSONDecoder().raw_decode(fragment, brace)
            if isinstance(v, dict):
                return v
        except json.JSONDecodeError:
            pass
    try:
        v = json_repair.loads(fragment)
        if isinstance(v, tuple):
            v = v[0]
        if isinstance(v, dict):
            return v
    except Exception:
        pass
    if brace != -1:
        try:
            v = json_repair.loads(fragment[brace:])
            if isinstance(v, tuple):
                v = v[0]
            if isinstance(v, dict):
                return v
        except Exception:
            pass
    return None


def parse_json_object_from_llm_text(raw: str) -> dict | None:
    """Return the first JSON object found using a multi-step parse pipeline."""
    for cand in _json_object_candidates(raw):
        obj = _decode_json_object_attempts(cand)
        if obj is not None:
            return obj
    return None


def _parse_llm_json_object(result: dict) -> dict | None:
    text = result.get("text")
    if not isinstance(text, str):
        return None
    return parse_json_object_from_llm_text(text)


def _parse_llm_tool_object(result: dict) -> dict | None:
    tool_calls = result.get("tool_calls")
    if not isinstance(tool_calls, list):
        return None
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function")
        if not isinstance(fn, dict):
            continue
        if fn.get("name") != "submit_translations":
            continue
        args = fn.get("arguments")
        if isinstance(args, dict):
            return args
        if not isinstance(args, str):
            continue
        obj = _decode_json_object_attempts(args)
        if obj is not None:
            return obj
    return None


def _extract_tibetan_items(data: dict, expected_count: int) -> list[str]:
    items = data.get("items")
    if not isinstance(items, list):
        raise RuntimeError('Expected JSON key "items" to be a list')
    if len(items) != expected_count:
        raise RuntimeError(f"Expected {expected_count} translations, got {len(items)}")
    out: list[str] = []
    for i, row in enumerate(items):
        if not isinstance(row, dict):
            raise RuntimeError(f"items[{i}] is not an object")
        t = row.get("tibetan")
        if not isinstance(t, str) or not t.strip():
            raise RuntimeError(f"Missing or empty tibetan for items[{i}]")
        out.append(t.strip())
    return out


def _is_retryable_batch_error(exc: Exception) -> bool:
    msg = str(exc)
    if "could not be parsed as JSON" in msg:
        return True
    if 'Expected JSON key "items" to be a list' in msg:
        return True
    if re.search(r"Expected \d+ translations, got \d+", msg):
        return True
    return False


def _source_lines_sha256(lines: list[str]) -> str:
    joined = "\n".join(lines).encode("utf-8", errors="replace")
    return hashlib.sha256(joined).hexdigest()


def _load_translation_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _save_translation_checkpoint(
    *,
    path: Path,
    lines: list[str],
    translated: list[str],
    service: str,
    batch_size: int,
    completed: bool,
) -> None:
    payload = {
        "source_sha256": _source_lines_sha256(lines),
        "source_count": len(lines),
        "translated_count": len(translated),
        "service": service,
        "batch_size": batch_size,
        "completed": completed,
        "items": [
            {"english": en, "tibetan": bo}
            for en, bo in zip(lines[: len(translated)], translated, strict=True)
        ],
    }
    path.parent.makedirs_p()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


async def _translate_batch_with_client(
    client: Any,
    lines: list[str],
    *,
    verbose: bool = False,
    batch_label: str | None = None,
    max_retries: int = 2,
) -> list[str]:
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(lines))
    user_content = TIBETAN_USER_PROMPT_TEMPLATE.format(numbered_lines=numbered)
    messages = [
        {"role": "system", "content": TIBETAN_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                user_content
                + "\n\nReturn by calling function submit_translations with JSON arguments."
            ),
        },
    ]
    total_attempts = max(1, max_retries + 1)
    label = batch_label or "?"
    for attempt in range(1, total_attempts + 1):
        result = await client.get_completion(
            messages,
            tools=TIBETAN_TRANSLATION_TOOL,
            temperature=0.2,
            max_tokens=8192,
        )
        completion_text = result.get("text")
        if not isinstance(completion_text, str):
            completion_text = ""
        metadata = result.get("metadata")
        usage = metadata.get("usage") if isinstance(metadata, dict) else {}
        finish_reason = (
            metadata.get("finish_reason") if isinstance(metadata, dict) else None
        )
        tool_calls = result.get("tool_calls")
        n_tool_calls = len(tool_calls) if isinstance(tool_calls, list) else 0
        if verbose:
            _log_translate(
                (
                    f"batch [bold]{label}[/] attempt [bold]{attempt}/{total_attempts}[/] "
                    f"finish_reason={finish_reason!r} "
                    f"text_len={len(completion_text)} "
                    f"tool_calls={n_tool_calls}"
                ),
                style="bright_blue",
            )
            _log_translate_usage(usage, quiet=not verbose)
        if completion_text.strip().startswith("Error:"):
            err: Exception = RuntimeError(completion_text)
        else:
            data = _parse_llm_tool_object(result) or _parse_llm_json_object(result)
            if not data:
                snippet = completion_text.strip().replace("\n", " ")
                if len(snippet) > 400:
                    snippet = snippet[:400] + "…"
                err = RuntimeError(
                    "LLM response could not be parsed as JSON"
                    + (f". Raw (truncated): {snippet!r}" if snippet else "")
                )
            else:
                try:
                    return _extract_tibetan_items(data, len(lines))
                except RuntimeError as e:
                    err = e
        if attempt < total_attempts and _is_retryable_batch_error(err):
            if verbose:
                _log_translate(
                    f"batch [bold]{label}[/] retrying after error: {err}",
                    style="yellow",
                )
            await asyncio.sleep(min(2.0, 0.5 * attempt))
            continue
        raise err
    raise RuntimeError("unreachable")


async def _translate_lines_to_tibetan(
    lines: list[str],
    *,
    service: str,
    batch_size: int,
    batch_retries: int,
    verbose: bool,
    checkpoint_path: Path | None = None,
) -> list[str]:
    from microeval.llm import get_llm_client

    if not lines:
        return []
    bs = max(1, batch_size)
    n_batches = (len(lines) + bs - 1) // bs
    merged: list[str] = []
    source_sha = _source_lines_sha256(lines)
    if checkpoint_path is not None:
        ck = _load_translation_checkpoint(checkpoint_path)
        if ck is not None and ck.get("source_sha256") == source_sha:
            items = ck.get("items")
            if isinstance(items, list):
                resumed: list[str] = []
                for row in items:
                    if isinstance(row, dict) and isinstance(row.get("tibetan"), str):
                        resumed.append(row["tibetan"].strip())
                merged = resumed[: len(lines)]
                if verbose and merged:
                    _log_translate(
                        f"resumed [bold]{len(merged)}[/] line(s) from [dim]{checkpoint_path}[/dim]",
                        style="magenta",
                    )
    async with get_llm_client(service) as client:
        model = getattr(client, "model", "?")
        svc_name = getattr(client, "service", service)
        if verbose:
            _log_translate(
                (
                    f"{len(lines)} line(s), {n_batches} batch(es) of up to {bs}, "
                    f"service={svc_name!r} model={model!r}"
                ),
                style="cyan",
            )
        start_index = len(merged)
        batch_num = start_index // bs
        for i in range(start_index, len(lines), bs):
            batch = lines[i : i + bs]
            batch_num += 1
            line_start = i + 1
            line_end = i + len(batch)
            if verbose:
                _log_translate(
                    f"batch [bold]{batch_num}/{n_batches}[/] (lines {line_start}-{line_end}) ...",
                    style="blue",
                )
            t0 = time.perf_counter()
            part = await _translate_batch_with_client(
                client,
                batch,
                verbose=verbose,
                batch_label=f"{batch_num}/{n_batches}",
                max_retries=batch_retries,
            )
            merged.extend(part)
            if checkpoint_path is not None:
                _save_translation_checkpoint(
                    path=checkpoint_path,
                    lines=lines,
                    translated=merged,
                    service=service,
                    batch_size=bs,
                    completed=False,
                )
            if verbose:
                dt = time.perf_counter() - t0
                _log_translate(
                    f"batch [bold]{batch_num}/{n_batches}[/] done in {dt:.2f}s",
                    style="green",
                )
    if checkpoint_path is not None:
        _save_translation_checkpoint(
            path=checkpoint_path,
            lines=lines,
            translated=merged,
            service=service,
            batch_size=bs,
            completed=len(merged) == len(lines),
        )
    return merged


def translate_lines_to_tibetan_sync(
    lines: list[str],
    *,
    service: str,
    batch_size: int,
    batch_retries: int,
    verbose: bool,
    checkpoint_path: Path | None = None,
) -> list[str]:
    return asyncio.run(
        _translate_lines_to_tibetan(
            lines,
            service=service,
            batch_size=batch_size,
            batch_retries=batch_retries,
            verbose=verbose,
            checkpoint_path=checkpoint_path,
        )
    )


def open_text(path: Path):
    suf = path.suffix.lower()
    if suf == ".bz2":
        return bz2.open(path, "rt", encoding="utf-8", errors="replace")
    if suf == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def complexity_key(text: str) -> tuple[int, int, int]:
    words = text.split()
    punct = sum(1 for c in text if c in ",;:()—–-")
    return (len(text), len(words), punct)


def better_candidate(a: str, b: str) -> str:
    ka, kb = complexity_key(a), complexity_key(b)
    return a if ka < kb else b


def greedy_near_dedup(
    simplest_first: list[str],
    *,
    target: int,
    threshold: float,
    min_chars: int,
) -> list[str]:
    kept: list[str] = []
    for s in simplest_first:
        if len(kept) >= target:
            break
        s_n = len(s.replace(" ", ""))
        skip = False
        for k in kept:
            if min_chars and (s_n < min_chars or len(k.replace(" ", "")) < min_chars):
                continue
            if word_similarity_fr(s, k) >= threshold:
                skip = True
                break
        if not skip:
            kept.append(s)
    return kept


def _word_count(text: str) -> int:
    return len(text.split())


def _looks_symbolic_or_single_letterish(text: str) -> bool:
    if re.search(r"\b[^\W\d_]\s*=\s*\S", text):
        return True
    for token in re.findall(r"[^\W\d_]+", text):
        if len(token) == 1 and token not in {"I", "a"}:
            return True
        if token.isupper() and len(token) <= 2 and token not in {"A", "I"}:
            return True
    return False


def _passes_similarity_gate(
    candidate: str,
    kept: list[str],
    *,
    threshold: float,
    min_chars: int,
) -> bool:
    if threshold <= 0:
        return True
    s_ns = len(candidate.replace(" ", ""))
    for k in kept:
        if min_chars and (
            s_ns < min_chars or len(k.replace(" ", "")) < min_chars
        ):
            continue
        if word_similarity_fr(candidate, k) >= threshold:
            return False
    return True


def _linear_target_word_count(
    index: int,
    *,
    count: int,
    min_words: int,
    max_words: int,
) -> float:
    if count <= 1:
        return float(min_words)
    span = max_words - min_words
    return min_words + span * (index / (count - 1))


def _select_linear_word_progression(
    unique_sentences: list[str],
    *,
    count: int,
    min_words: int,
    max_words: int,
    similarity_threshold: float,
    similarity_min_chars: int,
) -> list[str]:
    buckets: dict[int, deque[str]] = {
        w: deque() for w in range(min_words, max_words + 1)
    }
    for s in unique_sentences:
        n = _word_count(s)
        if min_words <= n <= max_words:
            buckets[n].append(s)
    for w in buckets:
        buckets[w] = deque(sorted(buckets[w], key=lambda t: (len(t), t)))

    out: list[str] = []
    for k in range(count):
        t = _linear_target_word_count(
            k, count=count, min_words=min_words, max_words=max_words
        )
        wc_order = sorted(
            range(min_words, max_words + 1),
            key=lambda w: (abs(w - t), w),
        )
        picked = False
        for w in wc_order:
            n_scan = len(buckets[w])
            scanned = 0
            while buckets[w] and scanned < n_scan + 1:
                candidate = buckets[w].popleft()
                scanned += 1
                if _passes_similarity_gate(
                    candidate,
                    out,
                    threshold=similarity_threshold,
                    min_chars=similarity_min_chars,
                ):
                    out.append(candidate)
                    picked = True
                    break
                buckets[w].append(candidate)
            if picked:
                break
        if not picked:
            break
    return out


@app.command
def select(
    *,
    source: Path = _ARCHIVE / "eng_sentences.tsv",
    destination: Path = _ARCHIVE / "english_simple_1000.txt",
    count: int = 1000,
    min_words: int = 4,
    max_words: int = 10,
    legacy: bool = False,
    exclude_symbolic: bool = True,
    similarity_threshold: float = 0.5,
    similarity_min_chars: int = 0,
) -> None:
    """Read Tatoeba-style ``eng_sentences.tsv`` (id, lang, text) and
    write one English sentence per line.

    By default, sentences have between ``min_words`` and ``max_words``
    (inclusive), chosen so word count rises **linearly** from the first
    line to the last. Use ``--legacy`` for the previous behaviour
    (globally shortest / least complex, then near-dedup).

    :param source: TSV path (tab-separated id, lang, text).
        ``.gz`` and ``.bz2`` are supported.
    :param destination: Output file, one sentence per line.
    :param count: How many sentences to emit.
    :param min_words: Minimum words per sentence (non-legacy mode).
    :param max_words: Maximum words per sentence (non-legacy mode).
    :param legacy: Shortest-first selection with ``greedy_near_dedup``.
    :param exclude_symbolic: Drop sentences with code-like or odd single-letter
        tokens (e.g. ``x = 1``, ``D``, ``F``, ``DJ``).
    :param similarity_threshold: Word Jaccard near-dedup; ``0`` disables.
    :param similarity_min_chars: Skip pairwise similarity when either
        sentence has fewer non-space characters than this value.
    """
    if min_words > max_words:
        print("--min-words cannot exceed --max-words", file=sys.stderr)
        raise SystemExit(1)

    best_by_norm: dict[str, str] = {}
    skipped_symbolic = 0
    with open_text(source) as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line:
                continue
            parts = line.split("\t", 2)
            if len(parts) < 3:
                continue
            _id, lang, text = parts[0], parts[1], parts[2]
            if lang != "eng":
                continue
            if exclude_symbolic and _looks_symbolic_or_single_letterish(text):
                skipped_symbolic += 1
                continue
            key = latin_text_dedup_key(text)
            if not key:
                continue
            prev = best_by_norm.get(key)
            if prev is None:
                best_by_norm[key] = text
            else:
                best_by_norm[key] = better_candidate(prev, text)

    unique = list(best_by_norm.values())
    n_unique = len(unique)

    if legacy:
        unique.sort(key=complexity_key)
        if similarity_threshold > 0:
            out_lines = greedy_near_dedup(
                unique,
                target=max(0, count),
                threshold=similarity_threshold,
                min_chars=similarity_min_chars,
            )
        else:
            out_lines = unique[: max(0, count)]
        mode = "legacy shortest / near-dedup"
    else:
        out_lines = _select_linear_word_progression(
            unique,
            count=max(0, count),
            min_words=min_words,
            max_words=max_words,
            similarity_threshold=similarity_threshold,
            similarity_min_chars=similarity_min_chars,
        )
        mode = (
            f"linear {min_words}–{max_words} words"
            + (
                f"; near-dedup on (threshold={similarity_threshold})"
                if similarity_threshold > 0
                else "; near-dedup off"
            )
        )

    destination.parent.makedirs_p()
    destination.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    msg = (
        f"Wrote {len(out_lines)} lines to {destination} "
        f"(from {n_unique} unique keys; {mode})"
    )
    if exclude_symbolic:
        msg += f"; filtered_symbolic={skipped_symbolic}"
    if not legacy and len(out_lines) < count:
        msg += (
            f" — only {len(out_lines)} of {count} filled "
            "(pool or similarity gate exhausted)"
        )
    print(msg)


@app.command
def translate_tibetan(
    *,
    source: Path = _ARCHIVE / "english_simple_1000.txt",
    destination: Path = _ARCHIVE / "english_tibetan_pairs_1000.tsv",
    checkpoint_json: Path = _ARCHIVE / "english_tibetan_pairs_1000.progress.json",
    service: str | None = None,
    batch_size: int = 20,
    batch_retries: int = 2,
    quiet: bool = False,
) -> None:
    """Translate English lines (one per line) to Tibetan via microeval LLM.

    Set ``LLM_SERVICE`` (e.g. ``openai``, ``groq``, ``ollama``, ``bedrock``)
    and the matching API credentials. Optional ``anki-tools/.env`` is loaded
    from the project root.

    Progress (batch index, line range, timing) is printed to stderr unless
    ``--quiet`` is set.

    :param source: Input file with one English sentence per line (default ``english_simple_1000.txt``).
    :param destination: Output TSV with columns ``english`` and ``tibetan``.
    :param checkpoint_json: JSON checkpoint file for intermediate/resume state.
    :param service: LLM backend; defaults to ``LLM_SERVICE`` or ``openai``.
    :param batch_size: Lines per LLM request (smaller if you hit token limits).
    :param batch_retries: Retries per batch for parse/count-mismatch failures.
    :param quiet: Disable per-batch progress on stderr.
    """
    svc = service or os.environ.get("LLM_SERVICE", "openai")
    text = source.read_text(encoding="utf-8", errors="replace")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        _log_translate(f"no non-empty lines in {source}", style="red")
        raise SystemExit(1)
    try:
        tibetan = translate_lines_to_tibetan_sync(
            lines,
            service=svc,
            batch_size=batch_size,
            batch_retries=batch_retries,
            verbose=not quiet,
            checkpoint_path=checkpoint_json,
        )
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        detail = str(e).strip() or repr(e)
        _log_translate(detail, style="red")
        _log_translate(
            "Set LLM_SERVICE and API keys (or use anki-tools/.env). "
            "See microeval / bhtool README for providers.",
            style="yellow",
        )
        raise SystemExit(1) from e
    rows = "\n".join(
        f"{en}\t{bo}"
        for en, bo in zip(lines, tibetan, strict=True)
    )
    destination.parent.makedirs_p()
    destination.write_text(rows + ("\n" if rows else ""), encoding="utf-8")
    print(f"Wrote {len(tibetan)} rows to {destination} (service={svc})")


def _parse_en_bo_tsv_rows(path: Path) -> list[tuple[str, str]]:
    """Return (english, tibetan) pairs from a two-column tab-separated file."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    rows: list[tuple[str, str]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        en, bo = parts[0].strip(), parts[1].strip()
        if not en or not bo:
            continue
        rows.append((en, bo))
    return rows


@app.command
def simple_anki(
    *,
    source: Path = _ARCHIVE / "english_tibetan_pairs_1000.tsv",
    destination: Path = _ROOT / "apkg" / "english_tibetan_listening_1000.apkg",
    audio_dir: Path = _ROOT / "audio" / "english_tibetan_1000",
    deck_name: str = "Simple English → Tibetan (listening)",
    delay: float = 0.5,
    gtts_lang: str = "en",
    gtts_tld: str = "com",
    quiet: bool = False,
) -> None:
    """Build an Anki deck: English **gTTS** audio on the front; back shows audio + text.

    Expects ``english_tibetan_pairs_1000.tsv`` (``translate-tibetan`` output):
    each line is English TAB Tibetan. Only
    the English line is sent to ``gTTS``; Tibetan appears as text on the back.

    :param source: Two-column TSV: English TAB Tibetan.
    :param destination: Output ``.apkg`` path.
    :param audio_dir: Directory for ``en_NNNNN.mp3`` (reused if present).
    :param deck_name: Deck title inside Anki.
    :param delay: Seconds between gTTS requests.
    :param gtts_lang: ``gTTS`` ``lang`` (default ``en``).
    :param gtts_tld: ``gTTS`` ``tld`` (default ``com``).
    :param quiet: Disable progress on stderr.
    """
    from gtts import gTTS

    rows = _parse_en_bo_tsv_rows(source)
    if not rows:
        print(f"No valid English–Tibetan rows in {source}", file=sys.stderr)
        raise SystemExit(1)

    audio_dir.makedirs_p()
    destination.parent.makedirs_p()

    if not quiet:
        print(
            f"simple-anki: {len(rows)} card(s); gTTS lang={gtts_lang!r} tld={gtts_tld!r}",
            file=sys.stderr,
            flush=True,
        )

    media_files: list[str] = []
    cards: list[dict[str, str]] = []

    for i, (en, bo) in enumerate(rows, start=1):
        fname = f"en_{i:05d}.mp3"
        out_mp3 = audio_dir / fname
        if not out_mp3.exists():
            try:
                tts = gTTS(text=en, lang=gtts_lang, tld=gtts_tld, slow=False)
                tts.save(str(out_mp3))
            except Exception as e:
                print(f"Row {i}: gTTS failed: {e}", file=sys.stderr)
                raise SystemExit(1) from e
            if not quiet:
                print(f"[{i}/{len(rows)}] wrote {out_mp3.name}", file=sys.stderr, flush=True)
            if delay > 0 and i < len(rows):
                time.sleep(delay)
        elif not quiet:
            print(f"[{i}/{len(rows)}] skip (exists): {out_mp3.name}", file=sys.stderr, flush=True)

        sound = f"[sound:{fname}]"
        media_files.append(os.path.abspath(str(out_mp3)))
        cards.append({"audio": sound, "english": en, "tibetan": bo})

    css_path = _ROOT / "templates" / "card.css"
    css_content = css_path.read_text(encoding="utf-8") if css_path.exists() else None

    AnkiPackage.create(
        destination,
        deck_name,
        fields=["audio", "english", "tibetan"],
        cards=cards,
        media_files=media_files,
        model_name="English audio + EN/BO text",
        question_format="{{audio}}",
        answer_format=(
            "{{FrontSide}}<hr id=answer>"
            "<div>{{english}}</div>"
            "<div style='font-size:2rem;line-height:1.4;margin-top:0.75em'>{{tibetan}}</div>"
        ),
        css=css_content,
        sort_field_index=1,
    )
    print(f"Wrote {destination} ({len(cards)} notes, audio in {audio_dir})")


if __name__ == "__main__":
    app()
