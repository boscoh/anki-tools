"""Build one HTML page that shows the real cards of an Anki package.

Run ``uv run anki preview --apkg deck.apkg``. The builder reads the note
fields, the card templates, and the model CSS of the package. It extracts the
media files that the cards use, so the audio controls and the images work in
the browser.
"""

from __future__ import annotations

import html
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, unquote

from anki_tools.apkg.package import AnkiPackage
from anki_tools.preview import (
    PreviewCard,
    PreviewSection,
    SOUND_PATTERN,
    build_page,
    render_template,
)

DEFAULT_OUTPUT_PATH = Path("preview/cards.html")

MEDIA_ATTRIBUTE_PATTERN = re.compile(
    r"(\b(?:src|href|poster)\s*=\s*[\"'])([^\"']+)([\"'])", re.IGNORECASE
)
CSS_URL_PATTERN = re.compile(r"url\(\s*([\"']?)([^)'\"]+)\1\s*\)", re.IGNORECASE)
SCHEME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
TAG_PATTERN = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class ApkgPreviewResult:
    """Result of one package preview build.

    :param output_path: HTML file that the builder wrote.
    :param cards: Number of cards on the page.
    :param sections: Number of sections on the page.
    :param media: Number of media files that the builder extracted.
    :param total: Number of cards in the package before the filters.
    """

    output_path: Path
    cards: int
    sections: int
    media: int
    total: int


class MediaCollector:
    """Map a media file name to a relative URL and record the files that cards use.

    :param mapping: Media mapping from :meth:`AnkiPackage.get_media_mapping`.
    :param media_prefix: Relative URL of the media directory. Use None to keep
        the ``[sound:...]`` placeholders.
    """

    def __init__(self, mapping: dict[str, str], media_prefix: str | None) -> None:
        self._ids = {filename: file_id for file_id, filename in mapping.items()}
        self._media_prefix = media_prefix
        self.needed: set[str] = set()

    def url(self, name: str) -> str | None:
        """Give the relative URL of one media file.

        :param name: File name as the note field writes it.
        :returns: URL of the file, or None when the package has no such file.
        """
        if self._media_prefix is None:
            return None
        name = unquote(name)
        if name not in self._ids:
            return None
        self.needed.add(name)
        relative = quote(name)
        if not self._media_prefix:
            return relative
        return f"{self._media_prefix}/{relative}"

    def rewrite(self, text: str) -> str:
        """Replace the sound tags and the media links in one card side.

        :param text: Rendered card HTML.
        :returns: HTML that points at the extracted media files.
        """

        def replace_sound(match: re.Match[str]) -> str:
            name = match.group(1)
            url = self.url(name)
            if url is None:
                escaped = html.escape(name, quote=True)
                return (
                    '<span class="preview-audio" title="' + escaped + '">'
                    '<span class="preview-play">&#9654;</span>'
                    f'<span class="preview-file">{escaped}</span></span>'
                )
            return (
                '<audio class="preview-sound" controls preload="none" '
                f'src="{html.escape(url, quote=True)}"></audio>'
            )

        text = SOUND_PATTERN.sub(replace_sound, text)
        return MEDIA_ATTRIBUTE_PATTERN.sub(self._rewrite_attribute, text)

    def _rewrite_attribute(self, match: re.Match[str]) -> str:
        value = match.group(2)
        if not value or value.startswith(("#", "//")) or SCHEME_PATTERN.match(value):
            return match.group(0)
        url = self.url(Path(unquote(value)).name)
        if url is None:
            return match.group(0)
        return f"{match.group(1)}{html.escape(url, quote=True)}{match.group(3)}"

    def rewrite_css(self, css: str) -> str:
        """Replace the media links in model CSS.

        :param css: Model CSS text.
        :returns: CSS that points at the extracted media files.
        """

        def replace_url(match: re.Match[str]) -> str:
            value = match.group(2).strip()
            if value.startswith("data:") or SCHEME_PATTERN.match(value):
                return match.group(0)
            url = self.url(Path(unquote(value)).name)
            if url is None:
                return match.group(0)
            return f'url("{url}")'

        return CSS_URL_PATTERN.sub(replace_url, css)


def _select_cards(
    package: AnkiPackage,
    models: dict,
    decks: dict,
    *,
    limit: int,
    deck: str | None,
    search: str | None,
) -> list:
    """Select the cards to show.

    :param package: Open package.
    :param models: Models from :meth:`AnkiPackage.get_models`.
    :param decks: Decks from :meth:`AnkiPackage.get_decks`.
    :param limit: Maximum card count. Zero shows every card.
    :param deck: Deck name must hold this text.
    :param search: One note field must hold this text.
    :returns: Selected card rows.
    """
    selected = []
    for card in package.get_cards():
        parsed = package.parse_card(card, models, decks)
        if deck and deck.lower() not in parsed["deck"].lower():
            continue
        if search:
            text = TAG_PATTERN.sub(
                " ", " ".join(str(v) for v in parsed["fields"].values())
            )
            if search.lower() not in text.lower():
                continue
        selected.append(card)
        if limit and len(selected) >= limit:
            break
    return selected


def _preview_card(
    card,
    models: dict,
    decks: dict,
    collector: MediaCollector,
) -> tuple[str, PreviewCard]:
    """Build one preview card.

    :param card: Card row from :meth:`AnkiPackage.get_cards`.
    :param models: Models from :meth:`AnkiPackage.get_models`.
    :param decks: Decks from :meth:`AnkiPackage.get_decks`.
    :param collector: Media URL resolver.
    :returns: Model id and rendered preview card.
    """
    model_id = str(card["mid"])
    model = models.get(model_id, {})
    templates = model.get("tmpls", [])
    template_index = int(card["ord"])
    if not 0 <= template_index < len(templates):
        template_index = 0
    template = templates[template_index] if templates else {}

    field_names = [field["name"] for field in model.get("flds", [])]
    values = card["flds"].split("\x1f")
    fields = dict(zip(field_names, values))

    front_source = render_template(template.get("qfmt", ""), fields)
    back_source = render_template(
        template.get("afmt", ""), fields, front_side=front_source
    )
    front = collector.rewrite(front_source)
    back = collector.rewrite(back_source)

    deck_name = decks.get(str(card["did"]), {}).get("name", "Unknown")
    template_name = template.get("name", "Card")
    tags = str(card["tags"] or "").strip()
    note = f"note {card['nid']} · card {card['id']}"
    if tags:
        note += f" · tags: {tags}"

    preview = PreviewCard(
        title=f"{deck_name} · {template_name}",
        front=front,
        back=back,
        css=collector.rewrite_css(model.get("css", "")),
        source=f"model {model.get('name', 'Unknown')} · template {template_name}",
        fields=", ".join(field_names),
        note=note,
    )
    return model_id, preview


def _build_sections(
    models: dict,
    decks: dict,
    cards: list,
    collector: MediaCollector,
) -> tuple[PreviewSection, ...]:
    """Group the selected cards by note model.

    :param models: Models from :meth:`AnkiPackage.get_models`.
    :param decks: Decks from :meth:`AnkiPackage.get_decks`.
    :param cards: Selected card rows.
    :param collector: Media URL resolver.
    :returns: One section for each model, in order of first use.
    """
    groups: dict[str, list[PreviewCard]] = {}
    names: dict[str, str] = {}
    for card in cards:
        model_id, preview = _preview_card(card, models, decks, collector)
        groups.setdefault(model_id, []).append(preview)
        names[model_id] = models.get(model_id, {}).get("name", "Unknown")

    sections = []
    for index, (model_id, previews) in enumerate(groups.items(), start=1):
        model = models.get(model_id, {})
        fields = ", ".join(field["name"] for field in model.get("flds", []))
        count = len(previews)
        label = "card" if count == 1 else "cards"
        description = f"{count} {label}. Fields: <code>{html.escape(fields)}</code>."
        sections.append(
            PreviewSection(
                anchor=f"model-{index}",
                title=names[model_id],
                description=description,
                cards=tuple(previews),
            )
        )
    return tuple(sections)


def _media_directory(
    output_path: Path, media_dir: str | Path | None
) -> tuple[Path, str]:
    """Resolve the media directory and its URL relative to the page.

    :param output_path: HTML file that the builder writes.
    :param media_dir: Requested directory, or None for a default.
    :returns: Absolute media directory and the relative URL prefix.
    """
    directory = (
        Path(media_dir)
        if media_dir is not None
        else (output_path.parent / f"{output_path.stem}_media")
    )
    relative = Path(os.path.relpath(directory, output_path.parent)).as_posix()
    if relative == ".":
        return directory, ""
    return directory, relative


def _extract_media(
    package: AnkiPackage, collector: MediaCollector, media_dir: Path
) -> int:
    """Copy the media files that the cards use into the media directory.

    :param package: Open package.
    :param collector: Media URL resolver with the needed file names.
    :param media_dir: Destination directory.
    :returns: Number of copied files.
    """
    if not collector.needed or package.temp_dir is None:
        return 0
    mapping = {
        filename: file_id for file_id, filename in package.get_media_mapping().items()
    }
    media_dir.mkdir(parents=True, exist_ok=True)
    root = media_dir.resolve()
    copied = 0
    for name in sorted(collector.needed):
        source = Path(package.temp_dir) / mapping[name]
        if not source.is_file():
            continue
        target = (media_dir / name).resolve()
        if not target.is_relative_to(root):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied += 1
    return copied


def write_apkg_preview(
    apkg_path: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    limit: int = 50,
    deck: str | None = None,
    search: str | None = None,
    extract_media: bool = True,
    media_dir: str | Path | None = None,
) -> ApkgPreviewResult:
    """Build a preview page for the cards of one package.

    :param apkg_path: Package file to read.
    :param output_path: HTML file to write.
    :param limit: Maximum card count. Zero shows every card.
    :param deck: Only show cards of a deck that holds this text.
    :param search: Only show cards with a field that holds this text.
    :param extract_media: Copy the used media files and link them in the page.
    :param media_dir: Directory for the media files. Defaults to a sibling
        directory named after the HTML file.
    :returns: Counts of the build.
    """
    apkg_path = Path(apkg_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with AnkiPackage(apkg_path) as package:
        models = package.get_models(include_templates=True)
        decks = package.get_decks()
        selected = _select_cards(
            package, models, decks, limit=limit, deck=deck, search=search
        )
        total = len(package.get_cards())

        prefix = None
        destination = None
        if extract_media:
            destination, prefix = _media_directory(output_path, media_dir)
        collector = MediaCollector(package.get_media_mapping(), prefix)
        sections = _build_sections(models, decks, selected, collector)
        media = 0
        if destination is not None:
            media = _extract_media(package, collector, destination)

    shown = sum(len(section.cards) for section in sections)
    title = f"Anki card previews · {apkg_path.name}"
    lead = (
        f"The cards of <code>{html.escape(apkg_path.name)}</code>. "
        "The page reads the note fields and the card templates of the package."
    )
    if shown != total:
        lead += f" Showing {shown} of {total} cards."
    if extract_media:
        lead += " The audio controls play the extracted media files."
    else:
        lead += " The audio control is a placeholder and does not play."
    source_note = (
        "Generated by <code>uv run anki preview --apkg "
        f"{html.escape(apkg_path.name)}</code>."
    )
    page = build_page(sections, title=title, lead=lead, source_note=source_note)
    output_path.write_text(page, encoding="utf-8")
    return ApkgPreviewResult(
        output_path=output_path,
        cards=shown,
        sections=len(sections),
        media=media,
        total=total,
    )
