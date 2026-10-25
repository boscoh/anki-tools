"""Build one HTML page that shows every card template in the repository.

Run ``uv run anki preview`` after a template edit. The builder reads the real
template and CSS files, fills each one with sample data, and writes
``preview/cards.html``. Open that file in a browser.
"""

from __future__ import annotations

import csv
import html
import re
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping

from anki_tools.apkg.templates import (
    DEFAULT_TEMPLATE_DIR,
    SHARED_CSS_PATH,
    TEMPLATES_DIR,
    load_char_links,
    load_css_layers,
    load_text,
)
from anki_tools.languages import (
    LANGUAGES,
    profile_languages,
    swadesh_languages,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "preview" / "cards.html"
VOCAB_DIR = ROOT / "vocab"
DEFAULT_TEMPLATE = DEFAULT_TEMPLATE_DIR
SHARED_SCRIPT = load_char_links()

SECTION_PATTERN = re.compile(
    r"\{\{([#^])\s*([^{}]+?)\s*\}\}(.*?)\{\{/\s*\2\s*\}\}", re.DOTALL
)
FIELD_PATTERN = re.compile(r"\{\{\{?\s*([^{}#/][^{}]*?)\s*\}\}\}?")
SOUND_PATTERN = re.compile(r"\[sound:([^\]]+)\]")
PARENTHETICAL_PATTERN = re.compile(r"\s*\([^)]*\)")

SENTENCE_LANGUAGES: tuple[tuple[str, str], ...] = tuple(
    (language.name, language.label) for language in profile_languages()
)

VOCAB_LANGUAGES: tuple[tuple[str, str], ...] = tuple(
    (name, LANGUAGES[name].label) for name in swadesh_languages()
)

CJK_FIELDS = {"characters", "hanzi", "cantonese", "mandarin", "chinese"}
SCRIPT_15EM_FIELDS = {"arabic", "hindi"}

PREVIEW_CARD_CSS = """
body {
    margin: 0;
    padding: 0;
    background: #ffffff;
    color: #111111;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.card {
    box-sizing: border-box;
    padding: 14px;
    overflow-wrap: anywhere;
}
.preview-audio {
    display: inline-flex;
    align-items: center;
    gap: 0.35em;
    max-width: 100%;
    padding: 0.25em 0.6em;
    border: 1px solid #d6dce5;
    border-radius: 999px;
    background: #eef2f7;
    color: #243447;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 0.55em;
    line-height: 1.2;
    vertical-align: middle;
}
.preview-play {
    color: #0b63c5;
    font-size: 1.1em;
}
.preview-file {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
hr#answer {
    margin: 0.8em 0;
    border: 0;
    border-top: 1px solid #d9dee6;
}
"""

PAGE_CSS = """
:root {
    color-scheme: light;
    --ink: #142033;
    --muted: #5c6b80;
    --line: #d9e0ea;
    --panel: #f6f8fb;
    --accent: #0b63c5;
}
* {
    box-sizing: border-box;
}
body {
    margin: 0;
    padding: 32px 20px 64px;
    background: #eef2f7;
    color: var(--ink);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    line-height: 1.5;
}
main {
    max-width: 1240px;
    margin: 0 auto;
}
header {
    margin-bottom: 24px;
}
h1 {
    margin: 0 0 8px;
    font-size: 30px;
}
h2 {
    margin: 42px 0 8px;
    padding-top: 8px;
    border-top: 1px solid var(--line);
    font-size: 22px;
}
h3 {
    margin: 0 0 4px;
    font-size: 16px;
}
p {
    margin: 0 0 10px;
}
code {
    padding: 1px 4px;
    border: 1px solid var(--line);
    border-radius: 4px;
    background: #ffffff;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.9em;
}
.lead {
    max-width: 900px;
    color: var(--muted);
}
.toc {
    display: flex;
    flex-wrap: wrap;
    gap: 8px 16px;
    margin: 16px 0 8px;
    padding: 12px 16px;
    border: 1px solid var(--line);
    border-radius: 8px;
    background: #ffffff;
}
.toc a {
    color: var(--accent);
    text-decoration: none;
}
.section-lead {
    max-width: 900px;
    color: var(--muted);
}
.card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(560px, 1fr));
    gap: 16px;
    margin-top: 14px;
}
.card-panel {
    min-width: 0;
    padding: 14px;
    border: 1px solid var(--line);
    border-radius: 10px;
    background: #ffffff;
    box-shadow: 0 1px 2px rgb(20 32 51 / 5%);
}
.sides {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-top: 10px;
}
.side {
    min-width: 0;
}
.side-label {
    display: inline-block;
    margin-bottom: 4px;
    color: var(--muted);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
iframe.card-preview {
    display: block;
    width: 100%;
    min-height: 90px;
    border: 1px solid var(--line);
    border-radius: 6px;
    background: #ffffff;
}
.meta {
    margin: 8px 0 0;
    color: var(--muted);
    font-size: 12px;
}
.meta-key {
    color: var(--ink);
    font-weight: 700;
}
.note {
    margin-top: 8px;
    color: var(--muted);
    font-size: 12px;
}
@media (max-width: 700px) {
    body {
        padding: 20px 12px 48px;
    }
    .card-grid {
        grid-template-columns: 1fr;
    }
    .sides {
        grid-template-columns: 1fr;
    }
}
@media print {
    body {
        background: #ffffff;
    }
    .card-panel {
        break-inside: avoid;
        box-shadow: none;
    }
}
"""

PAGE_SCRIPT = """
function fitCard(frame) {
    try {
        var doc = frame.contentDocument;
        if (!doc || !doc.documentElement) {
            return;
        }
        var bodyHeight = doc.body ? doc.body.getBoundingClientRect().height : 0;
        var height = bodyHeight || doc.documentElement.scrollHeight;
        frame.style.height = Math.ceil(height + 2) + "px";
    } catch (error) {
        // The frame can be unavailable while the page loads.
    }
}
function fitAllCards() {
    var frames = document.querySelectorAll("iframe.card-preview");
    for (var index = 0; index < frames.length; index++) {
        fitCard(frames[index]);
    }
}
window.addEventListener("load", fitAllCards);
window.addEventListener("resize", fitAllCards);
"""


@dataclass(frozen=True)
class PreviewCard:
    """One card template with rendered sample content.

    :param title: Card name shown above the preview.
    :param front: Rendered front HTML.
    :param back: Rendered back HTML.
    :param css: Card CSS that applies to the front and the back.
    :param source: Template files that make the card.
    :param fields: Note fields that the template reads.
    :param note: Extra text shown below the preview.
    """

    title: str
    front: str
    back: str
    css: str
    source: str
    fields: str
    note: str = ""


@dataclass(frozen=True)
class PreviewSection:
    """A group of cards that share one template family.

    :param anchor: HTML id of the section.
    :param title: Section heading.
    :param description: Section introduction.
    :param cards: Cards to show in the section.
    """

    anchor: str
    title: str
    description: str
    cards: tuple[PreviewCard, ...]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file with a header row.

    :param path: CSV file to read.
    :returns: One mapping for each data row.
    """
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_csv_rows_or_empty(path: Path) -> list[dict[str, str]]:
    """Read a CSV file, or give no rows when the file is absent.

    :param path: CSV file to read.
    :returns: One mapping for each data row.
    """
    if not path.exists():
        return []
    return read_csv_rows(path)


def audio_name(text: str) -> str:
    """Build the audio file name that the vocabulary builder uses.

    :param text: Target-language form of the entry.
    :returns: MP3 file name with the spaces replaced by underscores.
    """
    cleaned = PARENTHETICAL_PATTERN.sub("", text).split("/")[0].strip()
    return f"{cleaned.replace(' ', '_')}.mp3"


def render_template(
    template: str, fields: Mapping[str, str], front_side: str = ""
) -> str:
    """Render the mustache tags that an Anki card template uses.

    :param template: Card template text.
    :param fields: Note field values.
    :param front_side: Rendered front HTML for the ``FrontSide`` tag.
    :returns: Rendered HTML.
    """
    result = template
    while True:
        match = SECTION_PATTERN.search(result)
        if match is None:
            break
        marker = match.group(1)
        name = match.group(2).strip()
        body = match.group(3)
        value = fields.get(name, "")
        show = bool(value) == (marker == "#")
        replacement = render_template(body, fields, front_side) if show else ""
        result = f"{result[: match.start()]}{replacement}{result[match.end() :]}"

    def replace_field(match: re.Match[str]) -> str:
        name = match.group(1).strip()
        if name.startswith("!"):
            return ""
        if name == "FrontSide":
            return front_side
        return fields.get(name.split(":")[-1].strip(), "")

    return FIELD_PATTERN.sub(replace_field, result)


def replace_audio(text: str) -> str:
    """Replace each ``[sound:...]`` tag with a visible preview control.

    :param text: Rendered card HTML.
    :returns: HTML with the audio placeholders converted.
    """

    def replace_sound(match: re.Match[str]) -> str:
        name = html.escape(match.group(1), quote=True)
        return (
            '<span class="preview-audio" title="' + name + '">'
            '<span class="preview-play">&#9654;</span>'
            f'<span class="preview-file">{name}</span></span>'
        )

    return SOUND_PATTERN.sub(replace_sound, text)


def make_card(
    title: str,
    front_template: str,
    back_template: str,
    fields: Mapping[str, str],
    css: str,
    source: str,
    field_list: str,
    note: str = "",
) -> PreviewCard:
    """Render one front and one back from a pair of templates.

    :param title: Card name shown above the preview.
    :param front_template: Front template text.
    :param back_template: Back template text.
    :param fields: Note field values.
    :param css: Card CSS.
    :param source: Template file description.
    :param field_list: Note field names.
    :param note: Extra text shown below the preview.
    :returns: Rendered preview card.
    """
    front = render_template(front_template, fields)
    back = render_template(back_template, fields, front_side=front)
    return PreviewCard(
        title=title,
        front=replace_audio(front),
        back=replace_audio(back),
        css=css,
        source=source,
        fields=field_list,
        note=note,
    )


def find_word(language: str, target: str) -> dict[str, str] | None:
    """Find the reviewed word row of a sentence target.

    :param language: Language key, for example ``french``.
    :param target: Target-language form.
    :returns: Word table row, or None when the table has no row.
    """
    path = VOCAB_DIR / f"english-{language}.csv"
    for row in read_csv_rows_or_empty(path):
        if row.get(language) == target:
            return row
    return None


def gloss_for(language: str, row: dict[str, str]) -> str:
    """Build the target gloss of one sentence card.

    :param language: Language key, for example ``french``.
    :param row: Sentence CSV row.
    :returns: Gloss text with the reading or the gender where the pack adds it.
    """
    target = row["target"]
    word = find_word(language, target)
    label = word["english"] if word and word.get("english") else target
    if language == "cantonese":
        reading = word.get("transliteration", "") if word else ""
        return f"{label} · {reading}" if reading else label
    return label


def sentence_audio_name(row: dict[str, str]) -> str:
    """Build a sample sentence audio file name from a sentence CSV row.

    :param row: Sentence CSV row.
    :returns: MP3 file name with the source prefix.
    """
    if row.get("audio_source") == "native":
        return f"ta_{row.get('audio_id') or row['tatoeba_id']}.mp3"
    return f"tts_{row['tatoeba_id']}.mp3"


def sentence_card(language: str, title: str, row: dict[str, str]) -> PreviewCard:
    """Build one Swadesh sentence pack card.

    :param language: Language key, for example ``greek``.
    :param title: Card name shown above the preview.
    :param row: Sentence CSV row.
    :returns: Rendered preview card.
    """
    directory = TEMPLATES_DIR / "sentence_pack" / language
    front_template = load_text(directory / "front.html")
    back_template = load_text(directory / "back.html")
    if language in {"mandarin", "cantonese"} and SHARED_SCRIPT:
        back_template = f"{back_template}\n{SHARED_SCRIPT}"
    css = load_css_layers(
        SHARED_CSS_PATH,
        directory / "card.css",
    )
    target = row["target"]
    sentence = row["sentence"].replace(target, f"<b>{target}</b>", 1)
    fields = {
        "Audio": f"[sound:{sentence_audio_name(row)}]",
        "Sentence": sentence,
    }
    if language == "mandarin":
        fields["Pinyin"] = row.get("pinyin", "")
    if language == "cantonese":
        fields["Jyutping"] = row.get("jyutping", "")
    fields.update(
        {
            "English": row["english"],
            "Target": target,
            "Gloss": gloss_for(language, row),
            "WordAudio": f"[sound:word_{target}.mp3]",
            "Source": f"Tatoeba #{row['tatoeba_id']} · {row['tatoeba_author']}",
        }
    )
    field_list = ", ".join(fields)
    return make_card(
        title,
        front_template,
        back_template,
        fields,
        css,
        source=f"templates/sentence_pack/{language}/{{front,back,card}}",
        field_list=field_list,
        note="The target is bold in the sentence. The source row is from the pack CSV.",
    )


def default_card() -> PreviewCard:
    """Build the default deck style card.

    :returns: Rendered preview card.
    """
    front_template = load_text(DEFAULT_TEMPLATE / "front.html")
    back_template = load_text(DEFAULT_TEMPLATE / "back.html")
    if SHARED_SCRIPT and "char-links" in back_template:
        back_template = f"{back_template.rstrip()}\n{SHARED_SCRIPT}"
    css = load_css_layers(SHARED_CSS_PATH, DEFAULT_TEMPLATE / "card.css")
    fields = {
        "[a]": "[sound:wo_jintian_hen_mang.mp3]",
        "Sentence (Annotated)": "我 今天 很 忙。",
        "Sentence (Latin)": "wǒ jīn tiān hěn máng",
        "Sentence (Translation)": "I am very busy today.",
    }
    return make_card(
        "Chinese sentence deck",
        front_template,
        back_template,
        fields,
        css,
        source="templates/default/{front,back,card}",
        field_list="[a], Sentence (Annotated), Sentence (Latin), Sentence (Translation)",
        note="The style command maps these field names to the fields of the input deck.",
    )


def french_cull_card() -> PreviewCard:
    """Build the French 1000-card cull card.

    :returns: Rendered preview card.
    """
    directory = TEMPLATES_DIR / "french_cull"
    css = load_css_layers(
        SHARED_CSS_PATH,
        TEMPLATES_DIR / "sentence_pack" / "french" / "card.css",
    )
    fields = {
        "sound": "[sound:6c1f7a.mp3]",
        "French": "Je pensais qu'ils ne voudraient pas venir.",
        "Back": "I thought they wouldn't want to come.",
    }
    return make_card(
        "French 1000-card cull",
        load_text(directory / "front.html"),
        load_text(directory / "back.html"),
        fields,
        css,
        source="templates/french_cull/{front,back}",
        field_list="sound, French, Back",
        note="The deck keeps the model and the audio of the input deck.",
    )


def french_core_card() -> PreviewCard:
    """Build the French core-word deck card.

    :returns: Rendered preview card.
    """
    directory = TEMPLATES_DIR / "french_core"
    css = load_css_layers(
        SHARED_CSS_PATH,
        TEMPLATES_DIR / "sentence_pack" / "french" / "card.css",
    )
    fields = {
        "sound": "[sound:8f20b1.mp3]",
        "French": "Je ne t'en demanderai pas plus pour aujourd'hui.",
        "Back": "I won't ask you anything else today.",
        "Core": "ne",
        "CoreGloss": "not",
    }
    return make_card(
        "French core-word deck",
        load_text(directory / "front.html"),
        load_text(directory / "back.html"),
        fields,
        css,
        source="templates/french_core/{front,back}",
        field_list="sound, French, Back, Core, CoreGloss",
        note="The Core field is empty on a card with no assigned target.",
    )


def content_field_order(csv_fields: list[str], audio_field_name: str) -> list[str]:
    """Order the vocabulary fields as the builder does.

    :param csv_fields: All CSV column names.
    :param audio_field_name: Column of the target-language form.
    :returns: Target form, transliteration, and English gloss.
    """
    target = next(
        (field for field in csv_fields if field.lower() == audio_field_name.lower()),
        audio_field_name,
    )
    rest = [field for field in csv_fields if field != target]
    result = [target]
    for name in ("transliteration", "english"):
        found = next((field for field in rest if field.lower() == name), None)
        if found:
            result.append(found)
    return result


def vocab_card(language: str, title: str) -> PreviewCard:
    """Build one Swadesh vocabulary deck card.

    :param language: Language key, for example ``arabic``.
    :param title: Card name shown above the preview.
    :returns: Rendered preview card.
    """
    path = VOCAB_DIR / f"english-{language}.csv"
    rows = read_csv_rows(path)
    with path.open(encoding="utf-8", newline="") as handle:
        csv_fields = list(csv.DictReader(handle).fieldnames or ())
    row = next(
        (entry for entry in rows if entry.get("english", "").lower() == "water"),
        rows[0],
    )
    content_order = content_field_order(csv_fields, language)
    field_parts: list[str] = []
    for field in content_order:
        value = row.get(field, "")
        if field.lower() in CJK_FIELDS:
            field_parts.append(
                f"<div class='char-links' style='font-size: 3rem'>{value}</div>"
            )
        elif field.lower() in SCRIPT_15EM_FIELDS:
            field_parts.append(f"<div style='font-size: 1.5em'>{value}</div>")
        else:
            field_parts.append(f"<div style='font-size: 2rem'>{value}</div>")
    if SHARED_SCRIPT:
        field_parts.append(SHARED_SCRIPT)
    back_template = load_text(TEMPLATES_DIR / "vocab" / "back.html").replace(
        "{{fields}}", "\n".join(field_parts)
    )
    front_template = load_text(TEMPLATES_DIR / "vocab" / "front.html")
    language_css = TEMPLATES_DIR / "vocab" / language / "card.css"
    css = load_css_layers(
        SHARED_CSS_PATH,
        TEMPLATES_DIR / "vocab" / "card.css",
        language_css,
    )
    fields = {
        "audio": f"[sound:{audio_name(row[language])}]",
    }
    fields.update({field: row.get(field, "") for field in content_order})
    note = (
        "The card holds only the target form, the transliteration, and the "
        "English gloss."
    )
    if language in {"mandarin", "cantonese"}:
        note += " The shared script links each Han character."
    return make_card(
        title,
        front_template,
        back_template,
        fields,
        css,
        source="templates/vocab/{front,back,card}"
        + (f" + templates/vocab/{language}/card.css" if language_css.exists() else ""),
        field_list=", ".join(["audio", *content_order]),
        note=note,
    )


def build_sections() -> tuple[PreviewSection, ...]:
    """Build all preview sections from the template files.

    :returns: Sections in display order.
    """
    sentence_cards = []
    for language, title in SENTENCE_LANGUAGES:
        rows = read_csv_rows_or_empty(VOCAB_DIR / f"{language}_sentences.csv")
        if rows:
            sentence_cards.append(
                sentence_card(language, f"{title} sentence", rows[0])
            )

    vocab_cards = tuple(
        vocab_card(language, f"{title} vocabulary")
        for language, title in VOCAB_LANGUAGES
        if (VOCAB_DIR / f"english-{language}.csv").exists()
    )

    return (
        PreviewSection(
            anchor="default",
            title="Default deck style",
            description=(
                "The <code>anki style</code> command and <code>anki fix zh</code> "
                "use this pair. The back holds the annotated sentence, its Latin "
                "reading, and the translation."
            ),
            cards=(default_card(),),
        ),
        PreviewSection(
            anchor="sentence-packs",
            title="Swadesh sentence packs",
            description=(
                "Each language has its own front, back, and card CSS. The shared "
                "sheet comes first. Mandarin and Cantonese add a romanization "
                "line. The other packs use the same sentence block."
            ),
            cards=tuple(sentence_cards),
        ),
        PreviewSection(
            anchor="french-cull",
            title="French 1000-card cull",
            description=(
                "The cull keeps the note model and the audio of the input deck. "
                "It replaces the templates with this one-card pair."
            ),
            cards=(french_cull_card(),),
        ),
        PreviewSection(
            anchor="french-core",
            title="French core-word deck",
            description=(
                "The core deck adds the <code>Core</code> and "
                "<code>CoreGloss</code> fields to the French sentence model."
            ),
            cards=(french_core_card(),),
        ),
        PreviewSection(
            anchor="vocab",
            title="Swadesh vocabulary decks",
            description=(
                "The vocabulary builder writes one block for the target form, the "
                "transliteration, and the English gloss. Cantonese adds a Hong Kong "
                "font stack. Arabic adds a right-to-left rule."
            ),
            cards=vocab_cards,
        ),
    )


def iframe_document(card: PreviewCard, side: str) -> str:
    """Build the standalone document for one card side.

    :param card: Card to render.
    :param side: ``front`` or ``back``.
    :returns: Complete HTML document for the iframe.
    """
    body = card.front if side == "front" else card.back
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<style>{PREVIEW_CARD_CSS}\n{card.css}</style></head>"
        f"<body><div class='card'>{body}</div></body></html>"
    )


def iframe_tag(card: PreviewCard, side: str) -> str:
    """Build the iframe element of one card side.

    :param card: Card to render.
    :param side: ``front`` or ``back``.
    :returns: iframe HTML with an escaped ``srcdoc`` value.
    """
    document = iframe_document(card, side)
    title = html.escape(f"{card.title} {side}", quote=True)
    return (
        f'<iframe class="card-preview" title="{title}" '
        f'onload="fitCard(this)" srcdoc="{html.escape(document, quote=True)}"></iframe>'
    )


def card_panel(card: PreviewCard) -> str:
    """Build the panel of one card with the front and the back.

    :param card: Card to show.
    :returns: Panel HTML.
    """
    return (
        '<article class="card-panel">'
        f"<h3>{html.escape(card.title)}</h3>"
        '<div class="sides">'
        '<div class="side"><span class="side-label">Front</span>'
        f"{iframe_tag(card, 'front')}</div>"
        '<div class="side"><span class="side-label">Back</span>'
        f"{iframe_tag(card, 'back')}</div>"
        "</div>"
        f'<p class="meta"><span class="meta-key">Templates</span> '
        f"{html.escape(card.source)}</p>"
        f'<p class="meta"><span class="meta-key">Fields</span> '
        f"{html.escape(card.fields)}</p>"
        + (f'<p class="note">{html.escape(card.note)}</p>' if card.note else "")
        + "</article>"
    )


def section_html(section: PreviewSection) -> str:
    """Build the HTML of one preview section.

    :param section: Section to show.
    :returns: Section HTML.
    """
    panels = "\n".join(card_panel(card) for card in section.cards)
    return (
        f'<section id="{html.escape(section.anchor, quote=True)}">'
        f"<h2>{html.escape(section.title)}</h2>"
        f'<p class="section-lead">{section.description}</p>'
        f'<div class="card-grid">{panels}</div>'
        "</section>"
    )


def build_page(
    sections: tuple[PreviewSection, ...],
    *,
    title: str = "Anki card previews",
    lead: str = (
        "Every card type that the deck builders write. The page reads the "
        "real template and CSS files. The audio control is a placeholder "
        "and does not play."
    ),
    source_note: str = "Generated by <code>uv run anki preview</code>.",
) -> str:
    """Build the complete preview page.

    :param sections: Sections to show.
    :param title: Page title.
    :param lead: Introduction HTML for the page.
    :param source_note: HTML that names the command that made the page.
    :returns: Complete HTML page.
    """
    toc = "\n".join(
        f'<a href="#{html.escape(section.anchor, quote=True)}">'
        f"{html.escape(section.title)}</a>"
        for section in sections
    )
    body = "\n".join(section_html(section) for section in sections)
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html.escape(title)}</title>\n"
        f"<style>{PAGE_CSS}</style>\n"
        f"<script>{PAGE_SCRIPT}</script>\n"
        "</head>\n<body>\n<main>\n"
        f"<header>\n<h1>{html.escape(title)}</h1>\n"
        f'<p class="lead">{lead}</p>\n'
        f'<p class="lead">{source_note} Do not edit this file by hand.</p>\n'
        f'<nav class="toc">{toc}</nav>\n'
        "</header>\n"
        f"{body}\n"
        "</main>\n</body>\n</html>\n"
    )


def write_preview(output_path: Path = OUTPUT_PATH) -> tuple[int, int]:
    """Build the preview page and write it to a file.

    :param output_path: HTML file to write.
    :returns: Card count and section count.
    """
    sections = build_sections()
    page = build_page(sections)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")
    cards = sum(len(section.cards) for section in sections)
    return cards, len(sections)


def main() -> None:
    """Write the preview page to the default path and print the result."""
    cards, sections = write_preview()
    print(f"wrote {OUTPUT_PATH} with {cards} cards in {sections} sections")


if __name__ == "__main__":
    main()
