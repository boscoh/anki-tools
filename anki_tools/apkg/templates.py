"""Card template and CSS files shared by the deck builders."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from anki_tools.apkg.package import AnkiPackage

TEMPLATES_DIR = Path(__file__).parents[2] / "templates"
SHARED_TEMPLATE_DIR = TEMPLATES_DIR / "shared"
DEFAULT_TEMPLATE_DIR = TEMPLATES_DIR / "default"
SHARED_CSS_PATH = SHARED_TEMPLATE_DIR / "card.css"
CHAR_LINKS_PATH = SHARED_TEMPLATE_DIR / "char_links.html"
DEFAULT_CSS_PATH = DEFAULT_TEMPLATE_DIR / "card.css"
DEFAULT_FRONT_PATH = DEFAULT_TEMPLATE_DIR / "front.html"
DEFAULT_BACK_PATH = DEFAULT_TEMPLATE_DIR / "back.html"


def load_text(path: Path) -> str:
    """Read a template or CSS file.

    :param path: File to read.
    :raises FileNotFoundError: The file does not exist.
    :returns: The text of the file, without the final newline.
    """
    return Path(path).read_text(encoding="utf-8").strip()


def load_optional_text(path: Path) -> str:
    """Read a template or CSS file, or give an empty string when it is absent.

    :param path: File to read.
    :returns: The text of the file, or an empty string.
    """
    path = Path(path)
    if not path.exists():
        return ""
    return load_text(path)


def load_css_layers(*paths: Path) -> str:
    """Read the CSS files in order. A later rule overrides an earlier rule.

    :param paths: CSS files, the shared sheet first.
    :returns: The joined CSS text, empty when every file is absent.
    """
    return join_css(*(load_optional_text(path) for path in paths))


def load_char_links(path: Path = CHAR_LINKS_PATH) -> str:
    """Read the script that links each Han character to the CC-Canto dictionary.

    :param path: HTML partial that holds the ``<script>`` block.
    :returns: The script text, or an empty string when the file is absent.
    """
    return load_optional_text(path)


def add_char_links(back: str) -> str:
    """Append the character-link script when the back text holds the hook.

    :param back: Back template text.
    :returns: The back text, with the script appended.
    """
    script = load_char_links()
    if script and "char-links" in back:
        return f"{back.rstrip()}\n{script}"
    return back


def join_css(*layers: str) -> str:
    """Join the CSS texts in order. A later rule overrides an earlier rule.

    :param layers: CSS texts, the shared sheet first.
    :returns: The joined CSS text, empty when every layer is empty.
    """
    return "\n".join(layer for layer in layers if layer)


def load_css(css_path: Path, extra_css: str = "") -> str:
    """Read one CSS file and add the extra classes after it.

    :param css_path: Base CSS file. A missing file gives only the extra text.
    :param extra_css: CSS classes of the deck.
    :returns: Base CSS followed by the extra classes.
    """
    return join_css(load_optional_text(css_path), extra_css)


@dataclass(frozen=True)
class TemplateSpec:
    """The text and the switches of one note type.

    :param front: Front template text.
    :param back: Back template text.
    :param css_layers: CSS texts in order, the shared sheet first.
    :param fields: Field names of the note type.
    :param optional_blocks: Block names that the note type drops when the
        field is absent.
    """

    front: str
    back: str
    css_layers: Sequence[str] = ()
    fields: Sequence[str] = ()
    optional_blocks: Sequence[str] = ()


@dataclass(frozen=True)
class TemplatePair:
    """The template pair and the CSS of one note type.

    :param front: Front template text.
    :param back: Back template text.
    :param css: Joined CSS text.
    """

    front: str
    back: str
    css: str


def drop_block(text: str, name: str) -> str:
    """Remove one optional block from a template.

    :param text: Template text.
    :param name: Block name, for example ``Romanization``.
    :returns: The template without the block.
    """
    tag = re.escape(name)
    pattern = r"\{\{#" + tag + r"\}\}.*?\{\{/" + tag + r"\}\}"
    return re.sub(pattern, "", text, flags=re.DOTALL).strip()


def assemble_templates(spec: TemplateSpec) -> TemplatePair:
    """Build the template pair and the CSS of one note type.

    :param spec: Text and switches of the note type.
    :returns: The final templates and the joined CSS.
    """
    back = spec.back
    for name in spec.optional_blocks:
        if name not in spec.fields:
            back = drop_block(back, name)
    back = add_char_links(back)
    return TemplatePair(front=spec.front, back=back, css=join_css(*spec.css_layers))


def extract_template_fields(template: str) -> set[str]:
    """Give the field names that a card template reads.

    :param template: Template text, front or back.
    :returns: The field names, without the template builtins.
    """
    matches = re.findall(r"\{\{([^#/}]+)\}\}", template)
    builtins = {"FrontSide", "Tags", "Type", "Deck", "Subdeck", "Card"}
    return {name.strip() for name in matches if name.strip() not in builtins}


def find_best_match(field: str, deck_fields: set[str]) -> str | None:
    """Find the deck field that a template field name names.

    The function tries the exact name, then the name with the most shared
    words.

    :param field: Field name of the template.
    :param deck_fields: Field names of the deck models.
    :returns: The deck field name, or None when no name is close.
    """
    field_lower = field.lower()
    for deck_field in deck_fields:
        if deck_field.lower() == field_lower:
            return deck_field
    field_words = set(re.split(r"[^a-zA-Z]+", field_lower))
    best_match = None
    best_score = 0
    for deck_field in deck_fields:
        deck_words = set(re.split(r"[^a-zA-Z]+", deck_field.lower()))
        common = field_words & deck_words
        if common and len(common) > best_score:
            best_score = len(common)
            best_match = deck_field
    return best_match


def apply_field_mapping(template: str, mapping: dict[str, str]) -> str:
    """Replace the field references of a template with deck field names.

    :param template: Template text.
    :param mapping: Mapping from template field name to deck field name.
    :returns: The template text with the names replaced.
    """
    result = template
    for old_name, new_name in mapping.items():
        result = result.replace(f"{{{{{old_name}}}}}", f"{{{{{new_name}}}}}")
    return result


def apply_templates(
    target_apkg: Path,
    *,
    css_path: Path = DEFAULT_CSS_PATH,
    front_path: Path = DEFAULT_FRONT_PATH,
    back_path: Path = DEFAULT_BACK_PATH,
    verbose: bool = True,
) -> tuple[int, list[str]]:
    """Write the CSS and the card templates into every model of a package.

    The CSS is the shared sheet, then the deck sheet. A template field name
    that the model does not hold maps to the nearest model field name.

    :param target_apkg: APKG file to update in place.
    :param css_path: Deck CSS file, read after the shared sheet.
    :param front_path: Front template file.
    :param back_path: Back template file.
    :param verbose: Print the loaded files and the field mapping.
    :returns: The model count and the names of the applied parts.
    """
    with AnkiPackage(str(target_apkg)) as pkg:
        models = pkg.get_models(include_templates=True)
        all_fields: set[str] = set()
        for model in models.values():
            all_fields.update(f["name"] for f in model.get("flds", []))
        if verbose:
            print("Deck fields:")
            for _mid, model in models.items():
                print(f"  {', '.join(f['name'] for f in model['flds'])}")

    css_content = load_css_layers(SHARED_CSS_PATH, css_path)
    if css_content and verbose:
        print(f"\nLoaded: {SHARED_CSS_PATH} and {css_path}")
    elif verbose:
        print(f"\nNo CSS found in {SHARED_CSS_PATH} or {css_path}, skipping CSS")

    qfmt = None
    if front_path.exists():
        qfmt = front_path.read_text(encoding="utf-8")
        if verbose:
            print(f"Loaded: {front_path}")
        missing = extract_template_fields(qfmt) - all_fields
        if missing:
            mapping = {f: find_best_match(f, all_fields) for f in missing}
            mapping = {k: v for k, v in mapping.items() if v}
            if mapping and verbose:
                print(f"  Auto-mapping: {mapping}")
            qfmt = apply_field_mapping(qfmt, mapping)
    elif verbose:
        print(f"No {front_path} found, skipping front template")

    afmt = None
    if back_path.exists():
        afmt = back_path.read_text(encoding="utf-8")
        if verbose:
            print(f"Loaded: {back_path}")
        missing = extract_template_fields(afmt) - all_fields
        if missing:
            mapping = {f: find_best_match(f, all_fields) for f in missing}
            mapping = {k: v for k, v in mapping.items() if v}
            if mapping and verbose:
                print(f"  Auto-mapping: {mapping}")
            afmt = apply_field_mapping(afmt, mapping)
        afmt = add_char_links(afmt)
    elif verbose:
        print(f"No {back_path} found, skipping back template")

    if not css_content and not qfmt and not afmt:
        return (0, [])

    with AnkiPackage(str(target_apkg)) as pkg:
        models = pkg.get_models()
        for model_id in models:
            pkg.update_model_template(
                model_id=model_id, css=css_content, qfmt=qfmt, afmt=afmt
            )
        pkg.save(target_apkg)

    applied = []
    if css_content:
        applied.append("CSS")
    if qfmt:
        applied.append("front")
    if afmt:
        applied.append("back")
    if verbose:
        print(f"Updated {len(models)} model(s)")
    return (len(models), applied)
