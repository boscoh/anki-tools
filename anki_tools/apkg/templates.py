"""Card template and CSS files shared by the deck builders."""

from __future__ import annotations

import re
from pathlib import Path

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
    parts = [text for text in (load_optional_text(path) for path in paths) if text]
    return "\n".join(parts)


def load_char_links(path: Path = CHAR_LINKS_PATH) -> str:
    """Read the script that links each Han character to the CC-Canto dictionary.

    :param path: HTML partial that holds the ``<script>`` block.
    :returns: The script text, or an empty string when the file is absent.
    """
    return load_optional_text(path)


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
        script = load_char_links()
        if script and "char-links" in afmt:
            afmt = f"{afmt.rstrip()}\n{script}"
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
