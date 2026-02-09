"""
Build vocab/english-arabic.csv from Swadesh list and Wiktionary Module:Swadesh/data/ar.
Columns: english, arabic
"""

import re
import urllib.request
from pathlib import Path

SWADESH_PATH = Path(__file__).parent.parent / "vocab" / "swadesh.txt"
OUTPUT_PATH = Path(__file__).parent.parent / "vocab" / "english-arabic.csv"
MODULE_URL = (
    "https://en.wiktionary.org/w/index.php?title=Module:Swadesh/data/ar&action=raw"
)


def fetch_module() -> str:
    req = urllib.request.Request(MODULE_URL, headers={"User-Agent": "anki-tools/1.0"})
    with urllib.request.urlopen(req, timeout=30) as f:
        return f.read().decode("utf-8")


def parse_lua_module(lua: str) -> dict[int, str]:
    result: dict[int, str] = {}
    for m in re.finditer(
        r"m\[(\d+)\]\s*=\s*\{([^}]+(?:\{[^}]*\}[^}]*)*)\}", lua, re.DOTALL
    ):
        idx = int(m.group(1))
        block = m.group(2)
        term_match = re.search(r'term\s*=\s*"([^"]+)"', block)
        if term_match:
            term = term_match.group(1).split("<")[0].strip()
            result[idx] = term
    return result


def main() -> None:
    english_lines = SWADESH_PATH.read_text(encoding="utf-8").strip().splitlines()
    lua = fetch_module()
    arabic_by_idx = parse_lua_module(lua)

    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
        f.write("english,arabic\n")
        for i, eng in enumerate(english_lines, start=1):
            arabic = arabic_by_idx.get(i, "")
            f.write(f'"{eng}","{arabic}"\n')

    print(f"Wrote {OUTPUT_PATH} ({len(english_lines)} rows)")


if __name__ == "__main__":
    main()
