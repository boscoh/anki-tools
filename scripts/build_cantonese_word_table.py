#!/usr/bin/env python
"""Build the Cantonese word table and its review file (stage S1).

The build never runs this script. It reads the two committed CSV files. Run
this script only when the reviewed Cantonese forms change.

The script compares ``vocab/english-cantonese.csv`` with the terms of
``Module:Swadesh/data/yue``. The ``CHOSEN`` map holds the review decisions. A
row that keeps a form outside the module terms gets a review row with status
``keep``, so gate G15 of the pack audit stops a build with an open question.

Usage:
    uv run python scripts/build_cantonese_word_table.py
"""

from __future__ import annotations

import csv

from anki_tools.sentence_pack.languages.cantonese import (
    DEFAULT_CSV_PATH,
    DEFAULT_REVIEW_PATH,
    SWADESH_ROWS,
    WORD_REVIEW_FIELDS,
    fetch_module_terms,
    parse_module_terms,
    prepare_text,
    read_csv_rows,
    read_swadesh_labels,
)

CHOSEN = {
    8: ("嗰", "go2", "module term: 481 linked sentences"),
    42: ("阿媽", "aa3 maa1", "module term: 74 linked sentences"),
    43: ("阿爸", "aa3 baa1", "module term: 50 linked sentences"),
    48: ("蝨", "sat1", "module term replaces the compound 蝨乸"),
    62: ("皮", "pei4", "module term: 41 linked sentences"),
    66: ("脂肪", "zi1 fong1", "the noun sense; 肥 is the adjective"),
    69: ("尾", "mei5", "module term: 32 linked sentences"),
    86: ("腸", "coeng4", "module term: 25 linked sentences"),
    96: ("吐", "tou3", "module term; 吐痰 has no candidate"),
    97: ("嘔", "au2", "module term; 嘔吐 has no candidate"),
    101: ("睇", "tai2", "the Cantonese verb; 睇見 has no candidate"),
    102: ("聽", "teng1", "module term: 227 linked sentences"),
    105: ("聞", "man4", "module term: 17 linked sentences"),
    106: ("驚", "geng1", "module term: 61 linked sentences"),
    114: ("切", "cit3", "module term: 42 linked sentences"),
    119: ("游水", "jau4 seoi2", "the swim verb; 遊 means to roam"),
    123: ("瞓", "fan3", "module term: 76 linked sentences"),
    127: ("跌", "dit3", "module term; 跌倒 has no candidate"),
    129: ("揸", "zaa1", "module term: 31 linked sentences"),
    138: ("縫", "fung4", "the module term 聯 means to unite, not to sew"),
    139: ("數", "sou2", "module term; 數數 has no candidate"),
    162: ("天", "tin1", "module term: 203 linked sentences"),
    169: ("燒", "siu1", "module term: 31 linked sentences"),
    193: ("滑", "waat6", "module term; 平滑 has no candidate"),
    196: ("啱", "ngaam1", "module term: 97 linked sentences"),
}
KEPT = {
    7: "呢個 is unambiguous; 呢 is also a question particle",
    88: "背 has 34 linked sentences; 背脊 has 1",
    108: "生存 keeps the sense live; 生 also means birth",
    111: "打交 keeps the sense fight; 爭 means to contend",
    116: "刺 keeps the sense stab; 插 means insert",
    130: "搾 is the squeeze verb; 挾 means to carry under the arm",
    136: "掟 is the throw verb; it has 2 candidates",
    178: "日 is the basic noun; 日頭 has 2 candidates",
}


def main() -> None:
    """Write the reviewed word CSV and the word review CSV."""
    rows = read_csv_rows(DEFAULT_CSV_PATH)
    labels = read_swadesh_labels()
    if len(rows) != SWADESH_ROWS:
        raise SystemExit(f"expected {SWADESH_ROWS} rows, found {len(rows)}")
    cache_path = fetch_module_terms()
    terms = parse_module_terms(cache_path.read_text(encoding="utf-8"))

    table: list[dict[str, str]] = []
    review: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        current = row.get("cantonese", "").strip()
        reading = row.get("transliteration", "").strip()
        module = [prepare_text(term) for term in terms.get(index, ())]
        if index in CHOSEN:
            chosen, reading, issue = CHOSEN[index]
            table.append(
                {
                    "english": labels[index - 1],
                    "cantonese": chosen,
                    "transliteration": reading,
                }
            )
            review.append(
                {
                    "index": str(index),
                    "english": labels[index - 1],
                    "current": current,
                    "module_terms": " / ".join(module),
                    "chosen": chosen,
                    "issue": issue,
                    "fix": f"{chosen} {reading}",
                    "status": "fixed",
                }
            )
            continue
        table.append(
            {
                "english": labels[index - 1],
                "cantonese": current,
                "transliteration": reading,
            }
        )
        if current in module:
            continue
        review.append(
            {
                "index": str(index),
                "english": labels[index - 1],
                "current": current,
                "module_terms": " / ".join(module),
                "chosen": current if index in KEPT else "",
                "issue": KEPT.get(index, "the reviewed form is not a module term"),
                "fix": "",
                "status": "keep" if index in KEPT else "open",
            }
        )

    with DEFAULT_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["english", "cantonese", "transliteration"]
        )
        writer.writeheader()
        writer.writerows(table)

    DEFAULT_REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DEFAULT_REVIEW_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=WORD_REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(review)
    print(f"wrote {DEFAULT_CSV_PATH}")
    print(f"wrote {DEFAULT_REVIEW_PATH}: {len(review)} rows")


if __name__ == "__main__":
    main()
