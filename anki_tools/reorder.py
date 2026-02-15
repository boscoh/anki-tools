"""
Reorder an Anki deck based on sentence ranking.

Reads ranking from CSV and updates card order in the APKG file.
"""

import csv
import sys

from anki_tools.package import AnkiPackage
from anki_tools.rank import SIMILARITY_CONSIDER_DELETE_PENALTY

csv.field_size_limit(sys.maxsize)


def load_ranking(csv_path: str) -> tuple[dict[str, int], dict[int, int], set[int]]:
    """Load ranking from CSV file.

    :param csv_path: Path to the ranking CSV file.
    :returns: Tuple of (sentence to rank mapping, note_id to rank mapping, set of ranks to remove as high-similarity).
    """
    sentence_ranking = {}
    note_id_ranking = {}
    remove_ranks = set()
    sim_col = SIMILARITY_CONSIDER_DELETE_PENALTY

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rank = int(row["rank"])
            sentence = row["sentence"]
            sentence_ranking[sentence] = rank

            # If note_id column exists, use it for direct mapping
            if "note_id" in row and row["note_id"]:
                try:
                    note_id = int(row["note_id"])
                    note_id_ranking[note_id] = rank
                except (ValueError, TypeError):
                    pass

            if not (row.get("similar_to") or "").strip():
                continue
            raw = row.get("similarity") or row.get("similarity_penalty") or "0"
            try:
                if float(raw) > sim_col:
                    remove_ranks.add(rank)
            except ValueError:
                remove_ranks.add(rank)

    return sentence_ranking, note_id_ranking, remove_ranks


def reorder_deck(
    input_apkg: str,
    output_apkg: str,
    ranking_csv: str,
    remove_filtered: bool = True,
    remove_unranked: bool = False,
) -> dict:
    """Reorder deck based on ranking CSV.

    :param input_apkg: Path to input .apkg file.
    :param output_apkg: Path to output .apkg file.
    :param ranking_csv: Path to ranking CSV file.
    :param remove_filtered: If True, delete high-similarity cards (those with similar_to set in CSV).
    :param remove_unranked: If True, delete cards that are not in the ranking CSV.
    :returns: Dict with keys total_ranked, matched_notes, cards_removed, cards_reordered.
    """
    sentence_ranking, note_id_ranking, remove_ranks = load_ranking(ranking_csv)

    print(f"Loaded ranking for {len(sentence_ranking)} sentences")
    print(f"Marked {len(remove_ranks)} high-similarity cards for removal")

    with AnkiPackage(input_apkg) as pkg:
        cursor = pkg.conn.cursor()

        # Build note_ranks mapping, preferring note_id mapping if available
        note_ranks = {}

        # First, use direct note_id mapping if available
        if note_id_ranking:
            cursor.execute("SELECT id FROM notes")
            for row in cursor.fetchall():
                nid = row["id"]
                if nid in note_id_ranking:
                    note_ranks[nid] = note_id_ranking[nid]
            print(f"Matched {len(note_ranks)} notes by note_id")

        # Fall back to sentence matching for any unmatched notes
        if not note_ranks:
            cursor.execute("SELECT id, flds FROM notes")
            note_sentences = {}
            for row in cursor.fetchall():
                fields = row["flds"].split("\x1f")
                sentence = fields[0] if fields else ""
                note_sentences[row["id"]] = sentence

            for nid, sentence in note_sentences.items():
                if sentence in sentence_ranking:
                    note_ranks[nid] = sentence_ranking[sentence]
            print(f"Matched {len(note_ranks)} notes by sentence text")

        print(f"Total matched: {len(note_ranks)} notes to rankings")

        cursor.execute("SELECT id, nid, due FROM cards WHERE type = 0")
        cards = cursor.fetchall()

        cards_to_remove = []

        # Remove high-similarity cards if requested
        if remove_filtered:
            for card in cards:
                nid = card["nid"]
                if nid in note_ranks and note_ranks[nid] in remove_ranks:
                    cards_to_remove.append(card["id"])

        # Remove unranked cards if requested
        if remove_unranked:
            for card in cards:
                nid = card["nid"]
                if nid not in note_ranks:
                    cards_to_remove.append(card["id"])

        if cards_to_remove:
            if remove_filtered and remove_unranked:
                print(f"Removing {len(cards_to_remove)} cards (high-similarity + unranked)...")
            elif remove_filtered:
                print(f"Removing {len(cards_to_remove)} high-similarity cards...")
            elif remove_unranked:
                print(f"Removing {len(cards_to_remove)} unranked cards...")

            for card_id in cards_to_remove:
                pkg.delete_card(card_id, cleanup_audio=False)

        remaining_cards = []
        cursor.execute("SELECT id, nid FROM cards WHERE type = 0")
        for card in cursor.fetchall():
            nid = card["nid"]
            if nid in note_ranks:
                rank = note_ranks[nid]
                if rank not in remove_ranks:
                    remaining_cards.append((rank, card["id"]))

        remaining_cards.sort(key=lambda x: x[0])

        print(f"Updating order for {len(remaining_cards)} cards...")
        for new_due, (rank, card_id) in enumerate(remaining_cards, start=1):
            cursor.execute("UPDATE cards SET due = ? WHERE id = ?", (new_due, card_id))

        pkg.conn.commit()
        pkg._modified = True

        pkg.save(output_apkg)

        return {
            "total_ranked": len(sentence_ranking),
            "matched_notes": len(note_ranks),
            "cards_removed": len(cards_to_remove),
            "cards_reordered": len(remaining_cards),
        }
