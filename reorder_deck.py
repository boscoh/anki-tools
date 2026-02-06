#!/usr/bin/env python3
"""
Reorder an Anki deck based on sentence ranking.

Reads ranking from CSV and updates card order in the APKG file.
"""

import csv
import sys
from pathlib import Path

from anki_package import AnkiPackage

csv.field_size_limit(sys.maxsize)


def load_ranking(csv_path: str) -> tuple[dict[str, int], set[int]]:
    """
    Load ranking from CSV file.
    
    Returns:
        Tuple of (sentence -> rank mapping, ranks to remove)
    """
    ranking = {}
    remove_ranks = set()
    
    # Names to filter out
    name_indicators = [
        'Tracy', 'Jackie', 'Carrie', 'Perry', 'Gilbert', 
        'Krishna', 'A Ren', 'Bobby', 'Jimmy', 'Johnny'
    ]
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rank = int(row['rank'])
            sentence = row['sentence']
            freq = float(row['frequency'])
            english = row['english']
            
            ranking[sentence] = rank
            
            # Mark for removal
            if 'not valid' in english.lower():
                remove_ranks.add(rank)
            elif freq < 5:
                for name in name_indicators:
                    if name in english:
                        remove_ranks.add(rank)
                        break
    
    return ranking, remove_ranks


def reorder_deck(
    input_apkg: str,
    output_apkg: str,
    ranking_csv: str,
    remove_filtered: bool = True
) -> dict:
    """
    Reorder deck based on ranking CSV.
    
    Args:
        input_apkg: Path to input .apkg file
        output_apkg: Path to output .apkg file
        ranking_csv: Path to ranking CSV file
        remove_filtered: If True, delete cards marked for removal
    
    Returns:
        Dict with statistics
    """
    ranking, remove_ranks = load_ranking(ranking_csv)
    
    print(f"Loaded ranking for {len(ranking)} sentences")
    print(f"Marked {len(remove_ranks)} for removal")
    
    with AnkiPackage(input_apkg) as pkg:
        cursor = pkg.conn.cursor()
        
        # Build note_id -> sentence mapping
        cursor.execute('SELECT id, flds FROM notes')
        note_sentences = {}
        for row in cursor.fetchall():
            fields = row['flds'].split('\x1f')
            sentence = fields[0] if fields else ''
            note_sentences[row['id']] = sentence
        
        # Build note_id -> rank mapping
        note_ranks = {}
        for nid, sentence in note_sentences.items():
            if sentence in ranking:
                note_ranks[nid] = ranking[sentence]
        
        print(f"Matched {len(note_ranks)} notes to rankings")
        
        # Get cards to reorder
        cursor.execute('SELECT id, nid, due FROM cards WHERE type = 0')
        cards = cursor.fetchall()
        
        # Identify cards to remove
        cards_to_remove = []
        if remove_filtered:
            for card in cards:
                nid = card['nid']
                if nid in note_ranks and note_ranks[nid] in remove_ranks:
                    cards_to_remove.append(card['id'])
        
        # Remove filtered cards
        if cards_to_remove:
            print(f"Removing {len(cards_to_remove)} filtered cards...")
            for card_id in cards_to_remove:
                pkg.delete_card(card_id, cleanup_audio=False)
        
        # Update due values for remaining cards
        # Sort by rank and assign sequential due values
        remaining_cards = []
        cursor.execute('SELECT id, nid FROM cards WHERE type = 0')
        for card in cursor.fetchall():
            nid = card['nid']
            if nid in note_ranks:
                rank = note_ranks[nid]
                if rank not in remove_ranks:
                    remaining_cards.append((rank, card['id']))
        
        # Sort by rank and update due
        remaining_cards.sort(key=lambda x: x[0])
        
        print(f"Updating order for {len(remaining_cards)} cards...")
        for new_due, (rank, card_id) in enumerate(remaining_cards, start=1):
            cursor.execute(
                'UPDATE cards SET due = ? WHERE id = ?',
                (new_due, card_id)
            )
        
        pkg.conn.commit()
        pkg._modified = True
        
        # Save to new file
        pkg.save(output_apkg)
        
        return {
            'total_ranked': len(ranking),
            'matched_notes': len(note_ranks),
            'cards_removed': len(cards_to_remove),
            'cards_reordered': len(remaining_cards),
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Reorder Anki deck based on sentence ranking'
    )
    parser.add_argument('input_apkg', help='Input .apkg file')
    parser.add_argument('ranking_csv', help='Ranking CSV from rank_sentences.py')
    parser.add_argument('-o', '--output', help='Output .apkg file (default: input_ranked.apkg)')
    parser.add_argument('--keep-filtered', action='store_true',
                       help='Keep filtered cards (names, invalid) instead of removing')
    
    args = parser.parse_args()
    
    if args.output:
        output = args.output
    else:
        input_path = Path(args.input_apkg)
        output = str(input_path.parent / f"{input_path.stem}_ranked{input_path.suffix}")
    
    stats = reorder_deck(
        args.input_apkg,
        output,
        args.ranking_csv,
        remove_filtered=not args.keep_filtered
    )
    
    print(f"\nDone!")
    print(f"  Cards reordered: {stats['cards_reordered']}")
    print(f"  Cards removed: {stats['cards_removed']}")
    print(f"  Output: {output}")


if __name__ == '__main__':
    main()
