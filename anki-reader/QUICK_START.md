# Quick Start Guide: Reading Anki .apkg Files

## TL;DR - Which Method Should I Use?

```
Need to...                          → Use this method
─────────────────────────────────────────────────────────
Just read cards quickly             → Direct SQLite (current)
Export to Excel/CSV                 → anki-export
Analyze study patterns              → ankipandas
Create new decks programmatically   → genanki
Edit existing decks                 → ankisync2
```

## Method 1: Direct SQLite (Current - ✓ Recommended for reading)

**Install:** Nothing needed (uses Python stdlib)

```python
import sqlite3, zipfile, tempfile

with zipfile.ZipFile('deck.apkg', 'r') as z:
    z.extractall('/tmp/anki')

conn = sqlite3.connect('/tmp/anki/collection.anki2')
cursor = conn.cursor()
cursor.execute("SELECT * FROM cards JOIN notes ON cards.nid = notes.id")
for card in cursor.fetchall():
    print(card)
```

**Performance:** 0.073s for 576 cards

## Method 2: anki-export (Easy API)

**Install:** `uv add anki-export`

```python
from anki_export import ApkgReader

with ApkgReader('deck.apkg') as apkg:
    data = apkg.export()  # OrderedDict by card type
    for card_type, cards in data.items():
        print(f"{card_type}: {len(cards)} cards")
        # cards[0] = field names
        # cards[1:] = card data
```

**Performance:** 0.088s for 576 cards

## Method 3: ankipandas (Data Analysis)

**Install:** `uv add ankipandas`

```python
from ankipandas import Collection

col = Collection()  # Finds Anki DB automatically
notes = col.notes   # Pandas DataFrame
cards = col.cards   # Pandas DataFrame

# Analyze with pandas
cards.hist(column='creps', by='cdeck')
cards['cdeck'].value_counts()
```

**Note:** Reads from Anki's user directory, not standalone .apkg files

## Method 4: genanki (Creating Decks)

**Install:** `uv add genanki`

```python
import genanki

model = genanki.Model(1234, 'My Model',
    fields=[{'name': 'Question'}, {'name': 'Answer'}],
    templates=[...])

deck = genanki.Deck(5678, 'My Deck')
note = genanki.Note(model=model, fields=['Q1', 'A1'])
deck.add_note(note)

genanki.Package(deck).write_to_file('output.apkg')
```

**Note:** For creating, not reading

## File Format Reference

```
.apkg structure:
├── collection.anki2     (SQLite database)
├── collection.anki21    (newer format with zstd)
├── media               (JSON: ID → filename mapping)
├── 0                   (media file)
├── 1                   (media file)
└── ...

SQLite tables:
├── cards   (card instances)
├── notes   (note content, fields separated by \x1f)
├── col     (collection metadata, decks/models as JSON)
├── graves  (deleted items)
└── revlog  (review history)
```

## Common Tasks

### Read all cards with fields

```python
# Current method (Direct SQLite)
cursor.execute("""
    SELECT cards.*, notes.flds, notes.tags
    FROM cards JOIN notes ON cards.nid = notes.id
""")
for row in cursor:
    fields = row['flds'].split('\x1f')
    print(fields)
```

### Export to CSV

```python
# Using anki-export
from anki_export import ApkgReader
import csv

with ApkgReader('deck.apkg') as apkg:
    data = apkg.export()
    for card_type, cards in data.items():
        with open(f'{card_type}.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(cards)
```

### Get deck statistics

```python
# Using ankipandas
from ankipandas import Collection
col = Collection()
print(col.cards['cdeck'].value_counts())
print(col.cards.groupby('cdeck')['creps'].mean())
```

## Performance Comparison

| Method        | Time (576 cards) | Memory | Dependencies |
|---------------|------------------|--------|--------------|
| Direct SQLite | 0.073s          | Low    | None         |
| anki-export   | 0.088s          | Medium | 1 package    |
| ankipandas    | ~0.5s           | High   | pandas       |
| ankisync2     | ~0.1s           | Medium | 2 packages   |

## Try It

```bash
# Compare all methods
uv run compare_methods.py

# Read with current implementation
uv run read_anki.py

# Try anki-export
uv add anki-export
uv run examples_anki_export.py
```
