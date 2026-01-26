# Research: Different Ways to Read Anki .apkg Files

## Overview of .apkg Format

An Anki package (`.apkg`) is a **ZIP archive** containing:
- `collection.anki2` or `collection.anki21` - SQLite database with cards, notes, models, and decks
- `media` - JSON file mapping media IDs to filenames
- Numbered media files (1, 2, 3, etc.) for images/audio
- Database uses schema v11 (older) or v18 (newer, with zstd compression)

## Approaches to Reading .apkg Files

### 1. **Direct SQLite Access** (Current Implementation)
**What we're using now**

```python
import sqlite3
import zipfile
import tempfile

with zipfile.ZipFile('deck.apkg', 'r') as zip_ref:
    zip_ref.extractall(temp_dir)

conn = sqlite3.connect('collection.anki2')
cursor = conn.cursor()
cursor.execute("SELECT * FROM cards JOIN notes ON cards.nid = notes.id")
```

**Pros:**
- No external dependencies
- Full control over data extraction
- Lightweight and fast
- Works with any Anki version

**Cons:**
- Need to understand database schema
- Manual parsing of JSON fields
- More boilerplate code
- Need to handle field separator (\x1f) manually

---

### 2. **ankipandas** - DataFrame-Based Analysis
**Best for: Data analysis and visualization**

```python
from ankipandas import Collection

col = Collection()  # Auto-finds Anki database
notes = col.notes   # Pandas DataFrame
cards = col.cards   # Pandas DataFrame

# Filter and analyze
cards.hist(column="creps", by="cdeck")
notes.fields_as_columns()
```

**Pros:**
- Integrates with pandas for powerful analysis
- Great for statistics and visualization
- Auto-locates Anki database
- Rich querying with pandas syntax
- Merge cards with notes easily

**Cons:**
- Write functionality currently disabled
- Requires pandas (heavier dependency)
- Supports Python 3.7-3.10 only
- More overhead for simple reading

**Use Cases:**
- Statistical analysis of study patterns
- Finding difficult cards/leeches
- Visualizing deck statistics
- Data science workflows

---

### 3. **anki-export (ApkgReader)** - Simple Export Tool
**Best for: Converting to other formats**

```python
from anki_export import ApkgReader
import pyexcel_xlsxwx

with ApkgReader('test.apkg') as apkg:
    # Export to Excel
    pyexcel_xlsxwx.save_data('test.xlsx', apkg.export(),
                             config={'format': None})
```

**Pros:**
- Specifically designed for reading .apkg files
- Context manager for clean resource handling
- Easy export to spreadsheet formats
- Lightweight and focused

**Cons:**
- Less control over individual fields
- Limited documentation
- Primarily export-focused
- Fewer features than alternatives

**Use Cases:**
- Converting decks to Excel/CSV
- Quick data extraction
- Creating backups in readable formats

---

### 4. **ankisync2** - ORM-Based Approach
**Best for: Creating and modifying decks**

```python
from ankisync2 import Apkg

with Apkg("example.apkg") as apkg:
    # Iterate through cards
    for card in apkg:
        print(card)

    # Create new content
    model = apkg.db.Models.create(name="foo", flds=["field1", "field2"])
    note = apkg.db.Notes.create(mid=model.id, flds=["data1", "data2"])
```

**Pros:**
- Uses Peewee ORM (cleaner than raw SQL)
- Can create and modify decks
- Good for programmatic deck generation
- Handles media files
- Safe database operations

**Cons:**
- ORM adds complexity for simple reads
- Learning curve for Peewee
- Overkill for read-only use cases
- More dependencies

**Use Cases:**
- Creating decks programmatically
- Editing existing cards
- Adding media to decks
- Building deck generators

---

### 5. **genanki** - Deck Generation Library
**Best for: Creating new decks (not reading)**

```python
import genanki

# Note: genanki is primarily for CREATING cards, not reading
model = genanki.Model(...)
deck = genanki.Deck(...)
note = genanki.Note(model=model, fields=['Front', 'Back'])
deck.add_note(note)
```

**Pros:**
- Simple API for deck creation
- Well-documented
- Popular and maintained
- Great for flashcard generation

**Cons:**
- **Not designed for reading existing decks**
- Write-only functionality
- Would need custom code to read

**Use Cases:**
- Generating flashcards from data
- Automated deck creation
- Converting other formats to Anki

---

### 6. **AnkiTools** - Multi-Format Converter
**Best for: Format conversion**

Supports .xlsx, .apkg, and .anki2 formats with conversion between them.

**Pros:**
- Multiple format support
- Conversion capabilities
- Editor functionality

**Cons:**
- Limited documentation
- Less popular/maintained
- Unclear API

---

### 7. **Official Anki Python Library**
**Best for: Full Anki functionality**

The official library from ankitects/anki repository.

**Pros:**
- Official implementation
- Most complete feature set
- Always up-to-date with format changes

**Cons:**
- Very complex API
- Requires building from source
- Heavy dependencies
- Steep learning curve
- Overkill for simple reading

---

## Recommendation Matrix

| Use Case | Recommended Approach | Reason |
|----------|---------------------|---------|
| Simple card reading | **Direct SQLite** | No dependencies, fast, full control |
| Data analysis/stats | **ankipandas** | Pandas integration, visualization |
| Format conversion | **anki-export** | Built for this purpose |
| Creating new decks | **genanki** or **ankisync2** | Purpose-built for generation |
| Modifying existing decks | **ankisync2** | ORM makes edits safer |
| Production app | **Official Anki lib** | Most robust, handles all edge cases |
| Quick scripting | **Direct SQLite** | Minimal setup |

## Database Schema Key Tables

```sql
-- Main tables in collection.anki2
notes:  id, mid (model id), flds (fields separated by \x1f), tags
cards:  id, nid (note id), did (deck id), ord (template order)
col:    models (JSON), decks (JSON), conf (JSON)
```

## File Format Versions

- **Old format**: `collection.anki2` (SQLite with deflate, schema v11)
- **New format**: `collection.anki21b` (SQLite with zstd, schema v18, Protobuf)

## Conclusion

**For your current project** (reading Chinese flashcards):
- ✅ **Direct SQLite** (current) - Good choice for simple, fast reading
- Consider **ankipandas** if you want to analyze study patterns
- Consider **anki-export** for easy Excel export

The direct SQLite approach is actually well-suited for read-only access without heavy dependencies.
