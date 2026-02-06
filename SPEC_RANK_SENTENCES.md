# Sentence Ranking Script Specification

Rank sentences in Anki decks (specifically "Neri's Chinese Sents Part 2.apkg") by complexity, word frequency, and uniqueness.

## Input

- **Deck**: `apkg/Neri's Chinese Sents Part 2.apkg`
- **Format**: Anki21b (zstd-compressed SQLite)
- **Model**: "0 2024 Sentences Deck" (ID: 1733774697943)
- **Cards**: 5009 Chinese sentences
- **Key Fields**:

| Index | Field Name            | Description                          |
|------:|:----------------------|:-------------------------------------|
|     0 | Sentence              | Chinese characters (e.g., "你好吗？") |
|     1 | Sentence (Annotated)  | Same as Sentence                     |
|     2 | Sentence (Latin)      | Pinyin (e.g., "Nǐ hǎo ma?")         |
|     3 | Sentence (Translation)| English translation                  |
|     6 | Explanation 1         | Word-by-word breakdown (HTML table)  |
|     8 | Related 1             | Similar sentences (HTML table)       |
|    10 | [a]                   | Audio reference                      |
|    15 | [counter 2]           | Original deck order (1-5009)         |


## Output

CSV file with ranked sentences containing:
- Original fields (Sentence, Pinyin, English, original_order)
- Scoring metrics (complexity_score, frequency_score, similarity_penalty)
- Final composite rank


## Scoring Components

### 1. Complexity Score (0-100)

Measures sentence difficulty based on structural and lexical features.

**Metrics**:

| Metric                    | Weight | Description                                    |
|:--------------------------|-------:|:-----------------------------------------------|
| Character count           |    15% | Longer sentences = higher complexity           |
| Unique character count    |    15% | More distinct characters = more to learn       |
| Word count (jieba)        |    20% | More words = more complex grammar              |
| Average word length       |    10% | Longer words tend to be less common            |
| Character stroke count    |    15% | Complex characters = harder to write           |
| HSK level of characters   |    25% | Higher HSK levels = more advanced vocabulary   |

**Implementation**:
```python
def complexity_score(sentence: str) -> float:
    char_count = len([c for c in sentence if '\u4e00' <= c <= '\u9fff'])
    words = list(jieba.cut(sentence))
    unique_chars = len(set(c for c in sentence if '\u4e00' <= c <= '\u9fff'))
    
    # Normalize each component to 0-1 range
    # Weight and combine
    # Return 0-100 score
```


### 2. Frequency Score (0-100)

Measures how common/useful the vocabulary is based on word frequency lists.

**Data Sources**:
- SUBTLEX-CH frequency list (film subtitles - spoken Chinese)
- HSK vocabulary lists (levels 1-6)
- Character frequency list (top 3000)

**Metrics**:

| Metric                      | Weight | Description                                  |
|:----------------------------|-------:|:---------------------------------------------|
| Average word frequency rank |    40% | Lower rank = more common words               |
| % words in top 1000         |    25% | Core vocabulary coverage                     |
| % words in top 5000         |    15% | Extended vocabulary coverage                 |
| HSK coverage                |    20% | % words appearing in HSK 1-4                 |

**Implementation**:
```python
def frequency_score(sentence: str, freq_dict: dict) -> float:
    words = list(jieba.cut(sentence))
    ranks = [freq_dict.get(w, 50000) for w in words]
    
    avg_rank = sum(ranks) / len(ranks)
    top_1000_pct = sum(1 for r in ranks if r <= 1000) / len(ranks)
    # ...
    # Higher score = more common vocabulary
```


### 3. Similarity Penalty (0-100)

Penalizes sentences that are too similar to higher-ranked ones.

**Metrics**:

| Metric                  | Description                                      |
|:------------------------|:-------------------------------------------------|
| Character overlap       | Jaccard similarity of character sets             |
| Word overlap            | Jaccard similarity of word sets                  |
| N-gram overlap          | Shared bigrams/trigrams                          |
| Semantic similarity     | (Optional) Embedding-based similarity            |

**Algorithm**:
1. Sort sentences by preliminary score (complexity + frequency)
2. For each sentence, compute similarity to all higher-ranked sentences
3. Apply penalty based on max similarity found
4. Penalty formula: `penalty = max_similarity * 50` (up to 50 points)

**Implementation**:
```python
def similarity(sent_a: str, sent_b: str) -> float:
    chars_a = set(c for c in sent_a if '\u4e00' <= c <= '\u9fff')
    chars_b = set(c for c in sent_b if '\u4e00' <= c <= '\u9fff')
    
    jaccard = len(chars_a & chars_b) / len(chars_a | chars_b)
    return jaccard

def compute_similarity_penalties(sentences: list[str], preliminary_ranks: list[int]) -> list[float]:
    penalties = []
    for i, sent in enumerate(sentences):
        higher_ranked = [s for j, s in enumerate(sentences) if preliminary_ranks[j] < preliminary_ranks[i]]
        if not higher_ranked:
            penalties.append(0)
            continue
        max_sim = max(similarity(sent, h) for h in higher_ranked[:100])  # Check top 100
        penalties.append(max_sim * 50)
    return penalties
```


## Final Ranking Formula

```
final_score = (complexity_score * 0.3) + (frequency_score * 0.5) - (similarity_penalty * 0.2)
```

**Rationale**:
- **Frequency (50%)**: Prioritize learning common, useful vocabulary
- **Complexity (30%)**: Progress from simple to complex
- **Similarity (-20%)**: Ensure variety, avoid redundant sentences

**Adjustable Parameters**:
```python
WEIGHTS = {
    'complexity': 0.3,
    'frequency': 0.5,
    'similarity_penalty': 0.2,
}

COMPLEXITY_PARAMS = {
    'char_count_weight': 0.15,
    'unique_char_weight': 0.15,
    'word_count_weight': 0.20,
    # ...
}
```


## Implementation Plan

| Phase | Issue ID    | Title                                            | Status   | Depends On        |
|------:|:------------|:-------------------------------------------------|:---------|:------------------|
|     1 | anki-5y2.1  | Data extraction from Anki21b format              | closed   | -                 |
|     2 | anki-5y2.2  | Load frequency data (SUBTLEX-CH, HSK vocab)      | closed   | -                 |
|     3 | anki-5y2.3  | Implement scoring functions (complexity, freq)   | closed   | 5y2.1, 5y2.2      |
|     4 | anki-5y2.4  | Implement similarity analysis and penalties      | closed   | 5y2.3             |
|     5 | anki-5y2.5  | Final ranking and CSV export                     | closed   | 5y2.4             |

### Phase 1: Data Extraction (anki-5y2.1)
- Read anki21b format (zstd decompress + SQLite)
- Extract sentences from model 1733774697943
- Parse relevant fields into dataclass

### Phase 2: Frequency Data (anki-5y2.2)
- Load/download SUBTLEX-CH word frequency list
- Load HSK vocabulary lists (built into jieba or separate file)
- Create character frequency lookup

### Phase 3: Scoring Functions (anki-5y2.3)
- Implement `complexity_score()`
- Implement `frequency_score()`
- Unit tests for edge cases

### Phase 4: Similarity Analysis (anki-5y2.4)
- Implement `similarity()` function
- Efficient batch similarity computation (avoid O(n^2) with clever filtering)
- Apply similarity penalties

### Phase 5: Ranking and Output (anki-5y2.5)
- Combine scores with configurable weights
- Sort by final score
- Export to CSV with all metrics


## Dependencies

```toml
# Add to pyproject.toml
jieba = ">=0.42.1"       # Already present
pypinyin = ">=0.55.0"    # Already present  
zstandard = ">=0.25.0"   # Already added
```

**External Data Files** (download or embed):
- `vocab/subtlex_ch.txt` - Word frequency list
- `vocab/hsk_vocab.json` - HSK 1-6 vocabulary


## CLI Interface

```bash
# Basic usage
uv run python rank_sentences.py "apkg/Neri's Chinese Sents Part 2.apkg"

# With options
uv run python rank_sentences.py "apkg/Neri's Chinese Sents Part 2.apkg" \
    --output ranked_sentences.csv \
    --complexity-weight 0.3 \
    --frequency-weight 0.5 \
    --similarity-weight 0.2 \
    --top-n 1000
```


## Example Output

```csv
rank,sentence,pinyin,english,original_order,complexity,frequency,similarity_penalty,final_score
1,你好,Nǐ hǎo,Hello,42,15.2,95.3,0.0,47.15
2,谢谢,Xiè xiè,Thank you,128,12.8,94.1,2.5,44.23
3,再见,Zài jiàn,Goodbye,89,14.5,92.7,3.1,43.89
...
```


## Performance Considerations

- **5009 sentences**: Similarity computation is O(n^2) worst case
- **Optimization**: Use character set hashing for fast pre-filtering
- **Batch processing**: Process in chunks to show progress
- Expected runtime: < 30 seconds for full analysis


## Future Enhancements

1. **Semantic similarity**: Use sentence embeddings for deeper similarity detection
2. **Grammar patterns**: Detect and score grammar structures (把, 被, 是...的)
3. **Topic clustering**: Group sentences by topic for varied learning
4. **Spaced repetition integration**: Output as new ranked .apkg file
