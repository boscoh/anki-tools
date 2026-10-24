# Anki Fix Input Sources

This page gives the local input file and the upstream source of each
`anki fix` command.

`anki fix` does not download a source. It reads the APKG file that you give
it. The command has a default input for French and Cantonese. Chinese and
Spanish need an explicit input.

## 1. Summary

| Command   | Local default input                            | Upstream source                                          |
| --------- | ---------------------------------------------- | -------------------------------------------------------- |
| `fix zh`  | none, the input is required                    | Neri's Chinese Sentences; Xefjord's Complete Mandarin    |
| `fix fr`  | `apkg/6000_French_Sentences_w_Audio.apkg`      | AnkiWeb 1263692251, `6000 French Sentences w/ Audio`     |
| `fix yue` | `apkg/LTL Cantonese Deck Level 1 - Short.apkg` | AnkiWeb 906382208, `LTL Cantonese Deck Level 1 - Short`  |
| `fix es`  | none, the input is required                    | AnkiWeb 1713698257, `9000 Spanish sentences`             |

## 2. French

Local file: `apkg/6000_French_Sentences_w_Audio.apkg`

Deck name: `5000+ French Sentences w/ Audio`

Upstream page: <https://ankiweb.net/shared/info/1263692251>

The deck description links to that page. The page says the author took a
6,000-sentence French deck from
<https://ankiweb.net/shared/info/1419404332> and added audio with HyperTTS.
The audio files start with `hypertts-`.

The repo spec records the source of each part:

| Part               | Source                                      |
| ------------------ | ------------------------------------------- |
| French sentence    | Tatoeba and the deck author                 |
| English `Back`     | Tatoeba and the deck author                 |
| Audio              | Microsoft Azure TTS through HyperTTS        |
| `freq ranks` field | Unknown. The cull ignores the field.        |

See `specs/french-1000-sentence-cull.md` section 2 for the input analysis.
All French modes use this input: rank, `--select`, and `--core`.

## 3. Cantonese

Local file: `apkg/LTL Cantonese Deck Level 1 - Short.apkg`

Deck name: `LTL Cantonese Deck Level 1 - Short`

Upstream page: <https://ankiweb.net/shared/info/906382208>

The source is LTL Flexi Classes. The deck has 999 cards. The audio files
start with `azure-`, so the audio is Azure TTS. The note type has the fields
`Front`, `Back`, `Audio`, `Jyutping`, and `Eng Audio`.

## 4. Spanish

Local file: `apkg/Spanish 9000 sentences with native audio.apkg`

Deck name: `Spanish 9000 sentences with native audio`

Upstream page: <https://ankiweb.net/shared/info/1713698257>

The deck description says the sentences come from Tatoeba.org. The audio is
a native recording by a speaker from central Spain. The note fields hold
`EnglishSentenceID` and `SpanishSentenceID`, which are Tatoeba ids. The deck
has 9,263 cards.

## 5. Chinese

`fix zh` has no default input. The repo holds these Chinese source decks:

| Local file                                                         | Source                                  |
| ------------------------------------------------------------------ | --------------------------------------- |
| `apkg/Mandarin (Chinese) (25-02-22).apkg`                          | Xefjord's Complete Mandarin (Chinese)   |
| `apkg/Neri's Chinese Sents Part 1.apkg`                            | Neri's Chinese Sentences, 2024 deck     |
| `apkg/Neri's Chinese Sents Part 2.apkg`                            | Neri's Chinese Sentences, 2024 deck     |
| `apkg/zh_Chinese_Commonly_Used_Sentences_w_Word_Explanations.apkg` | A derived deck from Neri's Part 1       |

Xefjord's deck is from <https://xefjord.wixsite.com/xefscompletelangs>. It
holds Hanzi, vocabulary, and basic phrase sections.

Neri's deck is from <https://frequencylists.blogspot.com>. The note model is
`0 2024 Sentences Deck`. It holds the sentence, the annotation, the Latin
reading, the translation, and the explanations.

The `zh_Chinese_Commonly_Used_Sentences_w_Word_Explanations` deck is not an
upstream deck. It is a prepared copy of Neri's Part 1. The `_processed.apkg`
file is a later local output.

## 6. Command examples

```bash
uv run anki fix zh apkg/input.apkg
uv run anki fix fr                     # uses the 6000-sentence deck
uv run anki fix fr --select            # same input, 1000-card cull
uv run anki fix fr --core              # same input, core deck
uv run anki fix yue                    # uses the LTL deck
uv run anki fix es apkg/input.apkg
```

`uv run anki fix <lang> --help` shows the default input of `fr` and `yue`.
