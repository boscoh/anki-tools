# Spec: Swadesh Vocabulary Correction Sweep

Status: done for the seven CSV files; the review file is open (O1)
Date: 2026-10-18
Owner: anki-tools
Commits: `42d43cd` (seven languages), `5b4d93c` (Spanish)

## 1. Purpose

`vocab/english-<language>.csv` maps one Swadesh English label to one form in
the target language, and to a romanization. A generator gives poor
translations. It picks the wrong sense, the wrong reading, the wrong
register, or a form in the wrong script.

This spec records the manual correction sweep of the seven files. It gives
the rules, the result, and the complete change list. The files are the source
for every Swadesh deck build, so this record belongs in the repository.

## 2. Scope

In scope:

- The seven files, 207 rows each.
- The English label, the target form, and the transliteration.
- The check against the sources.
- The review record.

Out of scope:

- New concepts or new languages.
- The sentence layer. See `specs/mandarin-swadesh-sentence-pack.md`.
- Audio and deck templates.
- Changes to the build code.

## 3. Sources

| Check             | Source                                    | License      | Use                            |
| ----------------- | ----------------------------------------- | ------------ | ------------------------------ |
| English label     | `vocab/swadesh.txt`                       | derived      | Label text at the same index   |
| Canonical form    | `Module:Swadesh/data/{lang}`              | CC BY-SA 4.0 | Row set of accepted forms      |
| Mandarin reading  | CC-CEDICT                                 | CC BY-SA 4.0 | Gloss match for each polyphone |
| Cantonese reading | `tojyutping` via `anki_tools/jyutping.py` | MIT          | Syllabary check                |
| Frequency         | `wordfreq`                                | Apache 2.0   | Flag a form below Zipf 3.0     |

### 3.1 Normalization for the canonical check

These steps apply before the form comparison:

1. Fold case.
2. Remove a parenthetical note.
3. Remove `<notes:...>` and `[[...]]` markup.
4. Remove a leading article for Greek, German, and Spanish.
5. Remove Arabic vowel marks and tatweel.
6. Split a module term on `//` and keep both forms.
7. Convert Chinese forms to Simplified.

The dump date is 2026-10-18. The revision ids are: ar 89015768, cmn
92539038, de 86670657, el 83779807, es 88706918, hi 83632291, yue
72271170.

The module is a check only. Do not copy a module gloss into a deck. The
`wordfreq` value is a flag only. It is not a validator.

## 4. Rules

1. Do not translate the English label. It equals `vocab/swadesh.txt` at the
   same index.
2. Take the form from the module row set. Keep a file form that differs only
   when it is the more common word, and record the reason in the review row.
3. A polyphone takes the reading whose CC-CEDICT gloss matches the label.
   Do not use the first pypinyin reading.
4. Cantonese uses the colloquial character, not the Mandarin equivalent
   (哋, 嚟, 畀, 脷). The reading comes from `tojyutping`.
5. A Greek adjective takes the masculine nominative form (κόκκινος). A noun
   takes the article of its own gender (η κοιλιά, not το κοιλιά).
6. The form set must cover the full label. When the label names both genders
   or several persons, the file lists every form, separated by `/`
   (أَنْتَ/أَنْتِ).
7. The transliteration holds the romanization of the target language only.
   No pinyin in a Cantonese row. No mixed script.
8. No script regenerates a CSV. A generator gives the same wrong reading
   that caused the defect.

## 5. Method

1. Compare the English labels with `vocab/swadesh.txt`.
2. Compare every form with the module row set.
3. For Mandarin, look up each polyphone in CC-CEDICT and take the gloss
   match.
4. For Cantonese, check each reading with `tojyutping`, and check each
   colloquial character.
5. For Greek, check the gender, the article, and the citation form of each
   verb.
6. For Hindi and Arabic, check the vowel marks and the gendered forms.
7. Write one review row for each row that changes.
8. Apply the change in a normal diff. Group the changes by language.

## 6. Result

| Language  | Rows | Forms | Unique forms | Canonical match | Rows changed |
| --------- | ---- | ----- | ------------ | --------------- | ------------ |
| Arabic    | 207  | 211   | 211          | 206 of 207      | 8            |
| Cantonese | 207  | 207   | 207          | 175 of 207      | 48           |
| German    | 207  | 209   | 208          | 201 of 207      | 1            |
| Greek     | 207  | 213   | 210          | 184 of 207      | 42           |
| Hindi     | 207  | 207   | 199          | 205 of 207      | 3            |
| Mandarin  | 207  | 211   | 211          | 193 of 207      | 15           |
| Spanish   | 207  | 253   | 252          | 194 of 207      | 8            |
| Total     | 1449 | 1511  | 1498         | 1358 of 1449    | 125          |

The row counts come from the two commits. The Forms column counts each
slash-separated form as one entry. The canonical match uses the
normalization in section 3.1. The counts were verified on 2026-10-18.

Two count notes:

- An earlier revision of the pack spec reported 16 Mandarin fixes and a total
  of 126. The commits hold 15 changed Mandarin rows and 125 changed rows in
  total. The earlier count was an issue count, not a row count.
- Spanish changed in two commits. `42d43cd` fixed the label of index 3.
  `5b4d93c` fixed 8 rows, and index 3 was one of them.

A reading-only fix does not change the canonical form match. The match
deltas are 1, 29, 1, 32, 3, 4, and 1. The total is 71. The 125 changed rows
and the 71 new matches are different measures.

The sweep does not change the frequency findings. Greek and Hindi scores
have low value, because the corpora store inflected forms, not citation
forms. For example, δαγκώνω scores 0.00 and δαγκώνει scores 3.56. The
Cantonese scan is void, because `wordfreq` has no `yue` list.

## 7. Defect classes

| Class             | Rule                                                                       | Example                                                              |
| ----------------- | -------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| Sense             | The form names a different thing                                           | 胸 → 乳房 (breast), 擦 → 揉 (to rub), drücken → quetschen (squeeze)  |
| Reading           | The romanization is a different word, or the tone is wrong                 | 长 zhǎng → cháng, 数 shù → shǔ, 嘴 hau2 → zeoi2, 眼 ngaam5 → ngaan5  |
| Register          | The form is bookish, or a Mandarin word in Cantonese                       | 许多 → 多, 少数 → 少, 人類 → 人, 丈夫 → 老公, 耳朵 → 耳仔, 嘻笑 → 笑 |
| Character         | The form uses a homophone or a Mandarin character                          | 我地 → 我哋, 黎 → 嚟, 比 → 畀, 舌頭 → 脷, 鳥 → 雀, 站立 → 企         |
| Gender and number | The form covers one gender only                                            | أَنْتَ → أَنْتَ/أَنْتِ, هُمْ → هُمْ/هُنَّ, él/ella → él/ella/ello    |
| Grammar class     | A noun or a participle stands where the label names an adjective or a verb | το κόκκινο → κόκκινος, το κοιλιά → η κοιλιά, δάγκω → δαγκώνω         |
| Orthography       | The spelling or the vowel mark is wrong                                    | दाएँ → दायाँ, बाएँ → बायाँ, सुस्त → कुंद, عِنْد → عِنْدَ, Yo → yo    |
| Label             | The English label is too narrow                                            | he/she → he/she/it/they (singular)                                   |

## 8. Review file

`vocab/swadesh_review.csv` holds columns `lang`, `index`, `english`,
`current`, `canonical`, `issue`, `fix`, `status`. `status` is `fixed`,
`kept`, or `open`. The build fails on any `open` row.

M1 adds the file to the repository (O1). Until then, the change lists in
this spec are the record.

## 9. Reproduce the changes

```bash
git show 42d43cd^:vocab/english-mandarin.csv > /tmp/before.csv
git diff 42d43cd^ HEAD -- vocab/
```

The source checks in section 3 need the module dump and the CC-CEDICT
export. No script is kept. The one-off scripts in `archive/scripts/` are
stale and must not run. They would overwrite the corrected files. The module
revision ids in section 3.1 pin the counts. Fetch each page with
`action=raw` to rebuild the dump.

## 10. Residual findings

- Duplicate forms across rows remain in four files. Greek has 3: αυτό
  (index 3 and 7), το φτερό (70 and 84), το πόδι (80 and 81). German has 1:
  sie (3 and 6). Spanish has 1: en (201 and 202). Hindi has 8: तुम, वह,
  छोटा, पंख, काटना, मारना, तैरना, बर्फ़. A whole-cell duplicate makes two
  cards with the same answer: Greek 70/84 and 80/81, and all 8 Hindi pairs.
  The other cases share a form inside a slash row, so the two cards differ.
  Decide each case: keep both, or add a sense note to the gloss.
- The counts were verified on 2026-10-18 with the live module dumps in
  section 3.1. O3 is closed.
- The review file does not exist. The pack spec gates G15 on it.
- Cantonese index 138 `聯 lyun4` is suspect. 聯 means to unite. To sew is
  縫 fung4 or 車 ce1. Verify before release (O6).

## 11. Rules for the next edit

1. Change one language per commit.
2. Add a review row for each change, with the reason.
3. Give the new form and its reading in the same edit.
4. Keep the old form next to the new form when both are common (老/旧).
5. Do not run a generator over a CSV.

## 12. Acceptance

| Gate | Check                                                            |
| ---- | ---------------------------------------------------------------- |
| S1   | Every English label equals `vocab/swadesh.txt` at the same index |
| S2   | Every transliteration cell is filled                             |
| S3   | For a single-form row, the syllable count equals the Han count   |
| S4   | `english-mandarin.csv` has no duplicate form (Mandarin scope)    |
| S5   | The review file exists and has no `open` row                     |
| S6   | A build reads the CSV and never writes it                        |

Checked on 2026-10-18 for `english-mandarin.csv`: S1 gives 0 mismatches, S2
gives 0 empty cells, S3 gives 0 mismatches, S4 gives 0 duplicates.

## 13. Open items

| Id  | Item                                                            |
| --- | --------------------------------------------------------------- |
| O1  | Commit `vocab/swadesh_review.csv`                               |
| O2  | Add `anki swadesh review <lang>` to run the section 12 checks   |
| O3  | Closed: counts verified on 2026-10-18 (section 3.1)             |
| O4  | Closed: 15 rows changed; the earlier 16 was an issue count      |
| O5  | Review the duplicate forms in Greek, German, Spanish, and Hindi |
| O6  | Verify the Cantonese form for index 138 (聯)                    |

## Appendix A: Complete change list

The list comes from `git diff 42d43cd^ HEAD -- vocab/`. It holds every row
that changed. The `Before` and `After` cells give the form and the
transliteration. `Changed` names the fields that differ.

### A.1 Mandarin (15 rows)

| Index | English                   | Before          | After               | Changed              |
| ----- | ------------------------- | --------------- | ------------------- | -------------------- |
| 3     | he/she/it/they (singular) | 他/她 · tā tā   | 他/她/它 · tā tā tā | label, form, reading |
| 11    | who                       | 谁 · shuí       | 谁 · shéi           | reading              |
| 18    | many                      | 许多 · xǔ duō   | 多 · duō            | form, reading        |
| 20    | few                       | 少数 · shǎo shù | 少 · shǎo           | form, reading        |
| 28    | long                      | 长 · zhǎng      | 长 · cháng          | reading              |
| 85    | belly                     | 肚子 · dǔ zi    | 肚子 · dù zi        | reading              |
| 89    | breast                    | 胸 · xiōng      | 乳房 · rǔ fáng      | form, reading        |
| 131   | to rub                    | 擦 · cā         | 揉 · róu            | form, reading        |
| 138   | to sew                    | 缝 · fèng       | 缝 · féng           | reading              |
| 139   | to count                  | 数 · shù        | 数 · shǔ            | reading              |
| 180   | warm                      | 暖 · nuǎn       | 暖和 · nuǎn huo     | form, reading        |
| 184   | old                       | 旧 · jiù        | 老/旧 · lǎo jiù     | form, reading        |
| 188   | dirty                     | 脏 · zàng       | 脏 · zāng           | reading              |
| 195   | dry                       | 干 · gàn        | 干 · gān            | reading              |
| 203   | with                      | 和 · hé         | 跟 · gēn            | form, reading        |

### A.2 Cantonese (48 rows)

| Index | English       | Before              | After               | Changed       |
| ----- | ------------- | ------------------- | ------------------- | ------------- |
| 4     | we            | 我地 · ngo5 dei6    | 我哋 · ngo5 dei6    | form          |
| 5     | you (plural)  | 你地 · nei5 dei6    | 你哋 · nei5 dei6    | form          |
| 6     | they (plural) | 佢地 · keoi5 dei6   | 佢哋 · keoi5 dei6   | form          |
| 11    | who           | 誰 · seoi4          | 邊個 · bin1 go3     | form, reading |
| 16    | not           | 唔係 · m4 hai6      | 唔 · m4             | form, reading |
| 19    | some          | 一啲 · jat1 di1     | 啲 · di1            | form, reading |
| 20    | few           | 少啲 · siu2 di1     | 少 · siu2           | form, reading |
| 29    | wide          | 寬 · fun1           | 闊 · fut3           | form, reading |
| 31    | heavy         | 重 · zung5          | 重 · cung5          | reading       |
| 35    | thin          | 瘦 · sau3           | 薄 · bok6           | form, reading |
| 37    | man           | 人 · jan4           | 男人 · naam4 jan4   | form, reading |
| 38    | human being   | 人類 · jan4 leoi6   | 人 · jan4           | form, reading |
| 39    | child         | 細路 · sai2 lou6    | 細路 · sai3 lou6    | reading       |
| 41    | husband       | 丈夫 · zoeng6 fu1   | 老公 · lou5 gung1   | form, reading |
| 42    | mother        | 媽咪 · maa1 mi2     | 媽咪 · maa1 mi1     | reading       |
| 46    | bird          | 鳥 · niu5           | 雀 · zoek3          | form, reading |
| 53    | stick         | 條 · tiu4 / zhi1    | 棍 · gwan3          | form, reading |
| 54    | fruit         | 果 · gwo2           | 生果 · saang1 gwo2  | form, reading |
| 55    | seed          | 種子 · zung1 zi2    | 種子 · zung2 zi2    | reading       |
| 69    | tail          | 尾巴 · mei5 ba      | 尾巴 · mei5 baa1    | reading       |
| 70    | feather       | 羽毛 · jyu4 mou5    | 羽毛 · jyu5 mou4    | reading       |
| 71    | hair          | 頭髮 · tou4 faat3   | 頭髮 · tau4 faat3   | reading       |
| 72    | head          | 頭 · tou4           | 頭 · tau4           | reading       |
| 73    | ear           | 耳朵 · ji5 dou6     | 耳仔 · ji5 zai2     | form, reading |
| 74    | eye           | 眼 · ngaam5         | 眼 · ngaan5         | reading       |
| 76    | mouth         | 嘴 · hau2           | 嘴 · zeoi2          | reading       |
| 78    | tongue        | 舌頭 · zi2 gaa2     | 脷 · lei6           | form, reading |
| 82    | knee          | 膝蓋 · sat1 goi3    | 膝頭 · sat1 tau4    | form, reading |
| 84    | wing          | 翼 · yik6           | 翼 · jik6           | reading       |
| 85    | belly         | 肚腍 · tou5 nam4    | 肚 · tou5           | form, reading |
| 87    | neck          | 頸 · bui3           | 頸 · geng2          | reading       |
| 100   | to laugh      | 嘻笑 · hei1 siu3    | 笑 · siu3           | form, reading |
| 103   | to know       | 知道 · zi1 dou6     | 知道 · zi1 dou3     | reading       |
| 104   | to think      | 諗 · lam2           | 諗 · nam2           | reading       |
| 122   | to come       | 黎 · lei4           | 嚟 · lai4           | form, reading |
| 125   | to stand      | 站立 · zaam6 laap6  | 企 · kei5           | form, reading |
| 128   | to give       | 比 · bei2           | 畀 · bei2           | form          |
| 131   | to rub        | 磨 · mo4            | 擦 · caat3          | form, reading |
| 138   | to sew        | 縫紉 · fung4 jan6   | 聯 · lyun4          | form, reading |
| 143   | to float      | 浮 · faau4          | 浮 · fau4           | reading       |
| 145   | to freeze     | 凍 · dung3          | 結冰 · git3 bing1   | form, reading |
| 146   | to swell      | 漲 · zoeng3         | 脹 · zoeng3         | form          |
| 148   | moon          | 月亮 · jyut6 loeng5 | 月亮 · jyut6 loeng6 | reading       |
| 159   | earth         | 地球 · dei6 kau4    | 泥 · nai4           | form, reading |
| 175   | white         | 白 · baak3          | 白 · baak6          | reading       |
| 191   | sharp         | 尖 · zim1           | 利 · lei6           | form, reading |
| 193   | smooth        | 平滑 · ping4 wun6   | 平滑 · ping4 waat6  | reading       |
| 200   | left          | 左 · zuo2           | 左 · zo2            | reading       |

### A.3 Greek (42 rows)

| Index | English                | Before     | After      | Changed |
| ----- | ---------------------- | ---------- | ---------- | ------- |
| 16    | not                    | όχι        | δεν        | form    |
| 33    | short                  | σύντομος   | κοντός     | form    |
| 48    | louse                  | η φθείρα   | η ψείρα    | form    |
| 53    | stick                  | το κλαδί   | το ραβδί   | form    |
| 54    | fruit                  | ο καρπός   | το φρούτο  | form    |
| 85    | belly                  | το κοιλιά  | η κοιλιά   | form    |
| 86    | guts                   | τα σκάφη   | τα σπλάχνα | form    |
| 87    | neck                   | το λαιμό   | ο λαιμός   | form    |
| 94    | to bite                | δάγκω      | δαγκώνω    | form    |
| 95    | to suck                | συναγμένος | ρουφώ      | form    |
| 97    | to vomit               | εμετίζω    | ξερνώ      | form    |
| 117   | to scratch             | χαϊδεύω    | ξύνω       | form    |
| 118   | to dig                 | σκαλίζω    | σκάβω      | form    |
| 123   | to lie down            | βάζομαι    | ξαπλώνω    | form    |
| 126   | to turn (intransitive) | στρίβω     | γυρίζω     | form    |
| 146   | to swell               | πρηστάνω   | πρήζομαι   | form    |
| 152   | river                  | ο ποτάμι   | το ποτάμι  | form    |
| 158   | dust                   | το σκόνη   | η σκόνη    | form    |
| 169   | to burn                | το καίω    | καίω       | form    |
| 172   | red                    | το κόκκινο | κόκκινος   | form    |
| 173   | green                  | το πράσινο | πράσινος   | form    |
| 174   | yellow                 | το κίτρινο | κίτρινος   | form    |
| 175   | white                  | το λευκό   | λευκός     | form    |
| 176   | black                  | το μαύρο   | μαύρος     | form    |
| 179   | year                   | το έτος    | ο χρόνος   | form    |
| 180   | warm                   | ζεστό      | ζεστός     | form    |
| 181   | cold                   | κρύο       | κρύος      | form    |
| 182   | full                   | γεμάτο     | γεμάτος    | form    |
| 183   | new                    | νέο        | νέος       | form    |
| 184   | old                    | παλιό      | παλιός     | form    |
| 185   | good                   | καλό       | καλός      | form    |
| 186   | bad                    | κακό       | κακός      | form    |
| 187   | rotten                 | σάπιο      | σάπιος     | form    |
| 188   | dirty                  | βρώμικο    | βρόμικος   | form    |
| 189   | straight               | ευθύς      | ίσιος      | form    |
| 190   | round                  | στρογγυλό  | στρογγυλός | form    |
| 191   | sharp                  | αιχμηρό    | αιχμηρός   | form    |
| 192   | dull                   | τυπικό     | στομωμένος | form    |
| 193   | smooth                 | λείο       | λείος      | form    |
| 194   | wet                    | βρεγμένο   | βρεγμένος  | form    |
| 195   | dry                    | ξηρό       | ξηρός      | form    |
| 196   | correct                | σωστό      | σωστός     | form    |

### A.4 German (1 row)

| Index | English    | Before  | After     | Changed |
| ----- | ---------- | ------- | --------- | ------- |
| 130   | to squeeze | drücken | quetschen | form    |

### A.5 Spanish (8 rows)

| Index | English                   | Before              | After        | Changed     |
| ----- | ------------------------- | ------------------- | ------------ | ----------- |
| 1     | I                         | Yo                  | yo           | form        |
| 3     | he/she/it/they (singular) | él/ella             | él/ella/ello | label, form |
| 22    | one                       | un/una              | uno          | form        |
| 134   | to pull                   | tirar               | tirar de     | form        |
| 192   | dull                      | desafilado/sin filo | romo/roma    | form        |
| 199   | right                     | derecho/derecha     | derecha      | form        |
| 200   | left                      | izquierdo/izquierda | izquierda    | form        |
| 201   | at                        | en                  | en/a         | form        |

### A.6 Hindi (3 rows)

| Index | English | Before       | After           | Changed       |
| ----- | ------- | ------------ | --------------- | ------------- |
| 192   | dull    | सुस्त · sust | कुंद · kund     | form, reading |
| 199   | right   | दाएँ · daan  | दायाँ · daayaan | form, reading |
| 200   | left    | बाएँ · baan  | बायाँ · baayaan | form, reading |

### A.7 Arabic (8 rows)

| Index | English                   | Before         | After                        | Changed       |
| ----- | ------------------------- | -------------- | ---------------------------- | ------------- |
| 2     | you (singular)            | أَنْتَ · anta  | أَنْتَ/أَنْتِ · anta/anti    | form, reading |
| 3     | he/she/it/they (singular) | هُوَ · huwa    | هُوَ/هِيَ · huwa/hiya        | form, reading |
| 6     | they (plural)             | هُمْ · hum     | هُمْ/هُنَّ · hum/hunna       | form, reading |
| 7     | this                      | هٰذَا · hadha  | هٰذَا/هٰذِهِ · hadha/hadhihi | form, reading |
| 187   | rotten                    | فَاسِد · fasid | مُتَعَفِّن · muta'affin      | form, reading |
| 201   | at                        | عِنْد · ind    | عِنْدَ · inda                | form, reading |
| 203   | with                      | مَع · ma'      | مَعَ · ma'a                  | form, reading |
| 205   | if                        | إِن · in       | إِنْ · in                    | form          |
