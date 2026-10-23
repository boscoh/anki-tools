"""Tests for the Spanish Swadesh sentence pack (S1, the matcher, and the gates)."""

from __future__ import annotations

from anki_tools.languages import resolve
from anki_tools.languages.french.profile import NOUN_INDICES
from anki_tools.languages.spanish import profile as spanish
from anki_tools.sentence_pack.guards import check_sentence_shape

NOUN_ROWS = 84


def test_reviewed_table_has_207_rows_and_207_forms() -> None:
    table = spanish.load_word_table()

    assert len(table) == 207
    assert len({form.spanish for form in table}) == 207
    assert spanish.check_word_table(table) == []


def test_review_file_holds_the_noun_genders_and_the_choices() -> None:
    review = spanish.read_word_review()
    gender_rows = [row for row in review if row["issue"] == "gender"]
    chosen = {
        int(row["index"]): row["chosen"]
        for row in review
        if row["issue"] not in ("gender", "zipf")
    }

    assert len(gender_rows) == NOUN_ROWS
    assert {int(row["index"]) for row in gender_rows} == set(NOUN_INDICES)
    assert chosen[37] == "hombre"
    assert chosen[38] == "persona"
    assert chosen[40] == "esposa"
    assert chosen[129] == "sostener"
    assert all(row["status"] != "open" for row in review)


def test_module_cleanup() -> None:
    assert spanish.clean_term("[[estar]] [[sentado]]") == "estar sentado"
    assert spanish.clean_term("tú (informal)") == "tú"
    assert len(spanish.module_terms()) == 207


def test_form_kinds() -> None:
    table = {form.swadesh_index: form for form in spanish.load_word_table()}

    assert table[125].spanish == "estar de pie"
    assert table[125].form_kind == "pair"
    assert table[134].spanish == "tirar de"
    assert table[134].form_kind == "pair"
    assert table[93].form_kind == "single"


def test_whitespace_normalization() -> None:
    assert spanish.prepare_text("¿Dónde\u00a0estás?  ") == "¿Dónde estás?"
    assert spanish.surface_key("  ÉL ") == "él"


def test_script_rule_accepts_spanish_letters_only() -> None:
    assert spanish.script_ok("¿Dónde está el niño?")
    assert not spanish.script_ok("Tengo 3 perros.")
    assert not spanish.script_ok("Ça va.")


def test_inflected_forms_match_by_lemma() -> None:
    assert spanish.token_matches("Ella bebió agua ayer.", "beber") == 1
    assert spanish.token_matches("Los perros son grandes.", "perro") == 1
    assert spanish.token_matches("Los perros son grandes.", "grande") == 1
    assert spanish.token_matches("Me gusta el perro.", "gato") == 0


def test_pronoun_matches_by_surface_only() -> None:
    assert spanish.token_matches("Él se sentó en la silla.", "él") == 1
    assert spanish.token_matches("Ella se sentó en la silla.", "él") == 0


def test_reflexive_form_matches_the_verb_without_the_clitic() -> None:
    assert spanish.token_matches("Ella se sentó y bebió agua.", "sentarse") == 1
    assert spanish.token_matches("Quiero sentarme aquí.", "sentarse") == 0


def test_pair_form_needs_adjacent_parts() -> None:
    assert spanish.token_matches("Ella está de pie junto a la puerta.", "estar de pie") == 1
    assert spanish.token_matches("Ella está en la puerta de pie.", "estar de pie") == 0


def test_hits_count_a_form_that_appears_one_time() -> None:
    forms = {"beber", "agua", "perro"}

    assert spanish.hits("El perro bebe agua.", forms) == {
        "beber": 1,
        "agua": 1,
        "perro": 1,
    }
    assert spanish.hits("El perro y el perro beben.", forms) == {"beber": 1}


def test_sense_filter_rejects_false_friend_english() -> None:
    assert spanish.sense_filter("sal", "Get out of here!")
    assert not spanish.sense_filter("sal", "Pass me the salt.")
    assert spanish.sense_filter("lengua", "He speaks three languages.")
    assert not spanish.sense_filter("lengua", "The dog has a long tongue.")


def test_highlight_marks_the_matched_token() -> None:
    assert spanish.highlight("Ella bebió agua.", "beber") == "Ella <b>bebió</b> agua."
    assert spanish.highlight("Ella bebió agua.", "gato") == "Ella bebió agua."


def test_unknown_word_filter_with_lemma_matching() -> None:
    known = {"el", "perro", "beber", "agua"}

    assert spanish.count_unknown("El perro bebe agua.", known) == 0
    assert spanish.count_unknown("El perro bebe cerveza.", known) == 1


def test_orphan_conjunction_and_predicate_rules() -> None:
    assert spanish.has_orphan("Y el perro bebe agua.")
    assert not spanish.has_orphan("Y el perro, que bebe agua, duerme.")
    assert spanish.has_predicate("El perro bebe agua.")
    assert not spanish.has_predicate("El perro grande.")


def test_shape_guards_apply_the_spanish_marks() -> None:
    profile = spanish.SPANISH

    low, high = profile.min_units, profile.max_units

    assert check_sentence_shape("¿Dónde está el perro?", profile, low, high) is None
    assert check_sentence_shape("El perro bebe agua", profile, low, high) is not None


def test_registry_resolves_spanish() -> None:
    assert resolve("Spanish") == "spanish"
    assert spanish.SPANISH.tatoeba_lang == "spa"
    assert spanish.SPANISH.tokenizer == "es_core_news_sm"
