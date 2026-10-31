# Assemble the card templates in one module

The three deck routes state the templates and the CSS in three places.

`cli._build_layout` (`cli.py:717`) reads `templates/vocab/front.html` and
`templates/vocab/back.html`, joins the shared sheet with the language sheet
through `load_css_layers`, appends the character-link script, and fills the
`{{fields}}` slot of the back template from a class table. `pack.build_pack`
joins `profile.front_template` and `profile.back_template`, appends the same
character-link script, and joins the CSS through its own `load_css`
(`pack.py:170`). `assemble.deck_templates` removes the
`{{#Romanization}}…{{/Romanization}}` block when the language has no
romanization field.

So two modules join CSS layers, two append the script, and the block rule sits
in the module that writes the package rather than the module that states the
template. The rules are the same for all three routes; only the text differs.

Status: accepted. Extends ADR 0013 and does not change the field order of any
note type.

## Decision

`anki_tools/apkg/templates.py` owns the assembly. One function:

```python
assemble_templates(spec: TemplateSpec) -> TemplatePair
```

`TemplateSpec` holds the front text, the back text, the CSS layer texts in
order, the field names of the note type, a `char_links` switch, and the names of
the optional blocks. `TemplatePair` holds the front text, the back text, and the
joined CSS.

- The module removes an optional block when the note type has no field of that
  name. `Romanization` is the only block today.
- The module appends the character-link script after the back text when
  `char_links` is true.
- The module joins the CSS layers with one newline. A later layer wins.
- The module reads no file. A caller reads the text and passes it.
- The vocab class table stays in the vocab adapter. That adapter builds the back
  text from the fields, then calls the module.
- `pack.load_css`, `assemble.ROMANIZATION_BLOCK`, and `assemble.deck_templates`
  are deleted. `load_css` moves into `templates.py`, because the five language
  profiles use it to state their card CSS. The file readers `load_text`,
  `load_optional_text`, and `load_char_links` stay there too, because the
  callers still use them.

## Consequences

One module states the block rule, the script rule, and the CSS join, so the
vocab deck and the sentence pack cannot drift from the fix routes. Four tests
hold the rules in `tests/test_card_templates.py`: the block rule, its keep case,
the script switch, and the layer order.

The seam is real: three routes cross it. The interface is one record and one
function, and the implementation holds the three rules.

The `char_links` switch stays a switch rather than a text parameter, because the
script path and the CC-Canto attribution are one decision, and only the Han
decks set it.

## Rejected

One function per route. The routes differ in the text they give, not in the
rules they apply.

The module reads the template files. The sentence pack and the fix routes hold
their templates as strings, so the module would need two loaders for one seam.

Passing the character-link text instead of a switch. The caller then decides
where the script sits in the back template, which is the rule this module owns.
