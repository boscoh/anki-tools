# Give the core deck a registry capability

The four core modules (`mandarin_core`, `french_core`, `spanish_core`,
`cantonese_core`) were reachable only through `anki fix <lang> --core`. The
registry already held the sentence-pack capability (`module`), the profile, and
the audit runner, so the core deck was the one capability outside the registry.

Status: accepted

## Decision

`CliLanguage` holds a fifth capability: `core: ModuleType | None`. The field is
set for mandarin, french, spanish, and cantonese, and stays None for greek,
german, arabic, and hindi. The `anki swadesh core <language>` command reads the
field, fills the unset tuning values from the module
(`DEFAULT_CORE_WORDS`, `DEFAULT_OPTIONS.per_word`, `DEFAULT_OPTIONS.tail_cards`),
and calls the shared `_run_core_command`. A language with no core module prints
the unsupported message and returns.

## Consequences

The core deck sits next to profile, module, and audit, so one place answers
which capability a language has. The four `fix <lang> --core` paths stay as
they are, and both paths share one skeleton. A new language needs one registry
line and no new command.

## Rejected

A generic `capabilities` mapping on `CliLanguage`. It moves the existing fields
and the call sites that read them, and it buys nothing while the language has
four capabilities and a fixed command set.
