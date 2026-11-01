# Resolve language packages lazily in the registry

Status: accepted

## Context

`import anki_tools.languages` took 0.6 s, because the registry imported every
language profile at the start. jieba (Mandarin), ToJyutping (Cantonese), and the
French and Spanish modules cost most of it. Every `anki` command paid it, and
`anki --help` took 1.05 s, even for a command that needs one language.

## Decision

`CliLanguage` and `FixCommand` hold dotted paths (`profile_path`, `module_path`,
`audit_path`, `core_path`, `cull_path`, as `package.module:attribute`). The
properties `profile`, `module`, `audit`, `core`, and `cull` import the target on
first read through one helper, `_load`. The public names and types of the
properties do not change.

`PROFILES`, `get_profile`, and `profile_names` register every profile before
they answer, because a profile registers itself when its module is imported.
`cli.py` builds the `fix` route inside the command, not at start. The default
word count of a `fix` command is a literal in the registry, because the CLI
needs it at start.

## Consequences

`import anki_tools.cli` takes 0.19 s, and `anki --help` takes 0.24 s without
`uv run`. A wrong path, or a language that fails to import, shows at the first
read of that language, not at start. `test_language_registry` reads every path,
checks that a bad module or attribute raises, and checks in a fresh interpreter
that the CLI import loads no language package, jieba, ToJyutping, or spaCy.

## Rejected

A plugin or entry-point registry. Five languages in one repository do not need
it.

Keeping `LANGUAGES` eager and making each language package lazy inside. The
registry itself held the references to the heavy modules, so each package would
need a proxy.
