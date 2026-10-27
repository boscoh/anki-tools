# One audit entry point for the five language adapters

The five `run_<lang>_audit` functions repeated the same adapter. Each one built
`AuditInputs(profile, ...)` with seven keyword paths and one word-table loader,
then called `run_gates(<lang>_gates(), inputs, profile)`. Only the profile, the
gate list, and the write time of the language table loader differed.

Status: accepted

## Decision

`run_pack_audit(profile, gates, *, <seven paths>, load_table, probe)` in
`anki_tools/sentence_pack/audit.py` owns the adapter. It fills each empty path
from the profile field (`output_apkg`, `sentence_csv`, `word_csv`,
`word_review_csv`, `audio_dir`, `manifest_path`), keeps `ATTRIBUTION_PATH` and
`probe_duration` as the defaults of the shared module, builds `AuditInputs`, and
calls `run_gates`.

A language module keeps its `<lang>_gates()` list and one adapter. The adapter
takes `**paths` and passes `load_table` from its own module, so a test that
patches the module name still reaches the audit. Mandarin also passes `probe`
for the same reason.

## Consequences

The gate table of every built pack stays byte-identical. A new language needs
one call, not one copy. `sentence_pack/audit.py` still imports no language
package, so `tests/test_language_layout.py` stays green.

## Rejected

Moving the five gate lists into one shared module. The gate lists call language
gate functions, so the module would import every language package.
