# The default output path is the only path

`anki fix` takes `--output`, so one run has two possible paths. The mode writes
the default path when the flag is absent, and `--check` reads whatever path the
caller passes. A caller who passes `--output` to the mode and forgets the flag
on `--check` audits a different file and reads a stale result.

The flag costs 17 doc lines, one test, two command declarations, and a parameter
through `fixrun.run_fix`, `_run_core`, and `_run_cull`. No user asked for it, and
ADR 0015 already made the default path the source of every name.

Status: accepted. Supersedes the `--output` row of ADR 0015 and the `--output`
row of ADR 0009.

## Decision

`anki fix` holds no `--output` option. `identity.default_output(language, mode)`
gives the only output path of a run, and `--check` reads that path.

The same rule applies to the Swadesh commands, which never had the flag.

## Consequences

A caller cannot name a second output. A copy of the written package covers the
one-off case, and a copy does not change the title, the note type, or the side
files, which is what the flag would have done.

The mode and the gate now read one path, so `--check` cannot audit a file that
the run did not write.

A change of the default path moves the artifact, its title, its note type, and
its side files together (ADR 0015).

## Rejected

Keep the flag for an experiment. The experiment is one `cp` after the run.

Derive the output path from the input path. The input name then leaks into the
output title, which ADR 0015 already rejected.
