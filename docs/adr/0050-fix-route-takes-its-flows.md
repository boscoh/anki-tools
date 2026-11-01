# Pass the flows and gates to `run_fix` as keyword arguments

`tests/test_cli_fix.py` held six `monkeypatch.setattr` calls on repo code. They
replaced `cullflow.run_cull`, `coreflow.run_core`, `check_cull`, and
`check_core` inside `fixrun`. The module docstring of `fixrun.py` and ADR 0006
name the CLI-to-flow boundary as the seam, but the seam had no adapter, so the
tests patched module attributes instead.

Status: accepted

## Decision

`fixrun.run_fix` takes four keyword arguments: `run_core`, `run_cull`,
`check_core`, and `check_cull`. The defaults are the real functions, so the
four deck commands pass nothing and behave as before. `_run_core` and
`_run_cull` receive the two functions they call. The tests pass recorders and
read the calls, and no test patches repo code.

## Consequences

The seam has two adapters: the real flows and gates, and a recorder in
`tests/test_cli_fix.py`. A route test names the language, the mode, the
profile, and the default output, and reads the request that the flow would get.
Zh, yue, and es routes, the `--select` error of a deck without a cull, and the
`--output`, `--words`, and gate-exit-code paths now have rows.

`cli.py` does not expose the four arguments. The command options do not change.
The wiring from a CLI option to `run_fix` is checked only for the alias and the
`--check` guard, which run before any flow.

## Rejected

Keep the six mocks. They pin the module attribute names of `fixrun`, so a
rename of an import breaks a test that no behavior change touched.

A `FixFlows` dataclass that holds the four functions. One argument replaces
four, but the type has one real value and one test value, and `run_fix` then
reads a field where it now reads a name.
