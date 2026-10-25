# AGENT.md - Development Guide

Follow these instructions precisely for all sessions.

## Project Tools

**Markdown**: Do not use emojis; always format tables properly with aligned columns

## Python
- Use `uv` for package management and running scripts.
- use sphinx docstrings

## Code

When writing code:
- don't include obvious comments or docstrings
- reuse existing functions if possible
- include types

## Design

Use the deep-module vocabulary: module, interface, depth, seam, adapter,
leverage, locality. Read `GLOSSARY.md` before you name a new concept. Put a
small interface in front of a large implementation. Put the seam where a test
can use it. Record one ADR in `docs/adr/` for each new seam or boundary
decision. The formats live in the `codebase-design` and `domain-modeling`
skills.

## Simplicity (ponytail)

Ponytail mode `full` is active in every session. The `pi-ponytail` extension
injects the ruleset into the system prompt. Ponytail limits how much you build.
It does not lower the depth of a module. Before you write code, climb the
ladder: does this need to exist (YAGNI); is it already in the codebase; does
the standard library hold it; is it native to the platform; is it an installed
dependency; is it one line; is it the minimum that works. For a review of a
diff, run `/ponytail-review`.

## File Edits

Do not edit files with a Python script, `sed`, `awk`, `perl`, or a heredoc. Use `replace` and `insert`, which check anchors and support undo. Use `write` for a new file or a complete rewrite only.


## Issue Tracking (Beads)

Work is tracked in **Beads** (`bd`, Dolt-backed). `bd prime` is the single
source of truth for the workflow; run it after compaction or in a new session.
The `beads` skill lives at `.agents/skills/beads/SKILL.md`. Never keep a second
task list in markdown.

```bash
bd ready                        # unblocked work
bd create "title" -t task -p 2  # new issue; add --json for structured output
bd update <id> --claim          # claim atomically
bd update <id> --append-notes="Progress..."
bd close <id> --reason "..."    # complete
bd sync                         # pull, conflict check, push (at handoff)
```

- Issues live in a local Dolt database (`.beads/embeddeddolt/`). Dolt refs on
  the git remote (`refs/dolt/data`) are the sync protocol. `.beads/issues.jsonl`
  is a passive local artifact, never a source of truth. Fresh clones run
  `bd bootstrap`.
- `bd sync` is a manual handoff step. Auto-push is off (`dolt.auto-push:
  false`), so run `bd sync` before you hand off. Never run `bd dolt push` by
  hand.
- Never use `bd edit`; it opens an interactive `$EDITOR`. Update a field with a
  flag, for example `bd update <id> --description ...`.
- Link discovered work with `--deps discovered-from:<id>`.
- Put `## Acceptance Criteria` in the issue description. A bug also gets
  `## Steps to Reproduce` and an epic gets `## Success Criteria`.
  `bd lint` checks these sections.
- A long-lived epic stays open while a child is open, so `bd doctor
  --check=conventions` reports it as an orphan. The flag is expected; never
  close the epic to clear it.
- Git hooks are shared and committable: `git config core.hooksPath
  .beads-hooks` (installed with `bd hooks install --shared`). Do not use
  `.git/hooks`.

### After Each Task (MANDATORY)

1. Close the issue: `bd close <id> --reason="..."`
2. Stage and commit: `git add -A && git commit -m "Close <id>: description"`
3. If a relevant spec or migration markdown file exists, update it too
4. Only then start the next task

This keeps one atomic commit per task and a clean git history.


## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync            # pull, conflict check, push the Dolt refs
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
