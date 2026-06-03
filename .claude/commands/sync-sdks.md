---
description: After Python SDK changes land, produce a porting plan markdown file per target SDK (JS, Go) so a developer can mirror the changes manually.
argument-hint: "[base-ref | --pr <num>]"
allowed-tools: Read, Grep, Glob, Bash(git diff:*), Bash(git log:*), Bash(git status:*), Bash(git rev-parse:*), Bash(git branch:*), Bash(gh pr view:*), Bash(date:*), Write
---

Invoked in `cyborgdb-py` after Python SDK changes are ready (branch built, optionally a PR open). Produces one porting plan markdown file per target SDK — `cyborgdb-js` and `cyborgdb-go` — that the developer reviews before manually mirroring the changes in those repos.

**Rationale:** per @ahellegit (2026-06-03), continuously updating SDKs alongside service changes creates churn. The agreed flow is: ship Python SDK first → make Python's test suite green → cascade to JS + Go using Claude-generated porting plans → human implements + reviews.

**Plans live in this repo** at `sync-plans/<id>/<target>.md`. That keeps the plan next to the change that caused it, gets it reviewed via standard PR flow, and gives the JS/Go owner a single artifact to work from when porting.

---

## Inputs

- `$1` (optional):
  - `--pr <num>` → diff the PR's commits in `cyborginc/cyborgdb-py`
  - Otherwise, treated as a base ref to diff against; default `origin/main`

## Steps

1. **Resolve the diff scope.**
   - If `--pr <num>`: `gh pr view <num> --json baseRefName,headRefOid,commits,url,title` — use `baseRefName` as base, `headRefOid` as head, capture title for the plan summary
   - Otherwise: base = `$1` or `origin/main`; head = current `HEAD`
   - Get the commit list: `git log --oneline <base>..<head>`

2. **Identify changed public-surface files.**
   - `git diff --name-only <base>..<head> -- 'cyborgdb/**/*.py'`
   - Exclude: `cyborgdb/test/`, `cyborgdb/openapi_client/` (auto-generated), any file starting with `_`
   - The remaining files are the public-surface candidates

3. **Extract public-surface deltas.** For each changed file:
   - Read the file at base (`git show <base>:<path>`) and at head (current contents)
   - Identify added / modified / removed:
     - Class declarations and their public methods (no leading `_`)
     - Top-level function declarations
     - Type aliases / Pydantic models / dataclasses
     - Public constants and Enum values
   - A signature change counts as a delta even if the function name is unchanged

4. **Generate a short identifier for the plan.**
   - If `--pr`: `pr-<num>`
   - Otherwise: kebab-cased version of the current branch name (strip leading `feat/`, `fix/`, etc.)
   - Truncate to ≤40 chars

5. **Write one plan file per target SDK** at `sync-plans/<id>/js.md` and `sync-plans/<id>/go.md`.

   Front matter:
   ```yaml
   ---
   title: "<id> — port to <target>"
   target_sdk: cyborgdb-js | cyborgdb-go
   source_branch: "<branch>"
   source_pr: "<PR url, if --pr was used>"
   source_commit: "<head SHA>"
   created: "YYYY-MM-DD"
   status: draft   # draft | reviewed | implemented | obsolete
   ---
   ```

   Body sections, in this order:

   - **`## Summary`** — 1–3 sentences on what the Python change does and why.
   - **`## Public surface deltas`** — bullet list of every public-symbol change. Format each as:
     ```
     - **<kind>:** `<symbol path>` — <added | modified | removed>
       - Before: `<signature at base>`
       - After:  `<signature at head>`
       - Notes: <behavior / docstring / default-value changes worth flagging>
     ```
   - **`## Target translation`** — for each delta above, the equivalent in the target language. TypeScript for JS, Go idioms for Go. Note type-mapping decisions (e.g. `Optional[int]` → `number | undefined` vs `number | null`; `dict[str, Any]` → `Record<string, unknown>` vs `Map<string, any>`).
   - **`## Files likely to touch`** — best-guess list of paths in the target repo (cyborgdb-js or cyborgdb-go). State these as guesses, not authoritative — the porter should verify. Map from `cyborgdb/client/index.py` to e.g. `src/client/index.ts` and `client/index.go`.
   - **`## Tests to mirror`** — list any tests in `cyborgdb/test/` that were added or changed, and the equivalent test the target SDK should add.
   - **`## Breaking changes`** — explicit section. If any signature change is binary-incompatible or removes a public symbol, list here and **flag prominently at the top of the file**. If none, write "None."
   - **`## Acceptance`** — observable check: the target SDK's existing test suite passes after porting; any newly mirrored tests also pass.

6. **Update the sync-plans index.**
   - Append a row to `sync-plans/README.md` under the "Plans" table: `| <id> | <created> | draft | <source PR url> |`

7. **Output:**
   - Paths of the plan files written
   - One-line next-step prompt: "Review via PR, then assign to the JS / Go SDK owners for implementation"

## When NOT to use

- For purely internal Python changes (no public-surface deltas) — produces an empty plan, just skip
- For docs-only or test-only changes
- For changes to `cyborgdb/openapi_client/` (auto-generated from the service's OpenAPI spec; the JS/Go SDKs regenerate from the same source independently)

## Guardrails

- **The skill does not push or open PRs.** It writes files locally; the developer commits + reviews via standard flow.
- **Plans are advisory.** The developer porting to JS/Go should verify file paths and idiom translations — the skill is a checklist, not authoritative source.
- **Breaking changes get a top-of-file flag.** If any public symbol is removed or has a binary-incompatible signature change, the plan should open with a `> ⚠️ Breaking change: <summary>` blockquote.
- **Don't try to merge with prior plans.** If a plan with the same `<id>` already exists, suffix with `-2`, `-3`, etc. Concurrent planning is rare but possible.
