# Phase 3 internal RLCR execution note

The user explicitly prohibited the external Codex connection after repeated
reconnection failures. Therefore Phase 3 does not launch Humanize's
`setup-rlcr-loop.sh`, because its implementation/review hooks call `codex exec`
and `codex review` (see the installed `humanize-rlcr` skill). The same round
discipline is retained locally:

1. execute the immutable acceptance criteria in `docs/plans/phase3-plan.md`;
2. keep each numerical/performance experiment attributable and log failed
   branches;
3. revert any round that leaves `tests/kernel` red;
4. obtain two independent internal-agent reviews, one GPT-5.6-Terra High and
   one different model;
5. close only on a green commit.

No `codex exec`, `codex review`, or `ask-codex` command is permitted.
