# Dromeus source of truth

The canonical project documentation is the Obsidian Dromeus folder:

`/Users/devesh1011/Obsidian-Vault-Local/10 Dromeus/`

For every M2 planning, implementation, review, benchmark, or status task, read these
two files first:

1. Current implementation and evidence:
   `/Users/devesh1011/Obsidian-Vault-Local/10 Dromeus/M2 Current Progress.md`
2. Scope, workstreams, decisions, and acceptance criteria:
   `/Users/devesh1011/Obsidian-Vault-Local/10 Dromeus/plans/M-2 implementation plan.md`

Read the supporting source that matches the task:

- Architecture and module ownership:
  `/Users/devesh1011/Obsidian-Vault-Local/10 Dromeus/Architecture.md`
- AXL behavior and transport constraints:
  `/Users/devesh1011/Obsidian-Vault-Local/10 Dromeus/How AXL works?.md`
- NoLoCo paper/source decisions:
  `/Users/devesh1011/Obsidian-Vault-Local/10 Dromeus/M2 Workstream 1 - Paper and Reference Audit.md`
- Benchmark model decision:
  `/Users/devesh1011/Obsidian-Vault-Local/10 Dromeus/M2 Benchmark Model Selection.md`
- Historical grant scope:
  `/Users/devesh1011/Obsidian-Vault-Local/10 Dromeus/Dromeus Final Proposal.md`

## Authority and maintenance

- The M2 plan defines intended scope and gates. The current-progress note records
  what is actually implemented, tested, or evidenced.
- Treat checkboxes as claims to verify against code, tests, and retained evidence.
- When documentation conflicts, use Architecture for technical invariants, the M2
  plan for milestone requirements, and M2 Current Progress for present status.
- Update `M2 Current Progress.md` after implementation or benchmark work.
- Update the M2 implementation plan when scope, frozen values, decisions, or
  acceptance criteria change.
- Keep implementation details in code and tests; keep project decisions and status
  in Obsidian. Repository Markdown is not the canonical milestone record.
