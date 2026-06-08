## What & why

<!-- What does this PR change, and why? Link any related issue (e.g. "Closes #12"). -->

## Type of change

- [ ] Bug fix
- [ ] New feature (metric / optimizer / strategy / report / provider)
- [ ] Refactor (no behaviour change)
- [ ] Docs / tests / chore

## Checklist

- [ ] `ruff check src/` passes
- [ ] `pytest test/ -q` passes, coverage stays ≥ 80%
- [ ] New behaviour has tests (offline — uses a fixture/CSV `DataProvider`, not live Yahoo)
- [ ] No look-ahead bias introduced (decisions at day *d* use only data up to *d−1*)
- [ ] CHANGELOG updated under `[Unreleased]` if user-facing
- [ ] Flagged any change to reported numbers or any breaking API change (SemVer)

## Notes for the reviewer

<!-- Anything non-obvious, trade-offs, or follow-ups. -->
