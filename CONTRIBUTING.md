# Contributing to NoNans

Thank you for your interest in NoNans.

This repository contains the **open-source detection layer** and **public
client** for the NoNans resolution runtime. Contributions are welcome on
everything that lives here. Contributions to the resolution runtime itself
are not accepted from outside collaborators.

## Where to contribute

We actively welcome:

- **Detection improvements.** New event kinds, better classifiers, smaller
  hook overhead.
- **Integrations.** Examples and adapters for additional frameworks
  (Megatron-Core, Mosaic, Composer, NeMo, Lightning, Ray Train).
- **Benchmark workloads.** Reproducible scripts that exercise specific
  numerical-stability regimes.
- **Documentation.** Anything that helps an ML engineer who has lost a
  training run to NaN understand what NoNans does and how to use it.
- **Bug reports** of all severities. The detection layer should never
  silently drop events; if you find one, we want to know.

We do **not** accept:

- Contributions to the resolution mechanism. The mechanism is patent-
  pending and ships only inside the licensed runtime; we maintain it
  internally and do not accept code that touches resolution logic.
- Reverse-engineered descriptions of the runtime binary's behavior.

## How to contribute

1. **Open an issue first** for non-trivial changes. We don't want anyone
   investing serious effort in a PR that conflicts with our roadmap.
2. **Fork, branch, and PR** as you would for any open source project.
3. **Sign your commits** (`git commit -s`). We require DCO sign-off; we
   do not require a CLA.
4. **Add tests.** New detection logic should ship with a test in
   `tests/detect/` that exercises the case it covers.
5. **Run the linter and tests locally** before pushing:
   ```bash
   pip install -e .[dev]
   ruff check .
   mypy nonans/
   pytest
   ```

## Style

- Python: ruff handles formatting and linting. Configuration in
  `pyproject.toml`.
- Type hints: required for new code in `nonans/`.
- Docstrings: required for new public functions; brief is fine.
- Commit messages: imperative mood, ~72 char subject, longer body if the
  change is non-obvious.

## Reporting bugs

Open a GitHub issue with:

- The version of NoNans you're running (`python -c "import nonans; print(nonans.__version__)"`).
- The PyTorch version, CUDA version, and GPU model.
- A minimal reproduction (a small Python script is ideal).
- The contents of `./.nonans/events.jsonl` from the affected run, if any.

For security issues, see [SECURITY.md](SECURITY.md). Do not file public
issues for security bugs.

## Code of conduct

Be kind, be technical, and assume good faith. We don't have a long code
of conduct because we don't think we need one. If something feels off,
mail engineering@nonans.com and a real person will read it.

---

By contributing, you agree that your contributions will be licensed under
the MIT license that covers this repository.
