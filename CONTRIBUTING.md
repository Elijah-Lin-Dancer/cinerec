# Contributing / 参与贡献

Thanks for your interest in CineRec! This is a research/portfolio project, so
contributions are light-touch: bug reports, documentation fixes and small,
well-scoped pull requests are all welcome.

感谢关注 CineRec！本项目以研究/展示定位为主，欢迎的问题反馈、文档修正与小而聚焦的
Pull Request。

## Development setup / 开发环境

```bash
# Runtime + dev tooling (lint, format, tests)
pip install -r requirements.txt -r requirements-dev.txt

# Full ML stack only if you touch training/inference code
pip install -r requirements-train.txt
```

## Before opening a PR / 提交前自查

```bash
ruff check .        # lint (must pass)
black --check .     # formatting
python -m pytest    # test suite (CI runs the same, plus e2e/a11y and Docker jobs)
```

Tests default to `-m 'not e2e'`; browser journeys are opt-in and run in a
dedicated CI job.

## Guidelines / 约定

- **One change, one intent**: keep pull requests focused; unrelated refactors
  split into separate PRs.
- **Numbers come from artefacts**: any metric you add or change must be
  reproducible from a committed results JSON (see `docs/README.md` for the
  evaluation protocol) — never hand-edit numbers in tables.
- **Style**: the existing voice is bilingual (中文 / English) for user-facing
  docs and English for code comments/docstrings; follow the file you edit.
- **Data**: MovieLens is redistributed under its non-commercial terms; do not
  commit third-party datasets beyond what `docs/adr/0002-artifact-storage.md`
  allows.

## Reporting bugs / 报告问题

Open a GitHub issue with: what you expected, what happened, your environment
(Python version, OS, `APP_MODE`), and reproduction steps. Screenshots or a
failing test snippet help a lot.
