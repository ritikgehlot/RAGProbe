# RAGProbe commit history - run in PowerShell from the project root.
# Prereq: git config --global user.name / user.email already set.
git init -b main
git add .gitignore
git commit -m "chore: scaffold project structure"
git add LICENSE pyproject.toml
git commit -m "chore: add MIT license and zero-dependency package config"
git add src/ragprobe/__init__.py src/ragprobe/dataset.py
git commit -m "feat: add SQuAD ingestion and grounded-answer eval-set builder"
git add src/ragprobe/perturb.py
git commit -m "feat: add four seeded contamination generators with sentence-level ground truth"
git add src/ragprobe/scorers/
git commit -m "feat: add scorer interface with lexical, NLI, and LLM-judge implementations"
git add src/ragprobe/runner.py
git commit -m "feat: add meta-evaluation runner (precision, recall, per-type, localization)"
git add src/ragprobe/report.py
git commit -m "feat: add self-contained HTML audit report with detection matrix and case files"
git add src/ragprobe/cli.py
git commit -m "feat: add ragprobe build/run CLI"
git add tests/
git commit -m "test: pin perturbation determinism, scorer behavior, and judge parsing"
git add README.md scripts_git_history.ps1
git commit -m "docs: add README with live meta-evaluation results and limitations"
git add -A
git commit -m "chore: include remaining project files"
git log --oneline
