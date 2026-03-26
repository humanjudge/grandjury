# Contributing to GrandJury

Thanks for your interest in contributing to the GrandJury Python SDK.

## Development Setup

```bash
git clone https://github.com/humanjudge/grandjury.git
cd grandjury
uv sync --dev
```

## Running Tests

```bash
uv run pytest tests/ -v
```

## Code Style

- Python 3.8+ compatible
- Type hints where practical
- Silent failure design: SDK exceptions must never crash the developer's app

## Making Changes

1. Fork the repo
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Make your changes
4. Run tests: `uv run pytest tests/ -v`
5. Commit with a clear message
6. Open a PR against `master`

## Project Structure

```
grandjury/
  __init__.py      # Exports: GrandJury, Span, GrandJuryClient, evaluate_model
  sdk.py           # Main client: trace submission, read path, namespaces
  api_client.py    # Analytics: scoring, histograms, confidence, distributions
tests/
  test_sdk.py      # SDK unit tests
  test_analytics.py # Analytics unit tests
examples/
  quickstart.ipynb
  jupyter_analysis.ipynb
```

## Release Process

1. Bump version in `pyproject.toml` and `grandjury/__init__.py`
2. Commit: `git commit -m "release: vX.Y.Z"`
3. Tag: `git tag vX.Y.Z`
4. Push: `git push && git push --tags`
5. GitHub Actions builds, tests across Python 3.8-3.12, and publishes to PyPI
