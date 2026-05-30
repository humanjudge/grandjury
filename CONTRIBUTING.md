# Contributing to GrandJury

Two ways to contribute:

- **[Apply to the R&D community →](#apply-to-the-rd-community)** — join the open research community around HumanJudge. Open a small PR with a challenge result and your streams of interest. Get a Discord invite after merge.
- **[Contribute SDK code →](#sdk-development)** — file an issue or PR against the Python package itself (bug fixes, features, docs).

---

## Apply to the R&D community

The application is a small PR — not a form, not a CV submission. Two steps:

### 1. Complete the challenge

Follow the walkthrough in [`/challenges/README.md`](challenges/README.md). It's one short task — count evaluations in any arena using the SDK. ~15 minutes start to finish. The walkthrough covers installing the SDK, creating an account, generating a token + connecting your GitHub (same profile page — both at once), and running the challenge against an arena of your choice.

### 2. Open the PR

Submit a PR that adds `/challenges/<your-github-handle>.md`. Two paths:

**Option A — GitHub web UI (no local clone needed):**

1. Fork this repo (top-right button on GitHub)
2. In your fork, navigate to `/challenges/`
3. Click **Add file → Create new file**
4. Name your file `<your-github-handle>.md`
5. Paste the contents of [`/challenges/TEMPLATE.md`](challenges/TEMPLATE.md) into the editor
6. Fill in:
   - **Streams of interest:** 2–4 sentences per stream — what interests you about it, your background, what you'd want to work on (see the [README](README.md) for stream descriptions)
   - **Challenge result:** the arena, your result, and a timestamp
7. Click **Propose new file** → GitHub takes you to the PR creation screen
8. Submit the PR

**Option B — Local clone path:**

```bash
# After forking on GitHub:
git clone https://github.com/<your-handle>/grandjury.git
cd grandjury

# Create your file
cp challenges/TEMPLATE.md challenges/<your-github-handle>.md
# Edit it (fill in streams + challenge result)

git add challenges/<your-github-handle>.md
git commit -m "challenge: <your-github-handle>"
git push origin master
```

Then open the PR from your fork's GitHub page.

### What happens next

- Reviewed personally within ~3 business days
- We'll notify you by email when your PR is merged
- After merge: instructions for joining our Discord arrive in your inbox
- Join Discord → verify in `#verification` → unlock the stream channels

---

## SDK development

The section below is for contributing to the **Python SDK itself** (bug fixes, features, package docs).

### Development Setup

```bash
git clone https://github.com/humanjudge/grandjury.git
cd grandjury
uv sync --dev
```

### Running Tests

```bash
uv run pytest tests/ -v
```

### Code Style

- Python 3.8+ compatible
- Type hints where practical
- Silent failure design: SDK exceptions must never crash the developer's app

### Making Changes

1. Fork the repo
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Make your changes
4. Run tests: `uv run pytest tests/ -v`
5. Commit with a clear message
6. Open a PR against `master`

### Project Structure

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

### Release Process

1. Bump version in `pyproject.toml` and `grandjury/__init__.py`
2. Commit: `git commit -m "release: vX.Y.Z"`
3. Tag: `git tag vX.Y.Z`
4. Push: `git push && git push --tags`
5. GitHub Actions builds, tests across Python 3.8-3.12, and publishes to PyPI
