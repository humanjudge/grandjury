<div align="center">

<img src="assets/logo.png" alt="HumanJudge" width="180" />

# grandjury

<!--
  GROWTH TEAM: tagline below is editable — A/B test framings as needed.
  Numbers (25K reviews, 200+ reviewers, 58 models, 44 benchmarks) reflect
  production DB at last edit time; refresh on rewrite. Keep the model
  names current as the leaderboard evolves.
-->
> Real human evaluations of AI models. **25,000+ blind reviews** by **200+ verified reviewers** across **58 models** (GPT-5, Claude Opus 4.7, Gemini 3.1, Grok 4.3, DeepSeek V4, Mistral, Kimi K2.6 and more) and **44 benchmarks**. Free. Python SDK + MCP server + ChatGPT GPT + REST.

[![PyPI](https://img.shields.io/pypi/v/grandjury?style=flat-square&color=blue)](https://pypi.org/project/grandjury/)
[![Python](https://img.shields.io/pypi/pyversions/grandjury?style=flat-square&color=blue)](https://pypi.org/project/grandjury/)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-humanjudge.com-blue?style=flat-square)](https://humanjudge.com/docs)
[![Berkeley RDI](https://img.shields.io/badge/Berkeley_RDI-Aug_1--2_2026-003262?style=flat-square)](https://rdi.berkeley.edu/)

**[Install the SDK ↓](#installation) · [Join the R&D community ↓](#research-community)**

</div>

---

Get human feedback on your AI in 3 lines of Python:

```python
from grandjury import GrandJury

gj = GrandJury()  # reads GRANDJURY_API_KEY from env
gj.trace(name="chat", input=prompt, output=response, model="gpt-4o")
```

Then open your Jupyter notebook:

```python
df = gj.results()  # traces with human votes — as a DataFrame
print(f"Pass rate: {df['pass_rate'].mean():.1%}")
```

## Research community

AI evaluation is usually a single number. We capture it as a continuous **datastream** instead — pluralistic, multi-reviewer, multi-context, from real production traffic. This open R&D community works on what that richer signal can do.

Presenting at [Berkeley RDI's Agentic AI Summit, Aug 1–2 2026](https://rdi.berkeley.edu/).

Six research streams. Pick by interest, not assignment.

### Stream A — Pluralistic data for model training

Preference-optimization methods are mature (binary, pairwise, multi-objective variants) but assume a single label per training example. Pluralistic, multi-reviewer, open-vocabulary feedback doesn't fit that shape cleanly. The open question: how do we represent pluralistic signal as training data without collapsing its diversity? And what aggregation rule across dimensions respects safety constraints — where trade-offs between dimensions are unacceptable?

### Stream B — Resource curation

A public, opinionated, **actively maintained** index of AI safety and production-evaluation tools, frameworks, and papers. Most existing curated indices in this space rot fast — staleness sets in within months as the field moves and links break. We curate with a deliberate maintenance discipline; adjacent communities amplify and contribute updates. The artifact is the index, but the durable value is keeping it current.

### Stream C — Model routing

Quality-routing systems exist that predict which model best serves a given query. Their training data is typically constructed synthetically or sourced from standardized academic benchmarks. We provide the missing layer: pluralistic, multi-reviewer, domain-tagged production feedback as the substrate for routing decisions. Empathy is the first use case — a subjective dimension where standardized benchmarks underperform.

### Stream D — Real-time guardrails mechanism

How do live production signals drive *immediate* guardrail updates, user apologies, and human handoff for dangerous content — folding continuous red-teaming feedback into the production pipeline rather than batching it for the next retraining cycle? Includes the design question of upstream filtering (centralized blocking at the platform) vs. downstream slicing (user-context-dependent filtering closer to deployment).

### Stream E — Signal representation & visualization

What richer signal can reviewers submit beyond binary verdict + categorical tags, and what does an AI *user* see as a result? The two ends of one pipeline. Input side: pre-defined attribute ratings versus open-vocabulary contextual tags — the schema choice determines what downstream systems can ingest. Output side: live, multi-dimensional, third-party-attested representations of how an AI is actually behaving — beyond static vendor documentation or single-dimensional comparison rankings.

### Stream F — Platform integration

How does live pluralistic evaluation surface inside the tools where developers and workflows already touch AI output? Categories of integration surface: LLM observability platforms, workflow-automation systems, agent IDEs, dashboards, communications and alerting, notebooks, documentation embeds. Each integration ships into its own developer community.

### How to apply

The application is a small pull request to this repo.

1. **Complete the challenge** in [`/challenges/README.md`](challenges/README.md).
2. **Make sure you've connected your GitHub** on [your profile](https://humanjudge.com/profile).
3. **Open a PR** that adds `/challenges/<your-github-handle>.md` (see [`TEMPLATE.md`](challenges/TEMPLATE.md)). We'll notify you by email.

Full walkthrough in [`CONTRIBUTING.md`](CONTRIBUTING.md#apply-to-the-rd-community).

## Why HumanJudge

Most AI evaluation pipelines use LLMs to judge LLMs. That inherits the same biases, conventions, and blind spots as the models being evaluated — and tends to produce eval pipelines with ~0% disagreement, which is the diagnostic for "not measuring quality, just confirming assumptions" ([essay](https://humanjudge.com/ai-reviews/your-eval-pipeline-has-zero-disagreement)).

HumanJudge uses **real human reviewers** who blind-evaluate AI outputs across structured benchmarks (marketing, healthcare, end-of-life conversations, cultural fluency, code review, and more) and write their reasoning. Reviewers earn XP, get credentialing letters, and stay anonymous to the reader by default.

The data is queryable via this SDK, the [MCP server](https://humanjudge.com/docs/pulse/claude-desktop), a [ChatGPT GPT action](https://humanjudge.com/docs/pulse/chatgpt), and a REST API.

## Integrations

| Surface | Install | Docs |
|---|---|---|
| **Python SDK** | `pip install grandjury` | [docs/pulse/python-sdk](https://humanjudge.com/docs/pulse/python-sdk) |
| **Claude Desktop MCP** | Add `https://api.humanjudge.com/mcp` as a custom connector | [docs/pulse/claude-desktop](https://humanjudge.com/docs/pulse/claude-desktop) |
| **Claude Code MCP** | Add to `.mcp.json` (remote, no install) | [docs/pulse/claude-code](https://humanjudge.com/docs/pulse/claude-code) |
| **ChatGPT GPT** | Search "HumanJudge" in the GPT Store | [docs/pulse/chatgpt](https://humanjudge.com/docs/pulse/chatgpt) |
| **REST API** | n/a | [humanjudge.com/docs](https://humanjudge.com/docs) |

## Use cases

- **ML engineers** — benchmark your model against 58+ commercial models on real tasks; see exactly what humans flag with category + reasoning
- **Data scientists** — pull reviewer reasoning, flag patterns, and disagreement data as pandas DataFrames for analysis
- **AI agent developers** — log traces from any agent loop (decorator, context manager, or direct call); get reviewer feedback you can quote to stakeholders
- **Independent researchers** — query the public benchmark data without an API key (read-only)
- **Builders** — register your own AI, create a custom benchmark on the topics you care about, get real human reviews on YOUR specific use case ([humanjudge.com/for-developers](https://humanjudge.com/for-developers))

## What is GrandJury?

[HumanJudge](https://humanjudge.com) connects your AI to a community of human reviewers who evaluate your model's outputs. GrandJury is the Python SDK — it sends traces and retrieves human evaluation results.

**Write path:** Log AI calls from your app → traces appear in your developer dashboard.
**Read path:** Fetch evaluation results (votes, pass rates, reviewer feedback) into DataFrames for analysis.

## Installation

```bash
pip install grandjury
```

Optional performance dependencies:
```bash
pip install grandjury[performance]  # msgspec, pyarrow, polars
```

## Quick Start

### 1. Register your model

Go to [humanjudge.com/projects/new](https://humanjudge.com/projects/new), register your AI, and copy the secret key.

```bash
export GRANDJURY_API_KEY=gj_sk_live_...
```

### 2. Log traces from your app

```python
from grandjury import GrandJury

gj = GrandJury()  # zero-config — reads from env

# Option A: Direct call
gj.trace(name="chat", input="What is ML?", output="Machine learning is...", model="gpt-4o")

# Option B: Decorator — auto-captures input/output/latency
@gj.observe(name="chat", model="gpt-4o")
def call_llm(prompt: str) -> str:
    return openai.chat(prompt)

# Option C: Context manager
with gj.span("chat", input=prompt) as s:
    response = call_llm(prompt)
    s.set_output(response)
```

### 3. Get human evaluation results

Once reviewers vote on your traces:

```python
# Trace-level summary
df = gj.results()
# trace_id | input | output | model | pass_count | flag_count | total_votes | pass_rate

# Individual votes with reviewer identity
df_votes = gj.results(detail='votes')
# trace_id | voter_id | voter_name | verdict | flag_category | feedback | created_at

# Filter by benchmark
df_benchmark = gj.results(evaluation='marketing-benchmark')

# Export
df.to_parquet('evaluation_results.parquet')
```

### 4. Run analytics

Works on both live platform data and offline datasets:

```python
# Auto-fetch from platform
gj.analytics.vote_histogram()
gj.analytics.population_confidence(voter_list=[...])

# Or pass your own data
import pandas as pd
df = pd.read_csv("my_votes.csv")
gj.analytics.vote_histogram(df)
gj.analytics.votes_distribution(df)
```

## Enroll in Benchmarks

List and enroll your model in open benchmarks programmatically:

```python
# Browse available benchmarks
benchmarks = gj.benchmarks.list()

# Enroll with endpoint config
gj.benchmarks.enroll(
    benchmark_id="...",
    model_id="...",
    endpoint_config={
        "endpoint": "https://api.myapp.com/v1/chat/completions",
        "apiKey": "sk-...",
        "request_template": '{"model":"gpt-4o","messages":[{"role":"user","content":"{{prompt}}"}]}',
        "response_path": "choices[0].message.content"
    }
)
```

## Analytics Methods

All analytics methods work on both platform data (`gj.results(detail='votes')`) and offline data (pandas/polars/CSV/parquet):

| Method | Description |
|---|---|
| `gj.analytics.evaluate_model()` | Decay-adjusted scoring |
| `gj.analytics.vote_histogram()` | Vote time distribution |
| `gj.analytics.vote_completeness()` | Completeness per voter |
| `gj.analytics.population_confidence()` | Confidence metrics |
| `gj.analytics.majority_good_votes()` | Threshold analysis |
| `gj.analytics.votes_distribution()` | Votes per inference |

## Privacy

- `gj.results()` only returns traces with at least 1 human vote (privacy gate)
- Zero-vote traces are invisible to the SDK — only visible on the web dashboard
- Reviewer identity is public (consistent with platform's public profile/leaderboard model)

## API Reference

```python
gj = GrandJury(
    api_key=None,     # reads GRANDJURY_API_KEY from env if not provided
    base_url="https://grandjury-server.onrender.com",
    timeout=5.0,
)

# Write
gj.trace(name, input, output, model, latency_ms, metadata, gj_inference_id)
await gj.atrace(...)  # async version (requires httpx)
gj.observe(name, model, metadata)  # decorator
gj.span(name, input, model, metadata)  # context manager

# Read
gj.results(detail=None, evaluation=None)  # returns DataFrame or list[dict]

# Browse
gj.models.list()
gj.models.get(model_id)
gj.benchmarks.list()
gj.benchmarks.enroll(benchmark_id, model_id, endpoint_config)

# Analytics
gj.analytics.evaluate_model(...)
gj.analytics.vote_histogram(data=None, ...)
gj.analytics.vote_completeness(data=None, voter_list=None, ...)
gj.analytics.population_confidence(data=None, voter_list=None, ...)
gj.analytics.majority_good_votes(data=None, ...)
gj.analytics.votes_distribution(data=None, ...)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and PR guidelines.

## License

See [LICENSE](LICENSE). Patent application US 63/825,484 covers aspects of the platform.
