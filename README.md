# grandjury

Get human feedback on your AI in 3 lines of Python.

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

**Patent Pending.**

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

See [LICENSE](LICENSE).
