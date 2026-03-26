"""
Jupyter Analysis — fetch evaluation results and analyze in a notebook.

Run this in a Jupyter notebook or as a script.

Prerequisites:
  pip install grandjury pandas
  export GRANDJURY_API_KEY=gj_sk_live_...
"""

from grandjury import GrandJury

gj = GrandJury()

# --- Trace-level results ---
df = gj.results()
print(f"Total evaluated traces: {len(df)}")
print(f"Average pass rate: {df['pass_rate'].mean():.1%}")
print()
print(df.head())

# --- Vote-level detail ---
df_votes = gj.results(detail='votes')
print(f"\nTotal individual votes: {len(df_votes)}")
print(f"Unique voters: {df_votes['voter_id'].nunique()}")
print()

# --- Analytics on live data ---
histogram = gj.analytics.vote_histogram()
print(f"\nVote histogram: {histogram}")

# --- Filter by benchmark ---
# df_benchmark = gj.results(evaluation='marketing-benchmark')

# --- Export ---
df.to_parquet('evaluation_results.parquet')
print("\nExported to evaluation_results.parquet")
