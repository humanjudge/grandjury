"""
GrandJury Quick Start — 3 lines to human evaluation.

Prerequisites:
  pip install grandjury
  export GRANDJURY_API_KEY=gj_sk_live_...

Register your model at: https://humanjudge.com/projects/new
"""

from grandjury import GrandJury

gj = GrandJury()  # reads GRANDJURY_API_KEY from env

# 1. Send a trace
inference_id = gj.trace(
    name="chat",
    input="What makes a good Instagram post for Gen Z?",
    output="A good IG post for Gen Z should be authentic, use trending audio...",
    model="gpt-4o",
)
print(f"Trace submitted: {inference_id}")

# 2. Check results (once reviewers have voted)
df = gj.results()
if len(df) > 0:
    print(f"\nResults: {len(df)} traces evaluated")
    print(f"Overall pass rate: {df['pass_rate'].mean():.1%}")
else:
    print("\nNo votes yet — reviewers will evaluate your traces on HumanJudge.")
