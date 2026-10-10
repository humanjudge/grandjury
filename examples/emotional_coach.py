"""
Emotional Support Coach — GrandJury 101 example.

Purpose:
  An AI coach that responds to student emotional concerns with empathy
  and practical suggestions. Every response is tracked via GrandJury
  so human reviewers can evaluate the quality of the coaching.

Prerequisites:
  pip install langchain-openai python-dotenv grandjury

Environment:
  Copy examples/.env.example to .env and fill in your keys.
  Then run:
    python examples/emotional_coach.py
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from grandjury import GrandJury

load_dotenv()
gj = GrandJury()

llm = ChatOpenAI(
    model=os.environ["MODEL"],
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    openai_api_base=os.environ.get("OPENAI_API_BASE"),
)

SYSTEM_PROMPT = """You are a compassionate emotional support coach for students.
Respond with empathy, validate their feelings, and offer practical suggestions.
Never give medical advice. Keep responses warm and supportive."""

scenarios = [
    "A student says: 'I'm feeling overwhelmed with my coursework and falling behind.'",
    "A student says: 'I got a bad grade on my exam and feel like a failure.'",
    "A student says: 'I'm having trouble making friends at my new school.'",
    "A student says: 'I'm stressed about choosing a career path after graduation.'",
    "A student says: 'I feel like I'm not smart enough for this program.'",
]


for i, scenario in enumerate(scenarios):
    prompt = f"{SYSTEM_PROMPT}\n\nStudent's concern: {scenario}"
    response = llm.invoke(prompt)

    inference_id = gj.trace(
        name="emotional-support-coach",
        input=scenario,
        output=response.content if hasattr(response, "content") else str(response),
        model=os.environ["MODEL"],
    )
    print(f"Trace {i+1}: {inference_id}")
