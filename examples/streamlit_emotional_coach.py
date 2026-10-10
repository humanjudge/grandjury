"""
Streamlit Emotional Support Coach — GrandJury integration demo.

Run:
  streamlit run examples/streamlit_emotional_coach.py

Requires:
  .env with GRANDJURY_API_KEY, OPENAI_API_KEY, OPENAI_API_BASE, MODEL
"""

import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from grandjury import GrandJury

load_dotenv()

st.set_page_config(page_title="Emotional Support Coach", page_icon="🧘", layout="centered")

SYSTEM_PROMPT = """You are a compassionate emotional support coach for students.
Respond with empathy, validate their feelings, and offer practical suggestions.
Never give medical advice. Keep responses warm and supportive."""

def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "gj" not in st.session_state:
        st.session_state.gj = GrandJury()
    if "llm" not in st.session_state:
        st.session_state.llm = ChatOpenAI(
            model=os.environ["MODEL"],
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            openai_api_base=os.environ.get("OPENAI_API_BASE"),
            temperature=0.7,
        )

init_session()

st.title("🧘 Emotional Support Coach")
st.caption("Powered by HumanJudge GrandJury tracing")

with st.sidebar:
    st.header("Configuration")
    st.write(f"Model: `{os.environ.get('MODEL', 'not set')}`")
    st.write(f"GrandJury: `{'✅ connected' if st.session_state.gj._api_key else '❌ no API key'}`")
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("trace_id"):
            st.caption(f"Trace: {msg['trace_id']}")

if prompt := st.chat_input("Share what's on your mind..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                full_prompt = f"{SYSTEM_PROMPT}\n\nStudent's concern: {prompt}"
                response = st.session_state.llm.invoke(full_prompt)
                response_text = response.content if hasattr(response, "content") else str(response)
                
                st.markdown(response_text)
                
                inference_id = st.session_state.gj.trace(
                    name="emotional-support-coach",
                    input=prompt,
                    output=response_text,
                    model=os.environ["MODEL"],
                )
                
                if inference_id:
                    st.caption(f"✅ Traced: {inference_id}")
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response_text,
                        "trace_id": inference_id
                    })
                else:
                    st.warning("⚠️ Trace submission failed (check API key)")
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})