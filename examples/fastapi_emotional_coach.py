"""
FastAPI Emotional Support Coach — GrandJury integration demo.

Run:
  uvicorn examples.fastapi_emotional_coach:app --reload --port 8000

Requires:
  .env with GRANDJURY_API_KEY, OPENAI_API_KEY, OPENAI_API_BASE, MODEL
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import uuid
import time
from typing import List, Optional
from langchain_openai import ChatOpenAI
from grandjury import GrandJury

load_dotenv()

app = FastAPI(title="Emotional Support Coach API")

gj = GrandJury()
llm = ChatOpenAI(
    model=os.environ["MODEL"],
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    openai_api_base=os.environ.get("OPENAI_API_BASE"),
    temperature=0.7,
)

SYSTEM_PROMPT = """You are a compassionate emotional support coach for students.
Respond with empathy, validate their feelings, and offer practical suggestions.
Never give medical advice. Keep responses warm and supportive."""


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    trace_id: str | None = None


# OpenAI-compatible models for benchmark enrollment
class OpenAIMessage(BaseModel):
    role: str
    content: str


class OpenAIChatRequest(BaseModel):
    model: str
    messages: List[OpenAIMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


class OpenAIChatChoice(BaseModel):
    index: int
    message: OpenAIMessage
    finish_reason: str = "stop"


class OpenAIChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[OpenAIChatChoice]
    usage: dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


async def generate_response(prompt: str, model_name: str) -> tuple[str, str | None]:
    """Returns (response_text, trace_id)"""
    full_prompt = f"{SYSTEM_PROMPT}\n\nStudent's concern: {prompt}"
    response = llm.invoke(full_prompt)
    response_text = response.content if hasattr(response, "content") else str(response)

    trace_id = gj.trace(
        name="emotional-support-coach",
        input=prompt,
        output=response_text,
        model=model_name,
    )
    if trace_id is None:
        print("⚠️ Trace submission failed - check GRANDJURY_API_KEY and network")
    else:
        print(f"✅ Trace submitted: {trace_id}")

    return response_text, trace_id


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    response_text, trace_id = await generate_response(req.message, os.environ["MODEL"])
    return ChatResponse(response=response_text, trace_id=trace_id)


@app.get("/v1/chat/completions")
async def openai_chat_completions_info():
    """Endpoint info for validation checks."""
    return {
        "status": "ok",
        "endpoint": "/v1/chat/completions",
        "method": "POST",
        "format": "OpenAI chat completions",
        "model": "emotional-coach"
    }


@app.post("/v1/chat/completions", response_model=OpenAIChatResponse)
async def openai_chat_completions(req: OpenAIChatRequest):
    # Extract last user message as prompt
    user_messages = [m for m in req.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(400, "No user message found")
    prompt = user_messages[-1].content

    # Generate response (reuses tracing)
    response_text, _ = await generate_response(prompt, req.model)

    # Return OpenAI format
    return OpenAIChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=req.model,
        choices=[OpenAIChatChoice(
            index=0,
            message=OpenAIMessage(role="assistant", content=response_text),
            finish_reason="stop"
        )],
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )


@app.get("/health")
async def health():
    return {"status": "ok", "grandjury_connected": bool(gj._api_key)}


@app.get("/", response_class=HTMLResponse)
async def chat_ui():
    """Simple chat UI for reviewers to interact with the coach."""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Emotional Support Coach</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 700px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
            .container { background: white; border-radius: 12px; padding: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; margin-bottom: 8px; }
            .subtitle { color: #666; margin-bottom: 24px; }
            #chat { min-height: 300px; max-height: 500px; overflow-y: auto; margin-bottom: 20px; padding: 16px; background: #fafafa; border-radius: 8px; border: 1px solid #eee; }
            .message { margin-bottom: 16px; padding: 12px 16px; border-radius: 12px; max-width: 85%; animation: fadeIn 0.3s ease; }
            .message.user { background: #3498db; color: white; margin-left: auto; border-bottom-right-radius: 4px; }
            .message.assistant { background: #e8f5e9; color: #2c3e50; margin-right: auto; border-bottom-left-radius: 4px; }
            .message .label { font-size: 12px; font-weight: 600; margin-bottom: 4px; opacity: 0.8; }
            .message .content { line-height: 1.5; }
            .trace-id { font-size: 11px; color: #999; margin-top: 4px; font-family: monospace; }
            .input-area { display: flex; gap: 12px; }
            #msg { flex: 1; padding: 14px 18px; border: 2px solid #e0e0e0; border-radius: 24px; font-size: 16px; outline: none; transition: border-color 0.2s; }
            #msg:focus { border-color: #3498db; }
            button { padding: 14px 28px; background: #3498db; color: white; border: none; border-radius: 24px; font-size: 16px; font-weight: 600; cursor: pointer; transition: background 0.2s; }
            button:hover { background: #2980b9; }
            button:disabled { background: #bdc3c7; cursor: not-allowed; }
            @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
            .loading { display: none; text-align: center; color: #999; padding: 8px; }
            .spinner { display: inline-block; width: 16px; height: 16px; border: 2px solid #3498db; border-radius: 50%; border-top-color: transparent; animation: spin 0.8s linear infinite; margin-right: 8px; }
            @keyframes spin { to { transform: rotate(360deg); } }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧘 Emotional Support Coach</h1>
            <p class="subtitle">Share what's on your mind — I'm here to listen and support you.</p>
            
            <div id="chat"></div>
            
            <div class="loading" id="loading">
                <span class="spinner"></span> Thinking...
            </div>
            
            <div class="input-area">
                <input type="text" id="msg" placeholder="Type your message here..." autocomplete="off">
                <button id="sendBtn" onclick="sendMessage()">Send</button>
            </div>
        </div>

        <script>
            const chat = document.getElementById('chat');
            const msgInput = document.getElementById('msg');
            const sendBtn = document.getElementById('sendBtn');
            const loading = document.getElementById('loading');

            function addMessage(role, content, traceId = null) {
                const div = document.createElement('div');
                div.className = 'message ' + role;
                let label = role === 'user' ? 'You' : 'Coach';
                let traceHtml = traceId ? `<div class="trace-id">Trace: ${traceId}</div>` : '';
                div.innerHTML = `<div class="label">${label}</div><div class="content">${escapeHtml(content)}</div>${traceHtml}`;
                chat.appendChild(div);
                chat.scrollTop = chat.scrollHeight;
            }

            function escapeHtml(text) {
                const div = document.createElement('div');
                div.textContent = text;
                return div.innerHTML;
            }

            async function sendMessage() {
                const message = msgInput.value.trim();
                if (!message) return;

                addMessage('user', message);
                msgInput.value = '';
                sendBtn.disabled = true;
                loading.style.display = 'block';

                try {
                    const res = await fetch('/v1/chat/completions', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            model: 'emotional-coach',
                            messages: [{ role: 'user', content: message }],
                            temperature: 0.7
                        })
                    });

                    const data = await res.json();
                    
                    if (data.choices && data.choices[0] && data.choices[0].message) {
                        const responseText = data.choices[0].message.content;
                        const traceId = data.id; // chatcmpl-xxx
                        addMessage('assistant', responseText, traceId);
                    } else {
                        addMessage('assistant', 'Sorry, I encountered an error. Please try again.');
                    }
                } catch (err) {
                    console.error(err);
                    addMessage('assistant', 'Sorry, something went wrong. Please try again.');
                } finally {
                    sendBtn.disabled = false;
                    loading.style.display = 'none';
                }
            }

            msgInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') sendMessage();
            });

            // Focus on load
            msgInput.focus();
        </script>
    </body>
    </html>
    """)