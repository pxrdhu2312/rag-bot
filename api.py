"""
api.py
-------
FastAPI version of the RAG chatbot, for deployment / hosting.
Reuses all core logic from app.py (SQLite lookup, FAISS retrieval,
prompt building, Groq call, error handling) and exposes it as a
POST /chat endpoint, plus a GET / health check.

Local run:
    uvicorn api:app --reload

Interactive test UI once running: http://127.0.0.1:8000/docs
"""

import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_groq import ChatGroq

from app import (
    load_vector_store,
    answer_query,
    GROQ_MODEL,
)

app = FastAPI(
    title="NimbusCart AI Customer Support (RAG Bot)",
    description="Context-aware, membership-personalized customer support "
    "chatbot built with LangChain + Groq + FAISS + SQLite.",
    version="1.0.0",
)

# Loaded once at startup, reused across requests (avoids reloading the
# embedding model / vector store on every single API call).
_vector_store = None
_llm = None


class ChatRequest(BaseModel):
    user_id: str
    user_query: str


class ChatResponse(BaseModel):
    answer: str


@app.on_event("startup")
def startup_event():
    global _vector_store, _llm
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key.strip() == "" or api_key == "your_groq_api_key_here":
        # Don't crash the whole server on startup — surface the error per
        # request instead, so the health check (/) still responds.
        print(
            "[WARNING] GROQ_API_KEY is not set. /chat requests will fail "
            "until it is configured."
        )
    else:
        _llm = ChatGroq(model=GROQ_MODEL, api_key=api_key, temperature=0.2)

    _vector_store = load_vector_store()


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "NimbusCart RAG chatbot is running. POST to /chat or visit /docs.",
        "model": GROQ_MODEL,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    global _llm
    if _llm is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key.strip() == "" or api_key == "your_groq_api_key_here":
            return ChatResponse(
                answer="[ERROR] GROQ_API_KEY is missing or not set on the server."
            )
        _llm = ChatGroq(model=GROQ_MODEL, api_key=api_key, temperature=0.2)

    answer = answer_query(request.user_id, request.user_query, _vector_store, _llm)
    return ChatResponse(answer=answer)
