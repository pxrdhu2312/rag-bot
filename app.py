"""
app.py
-------
Main RAG chatbot application. Terminal-based loop.
"""

import os
import sqlite3
import sys

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
load_dotenv()  # load .env early so GROQ_MODEL below picks up any override

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "users.db")
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "vectorstore")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")

NO_CONTEXT_MESSAGE = (
    "I do not have enough information in the provided knowledge base to answer this."
)
USER_NOT_FOUND_MESSAGE = "User not found. Please enter a valid user_id."
TOP_K = 3

PROMPT_TEMPLATE = """You are an AI customer support assistant.

You are speaking with:
Name: {name}
Membership Tier: {membership_tier}

Answer the user's question using only the context provided below.

If the answer is not available in the context, say:
"I do not have enough information in the provided knowledge base to answer this."

Context:
{retrieved_chunks}

User Question:
{user_query}

Answer:"""


def load_env_and_check_key():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key.strip() == "" or api_key == "your_groq_api_key_here":
        print(
            "\n[ERROR] GROQ_API_KEY is missing or not set.\n"
            "Please create a .env file (copy .env.example to .env) and set:\n"
            "  GROQ_API_KEY=your_actual_key_here\n"
        )
        sys.exit(1)
    return api_key


def get_user(user_id: str):
    try:
        user_id_int = int(user_id)
    except (ValueError, TypeError):
        return None

    if not os.path.exists(DB_PATH):
        print(f"[ERROR] Database not found at {DB_PATH}. Run 'python create_db.py' first.")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT user_id, name, membership_tier FROM users WHERE user_id = ?",
        (user_id_int,),
    )
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None
    return {"user_id": row[0], "name": row[1], "membership_tier": row[2]}


def load_vector_store():
    if not os.path.exists(VECTOR_STORE_DIR):
        print(f"[ERROR] Vector store not found at {VECTOR_STORE_DIR}. Run 'python ingest.py' first.")
        sys.exit(1)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = FAISS.load_local(
        VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True
    )
    return vector_store


def retrieve_context(vector_store, query: str) -> str:
    if not query or not query.strip():
        return ""
    results = vector_store.similarity_search(query, k=TOP_K)
    if not results:
        return ""
    return "\n\n".join(doc.page_content for doc in results)


def build_prompt(name, membership_tier, retrieved_chunks, user_query):
    return PROMPT_TEMPLATE.format(
        name=name,
        membership_tier=membership_tier,
        retrieved_chunks=retrieved_chunks,
        user_query=user_query,
    )


def call_groq(llm, prompt: str) -> str:
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        err_text = str(e).lower()
        if "rate limit" in err_text or "429" in err_text:
            return "[ERROR] Groq API rate limit reached. Please wait a moment and try again."
        if "401" in err_text or "unauthorized" in err_text or "invalid api key" in err_text:
            return "[ERROR] Groq API rejected the request — check your GROQ_API_KEY in .env."
        return f"[ERROR] Something went wrong while calling the Groq API: {e}"


def answer_query(user_id: str, user_query: str, vector_store, llm) -> str:
    user = get_user(user_id)
    if user is None:
        return USER_NOT_FOUND_MESSAGE

    context = retrieve_context(vector_store, user_query)
    if not context.strip():
        return NO_CONTEXT_MESSAGE

    prompt = build_prompt(user["name"], user["membership_tier"], context, user_query)
    return call_groq(llm, prompt)


def main():
    print("=" * 60)
    print(" NimbusCart AI Customer Support (RAG Chatbot)")
    print("=" * 60)

    api_key = load_env_and_check_key()
    print("Loading vector store (first run may take a moment) ...")
    vector_store = load_vector_store()

    llm = ChatGroq(model=GROQ_MODEL, api_key=api_key, temperature=0.2)

    print("\nReady! Type 'exit' at any prompt to quit.\n")

    while True:
        user_id = input("Enter user_id: ").strip()
        if user_id.lower() == "exit":
            break

        user_query = input("Enter your question: ").strip()
        if user_query.lower() == "exit":
            break

        answer = answer_query(user_id, user_query, vector_store, llm)
        print("\n--- Answer ---")
        print(answer)
        print("--------------\n")


if __name__ == "__main__":
    main()