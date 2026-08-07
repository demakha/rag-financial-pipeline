"""
RAG Chain: Combines retrieval + LLM answer generation.
Uses OpenRouter API to access DeepSeek for free.
"""

import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'embedding'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'serving'))

from retriever import retrieve_relevant_chunks

# OpenRouter client setup
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)


def generate_answer(question: str, top_k: int = 3) -> dict:
    """
    Full RAG pipeline:
    1. Retrieve relevant chunks for the question
    2. Build a prompt with those chunks as context
    3. Send to DeepSeek via OpenRouter
    4. Return answer + sources
    """

    # Step 1: Retrieve relevant chunks
    chunks = retrieve_relevant_chunks(question, top_k=top_k)

    if not chunks:
        return {
            "question": question,
            "answer": "No relevant information found in the database.",
            "sources": []
        }

    # Step 2: Build context from retrieved chunks
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"\n[Source {i+1} - {chunk['company']} {chunk['section']}]\n"
        context += chunk['chunk_text']
        context += "\n"

    # Step 3: Build prompt
    prompt = f"""You are a financial analyst assistant. Answer the question based ONLY on the provided context from SEC 10-K filings. If the answer is not in the context, say so clearly.

Context:
{context}

Question: {question}

Provide a clear, concise answer based on the context above. Cite which source(s) you used."""

    # Step 4: Call DeepSeek via OpenRouter
    response = client.chat.completions.create(
        model="deepseek/deepseek-chat",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )

    answer = response.choices[0].message.content

    # Step 5: Format sources
    sources = [
        f"{chunk['company']} - {chunk['section']} (similarity: {chunk['similarity_score']})"
        for chunk in chunks
    ]

    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }


if __name__ == "__main__":
    # Test with real questions about Apple's 10-K
    questions = [
        "What are Apple's main business risks?",
        "What products does Apple sell?",
        "How does Apple generate revenue?"
    ]

    for question in questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}")

        result = generate_answer(question)

        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nSources used:")
        for source in result['sources']:
            print(f"  - {source}")