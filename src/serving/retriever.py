"""
Retrieval: Given a user question, find the most relevant chunks
from pgvector using similarity search.
"""

import os
import sys
import psycopg2

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'embedding'))

from embedder import get_embedding

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "rag_pipeline",
    "user": "postgres",
    "password": "password123"
}


def retrieve_relevant_chunks(question: str, top_k: int = 3) -> list:
    """
    Find the most relevant chunks for a given question.
    
    Steps:
    1. Convert question to embedding vector
    2. Search pgvector for most similar chunk embeddings
    3. Return top_k most relevant chunks with their metadata
    """
    # Step 1: Embed the question
    question_embedding = get_embedding(question)

    # Step 2: Search pgvector using cosine similarity
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT 
            company,
            section,
            chunk_text,
            source_url,
            1 - (embedding <=> %s::vector) AS similarity_score
        FROM chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (question_embedding, question_embedding, top_k))

    results = cursor.fetchall()
    cursor.close()
    conn.close()

    # Step 3: Format results
    chunks = []
    for row in results:
        chunks.append({
            "company": row[0],
            "section": row[1],
            "chunk_text": row[2],
            "source_url": row[3],
            "similarity_score": round(float(row[4]), 4)
        })

    return chunks


if __name__ == "__main__":
    # Test with a real question about Apple's 10-K
    question = "What are Apple's main business risks?"

    print(f"Question: {question}")
    print("\nSearching for relevant chunks...")

    chunks = retrieve_relevant_chunks(question, top_k=3)

    print(f"\nTop 3 most relevant chunks found:\n")
    for i, chunk in enumerate(chunks):
        print(f"--- Result {i+1} ---")
        print(f"Company: {chunk['company']}")
        print(f"Section: {chunk['section']}")
        print(f"Similarity Score: {chunk['similarity_score']}")
        print(f"Text preview: {chunk['chunk_text'][:200]}...")
        print()