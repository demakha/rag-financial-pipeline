"""
Store all embedded chunks into pgvector database permanently.
"""

import os
import sys
import psycopg2

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'chunking'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ingestion'))

from chunker import chunk_document
from extract_10k_text import fetch_document, extract_text
from ticker_to_10k import get_latest_10k_url
from embedder import get_embedding

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "rag_pipeline",
    "user": "postgres",
    "password": "password123"
}


def store_chunks(embedded_chunks: list):
    """Store all embedded chunks into the database."""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    for chunk in embedded_chunks:
        cursor.execute("""
            INSERT INTO chunks 
            (company, section, chunk_index, chunk_text, source_url, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            chunk['company'],
            chunk['section'],
            chunk['chunk_index'],
            chunk['chunk_text'],
            chunk['source_url'],
            chunk['embedding']
        ))

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Stored {len(embedded_chunks)} chunks successfully!")


if __name__ == "__main__":
    ticker = "AAPL"
    print(f"Processing {ticker}...")

    url = get_latest_10k_url(ticker)
    html = fetch_document(url)
    clean_text = extract_text(html)
    chunks = chunk_document(clean_text, company=ticker, source_url=url)

    print(f"Embedding {len(chunks)} chunks...")
    embedded_chunks = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk['chunk_text'])
        embedded_chunks.append({**chunk, "embedding": embedding})
        if (i + 1) % 20 == 0:
            print(f"Progress: {i + 1}/{len(chunks)}")

    print("Storing in database...")
    store_chunks(embedded_chunks)

    print("\nDone! Verifying...")
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM chunks;")
    count = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    print(f"Total chunks in database: {count}")