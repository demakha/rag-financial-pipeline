"""
Generate embeddings for text chunks using OpenAI's embeddings API.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-small"  # cheap, good quality, 1536 dimensions


def get_embedding(text: str) -> list[float]:
    """Convert a piece of text into an embedding vector."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


if __name__ == "__main__":
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'chunking'))
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ingestion'))

    from chunker import chunk_document
    from extract_10k_text import fetch_document, extract_text
    from ticker_to_10k import get_latest_10k_url

    ticker = "AAPL"
    url = get_latest_10k_url(ticker)
    html = fetch_document(url)
    clean_text = extract_text(html)
    chunks = chunk_document(clean_text, company=ticker, source_url=url)

    print(f"Total chunks available: {len(chunks)}")

    # Just test on the FIRST chunk for now - don't embed all 188 yet, that costs money each run
    first_chunk = chunks[0]
    print(f"\nGetting embedding for first chunk...")
    print(f"Chunk text preview: {first_chunk['chunk_text'][:100]}...")

    embedding = get_embedding(first_chunk['chunk_text'])

    print(f"\nEmbedding generated successfully!")
    print(f"Embedding length (dimensions): {len(embedding)}")
    print(f"First 10 numbers of the embedding: {embedding[:10]}")