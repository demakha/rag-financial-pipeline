"""
Generate embeddings for text chunks using sentence-transformers (free, local).
No API key or payment required.
"""

import os
import sys

_model = None

def get_model():
    """Lazy-load the model so it only downloads/loads when actually needed."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print("Loading local embedding model...")
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model


def get_embedding(text: str) -> list:
    """Convert a piece of text into an embedding vector."""
    model = get_model()
    embedding = model.encode(text)
    return embedding.tolist()


if __name__ == "__main__":
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

    first_chunk = chunks[0]
    print(f"\nChunk preview: {first_chunk['chunk_text'][:100]}...")

    print("\nGenerating embedding for first chunk...")
    embedding = get_embedding(first_chunk['chunk_text'])

    print(f"\nEmbedding generated successfully!")
    print(f"Embedding dimensions: {len(embedding)}")
    print(f"First 10 values: {embedding[:10]}")
    