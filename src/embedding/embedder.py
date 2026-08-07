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
    if __name__ == "__main__":
        import time
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

        print(f"Total chunks to embed: {len(chunks)}")
        print("\nEmbedding all chunks...")

        start = time.time()
        embedded_chunks = []

        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk['chunk_text'])
            embedded_chunks.append({
                **chunk,           # all existing metadata (company, section, source_url, chunk_index)
                "embedding": embedding  # add the embedding vector
            })

            # Show progress every 20 chunks
            if (i + 1) % 20 == 0:
                print(f"Progress: {i + 1}/{len(chunks)} chunks embedded")

        elapsed = time.time() - start

        print(f"\n All {len(embedded_chunks)} chunks embedded successfully!")
        print(f"Time taken: {elapsed:.1f} seconds")
        print(f"Embedding dimensions: {len(embedded_chunks[0]['embedding'])}")
        print(f"\nSample chunk metadata:")
        sample = embedded_chunks[50]
        print(f"  Company: {sample['company']}")
        print(f"  Section: {sample['section']}")
        print(f"  Chunk index: {sample['chunk_index']}")
        print(f"  Text preview: {sample['chunk_text'][:100]}...")
        print(f"  Embedding (first 5 values): {sample['embedding'][:5]}")
    