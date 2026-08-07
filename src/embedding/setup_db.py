"""
Set up Postgres database schema for storing chunks and embeddings.
Creates the pgvector extension and the chunks table.
"""

import psycopg2

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "rag_pipeline",
    "user": "postgres",
    "password": "password123"
}


def setup_database():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # Enable pgvector extension
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # Create chunks table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id SERIAL PRIMARY KEY,
            company TEXT NOT NULL,
            section TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            source_url TEXT NOT NULL,
            embedding vector(384)
        );
    """)

    conn.commit()
    cursor.close()
    conn.close()
    print("Database setup complete!")
    print("- pgvector extension enabled")
    print("- chunks table created")


if __name__ == "__main__":
    setup_database()