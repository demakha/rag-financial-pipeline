"""
Structure-aware chunking for SEC 10-K filings.

Strategy:
1. Find where the Table of Contents ends and real content begins
2. Split real content into sections based on "Item X" headers
3. Within each section, use LangChain's RecursiveCharacterTextSplitter
   to further split into smaller chunks WITHOUT breaking sentences/paragraphs
4. Attach metadata (company, section, chunk index) to every chunk
"""

import re
from langchain_text_splitters import RecursiveCharacterTextSplitter



def find_real_content_start(text: str) -> int:
    """
    Find where the Table of Contents ends and real content begins.
    Real section headers are reliably in ALL CAPS ('PART I'),
    while ToC formatting varies by company (some use 'Part I', some use 'PART I').
    """
    part_i_pattern = r'PART\s+I\b'
    positions = [m.start() for m in re.finditer(part_i_pattern, text)]

    if len(positions) == 0:
        return 0
    elif len(positions) == 1:
        return positions[0]
    else:
        return positions[1]
    


def split_into_sections(text: str) -> list[dict]:
    """
    Split real content (after ToC) into sections based on 'Item X.' headers.
    Returns a list of {"section_name": ..., "section_text": ...} dicts.
    """
    # Pattern to find section headers like "Item 1.", "Item 1A.", "Item 7A."
    section_pattern = r'(Item\s+\d+[A-Z]?\.)'

    # Split the text at each section header, keeping the header itself
    parts = re.split(section_pattern, text)

    sections = []
    # parts[0] is text before the first match (should be empty/minimal after ToC cutoff)
    # parts[1], parts[3], parts[5]... are the headers
    # parts[2], parts[4], parts[6]... are the content following each header

    for i in range(1, len(parts) - 1,2):
        header = parts[i].strip()
        content = parts[i + 1].strip()
        sections.append({
            "section_name": header,
            "section_text": content
        })

    return sections



def chunk_document(full_text: str, company: str, source_url: str) -> list[dict]:
    """
    Main function: takes full extracted document text and returns
    a list of chunks, each with text + metadata.
    """
    # Step 1: Skip Table of Contents
    start_pos = find_real_content_start(full_text)
    real_content = full_text[start_pos:]

    # Step 2: Split into sections by "Item X." headers
    sections = split_into_sections(real_content)

    # Step 3: Within each section, split further using LangChain's splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    all_chunks = []
    for section in sections:
        sub_chunks = splitter.split_text(section["section_text"])

        for idx, chunk_text in enumerate(sub_chunks):
            all_chunks.append({
                "chunk_text": chunk_text,
                "company": company,
                "section": section["section_name"],
                "chunk_index": idx,
                "source_url": source_url
            })

    return all_chunks



if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ingestion'))
    from extract_10k_text import fetch_document, extract_text
    from ticker_to_10k import get_latest_10k_url

    ticker = "TSLA"
    url = get_latest_10k_url(ticker)
    html = fetch_document(url)
    clean_text = extract_text(html)

    chunks = chunk_document(clean_text, company=ticker, source_url=url)



    print(f"Total chunks created: {len(chunks)}")
    print(f"\n--- First chunk ---")
    print(chunks[0])
    print(f"\n--- Chunk from a later section (index 50) ---")
    print(chunks[50] if len(chunks) > 50 else "Not enough chunks")