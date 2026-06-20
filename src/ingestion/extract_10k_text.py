"""
Phase 1 - Step 3: Download the actual 10-K document and extract readable text.
"""

import requests
from bs4 import BeautifulSoup
from ticker_to_10k import get_latest_10k_url

HEADERS = {
    "User-Agent": "Yashwant Singh yash09179@gmail.com"
}


def fetch_document(url: str) -> str:
    """Download the raw HTML of the actual 10-K filing."""
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.text


def extract_text(html: str) -> str:
    """Parse HTML and extract clean, readable text."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove script and style tags - they contain no readable content
    for tag in soup(["script", "style","ix:header","ix:hidden"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    return text


if __name__ == "__main__":
    tickers = ["AAPL","MSFT","TSLA"]

    for ticker in tickers:
        print(f"\n{'='*50}")
        print(f"Getting latest 10-K URL for {ticker}...")
        doc_url = get_latest_10k_url(ticker)
        print(f"URL: {doc_url}")
        print(f"{'='*50}")

        print("\nDownloading actual document (this may take a few seconds, it's a large file)...")
        html_content = fetch_document(doc_url)

        print(f"Raw HTML size: {len(html_content):,} characters")

        print("\nExtracting text...")
        clean_text = extract_text(html_content)

        print(f"Extracted text size: {len(clean_text):,} characters")

        print("\n--- FIRST 1500 CHARACTERS ---\n")
        print(clean_text[:1500])