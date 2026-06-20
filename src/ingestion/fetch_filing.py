"""
Phase 1 - Step 1: Download a single SEC filing and extract its text.
"""

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Yashwant Singh yash09179@gmail.com"
}

FILING_URL = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000320193&type=10-K&dateb=&owner=include&count=10"


def fetch_filing_list(url: str) -> str:
    """Download the raw HTML of the filing list page."""
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()  # throws an error if request failed (e.g. 404, 403)
    return response.text


def extract_text(html: str) -> str:
    """Parse HTML and extract clean, readable text."""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    return text


if __name__ == "__main__":
    print("Fetching filing list from SEC EDGAR...")
    html_content = fetch_filing_list(FILING_URL)

    print("Extracting text...")
    clean_text = extract_text(html_content)

    print("\n--- EXTRACTED TEXT (first 1000 characters) ---\n")
    print(clean_text[:1000])