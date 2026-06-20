"""
Combined: Ticker -> CIK -> Latest 10-K document URL
No manual CIK entry needed - works for ANY public company.
"""

import requests

HEADERS = {
    "User-Agent": "Yashwant Singh yash09179@gmail.com"
}

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


def get_cik_for_ticker(ticker: str) -> str:
    response = requests.get(TICKERS_URL, headers=HEADERS)
    response.raise_for_status()
    data = response.json()

    ticker = ticker.upper()
    for entry in data.values():
        if entry["ticker"] == ticker:
            return str(entry["cik_str"]).zfill(10)
    return None


def get_submissions(cik: str) -> dict:
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.json()


def find_latest_10k(data: dict) -> dict:
    recent = data["filings"]["recent"]
    for i, form_type in enumerate(recent["form"]):
        if form_type == "10-K":
            return {
                "accessionNumber": recent["accessionNumber"][i],
                "filingDate": recent["filingDate"][i],
                "primaryDocument": recent["primaryDocument"][i],
            }
    return None


def build_document_url(cik: str, accession_number: str, primary_document: str) -> str:
    accession_no_dashes = accession_number.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_no_dashes}/{primary_document}"


def get_latest_10k_url(ticker: str) -> str:
    """The main function - give it ANY ticker, get back the 10-K URL."""
    cik = get_cik_for_ticker(ticker)
    if not cik:
        print(f"Ticker {ticker} not found.")
        return None

    submissions = get_submissions(cik)
    latest_10k = find_latest_10k(submissions)

    if not latest_10k:
        print(f"No 10-K found for {ticker}.")
        return None

    return build_document_url(cik, latest_10k["accessionNumber"], latest_10k["primaryDocument"])


if __name__ == "__main__":
    test_tickers = ["AAPL", "MSFT", "TSLA"]

    for ticker in test_tickers:
        url = get_latest_10k_url(ticker)
        print(f"{ticker}: {url}")