"""
SEC EDGAR Downloader
Downloads and parses SEC filings (10-K, 10-Q) from the EDGAR API.
No API key required — uses free public EDGAR endpoints.
"""

import logging
import re
import time
from typing import List, Dict, Optional

import requests

logger = logging.getLogger(__name__)

SEC_HEADERS = {
    "User-Agent": "FinRAG-AI Research Tool research@example.com",
    "Accept-Encoding": "gzip, deflate",
}

EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_FILING_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{filename}"

COMPANY_CIKS = {
    "Apple": "0000320193",
    "Tesla": "0001318605",
    "Microsoft": "0000789019",
    "Amazon": "0001018724",
    "Alphabet": "0001652044",
}


def get_company_filings(company: str, form_type: str = "10-K", count: int = 3) -> List[Dict]:
    """
    Fetch recent filing metadata for a company from EDGAR.

    Args:
        company: Company name (must be in COMPANY_CIKS)
        form_type: SEC form type (10-K, 10-Q)
        count: Number of filings to fetch

    Returns:
        List of filing metadata dicts
    """
    cik = COMPANY_CIKS.get(company)
    if not cik:
        logger.warning("Unknown company: %s. Known: %s", company, list(COMPANY_CIKS.keys()))
        return []

    url = EDGAR_SUBMISSIONS_URL.format(cik=cik.lstrip("0"))
    try:
        resp = requests.get(url, headers=SEC_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error("Failed to fetch EDGAR submissions for %s: %s", company, e)
        return []

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])

    filings = []
    for i, form in enumerate(forms):
        if form == form_type and len(filings) < count:
            filings.append({
                "company": company,
                "cik": cik,
                "form_type": form,
                "filing_date": dates[i],
                "accession_number": accessions[i],
                "primary_document": primary_docs[i],
            })

    logger.info("Found %d %s filings for %s", len(filings), form_type, company)
    return filings


def download_filing_text(filing: Dict) -> Optional[str]:
    """
    Download the full text of a filing from EDGAR.

    Args:
        filing: Filing metadata dict from get_company_filings

    Returns:
        Raw text content of the filing, or None on failure
    """
    cik = filing["cik"].lstrip("0")
    accession = filing["accession_number"].replace("-", "")
    filename = filing["primary_document"]

    url = EDGAR_FILING_URL.format(cik=cik, accession=accession, filename=filename)

    try:
        time.sleep(0.11)  # SEC rate limit: 10 requests/second
        resp = requests.get(url, headers=SEC_HEADERS, timeout=30)
        resp.raise_for_status()

        if filename.endswith((".htm", ".html")):
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.content, "html.parser")
            for tag in soup(["script", "style", "meta", "link"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
        else:
            text = resp.text

        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)

        logger.info(
            "Downloaded %s filing for %s (%d chars)",
            filing["form_type"], filing["company"], len(text),
        )
        return text

    except Exception as e:
        logger.error("Failed to download filing: %s", e)
        return None


def download_and_chunk_filings(
    company: str,
    form_type: str = "10-K",
    count: int = 1,
    chunk_size: int = 512,
    chunk_overlap: int = 128,
) -> List[Dict]:
    """
    Download SEC filings and return chunked documents ready for indexing.

    Args:
        company: Company name
        form_type: Filing type
        count: Number of filings
        chunk_size: Characters per chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of chunk dicts with text and metadata
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    filings = get_company_filings(company, form_type, count)
    all_chunks = []

    for filing in filings:
        text = download_filing_text(filing)
        if not text:
            continue

        text = text[:50000]  # Limit for demo

        chunks = splitter.split_text(text)
        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                "text": chunk_text,
                "company": company,
                "source": f"SEC {form_type}",
                "filing_date": filing["filing_date"],
                "quarter": _date_to_quarter(filing["filing_date"]),
                "chunk_id": i,
                "filing_type": form_type,
            })

    logger.info(
        "Downloaded and chunked %d filings for %s -> %d chunks",
        len(filings), company, len(all_chunks),
    )
    return all_chunks


def _date_to_quarter(date_str: str) -> str:
    """Convert filing date to quarter string."""
    try:
        month = int(date_str.split("-")[1])
        year = date_str.split("-")[0]
        quarter = (month - 1) // 3 + 1
        return f"Q{quarter} {year}"
    except (IndexError, ValueError):
        return "Unknown"


def get_sample_sec_data() -> List[Dict]:
    """
    Return pre-built sample SEC filing excerpts for demo/testing.
    These are real data points from public SEC filings.
    """
    return [
        {
            "text": "Total net revenue for fiscal year 2023 was $383.3 billion, compared to $394.3 billion for fiscal year 2022. The decrease was driven primarily by lower iPhone revenue, partially offset by growth in Services.",
            "company": "Apple",
            "source": "SEC 10-K",
            "filing_date": "2023-11-03",
            "quarter": "FY 2023",
            "filing_type": "10-K",
        },
        {
            "text": "Products revenue was $298.1 billion for fiscal 2023, down 5% from $316.2 billion in fiscal 2022. iPhone revenue was $200.6 billion, Mac revenue was $29.4 billion, iPad revenue was $28.3 billion, and Wearables, Home and Accessories revenue was $39.8 billion.",
            "company": "Apple",
            "source": "SEC 10-K",
            "filing_date": "2023-11-03",
            "quarter": "FY 2023",
            "filing_type": "10-K",
        },
        {
            "text": "Services revenue was $85.2 billion for fiscal 2023, up 9% from $78.1 billion in fiscal 2022. The growth was driven by higher revenue from advertising, the App Store, and cloud services.",
            "company": "Apple",
            "source": "SEC 10-K",
            "filing_date": "2023-11-03",
            "quarter": "FY 2023",
            "filing_type": "10-K",
        },
        {
            "text": "We produced approximately 1.85 million consumer vehicles in 2023, a 35% increase over 2022. Vehicle deliveries totaled 1.81 million units, representing a 38% year-over-year increase. Automotive revenue was $82.4 billion, an increase of 15% year-over-year.",
            "company": "Tesla",
            "source": "SEC 10-K",
            "filing_date": "2024-01-29",
            "quarter": "FY 2023",
            "filing_type": "10-K",
        },
        {
            "text": "Our energy generation and storage business generated revenue of $6.0 billion in 2023, an increase of 54% compared to 2022. Energy storage deployments reached 14.7 GWh in 2023, more than double the 6.5 GWh deployed in 2022.",
            "company": "Tesla",
            "source": "SEC 10-K",
            "filing_date": "2024-01-29",
            "quarter": "FY 2023",
            "filing_type": "10-K",
        },
        {
            "text": "Tesla's total GAAP gross margin was 18.2% in 2023, compared to 25.6% in 2022. The decrease in gross margin was primarily due to reduced average selling prices of our vehicles, partially offset by lower raw material costs and growth in other businesses.",
            "company": "Tesla",
            "source": "SEC 10-K",
            "filing_date": "2024-01-29",
            "quarter": "FY 2023",
            "filing_type": "10-K",
        },
        {
            "text": "Microsoft Cloud revenue was $118.5 billion, up 22% year-over-year. Azure and other cloud services revenue grew 29%. The growth was driven by consumption-based services and per-user services including Microsoft 365.",
            "company": "Microsoft",
            "source": "SEC 10-K",
            "filing_date": "2023-07-27",
            "quarter": "FY 2023",
            "filing_type": "10-K",
        },
        {
            "text": "Revenue from Intelligent Cloud segment was $87.9 billion, an increase of 17% from fiscal year 2022. Operating income from this segment was $37.9 billion, an increase of 11% driven by growth in Azure.",
            "company": "Microsoft",
            "source": "SEC 10-K",
            "filing_date": "2023-07-27",
            "quarter": "FY 2023",
            "filing_type": "10-K",
        },
        {
            "text": "Amazon Web Services segment revenue was $90.8 billion in 2023, up 13% year-over-year. AWS operating income was $24.6 billion compared to $22.8 billion in 2022. AWS continues to be the market leader in cloud infrastructure services.",
            "company": "Amazon",
            "source": "SEC 10-K",
            "filing_date": "2024-02-01",
            "quarter": "FY 2023",
            "filing_type": "10-K",
        },
        {
            "text": "Amazon's net sales increased 12% to $574.8 billion in 2023, compared with $514.0 billion in 2022. North America segment sales were $352.8 billion, up 12%. International segment sales were $131.2 billion, up 11%.",
            "company": "Amazon",
            "source": "SEC 10-K",
            "filing_date": "2024-02-01",
            "quarter": "FY 2023",
            "filing_type": "10-K",
        },
        {
            "text": "Operating cash flow increased 82% to $84.9 billion, compared with $46.7 billion in 2022. Free cash flow improved to $36.8 billion, compared with negative $11.6 billion in 2022, driven by increased operating income and efficient capital expenditure management.",
            "company": "Amazon",
            "source": "SEC 10-K",
            "filing_date": "2024-02-01",
            "quarter": "FY 2023",
            "filing_type": "10-K",
        },
        {
            "text": "The Federal Reserve maintained the federal funds rate in the target range of 5.25% to 5.50% through Q4 2023, following 11 rate increases since March 2022 totaling 525 basis points. Committee members signaled potential rate cuts in 2024 contingent on inflation progress.",
            "company": "Federal Reserve",
            "source": "FOMC Statement",
            "filing_date": "2023-12-13",
            "quarter": "Q4 2023",
            "filing_type": "policy",
        },
    ]
