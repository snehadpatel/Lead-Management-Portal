import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# Set up directories
BASE_DIR = "/Users/snehapatel/BigData"
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
PDF_DIR = os.path.join(DATASETS_DIR, "prospectus_pdfs")
UNSTRUCTURED_DIR = os.path.join(DATASETS_DIR, "unstructured", "prospectus_texts")
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(UNSTRUCTURED_DIR, exist_ok=True)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
}

# Verified working AMC pages that contain actual PDF links
AMC_SOURCES = [
    {
        "name": "SBI_MF",
        "pages": [
            "https://www.sbimf.com",
            "https://www.sbimf.com/en-us/information-centre",
        ],
        "base_url": "https://www.sbimf.com",
    },
    {
        "name": "HDFC_MF",
        "pages": [
            "https://www.hdfcfund.com",
            "https://www.hdfcfund.com/about-us",
            "https://www.hdfcfund.com/statutory-disclosure",
        ],
        "base_url": "https://www.hdfcfund.com",
    },
    {
        "name": "Nippon_India_MF",
        "pages": [
            "https://mf.nipponindiaim.com",
        ],
        "base_url": "https://mf.nipponindiaim.com",
    },
    {
        "name": "Tata_MF",
        "pages": [
            "https://www.tatamutualfund.com",
        ],
        "base_url": "https://www.tatamutualfund.com",
    },
    {
        "name": "Baroda_BNP_MF",
        "pages": [
            "https://www.barodabnpparibasmf.in",
        ],
        "base_url": "https://www.barodabnpparibasmf.in",
    },
]


def scrape_all_pdf_links():
    """Scrape all AMC websites for PDF links."""
    all_links = []
    
    for source in AMC_SOURCES:
        name = source["name"]
        base = source["base_url"]
        
        for page_url in source["pages"]:
            print(f"  Scraping {name}: {page_url}")
            try:
                r = requests.get(page_url, headers=HEADERS, timeout=15)
                soup = BeautifulSoup(r.text, 'html.parser')
                
                for a in soup.find_all('a', href=True):
                    href = a['href'].strip()
                    if '.pdf' in href.lower():
                        if href.startswith('http'):
                            full_url = href
                        elif href.startswith('//'):
                            full_url = 'https:' + href
                        elif href.startswith('/'):
                            full_url = base + href
                        else:
                            full_url = urljoin(page_url, href)
                        all_links.append((name, full_url))
                        
                time.sleep(1)
            except Exception as e:
                print(f"    Error: {e}")
    
    # Deduplicate by URL
    seen = set()
    unique = []
    for name, url in all_links:
        if url not in seen:
            seen.add(url)
            unique.append((name, url))
    
    print(f"\n  Total unique PDF links found: {len(unique)}")
    return unique


def download_and_extract(links):
    """Download PDFs and extract text."""
    downloaded = 0
    extracted = 0
    
    for name, url in links:
        try:
            # Create filename
            parsed = urlparse(url)
            raw_name = os.path.basename(parsed.path)
            if not raw_name or not raw_name.endswith('.pdf'):
                raw_name = raw_name + '.pdf' if raw_name else 'document.pdf'
            filename = f"{name}_{raw_name}".replace('%20', '_').replace(' ', '_')
            file_path = os.path.join(PDF_DIR, filename)
            
            if os.path.exists(file_path):
                downloaded += 1
                continue
            
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200 and (b'%PDF' in r.content[:10] or 'pdf' in r.headers.get('content-type', '')):
                with open(file_path, 'wb') as f:
                    f.write(r.content)
                downloaded += 1
                
                # Extract text immediately
                try:
                    import fitz
                    doc = fitz.open(file_path)
                    text = ""
                    for i, page in enumerate(doc):
                        text += f"\n--- Page {i+1} ---\n{page.get_text()}"
                    doc.close()
                    
                    txt_name = os.path.splitext(filename)[0] + '.txt'
                    txt_path = os.path.join(UNSTRUCTURED_DIR, txt_name)
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(text)
                    extracted += 1
                except Exception:
                    pass
                    
            time.sleep(0.5)
        except Exception:
            pass
    
    return downloaded, extracted


def main():
    print("=" * 60)
    print("Official AMC Document Collector")
    print("=" * 60)
    
    links = scrape_all_pdf_links()
    
    if not links:
        print("No PDF links found!")
        return
    
    downloaded, extracted = download_and_extract(links)
    
    print(f"\n{'=' * 60}")
    print(f"PDFs downloaded: {downloaded}")
    print(f"Texts extracted: {extracted}")
    print(f"PDF dir: {PDF_DIR}")
    print(f"Text dir: {UNSTRUCTURED_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
