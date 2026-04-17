import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# Set up directories
BASE_DIR = "/Users/snehapatel/BigData"
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
IMAGES_DIR = os.path.join(DATASETS_DIR, "unstructured", "fund_images")
os.makedirs(IMAGES_DIR, exist_ok=True)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
}

# Official AMC websites to scrape images from
AMC_PAGES = [
    {
        "name": "SBI_MF",
        "urls": [
            "https://www.sbimf.com/en-us",
            "https://www.sbimf.com/en-us/about-us",
        ],
        "base_url": "https://www.sbimf.com",
    },
    {
        "name": "HDFC_MF",
        "urls": [
            "https://www.hdfcfund.com",
            "https://www.hdfcfund.com/about-us",
        ],
        "base_url": "https://www.hdfcfund.com",
    },
    {
        "name": "ICICI_Pru_MF",
        "urls": [
            "https://www.icicipruamc.com",
            "https://www.icicipruamc.com/about-us",
        ],
        "base_url": "https://www.icicipruamc.com",
    },
    {
        "name": "Nippon_India_MF",
        "urls": [
            "https://mf.nipponindiaim.com",
        ],
        "base_url": "https://mf.nipponindiaim.com",
    },
    {
        "name": "Axis_MF",
        "urls": [
            "https://www.axismf.com",
            "https://www.axismf.com/about-us",
        ],
        "base_url": "https://www.axismf.com",
    },
    {
        "name": "Kotak_MF",
        "urls": [
            "https://www.kotakmf.com",
        ],
        "base_url": "https://www.kotakmf.com",
    },
    {
        "name": "AMFI",
        "urls": [
            "https://www.amfiindia.com",
        ],
        "base_url": "https://www.amfiindia.com",
    },
]

VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.svg', '.gif')


def scrape_images(source):
    """Scrape images from AMC website pages."""
    name = source["name"]
    base = source["base_url"]
    image_urls = []
    
    for url in source["urls"]:
        print(f"  Scraping images from: {url}")
        try:
            r = requests.get(url, headers={
                'User-Agent': HEADERS['User-Agent'],
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            }, timeout=15)
            
            if r.status_code != 200:
                print(f"    Got HTTP {r.status_code}, skipping.")
                continue
                
            soup = BeautifulSoup(r.text, 'html.parser')
            
            # Find all <img> tags
            for img in soup.find_all('img', src=True):
                src = img['src'].strip()
                if any(src.lower().endswith(ext) for ext in VALID_EXTENSIONS):
                    if src.startswith('http'):
                        image_urls.append(src)
                    elif src.startswith('//'):
                        image_urls.append('https:' + src)
                    elif src.startswith('/'):
                        image_urls.append(base + src)
                    else:
                        image_urls.append(urljoin(url, src))
                        
            # Also check CSS background images and srcset
            for tag in soup.find_all(attrs={"srcset": True}):
                srcset = tag['srcset']
                for part in srcset.split(','):
                    src = part.strip().split(' ')[0]
                    if any(src.lower().endswith(ext) for ext in VALID_EXTENSIONS):
                        if src.startswith('http'):
                            image_urls.append(src)
                        elif src.startswith('/'):
                            image_urls.append(base + src)
                            
            time.sleep(1)  # Be polite
            
        except Exception as e:
            print(f"    Error: {e}")
    
    # Deduplicate
    image_urls = list(set(image_urls))
    
    # Filter out tiny tracking pixels and icons (by URL heuristic)
    filtered = []
    for u in image_urls:
        lower = u.lower()
        # Skip common non-useful images
        if any(skip in lower for skip in ['tracking', 'pixel', '1x1', 'spacer', 'blank', 'favicon']):
            continue
        filtered.append(u)
    
    print(f"    Found {len(filtered)} potential images from {name}")
    return filtered


def download_image(url, prefix, idx):
    """Download a single image."""
    try:
        parsed = urlparse(url)
        ext = os.path.splitext(parsed.path)[1].lower()
        if ext not in VALID_EXTENSIONS:
            ext = '.png'
            
        filename = f"{prefix}_img_{idx:03d}{ext}"
        file_path = os.path.join(IMAGES_DIR, filename)
        
        if os.path.exists(file_path):
            return True
            
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200 and len(r.content) > 1000:  # Skip tiny images (<1KB)
            with open(file_path, 'wb') as f:
                f.write(r.content)
            return True
            
    except Exception:
        pass
    return False


def main():
    print("=" * 60)
    print("Official AMC Fund Image Collector")
    print("=" * 60)
    
    total_downloaded = 0
    
    for source in AMC_PAGES:
        urls = scrape_images(source)
        
        if not urls:
            continue
            
        prefix = source["name"]
        downloaded = 0
        
        for i, url in enumerate(urls):
            if download_image(url, prefix, i):
                downloaded += 1
            time.sleep(0.3)
        
        print(f"    Downloaded {downloaded} images from {prefix}")
        total_downloaded += downloaded
        
    print(f"\n{'=' * 60}")
    print(f"Total images collected: {total_downloaded}")
    print(f"Saved to: {IMAGES_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
