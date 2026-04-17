import feedparser
import json
import os
import ssl
from bs4 import BeautifulSoup
import re

# Set up directories
BASE_DIR = "/Users/snehapatel/BigData"
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
UNSTRUCTURED_DIR = os.path.join(DATASETS_DIR, "unstructured", "financial_news")
os.makedirs(UNSTRUCTURED_DIR, exist_ok=True)

# Ignore SSL certificate errors for RSS feeds
if hasattr(ssl, '_create_unverified_context'):
    ssl._create_default_https_context = ssl._create_unverified_context

def clean_html(raw_html):
    if not isinstance(raw_html, str):
        return ""
    # Remove HTML tags using beautiful soup
    soup = BeautifulSoup(raw_html, "html.parser")
    text = soup.get_text(separator=' ')
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def fetch_financial_news():
    print("Fetching real Financial News from RSS feeds...")

    # Legal RSS feeds for financial news
    rss_urls = [
        "https://www.moneycontrol.com/rss/business.xml",
        "https://www.moneycontrol.com/rss/marketreports.xml",
        "https://www.moneycontrol.com/rss/mutualfunds.xml",
        "https://economictimes.indiatimes.com/markets/rssfeeds/19770215.cms", 
        "https://economictimes.indiatimes.com/mutual-funds/rssfeeds/8972232.cms" 
    ]

    count = 0

    for url in rss_urls:
        print(f"  Parsing {url}...")
        try:
            feed = feedparser.parse(url)
            source = "Moneycontrol" if "moneycontrol" in url else "Economic Times"
            category = "Mutual Funds" if "mutual" in url else "Markets_Business"
            
            for i, entry in enumerate(feed.entries):
                title = clean_html(entry.title) if hasattr(entry, 'title') else ""
                
                # We need a safe filename
                safe_title = re.sub(r'[^\w\s-]', '', title)[:50].strip().replace(' ', '_')
                if not safe_title:
                    safe_title = f"article_{i}"
                
                article = {
                    'Source': source,
                    'Category': category,
                    'Title': title,
                    'Link': entry.link if hasattr(entry, 'link') else "",
                    'Published_Date': entry.published if hasattr(entry, 'published') else "",
                    'Body_Text': clean_html(entry.description) if hasattr(entry, 'description') else ""
                }
                
                # Save purely unstructured JSON file per article
                file_name = f"{source}_{category}_{safe_title}.json"
                out_path = os.path.join(UNSTRUCTURED_DIR, file_name)
                
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(article, f, indent=4)
                
                # Also save the raw text for pure NLP
                text_out_path = os.path.join(UNSTRUCTURED_DIR, f"{source}_{category}_{safe_title}.txt")
                with open(text_out_path, 'w', encoding='utf-8') as f:
                    f.write(f"{title}\n\n{article['Body_Text']}")

                count += 1
                
        except Exception as e:
            print(f"    Failed to parse {url}: {e}")

    print(f"Successfully saved {count} raw unstructured financial news files (JSON/TXT) to {UNSTRUCTURED_DIR}")

if __name__ == "__main__":
    fetch_financial_news()
