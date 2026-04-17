import os
import subprocess
import shutil

# Set up directories
BASE_DIR = "/Users/snehapatel/BigData"
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
KAGGLE_DIR = os.path.join(DATASETS_DIR, "kaggle")
os.makedirs(KAGGLE_DIR, exist_ok=True)

def download_kaggle_dataset(dataset_slug, download_path):
    print(f"Downloading Kaggle dataset: {dataset_slug}...")
    try:
        # We use the Kaggle CLI which is installed via pip (Requires ~/.kaggle/kaggle.json user setup)
        command = [
            "kaggle", "datasets", "download", 
            "-d", dataset_slug,
            "-p", download_path,
            "--unzip"
        ]
        
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Successfully downloaded and unzipped {dataset_slug} to {download_path}")
        else:
            print(f"Failed to download {dataset_slug}. Error:\n{result.stderr}")
            print(f"Please ensure you have configured your Kaggle API key at ~/.kaggle/kaggle.json")
    
    except FileNotFoundError:
        print("Error: 'kaggle' command not found. Have you installed it with 'pip install kaggle'?")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    print("Fetching massive unstructured datasets from Kaggle...")
    
    # 1. Social Sentiment Data (Twitter Financial Sentiment)
    # This dataset contains thousands of finance related tweets with sentiment labels
    sentiment_dataset = "sbhatti/financial-sentiment-analysis"
    sentiment_dir = os.path.join(KAGGLE_DIR, "social_sentiment")
    os.makedirs(sentiment_dir, exist_ok=True)
    download_kaggle_dataset(sentiment_dataset, sentiment_dir)
    
    # 2. Semi-Structured Data (JSON / Web Clickstream)
    # Real web interaction logs tracking page clicks, dwell time, and user session IDs.
    # This precisely acts as our clickstream data for Engagement Scoring.
    interaction_dataset = "tunguz/clickstream-data-for-online-shopping"
    interaction_dir = os.path.join(KAGGLE_DIR, "financial_user_behaviors")
    os.makedirs(interaction_dir, exist_ok=True)
    download_kaggle_dataset(interaction_dataset, interaction_dir)
    # 3. Massive Unstructured Text (Financial News)
    # 1.2 Gigabytes of raw financial news articles, representing genuine unstructured Big Data
    # Useful for training NLP sentiment and contextual models
    news_dataset = "jeet2016/us-financial-news-articles"
    news_dir = os.path.join(KAGGLE_DIR, "massive_financial_news")
    os.makedirs(news_dir, exist_ok=True)
    download_kaggle_dataset(news_dataset, news_dir)
    
    # 4. Unstructured Image Data (Candlestick Charts)
    # Candlestick chart images suitable for computer vision models in finance
    images_dataset = "raimiazeezbabatunde/candle-image-data"
    images_dir = os.path.join(KAGGLE_DIR, "candlestick_images")
    os.makedirs(images_dir, exist_ok=True)
    download_kaggle_dataset(images_dataset, images_dir)
    
    # 5. Unstructured PDF Data (Financial Receipts)
    # Scanned receipt PDFs suitable for OCR and document parsing tasks
    pdfs_dataset = "jenswalter/receipts"
    pdfs_dir = os.path.join(KAGGLE_DIR, "receipt_pdfs")
    os.makedirs(pdfs_dir, exist_ok=True)
    download_kaggle_dataset(pdfs_dataset, pdfs_dir)

    print("\nKaggle dataset pull complete.")
    print("Please verify the downloaded files in the datasets/kaggle/ directory.")

if __name__ == "__main__":
    main()
