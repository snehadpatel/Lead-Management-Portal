# Studio-Based Project — AI, Cloud, and Big Data
# Dataset Documentation

**Project Title:** AI-Powered Lead Recommendation System for Mutual Fund Distributors

**Project Group Details:**
- *(Fill in your group member names and roll numbers)*

**Submission Date:** *(Fill in)*

---

## 1. Basic Dataset Information

| Field | Details |
|---|---|
| **Dataset Name** | Indian Financial Markets Composite Dataset |
| **Domain / Application Area** | Finance — Mutual Fund Lead Generation, Investor Behavior Analytics, NAV Prediction, Market Sentiment Analysis |
| **Type of Data** | Structured, Semi-Structured, and Unstructured |
| **Sources** | AMFI India (amfiindia.com), NSE India (nseindia.com), World Bank Open Data, Kaggle, Moneycontrol RSS, Economic Times RSS, Official AMC Websites (SBI MF, HDFC MF, Nippon India MF, Tata MF, Axis MF, Kotak MF, Baroda BNP Paribas MF) |
| **Dataset Links / References** | See Section 1.1 below |
| **License / Usage Rights** | Kaggle datasets: CC-BY-NC-SA-4.0 and CC0 Public Domain; AMFI/NSE data: Publicly available for educational and research use; RBI data: Open Government License — India |
| **Creator(s) and Ownership** | AMFI (Association of Mutual Funds in India), NSE (National Stock Exchange of India), World Bank Group, Kaggle community contributors, Official AMC fund houses |
| **Publication & Access Dates** | Data spans 1990–2026; Acquired February 2026 |
| **Total Dataset Size** | **30 GB** |

### 1.1 Source References (APA Format)

1. Association of Mutual Funds in India. (2026). *AMFI NAV Data Feed*. Retrieved February 2026, from https://www.amfiindia.com/spages/NAVAll.txt
2. National Stock Exchange of India. (2026). *NSE Market Status API*. Retrieved February 2026, from https://www.nseindia.com/api/marketStatus
3. World Bank Group. (2024). *World Development Indicators — India*. Retrieved February 2026, from https://data.worldbank.org/country/india
4. Debashis. (2026). *ALGO TRADING DATA — Nifty 500 Intraday Data with Indicators* [Dataset]. Kaggle. https://www.kaggle.com/datasets/debashis74017/algo-trading-data-nifty-100-data-with-indicators
5. Hk7797. (2021). *Stock Market India* [Dataset]. Kaggle. https://www.kaggle.com/datasets/hk7797/stock-market-india
6. Jeet2016. (2020). *US Financial News Articles* [Dataset]. Kaggle. https://www.kaggle.com/datasets/jeet2016/us-financial-news-articles
7. Nishanth Salian. (2021). *Indian Stock Index EOD Data 1990 Onwards* [Dataset]. Kaggle. https://www.kaggle.com/datasets/nishanthsalian/indian-stock-index-eod-data1990-onwards
8. Nishanth Salian. (2021). *Indian Stock Index 1-Minute Data 2008-2020* [Dataset]. Kaggle. https://www.kaggle.com/datasets/nishanthsalian/indian-stock-index-1minute-data-2008-2020
9. Tunguz. (2020). *Clickstream Data for Online Shopping* [Dataset]. Kaggle. https://www.kaggle.com/datasets/tunguz/clickstream-data-for-online-shopping
10. Sbhatti. (2020). *Financial Sentiment Analysis* [Dataset]. Kaggle. https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis
11. Moneycontrol. (2026). *RSS Feeds — Markets & Business*. Retrieved February 2026, from https://www.moneycontrol.com/rss/MCtopnews.xml
12. Economic Times. (2026). *RSS Feeds — Markets*. Retrieved February 2026, from https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms
13. NJGroup. (2023). *Mutual Fund Investors Behaviour* [Dataset]. Kaggle. https://www.kaggle.com/datasets/njgroup/mutual-fund-investors-behaviour
14. Chatterjee, A. (2020). *Lead Scoring Dataset* [Dataset]. Kaggle. https://www.kaggle.com/datasets/amritachatterjee09/lead-scoring-dataset
15. NickEinstein. (2025). *Insurance Leads* [Dataset]. Kaggle. https://www.kaggle.com/datasets/nickeinstein/insurance-leads
16. Bhone, A. (2022). *Lead Score Case Study* [Dataset]. Kaggle. https://www.kaggle.com/datasets/amolbhone/lead-score-case-study

---

## 2. Data Characteristics

### 2.1 Structured Data

#### A. AMFI Mutual Fund Scheme List
| Attribute | Details |
|---|---|
| **File** | `structured/mutual_funds/amfi_scheme_list.csv` |
| **Records (Rows)** | 14,275 |
| **Features (Columns)** | 7 |
| **File Size** | 1.9 MB |
| **Target Variable** | Net_Asset_Value (for NAV prediction) |

| Column Name | Data Type | Description |
|---|---|---|
| Category | String | Fund category (e.g., Equity, Debt, Hybrid) |
| Scheme_Code | Integer | Unique AMFI scheme identifier |
| ISIN_Div_Payout | String | ISIN for dividend payout option |
| ISIN_Div_Reinvestment | String | ISIN for dividend reinvestment option |
| Scheme_Name | String | Full name of the mutual fund scheme |
| Net_Asset_Value | Float | Latest NAV in INR |
| Date | String (Date) | Date of the NAV value |

#### B. Mutual Fund NAV History
| Attribute | Details |
|---|---|
| **Directory** | `structured/mutual_funds/nav_history/` |
| **Total Files** | 2,965 CSV files |
| **Total Rows** | 7,346,444 |
| **Columns per File** | 2 (date, nav) |
| **File Size** | 151 MB total |
| **Time Period** | Varies per scheme (up to 20+ years) |

| Column Name | Data Type | Description |
|---|---|---|
| date | String (Date) | NAV date in YYYY-MM-DD format |
| nav | Float | Net Asset Value on that date |

#### C. NIFTY 50 Historical Index
| Attribute | Details |
|---|---|
| **File** | `structured/stock_prices/nifty50_index/nse_nifty50_historical_merged.csv` |
| **Records** | 745 |
| **Features** | 7 |
| **File Size** | 48 KB |
| **Time Period** | Feb 2023 – Feb 2026 |

| Column Name | Data Type | Description |
|---|---|---|
| Date | String (Date) | Trading date |
| Open | Float | Opening index value |
| High | Float | Highest index value of the day |
| Low | Float | Lowest index value of the day |
| Close | Float | Closing index value |
| Shares Traded | Integer | Total shares traded |
| Turnover (₹ Cr) | Float | Trading turnover in crores |

#### D. EOD Bhav Copy Data (NSE)
| Attribute | Details |
|---|---|
| **Directory** | `structured/stock_prices/eod_bhav_copies/` |
| **Total Files** | 2,235 CSV files (one per stock) |
| **Total Rows (all files)** | 6,367,114 |
| **Columns per File** | 6 (Price, Close, High, Low, Open, Volume) |
| **Total Size** | 529 MB |

#### E. NIFTY 500 Intraday Data with Technical Indicators
| Attribute | Details |
|---|---|
| **Directory** | `structured/stock_prices/nifty500_intraday/` |
| **Total Files** | 499 CSV files (one per stock) |
| **Rows per File** | 101,665 – 1,015,399 (minute-level candles) |
| **Columns per File** | 6 (date, open, high, low, close, volume) |
| **Total Size** | 19 GB |
| **Time Period** | Multi-year intraday data (2008–2026) |

#### F. Stock Market India (All Listed Companies)
| Attribute | Details |
|---|---|
| **Directory** | `structured/stock_prices/stock_market_india/` |
| **Total Files** | 162 |
| **Total Size** | 5.8 GB |

#### G. NSE EOD Data (1990 Onwards)
| Attribute | Details |
|---|---|
| **Directory** | `structured/stock_prices/nse_eod_1990/` |
| **Total Files** | 3,189 CSV files |
| **Total Size** | 512 MB |
| **Time Period** | 1990 – 2021 |

#### H. NSE Intraday Minute Data (2008–2020)
| Attribute | Details |
|---|---|
| **Directory** | `structured/stock_prices/nse_intraday_minute/` |
| **Total Files** | 26 |
| **Total Size** | 239 MB |
| **Time Period** | 2008 – 2020 |

#### I. World Bank Macroeconomic Indicators (India)
| Attribute | Details |
|---|---|
| **File** | `structured/macroeconomic/world_bank_indicators.csv` |
| **Records** | 22 |
| **Features** | 4 |
| **Time Period** | 2003 – 2024 |

| Column Name | Data Type | Description |
|---|---|---|
| Year | Integer | Calendar year |
| Inflation_Annual_Pct | Float | Annual inflation rate (%) |
| Real_Interest_Rate_Pct | Float | Real interest rate (%) |
| GDP_Growth_Annual_Pct | Float | Annual GDP growth rate (%) |

#### J. Mutual Fund Investor Behavior
| Attribute | Details |
|---|---|
| **File** | `structured/leads/mf_investor_behavior/MF_Behavior.xlsx` |
| **Records** | 1,000 |
| **Features** | 15 |
| **Target Variable** | AUM (Assets Under Management — indicates lead quality) |

| Column Name | Data Type | Description |
|---|---|---|
| Investor_ID | Integer | Unique investor identifier |
| Longevity | Float | Duration of investment relationship |
| Female | Binary (0/1) | Gender indicator |
| Age | Integer | Investor age |
| Income | Float | Annual income |
| ProfManage | Float | Preference for professional management |
| Diversification | Float | Preference for portfolio diversification |
| Affordability | Float | Price sensitivity score |
| Liquidity | Float | Preference for liquid investments |
| Growth | Float | Preference for growth-oriented funds |
| Trustworthiness | Float | Trust factor score |
| Technology | Float | Technology adoption score |
| Integrity | Float | Perceived integrity score |
| BrandValue | Float | Brand importance score |
| AUM | Float | Assets Under Management (investment size) |

#### K. Lead Scoring Dataset
| Attribute | Details |
|---|---|
| **File** | `structured/leads/lead_scoring/Lead Scoring.csv` |
| **Records** | 9,240 |
| **Features** | 37 |
| **Target Variable** | Converted (binary: 1 = converted to customer, 0 = did not convert) |

| Key Columns | Data Type | Description |
|---|---|---|
| Prospect ID | String | Unique lead identifier |
| Lead Origin | String | Source channel (API, Landing Page, etc.) |
| Lead Source | String | How the lead was acquired |
| Converted | Binary (0/1) | **Target** — whether the lead converted |
| TotalVisits | Integer | Number of website visits |
| Total Time Spent on Website | Float | Engagement duration (minutes) |
| Page Views Per Visit | Float | Pages browsed per session |
| Last Activity | String | Most recent interaction type |
| Country | String | Geographic location |
| Current Occupation | String | Employment status |
| What matters most | String | Key decision factor |

#### L. Insurance Leads (Financial Product Lead Prediction)
| Attribute | Details |
|---|---|
| **File** | `structured/leads/insurance_leads/insurance_leads_training.csv` |
| **Records** | 5,000 |
| **Features** | 12 |
| **Target Variable** | conversion_score (lead quality score) |

| Column Name | Data Type | Description |
|---|---|---|
| lead_id | Integer | Unique lead identifier |
| age | Integer | Prospect age |
| income | Float | Annual income |
| policy_type | String | Type of financial product interest |
| quote_requests_30d | Integer | Engagement signal — quote requests in last 30 days |
| social_engagement_score | Float | Social media interaction score |
| location_risk_score | Float | Geographic risk factor |
| previous_insurance | Binary (0/1) | Has existing financial product |
| credit_score_proxy | Float | Credit worthiness indicator |
| consent_given | Binary (0/1) | Marketing consent status |
| consent_timestamp | DateTime | When consent was given |
| conversion_score | Float | **Target** — predicted lead quality |

#### M. Lead Score Case Study (UpGrad)
| Attribute | Details |
|---|---|
| **Directory** | `structured/leads/lead_score_upgrad/` |
| **Files** | Leads.csv (9,240 rows), Data Dictionary, Case Study PDF, Presentation PDF, Jupyter Notebook |
| **Total Size** | 22.8 MB |
| **Content** | Complete lead scoring case study with analysis, same schema as Lead Scoring dataset |

### 2.2 Semi-Structured Data

#### A. Clickstream / User Interaction Data
| Attribute | Details |
|---|---|
| **Directory** | `semi_structured/clickstream/` |
| **Files** | `events.csv` (2,756,101 rows), `item_properties_part1.csv`, `item_properties_part2.csv`, `category_tree.csv` |
| **Total Size** | 942 MB |
| **Format** | CSV (semi-structured event logs) |

| Column Name (events.csv) | Data Type | Description |
|---|---|---|
| timestamp | Integer (Unix) | Event timestamp |
| visitorid | Integer | Unique visitor identifier |
| event | String | Event type (view, addtocart, transaction) |
| itemid | Integer | Item interacted with |
| transactionid | Integer | Transaction ID (if purchase) |

#### B. Financial News RSS Feed
| Attribute | Details |
|---|---|
| **Directory** | `semi_structured/financial_news_rss/` |
| **Files** | 35 (1 CSV + 17 JSON + 17 TXT) |
| **Format** | CSV, JSON, TXT |
| **Sources** | Moneycontrol, Economic Times |

#### C. Social Sentiment Data
| Attribute | Details |
|---|---|
| **File** | `semi_structured/social_sentiment/data.csv` |
| **Records** | 5,842 |
| **Columns** | 2 (Sentence, Sentiment) |
| **Labels** | positive, negative, neutral |

#### D. User Behavior Data
| Attribute | Details |
|---|---|
| **File** | `semi_structured/user_behaviors/e-shop clothing 2008.csv` |
| **Records** | 165,474 |
| **Columns** | 14 |
| **Annotation** | Labeled with session, category, and behavior data |

### 2.3 Unstructured Data

#### A. Financial News Corpus
| Attribute | Details |
|---|---|
| **Directory** | `unstructured/financial_news_corpus/` |
| **Total Files** | 306,247 (JSON + TXT pairs) |
| **Total Size** | 2.5 GB |
| **Format** | JSON (metadata) and TXT (full article text) |
| **Annotation** | Unlabeled raw text |

#### B. AMC Prospectus PDFs
| Attribute | Details |
|---|---|
| **Directory** | `unstructured/prospectus_pdfs/` |
| **Total Files** | 33 PDF documents |
| **Total Size** | 45 MB |
| **Sources** | SBI MF, HDFC MF, Nippon India MF, Tata MF, Baroda BNP Paribas MF, AMFI |
| **Content** | Factsheets, yearbooks, SAI documents, valuation policies, annual reports |

#### C. Extracted PDF Texts
| Attribute | Details |
|---|---|
| **Directory** | `unstructured/prospectus_texts/` |
| **Total Files** | 34 TXT files |
| **Format** | Plain text extracted via PyMuPDF |

#### D. Fund House Images
| Attribute | Details |
|---|---|
| **Directory** | `unstructured/fund_images/` |
| **Total Files** | 84 images |
| **Formats** | PNG, JPG, JPEG, WebP, SVG |
| **Sources** | HDFC MF, Nippon India MF, Axis MF, Kotak MF, AMFI India |
| **Content** | Fund house logos, banners, financial infographics |

---

## 3. Big Data Perspective (5 Vs)

### Volume
The dataset comprises **30 GB** of data across **315,538 files** and **millions of records**. Key contributors:
- 19 GB of NIFTY 500 intraday minute-level data (499 stocks, 101K–1.01M rows each)
- 5.8 GB of Stock Market India data (all listed companies)
- 7,346,444 rows of mutual fund NAV history across 2,965 funds
- 2.75 million clickstream events
- 15,240 lead records with conversion labels across 3 lead scoring datasets
- 1,000 mutual fund investor behavior profiles with demographics and AUM
- 306,247 financial news articles (2.5 GB)

This scale requires distributed processing tools (PySpark, Hadoop) and cannot be efficiently handled by traditional single-machine databases.

### Velocity
Real-time and near-real-time data is demonstrated through three live streaming sources:
- **AMFI Live NAV Feed**: Complete NAV data for 14,275 schemes refreshed daily at market close (17,546 lines per snapshot)
- **NSE Market Pulse**: NIFTY 50 index polled every 5 seconds via the official NSE API, capturing real-time price, variation, and market status
- **Live Financial News Stream**: 5 RSS feeds (Moneycontrol, Economic Times, Livemint) polled every 30 seconds, capturing breaking headlines with microsecond ingestion timestamps

In a single 2-minute burst, the system ingested 17,546-line NAV snapshots, 24 market pulse ticks, and 118 live headlines — demonstrating genuine high-frequency data ingestion.

### Variety
The dataset spans all three categories of data:
- **Structured**: CSV files with tabular data — stock prices (OHLCV), mutual fund NAVs, macroeconomic indicators, scheme metadata
- **Semi-Structured**: JSON news articles, XML/RSS feeds, CSV-based clickstream event logs with nested session data, labeled sentiment data
- **Unstructured**: Raw PDF documents (mutual fund factsheets, annual reports), fund house images (PNG/JPG/SVG), and plain text extracts from 306K+ financial news articles

### Veracity
Data quality is ensured through official and verified sources:
- **AMFI India**: The statutory regulatory body for mutual funds; NAV data is directly from fund houses
- **NSE India**: Official stock exchange data — audited and regulated
- **World Bank**: Internationally standardized macroeconomic indicators
- **Kaggle**: Community-vetted datasets with peer review and licensing
- Duplicate detection confirmed only **2.97 MB** of redundant data out of 30 GB (0.01%), primarily due to mutual fund plans with identical NAV histories (direct vs. regular plans)
- Missing value analysis shows **zero missing values** in core datasets (AMFI scheme list, NIFTY 50 index)

### Value
The dataset enables the core business objective — **recommending high-quality leads to mutual fund distributors** — along with supporting analytics:
- **Lead Scoring & Recommendation**: Lead Scoring dataset (9,240 leads with `Converted` labels) and Insurance Leads (5,000 leads with `conversion_score`) train ML models to predict which prospects will convert — MF distributors receive ranked lead lists sorted by conversion probability
- **Investor Profiling & Segmentation**: MF Investor Behavior data (1,000 profiles with Age, Income, AUM, risk preferences) enables K-Means clustering to build investor personas — distributors target leads matching high-AUM investor profiles
- **NAV Prediction**: LSTM time-series forecasting of mutual fund performance using historical NAV data — helps distributors pitch funds with strong predicted returns
- **Sentiment-Driven Lead Timing**: NLP sentiment analysis on 306K+ news articles and 5,842 labeled tweets gauges market mood — distributors can time outreach when investor sentiment is positive
- **Fund-Investor Matching**: Combining fund characteristics (category, risk, NAV trend) with investor profiles to recommend the right fund to the right lead
- **Document Intelligence**: NLP-based risk scoring and fund classification from prospectus PDFs — auto-generates fund summaries for distributor pitches

---

## 4. Data Quality Assessment

### Missing Values Analysis
| Dataset | Missing Values | Details |
|---|---|---|
| AMFI Scheme List (14,275 rows) | **0** | All 7 columns are fully populated |
| NIFTY 50 Index (745 rows) | **0** | All 7 columns are fully populated |
| MF NAV History (7,346,444 rows) | **Minimal** | Some older funds have shorter histories; no NULL values within files |
| World Bank Indicators (22 rows) | **2** | Real_Interest_Rate_Pct missing for 2023 and 2024 (not yet published) |
| Social Sentiment (5,842 rows) | **0** | Fully labeled with Sentence and Sentiment |
| Clickstream Events (2.75M rows) | **Partial** | `transactionid` is NULL for non-purchase events (by design) |

### Duplicate Records
A comprehensive MD5-hash-based duplicate scan across all 30 GB identified:
- **43 duplicate groups** totaling only **2.97 MB** of redundant data
- **40 groups** in MF NAV history: Different scheme codes (e.g., Growth vs. Dividend payout options) sharing identical NAV trajectories
- **3 groups** in NSE EOD 1990 data: Company name changes on NSE (e.g., JSL ↔ JSLHISAR)
- These are **legitimate data entries** representing real market relationships, not data collection errors

### Outliers
- Stock price data may contain circuit-limit hits (upper/lower price bands) which appear as extreme daily moves — these are real market events, not data errors
- Macro indicators: COVID-19 period (2020) shows GDP Growth of −5.78% — a genuine outlier reflecting the pandemic

### Noise / Inconsistencies
- Date formats vary across datasets: YYYY-MM-DD (NAV history), DD-Mon-YYYY (NSE API), Unix timestamps (clickstream)
- Currency units: Stock prices in INR, turnover in ₹ Crores — standardization required during preprocessing

### Data Imbalance
- Social Sentiment: Distribution across positive/negative/neutral may be unbalanced — to be analyzed during EDA
- Clickstream events: "view" events heavily outweigh "transaction" events (typical e-commerce funnel)

### Bias Considerations
- Geographic bias: All financial data is India-centric (NSE/BSE listed companies, Indian mutual funds)
- Temporal bias: Intraday data is denser for recent years; some older funds have limited history
- The financial news corpus is English-language only, excluding regional language financial coverage

---

## 5. Data Preprocessing Details

### Data Cleaning Techniques Used
- **Duplicate detection**: MD5-hash-based file content comparison across 30 GB; identified 2.97 MB of content-identical files
- **Irrelevant data removal**: Deleted 678 synthetically generated candlestick images and 50 RBI regulation PDFs that were not relevant to mutual fund analytics
- **File format standardization**: All structured data normalized to CSV format with consistent headers

### Handling Missing Values
- World Bank indicators: 2 missing values (Real_Interest_Rate_Pct for 2023–2024) to be handled via forward-fill interpolation
- Clickstream `transactionid`: NULL values are structurally meaningful (non-purchase events) — no imputation needed
- MF NAV gaps (non-trading days): No interpolation applied; gaps represent weekends and market holidays

### Encoding Techniques
- Categorical encoding for fund categories (Equity, Debt, Hybrid, etc.) — planned using Label Encoding and One-Hot Encoding
- Sentiment labels (positive/negative/neutral) — Label Encoding for model input

### Feature Scaling / Normalization
- NAV values: Min-Max scaling applied for LSTM input (values normalized to [0, 1])
- Stock OHLCV data: StandardScaler (Z-score normalization) planned for ML feature engineering
- Macro indicators: Already in percentage format; additional scaling during model training

### Transformation Steps
- Date parsing: All dates converted to `datetime` objects and standardized to YYYY-MM-DD format
- Text extraction: PyMuPDF used to extract raw text from 33 AMC PDF documents
- News article parsing: Raw JSON files contain metadata; TXT files contain cleaned article body text
- NAV history: Raw AMFI text file parsed into individual per-scheme CSV files

---

## 6. Exploratory Data Analysis (EDA)

### Summary Statistics
| Metric | Value |
|---|---|
| Total dataset size | 30 GB |
| Total files | 315,538 |
| Total structured records | 13,713,558+ rows (NAV: 7,346,444 + EOD: 6,367,114) excluding intraday |
| Unique mutual fund schemes | 14,275 |
| Unique stocks (EOD) | 2,235 |
| NIFTY 500 intraday stocks | 499 |
| Financial news articles | 306,247 |
| Clickstream events | 2,756,101 |
| Date range coverage | 1990 – 2026 |

### Visualization Techniques Used (Planned)
- **Time-series plots**: NAV trends, NIFTY 50 index movement, intraday price patterns
- **Heatmaps**: Correlation matrices for stock prices, feature importance
- **Bar charts**: Fund category distribution, sentiment class distribution
- **Candlestick charts**: OHLCV data visualization using mplfinance
- **Word clouds**: Most frequent terms in financial news corpus
- **Box plots**: Outlier detection in stock returns and NAV changes

### Correlation Analysis (Planned)
- Inter-stock correlation analysis using EOD closing prices
- NAV correlation with NIFTY 50 index movement
- Sentiment score correlation with next-day market returns
- Macro indicator correlation with fund category performance

### Key Observations
- The AMFI scheme list contains 14,275 unique schemes across Equity, Debt, Hybrid, and Solution-Oriented categories
- NIFTY 50 index ranged from 17,000 to 26,000 during 2023–2026, showing significant market growth
- The NIFTY 500 intraday dataset alone is 19 GB — truly Big Data scale requiring distributed processing
- Clickstream data shows classic e-commerce funnel patterns (many views → few purchases)

---

## 7. Tools and Technologies Used

| Category | Tools |
|---|---|
| **Programming Languages** | Python 3.13 |
| **Data Processing Libraries** | Pandas, NumPy, PySpark (planned) |
| **Web Scraping** | Requests, BeautifulSoup4, Feedparser |
| **PDF Processing** | PyMuPDF (fitz) |
| **Machine Learning** | Scikit-learn, PyTorch |
| **Data Visualization** | Matplotlib, mplfinance (planned: Seaborn, Plotly) |
| **NLP** | NLTK / spaCy (planned) |
| **Data APIs** | mftool (AMFI), yfinance (NSE), World Bank API |
| **Dataset Management** | Kaggle CLI |
| **Version Control** | Git |
| **Big Data Tools (Planned)** | Apache Spark (PySpark), Hadoop HDFS |
| **Cloud (Planned)** | AWS / Google Cloud for distributed processing |

---

## 8. Ethical and Legal Considerations

| Aspect | Details |
|---|---|
| **Personal Data Involved** | No PII (Personally Identifiable Information). Clickstream data uses anonymized `visitorid`; no names, emails, or addresses. Sentiment data uses anonymized tweet text. |
| **Consent Obtained** | All data is publicly available. Kaggle datasets are shared under open licenses by their creators. AMFI/NSE data is publicly published for transparency. |
| **Anonymization Techniques** | Not needed — no personal data is collected. Clickstream visitor IDs are already anonymized at source. |
| **Compliance with Data Protection Norms** | Compliant with India's Digital Personal Data Protection Act, 2023 — no personal data is processed. All data is aggregated market data or anonymized behavioral data. |
| **Sensitive Attributes** | None. The dataset contains only financial market data (prices, NAVs, volumes) and anonymized behavioral data. No attributes related to race, gender, religion, caste, or any protected characteristics. |
| **Fair Use** | All data is used strictly for educational and academic research purposes as part of a university project. Kaggle datasets are used under their respective Creative Commons licenses (CC-BY-NC-SA-4.0, CC0). |

---

## 9. Data Storage and Management

| Aspect | Details |
|---|---|
| **Storage Type** | Local filesystem (development) + Cloud (planned for deployment) |
| **Storage Platform** | macOS local SSD (current); AWS S3 / Google Cloud Storage (planned) |
| **Data Format Stored** | CSV, JSON, TXT, PDF, PNG, JPG, WebP, SVG |
| **Total Storage Used** | 30 GB |

### Data Organization Structure

```
datasets/                           (30 GB)
├── structured/                     (26 GB)   — Tabular CSV data
│   ├── stock_prices/                         — 6 subdirectories
│   │   ├── nifty500_intraday/               (19 GB, 499 files)
│   │   ├── stock_market_india/              (5.8 GB, 162 files)
│   │   ├── eod_bhav_copies/                 (529 MB, 2,235 files)
│   │   ├── nse_eod_1990/                    (512 MB, 3,189 files)
│   │   ├── nse_intraday_minute/             (239 MB, 26 files)
│   │   └── nifty50_index/                   (48 KB, 1 file)
│   ├── mutual_funds/
│   │   ├── amfi_scheme_list.csv             (1.9 MB, 14,275 rows)
│   │   └── nav_history/                     (151 MB, 2,965 files)
│   ├── leads/                               — Lead recommendation data
│   │   ├── mf_investor_behavior/            (73 KB, 1,000 investors)
│   │   ├── lead_scoring/                    (2.3 MB, 9,240 leads)
│   │   ├── insurance_leads/                 (572 KB, 5,000 leads)
│   │   └── lead_score_upgrad/               (22.8 MB, case study)
│   └── macroeconomic/
│       └── world_bank_indicators.csv        (1.2 KB, 22 rows)
│
├── semi_structured/                (949 MB)  — JSON, RSS, Clickstream
│   ├── clickstream/                         (942 MB, 4 files)
│   ├── financial_news_rss/                  (144 KB, 35 files)
│   ├── social_sentiment/                    (728 KB, 1 file)
│   └── user_behaviors/                      (6.4 MB, 1 file)
│
├── unstructured/                   (2.5 GB)  — PDFs, Images, Raw Text
│   ├── financial_news_corpus/               (2.5 GB, 306,247 files)
│   ├── prospectus_pdfs/                     (45 MB, 33 PDFs)
│   ├── prospectus_texts/                    (2.9 MB, 34 TXT files)
│   └── fund_images/                         (4.5 MB, 84 images)
│
├── velocity/                       (1.6 MB)  — Real-time Streaming Data
│   ├── amfi_nav_snapshots/                  (NAV feed snapshots)
│   ├── nse_market_pulse.csv                 (NIFTY 50 tick data)
│   └── live_news_stream.csv                 (Live headline stream)
│
└── models/                         (176 KB)  — ML Artifacts
    ├── ml_features/
    ├── ml_scalers/
    └── saved_models/
```

| Aspect | Details |
|---|---|
| **Backup and Recovery Strategy** | Git version control for scripts and configuration; dataset files tracked via `.gitignore` with manual backup to external storage |
| **Security Measures** | Local SSD with macOS FileVault encryption; Kaggle API key stored in `~/.kaggle/kaggle.json` with 600 permissions; No sensitive data requiring additional encryption |
| **Scalability Considerations** | Folder structure supports horizontal scaling — new data sources can be added under `structured/`, `semi_structured/`, or `unstructured/` without restructuring. Planned migration to HDFS/S3 for distributed storage when processing with PySpark/Hadoop. |
| **Ingestion Method** | Python scripts for automated data collection: `fetch_amfi_nse.py` (AMFI/NSE), `fetch_mf_history.py` (MF NAV histories via mftool), `stream_velocity_data.py` (real-time streaming), `kaggle_dataset_downloader.py` (Kaggle CLI), `fetch_prospectus_pdfs.py` (web scraping AMC sites), `fetch_fund_images.py` (image scraping) |

---

> **Note:** Sections 10 (Problem Definition) and 11 (Data Splitting Strategy) will be completed after discussion with Prof. Jaideep sir.
