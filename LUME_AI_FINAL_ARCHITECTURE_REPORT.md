# Lume AI: Machine Learning Architecture & Technical Defense

This document serves as the formal architectural defense for the April 17th Studio-Based Project (SBP) Evaluation. It outlines our data engineering pipeline, justifies the core algorithmic choices, unpacks competitive rejections, and establishes the path for future structural scaling.

---

## 1. Data Ecosystem (What data do we have?)
Our architecture ingests three completely distinct paradigms of data to build the Dual-Sided Platform:
1. **Supervised CRM Interaction Data (Kaggle Lead Scoring)**: 9,000+ rows possessing heavy categorical and numerical behavioral data (e.g., `Lead Origin`, `Time Spent on Website`, `Occupation`) flagged with a binary `Converted` matrix.
2. **Unsupervised Psychological Survey Data**: Raw behavioral variables mapping human risk tolerances (`Trustworthiness`, `Growth`, `Liquidity`).
3. **Sequential Financial Time-Series Data**: Historical `NSE NIFTY 50` closing prices tracking deep-time market volatility.
4. **Textual Datasets**: The complete `AMFI` Mutual fund repository containing 14,275 raw English string combinations.

---

## 2. Model Rationale & Rejections (Why these models?)

### Engine 1: Random Forest (Lead Conversion Classifier)
* **Why we used it:** Human behavioral data (like clicking websites vs filling out forms) is inherently non-linear and messy. Random Forest isolates these overlapping variables into thousands of independent Decision Trees, preventing massive overfitting. Crucially, it allows us to natively extract **Explainable AI (XAI)** mathematical weights.
* **Why we rejected Logistic Regression:** Logistic Regression assumes a perfect linear mathematical relationship between variables. It would violently fail on our chaotic Kaggle dataset and would aggressively misclassify leads.

### Engine 2: K-Means (Investor Persona Clustering)
* **Why we used it:** Our psychological survey mapping matrix had no `Target` variable (meaning it was unlabelled). K-Means allowed us to organically define exactly $k=4$ distinct geometrical boundaries based on Euclidean similarity, natively mapping human behavior into 4 pitchable Personas.
* **Why we rejected DBSCAN:** DBSCAN forcefully ejects outliers into a "Noise" cluster. In a finance app, we cannot classify an investor as "Noise" and ignore them; every user must be routed to a specific fund.

### Engine 3: PyTorch LSTM (NAV Forecaster)
* **Why we used it:** A Long Short-Term Memory neural network fundamentally understands "Time". It possesses internal states that 'remember' the volatility of previous continuous days while attempting to project the future.
* **Why we rejected ARIMA:** Traditional statistical ARIMA models expect stationary, predictable patterns. NIFTY 50 financial spikes are wildly unpredictable, meaning ARIMA would flatline, whereas our LSTM algorithm gracefully rides the lagging volatility.

### Engine 4: TF-IDF Cosine Engine (Semantic Search)
* **Why we used it:** Converts human vocabulary to geometrical matrices effortlessly. Operating a `14,000 x N` array using Cosine Match Drop-off generates millisecond latency while successfully bypassing exact-string-match typing failures.
* **Why we rejected NLP BERT (Transformers):** While powerful, BERT requires massive parallel GPU computation resources. Processing 14,275 Mutual Funds through a 500-layer transformer inside a Streamlit application loop would crash the memory heap.

---

## 3. Empirical Results & Mathematical Validation
Our algorithms natively dumped their Scikit-Learn/PyTorch mathematical coefficients onto the interface:
- **Lead Classification Benchmark:** By injecting One-Hot Encoded variables and applying mathematical class-balancing, our Random Forest officially validated an **81.01% Accuracy Score** inside chaotic Human Interaction data (F1-Score: 0.82). 
- **Explainable AI Matrix:** We successfully generated XAI arrays proving that behavioral metrics fundamentally dictate the Mathematical Node decision outputs.
- **LSTM Progression Vector:** Established a powerful trajectory curve proving the network captures Deep-Time variances (mapping tightly with historical ~25,000 INR price thresholds).
- **K-Means Purity:** Generated Silhouette clusters mathematically separating boundaries across a 2-Dimensional PCA mapping grid.

---

## 4. Future Scalability (How can we improve?)
While the Data Science core is functionally complete, future deployment scaleings involve:
1. **Pipeline Decoupling (FastAPI):** Detaching the heavy Scikit-Learn `.pkl` binaries out of the Streamlit loop and hosting them on a dynamic async backend (e.g., Python FastAPI).
2. **Kafka Streaming (Real-Time Ingestion):** Utilizing extreme data-engineering to pipe real `NSE NIFTY` stock JSON loads dynamically from Indian Brokers directly into the PyTorch LSTM tensor weights. 
3. **Advanced NLP Shift:** Assuming we secure AWS GPU funding, upgrading the TF-IDF integer math completely into multi-dimensional Semantic LLM embeddings (like LangChain vectorstores) for conversational fund retrieval.
