import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# Suppress minor warnings for clean CLI output
warnings.filterwarnings('ignore')

class MutualFundSearchEngine:
    def __init__(self, amfi_csv_path):
        self.csv_path = amfi_csv_path
        self.df = None
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        self.tfidf_matrix = None
        
        self._load_and_vectorize()

    def _load_and_vectorize(self):
        print(f"Loading Mutual Fund dataset from {self.csv_path}...")
        try:
            self.df = pd.read_csv(self.csv_path)
            # Combine Scheme Name and Category to enrich the semantic search context
            self.df['search_corpus'] = self.df['Scheme_Name'].astype(str) + " " + self.df['Category'].astype(str)
            
            print("Vectorizing 14,000+ mutual funds using TF-IDF...")
            self.tfidf_matrix = self.vectorizer.fit_transform(self.df['search_corpus'])
            print("Search Engine Ready.\n")
        except Exception as e:
            print(f"Error initializing search engine: {e}")

    def query(self, search_text, top_k=5):
        """
        Executes a natural language query against the TF-IDF matrix using Cosine Similarity.
        """
        if self.df is None:
            return "Engine not initialized properly."
            
        # Convert user query to vector
        query_vec = self.vectorizer.transform([search_text])
        
        # Calculate Cosine Similarity between query vector and all fund vectors
        cosine_sim = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get the indices of the Top K most similar funds
        top_indices = cosine_sim.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            score = cosine_sim[idx]
            if score > 0.0:  # Only return relevant results
                fund = self.df.iloc[idx]
                results.append({
                    'Scheme_Code': fund['Scheme_Code'],
                    'Scheme_Name': fund['Scheme_Name'],
                    'Category': fund['Category'],
                    'Match_Score': round(score * 100, 2)
                })
                
        return results

if __name__ == "__main__":
    # Test execution for the Semantic Search
    AMFI_PATH = '../datasets/structured/mutual_funds/amfi_scheme_list.csv'
    
    engine = MutualFundSearchEngine(AMFI_PATH)
    
    test_queries = [
        "aggressive equity growth fund",
        "low risk liquid debt",
        "hdfc hybrid"
    ]
    
    for q in test_queries:
        print(f"--- QUERY: '{q}' ---")
        matches = engine.query(q, top_k=3)
        if matches:
            for match in matches:
                print(f"[{match['Match_Score']}%] {match['Scheme_Name']} ({match['Category']})")
        else:
            print("No matches found.")
        print("")
