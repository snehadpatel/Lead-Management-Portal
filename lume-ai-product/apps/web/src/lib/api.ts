export interface PredictionResponse {
  task: string;
  timestamp: string;
  prediction: any;
  confidence?: number;
  explanation: string;
  disclaimer: string;
  model_version: string;
}

export interface SearchResult {
  scheme_code: string;
  scheme_name: string;
  category: string;
  match_score: number;
}

export interface SearchResponse {
  timestamp: string;
  query: string;
  results: SearchResult[];
  disclaimer: string;
}

export interface ForecastData {
  days: number[];
  actual: number[];
  predicted: number[];
  label_actual: string;
  label_predicted: string;
  disclaimer: string;
  timestamp: string;
}

export interface MetricsData {
  timestamp: string;
  random_forest?: any;
  classification_report?: any;
  confusion_matrix?: any;
  kmeans?: any;
  kmeans_pca?: any;
  sentiment?: any;
  lstm?: any;
  lstm_holdout?: any;
  manifest?: any;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorDetail = 'An error occurred while communicating with the server.';
    try {
      const errorJson = await response.json();
      errorDetail = errorJson.detail || errorJson.error || errorDetail;
    } catch {
      // ignore
    }
    throw new Error(errorDetail);
  }
  return response.json() as Promise<T>;
}

export const api = {
  async predict(task: string, payload: { lead?: any; investor_behavior?: any; text?: string }): Promise<PredictionResponse> {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ task, ...payload }),
    });
    return handleResponse<PredictionResponse>(response);
  },

  async search(query: string, topK: number = 5): Promise<SearchResponse> {
    const response = await fetch(`${API_BASE_URL}/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, top_k: topK }),
    });
    return handleResponse<SearchResponse>(response);
  },

  async getForecastDemo(): Promise<ForecastData> {
    const response = await fetch(`${API_BASE_URL}/forecast/demo`);
    return handleResponse<ForecastData>(response);
  },

  async getMetrics(): Promise<MetricsData> {
    const response = await fetch(`${API_BASE_URL}/metrics`);
    return handleResponse<MetricsData>(response);
  },

  async checkHealth(): Promise<{ status: string; models_loaded: Record<string, boolean> }> {
    const response = await fetch(`${API_BASE_URL}/health`);
    return handleResponse(response);
  }
};
