"use client";

import { useState } from "react";
import { api, SearchResult } from "@/lib/api";
import { Search, AlertCircle, RefreshCw, Sparkles, Database } from "lucide-react";
import LoadingSkeleton from "@/components/LoadingSkeleton";
import Disclaimer from "@/components/Disclaimer";

const PRESETS = [
  "Aggressive growth equity schemes",
  "High safety liquid debt options",
  "Balanced tax saver funds",
  "Passive index trackers",
];

export default function SemanticSearchDemo() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<SearchResult[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleReset = () => {
    setQuery("");
    setResults(null);
    setError(null);
  };

  const handlePreset = (preset: string) => {
    setQuery(preset);
    setResults(null);
    setError(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.search(query.trim());
      setResults(res.results);
    } catch (err: any) {
      setError(err.message || "Failed to search mutual funds.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[var(--accent-mint-dim)] text-[var(--accent-mint)]">
          <Search className="h-5 w-5" />
        </div>
        <div>
          <h2 className="text-xl font-bold text-slate-100">Semantic Fund Search</h2>
          <p className="text-slate-400 text-xs mt-0.5">Lookup schemes by typing investment concepts rather than strict tickers/codes.</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          {/* Preset Buttons */}
          <div className="glass-card p-4 border-[var(--border-subtle)] space-y-3">
            <span className="text-xs font-semibold text-slate-455 flex items-center gap-1.5">
              <Sparkles className="h-3.5 w-3.5 text-[var(--accent-mint)]" /> Concept Search Presets
            </span>
            <div className="flex flex-wrap gap-2">
              {PRESETS.map((preset) => (
                <button
                  key={preset}
                  type="button"
                  onClick={() => handlePreset(preset)}
                  className="px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-750 text-slate-300 hover:text-slate-200 transition-all border border-slate-700/50"
                >
                  {preset}
                </button>
              ))}
            </div>
          </div>

          <form onSubmit={handleSubmit} className="glass-card p-6 border-[var(--border-subtle)] space-y-4">
            <div className="space-y-2">
              <div className="flex justify-between items-center text-xs font-medium text-slate-400">
                <label>Semantic Search Term</label>
                <span className="font-mono text-slate-500">{query.length}/200</span>
              </div>
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value.slice(0, 200))}
                className="input-field"
                placeholder="Type e.g., low-volatility inflation hedge or capital preservation..."
                required
              />
            </div>

            <div className="flex items-center gap-3 pt-2">
              <button type="submit" disabled={loading || !query.trim()} className="btn-primary flex-1 sm:flex-initial">
                {loading ? "Matching Embeddings..." : "Execute Search"}
              </button>
              <button type="button" onClick={handleReset} className="btn-secondary flex items-center justify-center p-2.5">
                <RefreshCw className="h-4 w-4" />
              </button>
            </div>
          </form>
        </div>

        <div className="flex flex-col gap-6">
          <div className="glass-card p-6 border-[var(--border-subtle)] bg-slate-900/10 flex-1 flex flex-col justify-center">
            {loading && <LoadingSkeleton />}

            {error && (
              <div className="p-4 rounded-xl border border-red-500/20 bg-red-500/5 text-slate-200 space-y-2">
                <div className="flex items-center gap-2 text-red-500 font-bold text-sm">
                  <AlertCircle className="h-4 w-4" /> Search Index Unavailable
                </div>
                <p className="text-xs leading-relaxed text-slate-400">
                  {error.includes("fund_embeddings.pkl") 
                    ? "The semantic index is currently offline. Ensure embeddings are built."
                    : error}
                </p>
              </div>
            )}

            {!loading && !results && !error && (
              <div className="text-center space-y-3 py-12">
                <Database className="h-10 w-10 text-slate-650 mx-auto" />
                <div className="space-y-1">
                  <h3 className="text-sm font-bold text-slate-350">Awaiting Query</h3>
                  <p className="text-slate-500 text-xs max-w-xs mx-auto leading-relaxed">
                    Submit a concepts phrase above to match it mathematically against the SBERT index.
                  </p>
                </div>
              </div>
            )}

            {!loading && results && (
              <div className="space-y-6">
                <div className="space-y-1">
                  <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider">Top Matches (Cosine Similarity)</h3>
                  <span className="text-xs text-slate-400">Showing top 5 matching schemes</span>
                </div>

                <div className="space-y-4">
                  {results.map((r, i) => (
                    <div key={r.scheme_code} className="space-y-1.5 p-3 rounded-lg bg-slate-900/30 border border-[var(--border-subtle)]">
                      <div className="flex justify-between items-start gap-2">
                        <div className="min-w-0">
                          <h4 className="text-xs font-bold text-slate-200 truncate">{r.scheme_name}</h4>
                          <span className="text-[10px] text-slate-400 font-medium">{r.category}</span>
                        </div>
                        <span className="text-[10px] font-bold font-mono text-[var(--accent-mint)] bg-[var(--accent-mint-dim)] px-1.5 py-0.5 rounded">
                          {(r.match_score * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="confidence-bar">
                        <div
                          className="confidence-bar-fill bg-[var(--accent-blue)]"
                          style={{ width: `${r.match_score * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>

                <Disclaimer />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
