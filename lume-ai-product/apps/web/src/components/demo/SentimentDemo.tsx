"use client";

import { useState } from "react";
import { api } from "@/lib/api";
import { MessageSquare, AlertCircle, RefreshCw, Sparkles, CheckCircle2 } from "lucide-react";
import LoadingSkeleton from "@/components/LoadingSkeleton";
import Disclaimer from "@/components/Disclaimer";

const PRESETS = [
  "Markets rally on strong GDP print and easing inflation fears",
  "Federal Reserve signals aggressive rate hikes to curb rising inflation",
  "Nippon Small Cap Fund records highest monthly inflows in a year",
  "Consolidated earnings fall short of expectations, raising margin concerns",
];

export default function SentimentDemo() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleReset = () => {
    setText("");
    setResult(null);
    setError(null);
  };

  const handlePreset = (preset: string) => {
    setText(preset);
    setResult(null);
    setError(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.predict("sentiment", { text: text.trim() });
      setResult(res);
    } catch (err: any) {
      setError(err.message || "Failed to classify sentiment.");
    } finally {
      setLoading(false);
    }
  };

  const getSentimentStyles = (sent: string) => {
    switch (sent.toLowerCase()) {
      case "positive":
        return { color: "text-[var(--accent-green)]", bg: "bg-[var(--accent-green)]", border: "border-green-500/20", text: "Positive/Bullish" };
      case "negative":
        return { color: "text-[var(--accent-red)]", bg: "bg-[var(--accent-red)]", border: "border-red-500/20", text: "Negative/Bearish" };
      default:
        return { color: "text-[var(--accent-amber)]", bg: "bg-[var(--accent-amber)]", border: "border-amber-500/20", text: "Neutral" };
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[var(--accent-mint-dim)] text-[var(--accent-mint)]">
          <MessageSquare className="h-5 w-5" />
        </div>
        <div>
          <h2 className="text-xl font-bold text-slate-100">Market News Sentiment</h2>
          <p className="text-slate-400 text-xs mt-0.5">Extract sentiment dynamics (positive, negative, neutral) from financial text.</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          {/* Preset Buttons */}
          <div className="glass-card p-4 border-[var(--border-subtle)] space-y-3">
            <span className="text-xs font-semibold text-slate-450 flex items-center gap-1.5">
              <Sparkles className="h-3.5 w-3.5 text-[var(--accent-mint)]" /> Financial News Presets
            </span>
            <div className="flex flex-col gap-2">
              {PRESETS.map((preset) => (
                <button
                  key={preset}
                  type="button"
                  onClick={() => handlePreset(preset)}
                  className="px-3 py-2 rounded-lg text-xs text-left font-medium bg-slate-800 hover:bg-slate-750 text-slate-300 hover:text-slate-200 transition-all border border-slate-700/50 truncate w-full"
                >
                  {preset}
                </button>
              ))}
            </div>
          </div>

          <form onSubmit={handleSubmit} className="glass-card p-6 border-[var(--border-subtle)] space-y-4">
            <div className="space-y-2">
              <div className="flex justify-between items-center text-xs font-medium text-slate-400">
                <label>Financial News Headline or Sentiment Prompt</label>
                <span className="font-mono text-slate-500">{text.length}/500</span>
              </div>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value.slice(0, 500))}
                className="input-field min-h-[120px] py-3 leading-relaxed resize-none"
                placeholder="Type market commentary or headlines here..."
                required
              />
            </div>

            <div className="flex items-center gap-3 pt-2">
              <button type="submit" disabled={loading || !text.trim()} className="btn-primary flex-1 sm:flex-initial">
                {loading ? "Analyzing Headline..." : "Analyze Sentiment"}
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
                  <AlertCircle className="h-4 w-4" /> Inference Failure
                </div>
                <p className="text-xs leading-relaxed text-slate-400">{error}</p>
              </div>
            )}

            {!loading && !result && !error && (
              <div className="text-center space-y-3 py-12">
                <MessageSquare className="h-10 w-10 text-slate-650 mx-auto" />
                <div className="space-y-1">
                  <h3 className="text-sm font-bold text-slate-350">Awaiting Sentiment</h3>
                  <p className="text-slate-500 text-xs max-w-xs mx-auto leading-relaxed">
                    Type a news headline or click a quick preset to run sentiment analysis.
                  </p>
                </div>
              </div>
            )}

            {!loading && result && (
              <div className="space-y-6">
                <div className="space-y-1">
                  <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider">Classification</h3>
                  <div className="flex items-baseline gap-2">
                    <span className={`text-2xl font-extrabold tracking-tight ${getSentimentStyles(result.prediction.sentiment).color}`}>
                      {getSentimentStyles(result.prediction.sentiment).text}
                    </span>
                    {result.confidence !== null && (
                      <span className="text-xs font-semibold text-slate-450">
                        ({(result.confidence * 100).toFixed(1)}% confidence)
                      </span>
                    )}
                  </div>
                </div>

                {result.confidence !== null && (
                  <div className="space-y-2">
                    <div className="flex justify-between items-center text-[10px] font-semibold text-slate-500 uppercase tracking-wider">
                      <span>Model Confidence</span>
                    </div>
                    <div className="confidence-bar">
                      <div
                        className={`confidence-bar-fill ${getSentimentStyles(result.prediction.sentiment).bg}`}
                        style={{ width: `${result.confidence * 100}%` }}
                      />
                    </div>
                  </div>
                )}

                <div className="space-y-2 border-t border-[var(--border-subtle)] pt-4">
                  <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-300">
                    <CheckCircle2 className="h-4 w-4 text-[var(--accent-mint)]" /> Natural Language Explanation
                  </div>
                  <p className="text-xs leading-relaxed text-slate-450">{result.explanation}</p>
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
