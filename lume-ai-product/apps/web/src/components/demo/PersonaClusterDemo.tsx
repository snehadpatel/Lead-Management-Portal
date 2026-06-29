"use client";

import { useState } from "react";
import { api } from "@/lib/api";
import { Users, AlertCircle, RefreshCw, Sparkles, CheckCircle2 } from "lucide-react";
import LoadingSkeleton from "@/components/LoadingSkeleton";
import Disclaimer from "@/components/Disclaimer";

const PRESETS = [
  {
    name: "Safe/Conservative",
    ProfManage: 8.0,
    Diversification: 9.0,
    Affordability: 7.0,
    Liquidity: 9.0,
    Growth: 2.0,
    Trustworthiness: 9.5,
    Technology: 4.0,
  },
  {
    name: "High Growth Equity",
    ProfManage: 4.0,
    Diversification: 3.0,
    Affordability: 8.0,
    Liquidity: 4.0,
    Growth: 9.5,
    Trustworthiness: 6.0,
    Technology: 8.5,
  },
  {
    name: "Balanced & Diversified",
    ProfManage: 7.0,
    Diversification: 8.0,
    Affordability: 6.0,
    Liquidity: 6.0,
    Growth: 7.0,
    Trustworthiness: 8.0,
    Technology: 7.0,
  },
];

const DEFAULT_BEHAVIOR = {
  ProfManage: 5.0,
  Diversification: 5.0,
  Affordability: 5.0,
  Liquidity: 5.0,
  Growth: 5.0,
  Trustworthiness: 5.0,
  Technology: 5.0,
};

export default function PersonaClusterDemo() {
  const [behavior, setBehavior] = useState(DEFAULT_BEHAVIOR);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSliderChange = (name: string, val: number) => {
    setBehavior((prev) => ({
      ...prev,
      [name]: val,
    }));
  };

  const loadPreset = (preset: typeof PRESETS[0]) => {
    const { name, ...data } = preset;
    setBehavior(data);
    setResult(null);
    setError(null);
  };

  const handleReset = () => {
    setBehavior(DEFAULT_BEHAVIOR);
    setResult(null);
    setError(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await api.predict("investor_cluster", { investor_behavior: behavior });
      setResult(res);
    } catch (err: any) {
      setError(err.message || "Failed to classify investor cluster.");
    } finally {
      setLoading(false);
    }
  };

  const sliders = [
    { name: "ProfManage", label: "Professional Management Needs" },
    { name: "Diversification", label: "Diversification Importance" },
    { name: "Affordability", label: "Affordability Preference" },
    { name: "Liquidity", label: "Liquidity Needs" },
    { name: "Growth", label: "Growth Appetite" },
    { name: "Trustworthiness", label: "Trustworthiness Importance" },
    { name: "Technology", label: "Technology Preference" },
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[var(--accent-mint-dim)] text-[var(--accent-mint)]">
          <Users className="h-5 w-5" />
        </div>
        <div>
          <h2 className="text-xl font-bold text-slate-100">Investor Persona Clustering</h2>
          <p className="text-slate-400 text-xs mt-0.5">Map psycho-graphic client traits into cluster personas using a K-Means model.</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          {/* Preset Buttons */}
          <div className="glass-card p-4 border-[var(--border-subtle)] space-y-3">
            <span className="text-xs font-semibold text-slate-450 flex items-center gap-1.5">
              <Sparkles className="h-3.5 w-3.5 text-[var(--accent-mint)]" /> Quick Presets
            </span>
            <div className="flex flex-wrap gap-2">
              {PRESETS.map((preset) => (
                <button
                  key={preset.name}
                  type="button"
                  onClick={() => loadPreset(preset)}
                  className="px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-750 text-slate-350 hover:text-slate-200 transition-all border border-slate-700/50"
                >
                  {preset.name}
                </button>
              ))}
            </div>
          </div>

          <form onSubmit={handleSubmit} className="glass-card p-6 border-[var(--border-subtle)] space-y-6">
            <div className="space-y-4">
              {sliders.map((s) => (
                <div key={s.name} className="space-y-2">
                  <div className="flex justify-between items-center text-xs font-medium text-slate-350">
                    <span>{s.label}</span>
                    <span className="font-mono bg-slate-800 px-1.5 py-0.5 rounded border border-slate-700/40 text-[var(--accent-mint)] font-bold">
                      {behavior[s.name as keyof typeof behavior].toFixed(1)}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="10"
                    step="0.5"
                    value={behavior[s.name as keyof typeof behavior]}
                    onChange={(e) => handleSliderChange(s.name, Number(e.target.value))}
                    className="w-full h-1.5 rounded-lg appearance-none bg-slate-800 accent-[var(--accent-mint)] cursor-pointer"
                  />
                </div>
              ))}
            </div>

            <div className="flex items-center gap-3 pt-2 border-t border-[var(--border-subtle)]">
              <button type="submit" disabled={loading} className="btn-primary flex-1 sm:flex-initial">
                {loading ? "Assigning Cluster..." : "Segment Investor"}
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
                <Users className="h-10 w-10 text-slate-650 mx-auto" />
                <div className="space-y-1">
                  <h3 className="text-sm font-bold text-slate-350">Awaiting Segment</h3>
                  <p className="text-slate-500 text-xs max-w-xs mx-auto leading-relaxed">
                    Set psycho-graphic slider values or select a preset to compute cluster placement.
                  </p>
                </div>
              </div>
            )}

            {!loading && result && (
              <div className="space-y-6">
                <div className="space-y-1">
                  <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider">Assigned Segment</h3>
                  <div className="flex items-baseline gap-2">
                    <span
                      className="text-2xl font-extrabold tracking-tight"
                      style={{ color: result.prediction.color || "var(--accent-mint)" }}
                    >
                      {result.prediction.persona}
                    </span>
                    <span className="text-xs font-semibold text-slate-400">
                      (Cluster #{result.prediction.cluster_id})
                    </span>
                  </div>
                </div>

                <div className="p-4 rounded-xl bg-slate-900/40 border border-[var(--border-subtle)]">
                  <p className="text-xs leading-relaxed text-slate-300">{result.prediction.description}</p>
                </div>

                <div className="space-y-2 border-t border-[var(--border-subtle)] pt-4">
                  <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-300">
                    <CheckCircle2 className="h-4 w-4 text-[var(--accent-mint)]" /> Natural Language Explanation
                  </div>
                  <p className="text-xs leading-relaxed text-slate-400">{result.explanation}</p>
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
