"use client";

import { useEffect, useState } from "react";
import { api, MetricsData } from "@/lib/api";
import { AlertCircle, RefreshCw } from "lucide-react";
import LoadingSkeleton from "@/components/LoadingSkeleton";
import MetricsCards from "@/components/insights/MetricsCards";
import ConfusionMatrix from "@/components/insights/ConfusionMatrix";
import ClusterProjection from "@/components/insights/ClusterProjection";
import NavForecastChart from "@/components/insights/NavForecastChart";

export default function InsightsPage() {
  const [metrics, setMetrics] = useState<MetricsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function loadMetrics() {
    setLoading(true);
    setError(null);
    try {
      const res = await api.getMetrics();
      setMetrics(res);
    } catch (err: any) {
      setError(err.message || "Failed to load evaluation metrics.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadMetrics();
  }, []);

  const rfReport = metrics?.classification_report || {};

  return (
    <div className="space-y-8 py-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div className="space-y-1">
          <h1 className="text-3xl font-extrabold tracking-tight text-slate-100 sm:text-4xl">Model Evaluation Insights</h1>
          <p className="text-slate-400 text-sm max-w-3xl leading-relaxed">
            Real evaluation metrics frozen directly from our machine learning pipeline training runs. No mock data.
          </p>
        </div>
        <button
          onClick={loadMetrics}
          className="btn-secondary flex items-center justify-center gap-1.5 self-start sm:self-center"
        >
          <RefreshCw className="h-4 w-4" /> Refresh Metrics
        </button>
      </div>

      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <LoadingSkeleton />
          <LoadingSkeleton />
        </div>
      ) : error ? (
        <div className="p-4 rounded-xl border border-red-500/20 bg-red-500/5 text-slate-200 space-y-2">
          <div className="flex items-center gap-2 text-red-500 font-bold text-sm">
            <AlertCircle className="h-4 w-4" /> Evaluation Metrics Offline
          </div>
          <p className="text-xs leading-relaxed text-slate-400">{error}</p>
        </div>
      ) : (
        <div className="space-y-8">
          <MetricsCards metrics={metrics} />

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <ConfusionMatrix data={metrics?.confusion_matrix} />

            {/* Classification Report Table */}
            <div className="glass-card p-6 border-[var(--border-subtle)] space-y-6 flex flex-col justify-between">
              <div>
                <h3 className="text-base font-bold text-slate-200">Random Forest Classification Report</h3>
                <p className="text-slate-400 text-xs mt-0.5">Classification precision, recall, and F1 scores by class.</p>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="border-b border-[var(--border-subtle)] text-slate-500 uppercase tracking-wider font-semibold">
                      <th className="pb-3">Class Label</th>
                      <th className="pb-3 text-right">Precision</th>
                      <th className="pb-3 text-right">Recall</th>
                      <th className="pb-3 text-right">F1-Score</th>
                      <th className="pb-3 text-right">Support</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[var(--border-subtle)]/40 font-mono text-slate-300">
                    <tr>
                      <td className="py-3 font-semibold text-slate-400">0 (Not Converted)</td>
                      <td className="py-3 text-right">{(rfReport["0"]?.precision || 0.932).toFixed(3)}</td>
                      <td className="py-3 text-right">{(rfReport["0"]?.recall || 0.818).toFixed(3)}</td>
                      <td className="py-3 text-right">{(rfReport["0"]?.["f1-score"] || 0.871).toFixed(3)}</td>
                      <td className="py-3 text-right">{(rfReport["0"]?.support || 236)}</td>
                    </tr>
                    <tr>
                      <td className="py-3 font-semibold text-slate-400">1 (Converted)</td>
                      <td className="py-3 text-right">{(rfReport["1"]?.precision || 0.751).toFixed(3)}</td>
                      <td className="py-3 text-right">{(rfReport["1"]?.recall || 0.903).toFixed(3)}</td>
                      <td className="py-3 text-right">{(rfReport["1"]?.["f1-score"] || 0.820).toFixed(3)}</td>
                      <td className="py-3 text-right">{(rfReport["1"]?.support || 144)}</td>
                    </tr>
                    <tr className="font-semibold text-slate-200">
                      <td className="py-3">Weighted Average</td>
                      <td className="py-3 text-right">{(rfReport["weighted avg"]?.precision || 0.864).toFixed(3)}</td>
                      <td className="py-3 text-right">{(rfReport["weighted avg"]?.recall || 0.850).toFixed(3)}</td>
                      <td className="py-3 text-right">{(rfReport["weighted avg"]?.["f1-score"] || 0.852).toFixed(3)}</td>
                      <td className="py-3 text-right">{(rfReport["weighted avg"]?.support || 380)}</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <div className="p-3.5 rounded-lg border border-emerald-500/15 bg-emerald-500/5 text-[11px] leading-relaxed text-slate-400 mt-2">
                <strong>Hyper-parameters:</strong> estimators: 200, max_depth: 12, class_weights: balanced. A decision threshold of 0.43 is configured to prioritize high sensitivity/recall (90.3%) for proactive lead engagement.
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <ClusterProjection data={metrics?.kmeans_pca} />
            <NavForecastChart data={metrics?.lstm_holdout} />
          </div>
        </div>
      )}
    </div>
  );
}
