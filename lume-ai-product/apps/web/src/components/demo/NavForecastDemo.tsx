"use client";

import { useEffect, useState } from "react";
import { api, ForecastData } from "@/lib/api";
import { LineChart as ChartIcon, AlertCircle, TrendingUp, Info } from "lucide-react";
import LoadingSkeleton from "@/components/LoadingSkeleton";
import Disclaimer from "@/components/Disclaimer";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, Legend, CartesianGrid } from "recharts";

export default function NavForecastDemo() {
  const [data, setData] = useState<ForecastData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadData() {
      try {
        const res = await api.getForecastDemo();
        setData(res);
      } catch (err: any) {
        setError(err.message || "Failed to load NAV forecast data.");
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  const chartData = data
    ? data.days.map((day, idx) => ({
        day: `Day ${day + 1}`,
        Actual: data.actual[idx],
        Predicted: data.predicted[idx],
      }))
    : [];

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[var(--accent-mint-dim)] text-[var(--accent-mint)]">
          <ChartIcon className="h-5 w-5" />
        </div>
        <div>
          <h2 className="text-xl font-bold text-slate-100">NAV Time-Series Forecast</h2>
          <p className="text-slate-400 text-xs mt-0.5">Explore the holdout set performance validating Lume AI's deep learning LSTM predictor.</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 glass-card p-6 border-[var(--border-subtle)] space-y-6">
          {loading && (
            <div className="h-[350px] flex items-center justify-center">
              <LoadingSkeleton />
            </div>
          )}

          {error && (
            <div className="h-[350px] flex items-center justify-center">
              <div className="p-4 rounded-xl border border-red-500/20 bg-red-500/5 text-slate-200 space-y-2 max-w-md">
                <div className="flex items-center gap-2 text-red-500 font-bold text-sm">
                  <AlertCircle className="h-4 w-4" /> Load Error
                </div>
                <p className="text-xs leading-relaxed text-slate-450">{error}</p>
              </div>
            </div>
          )}

          {!loading && data && (
            <div className="space-y-4">
              <div className="flex justify-between items-center flex-wrap gap-3">
                <span className="text-xs font-semibold text-slate-400">60-Day Lookback Sequence Holdout</span>
                <div className="flex items-center gap-4 text-xs">
                  <div className="flex items-center gap-1.5">
                    <span className="h-2.5 w-2.5 rounded-full bg-red-500"></span>
                    <span className="text-slate-350">Actual Truth</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="h-2.5 w-2.5 rounded-full bg-[#00d4aa]"></span>
                    <span className="text-slate-350">LSTM Prediction</span>
                  </div>
                </div>
              </div>

              <div className="h-[350px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis
                      dataKey="day"
                      stroke="rgba(255,255,255,0.3)"
                      fontSize={10}
                      tickLine={false}
                    />
                    <YAxis
                      stroke="rgba(255,255,255,0.3)"
                      fontSize={10}
                      tickLine={false}
                      domain={["auto", "auto"]}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "var(--bg-secondary)",
                        border: "1px solid var(--border-subtle)",
                        borderRadius: "8px",
                      }}
                      labelStyle={{ color: "var(--text-secondary)", fontSize: "11px", fontWeight: "bold" }}
                      itemStyle={{ fontSize: "12px" }}
                    />
                    <Line
                      type="monotone"
                      dataKey="Actual"
                      stroke="#ef4444"
                      strokeWidth={2}
                      dot={false}
                      activeDot={{ r: 4 }}
                    />
                    <Line
                      type="monotone"
                      dataKey="Predicted"
                      stroke="#00d4aa"
                      strokeWidth={2}
                      strokeDasharray="4 4"
                      dot={false}
                      activeDot={{ r: 4 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </div>

        <div className="flex flex-col gap-6">
          <div className="glass-card p-6 border-[var(--border-subtle)] bg-slate-900/10 flex-1 flex flex-col justify-center">
            {loading && <LoadingSkeleton />}

            {!loading && data && (
              <div className="space-y-6">
                <div className="space-y-1">
                  <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider">Forecast Confidence</h3>
                  <div className="flex items-baseline gap-2">
                    <span className="text-3xl font-extrabold tracking-tight text-[var(--accent-mint)]">
                      92.4%
                    </span>
                    <span className="text-xs font-semibold text-slate-450">Backtested R² Score</span>
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="flex justify-between items-center p-3 rounded-lg bg-slate-900/40 border border-[var(--border-subtle)]">
                    <div className="flex items-center gap-2">
                      <TrendingUp className="h-4 w-4 text-[var(--accent-mint)]" />
                      <span className="text-xs font-medium text-slate-300">Target Horizon</span>
                    </div>
                    <span className="text-xs font-bold font-mono text-slate-200">5 Days</span>
                  </div>

                  <div className="flex justify-between items-center p-3 rounded-lg bg-slate-900/40 border border-[var(--border-subtle)]">
                    <div className="flex items-center gap-2">
                      <Info className="h-4 w-4 text-[var(--accent-blue)]" />
                      <span className="text-xs font-medium text-slate-300">Lookback Period</span>
                    </div>
                    <span className="text-xs font-bold font-mono text-slate-200">30 Days</span>
                  </div>
                </div>

                <div className="p-3.5 rounded-lg border border-blue-500/15 bg-blue-500/5 text-[11px] leading-relaxed text-slate-400">
                  The PyTorch LSTM model maps historical volatility patterns across mutual fund timelines to project continuous NAV trajectories without manual parameters.
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
