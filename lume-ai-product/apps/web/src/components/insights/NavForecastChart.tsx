"use client";

import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from "recharts";

interface NavForecastChartProps {
  data: any;
}

export default function NavForecastChart({ data }: NavForecastChartProps) {
  const chartData = data
    ? data.days.map((day: number, idx: number) => ({
        day: `Day ${day + 1}`,
        Actual: data.actual[idx],
        Predicted: data.predicted[idx],
      }))
    : [];

  return (
    <div className="glass-card p-6 border-[var(--border-subtle)] space-y-6">
      <div className="flex justify-between items-start flex-wrap gap-4">
        <div>
          <h3 className="text-base font-bold text-slate-200">LSTM NAV Predictions</h3>
          <p className="text-slate-400 text-xs mt-0.5">LSTM regression model performance mapped against holdout target baseline.</p>
        </div>
        <div className="flex items-center gap-4 text-xs font-semibold">
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

      <div className="h-[300px] w-full">
        {chartData.length === 0 ? (
          <div className="h-full flex items-center justify-center text-xs text-slate-500">
            NAV trajectory data unavailable.
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis
                dataKey="day"
                stroke="rgba(255,255,255,0.2)"
                fontSize={10}
                tickLine={false}
              />
              <YAxis
                stroke="rgba(255,255,255,0.2)"
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
        )}
      </div>
    </div>
  );
}
