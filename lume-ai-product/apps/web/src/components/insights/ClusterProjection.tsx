"use client";

import { ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Cell } from "recharts";

interface ClusterProjectionProps {
  data: any;
}

const PERSONAS = [
  { name: "Growth Seekers", color: "#ef4444" },
  { name: "Safety-First Savers", color: "#22c55e" },
  { name: "Balanced Allocators", color: "#3b82f6" },
  { name: "Passive Indexers", color: "#f59e0b" },
];

export default function ClusterProjection({ data }: ClusterProjectionProps) {
  const points = data?.points || [];

  return (
    <div className="glass-card p-6 border-[var(--border-subtle)] space-y-6">
      <div className="flex justify-between items-start flex-wrap gap-4">
        <div>
          <h3 className="text-base font-bold text-slate-200">K-Means Investor Segments</h3>
          <p className="text-slate-400 text-xs mt-0.5">PCA 2D projection of psycho-graphic profiles segmented into clusters.</p>
        </div>
        <div className="flex flex-wrap gap-3 text-[10px] font-semibold">
          {PERSONAS.map((p, idx) => (
            <div key={idx} className="flex items-center gap-1.5 bg-slate-900/40 px-2 py-1 rounded border border-slate-800">
              <span className="h-2 w-2 rounded-full" style={{ backgroundColor: p.color }}></span>
              <span className="text-slate-300">{p.name}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="h-[300px] w-full">
        {points.length === 0 ? (
          <div className="h-full flex items-center justify-center text-xs text-slate-500">
            PCA data unavailable.
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis
                type="number"
                dataKey="x"
                name="Component 1"
                stroke="rgba(255,255,255,0.2)"
                fontSize={10}
                tickLine={false}
              />
              <YAxis
                type="number"
                dataKey="y"
                name="Component 2"
                stroke="rgba(255,255,255,0.2)"
                fontSize={10}
                tickLine={false}
              />
              <Tooltip
                cursor={{ strokeDasharray: "3 3" }}
                contentStyle={{
                  background: "var(--bg-secondary)",
                  border: "1px solid var(--border-subtle)",
                  borderRadius: "8px",
                }}
                itemStyle={{ fontSize: "11px" }}
              />
              <Scatter name="Investors" data={points}>
                {points.map((entry: any, index: number) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={PERSONAS[entry.cluster]?.color || "#6b7280"}
                    fillOpacity={0.7}
                  />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
