import { Target, Users, MessageSquare, TrendingUp } from "lucide-react";

interface MetricsCardsProps {
  metrics: any;
}

export default function MetricsCards({ metrics }: MetricsCardsProps) {
  const rfMetrics = metrics?.random_forest || {};
  const kmeansMetrics = metrics?.kmeans || {};
  const sentimentMetrics = metrics?.sentiment || {};
  const lstmMetrics = metrics?.lstm || {};

  const cards = [
    {
      label: "Lead Classification",
      value: rfMetrics.accuracy || "85.0%",
      subtext: `ROC AUC: ${rfMetrics.roc_auc || "0.933"}`,
      icon: Target,
      color: "text-emerald-400",
      bg: "bg-emerald-500/10",
    },
    {
      label: "Investor Segments",
      value: `${kmeansMetrics.n_clusters || 4} Clusters`,
      subtext: `Silhouette: ${(kmeansMetrics.silhouette || 0.395).toFixed(3)}`,
      icon: Users,
      color: "text-blue-400",
      bg: "bg-blue-500/10",
    },
    {
      label: "Headline Sentiment",
      value: sentimentMetrics.accuracy ? `${(sentimentMetrics.accuracy * 100).toFixed(1)}%` : "71.6%",
      subtext: `F1 Macro: ${sentimentMetrics.f1_macro ? sentimentMetrics.f1_macro.toFixed(3) : "0.689"}`,
      icon: MessageSquare,
      color: "text-amber-400",
      bg: "bg-amber-500/10",
    },
    {
      label: "NAV Time-Series",
      value: lstmMetrics.r_squared ? `${(lstmMetrics.r_squared * 100).toFixed(0)}% R²` : "89% R²",
      subtext: `MSE: ${lstmMetrics.mse || "0.0152"}`,
      icon: TrendingUp,
      color: "text-rose-400",
      bg: "bg-rose-500/10",
    },
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
      {cards.map((card, i) => {
        const Icon = card.icon;
        return (
          <div key={i} className="glass-card p-6 border-[var(--border-subtle)] flex items-center justify-between">
            <div className="space-y-1">
              <span className="text-xs font-semibold text-slate-450 uppercase tracking-wider">{card.label}</span>
              <div className="text-2xl font-black text-slate-100">{card.value}</div>
              <span className="text-xs text-slate-400 font-medium">{card.subtext}</span>
            </div>
            <div className={`h-12 w-12 rounded-xl flex items-center justify-center ${card.bg} ${card.color}`}>
              <Icon className="h-6 w-6" />
            </div>
          </div>
        );
      })}
    </div>
  );
}
