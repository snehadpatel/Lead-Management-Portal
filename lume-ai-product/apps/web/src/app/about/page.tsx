import { Code2, Database, Cpu, Server, ExternalLink } from "lucide-react";

export const metadata = {
  title: "About the Architecture — Lume AI Sandbox",
  description: "Learn about the architecture and technologies powering the Lume AI mutual fund intelligence platform.",
};

export default function AboutPage() {
  const steps = [
    {
      title: "1. Big Data Processing",
      description: "Extracting, cleaning, and aggregating structural parameters from mutual fund NAV timelines, stock market index datasets, and distributor lead catalogs.",
      tech: "PySpark • Pandas • Parquet",
      icon: Database,
    },
    {
      title: "2. Machine Learning Pipeline",
      description: "Offline model fitting on local/Colab infrastructure. Scikit-learn random forests for scoring, K-Means for investor profiling, SBERT for news sentiment and semantic lookup indexes, and PyTorch LSTMs for NAV trajectories.",
      tech: "Scikit-Learn • PyTorch • SBERT",
      icon: Cpu,
    },
    {
      title: "3. Serviced Inference API",
      description: "Unified FastAPI backend exposing prediction endpoints and loading pre-trained binary models. Strict Pydantic validations, CORS locks, and SlowAPI rate limiting safeguard the public sandbox.",
      tech: "FastAPI • Docker • Render",
      icon: Server,
    },
  ];

  return (
    <div className="space-y-10 py-6 max-w-4xl mx-auto">
      <div className="space-y-2">
        <h1 className="text-3xl font-extrabold tracking-tight text-slate-100 sm:text-4xl">System Architecture</h1>
        <p className="text-slate-400 text-sm leading-relaxed">
          Learn how Lume AI decouples offline big data processing and training pipelines from live sandbox serving.
        </p>
      </div>

      {/* Mermaid-style SVG flow diagram */}
      <section className="glass-card p-6 border-[var(--border-subtle)] space-y-4">
        <h3 className="text-sm font-bold text-slate-350 uppercase tracking-wider">Pipeline Flow</h3>
        <div className="w-full flex flex-col md:flex-row items-center justify-between gap-4 py-6 px-2">
          <div className="flex-1 text-center p-4 rounded-xl bg-slate-900/40 border border-slate-800 space-y-2 w-full">
            <Database className="h-6 w-6 text-emerald-400 mx-auto" />
            <div className="text-xs font-bold text-slate-200">Raw Data Storage</div>
            <div className="text-[10px] text-slate-500">Mutual Fund NAVs & Lead CSVs</div>
          </div>
          <div className="text-slate-600 font-bold rotate-90 md:rotate-0">➔</div>
          <div className="flex-1 text-center p-4 rounded-xl bg-slate-900/40 border border-slate-800 space-y-2 w-full">
            <Cpu className="h-6 w-6 text-blue-400 mx-auto" />
            <div className="text-xs font-bold text-slate-200">PySpark / PyTorch</div>
            <div className="text-[10px] text-slate-500">Offline training & Scalers</div>
          </div>
          <div className="text-slate-600 font-bold rotate-90 md:rotate-0">➔</div>
          <div className="flex-1 text-center p-4 rounded-xl bg-slate-900/40 border border-slate-800 space-y-2 w-full">
            <Server className="h-6 w-6 text-amber-400 mx-auto" />
            <div className="text-xs font-bold text-slate-200">FastAPI Registry</div>
            <div className="text-[10px] text-slate-500">Prediction Endpoints (No DB)</div>
          </div>
        </div>
      </section>

      {/* Architecture steps */}
      <section className="space-y-6">
        <h3 className="text-lg font-bold text-slate-200">Processing Pipeline</h3>
        <div className="space-y-4">
          {steps.map((s, idx) => {
            const Icon = s.icon;
            return (
              <div key={idx} className="flex gap-4 p-5 rounded-xl border border-[var(--border-subtle)] bg-[var(--bg-card)]">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-slate-800/80 border border-slate-700/50 text-[var(--accent-mint)]">
                  <Icon className="h-5 w-5" />
                </div>
                <div className="space-y-1">
                  <h4 className="text-sm font-bold text-slate-200">{s.title}</h4>
                  <p className="text-slate-400 text-xs leading-relaxed">{s.description}</p>
                  <div className="text-[10px] font-bold font-mono text-[var(--accent-mint)] pt-1">{s.tech}</div>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* Developer info */}
      <section className="glass-card p-6 border-[var(--border-subtle)] space-y-4">
        <div className="flex items-center gap-2 text-slate-200 font-bold text-sm">
          <Code2 className="h-5 w-5 text-[var(--accent-mint)]" /> Developer Profile
        </div>
        <p className="text-slate-400 text-xs leading-relaxed">
          This MVP sandbox was built and integrated by <strong>Sneha Patel</strong>. You can browse the source code repositories or portfolio sites using the external links below:
        </p>
        <div className="flex flex-wrap gap-4 pt-2">
          <a
            href="https://github.com/snehadpatel"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 text-xs font-bold text-[var(--accent-mint)] hover:underline"
          >
            GitHub Profile <ExternalLink className="h-3.5 w-3.5" />
          </a>
          <a
            href="https://snehapatel.vercel.app"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 text-xs font-bold text-[var(--accent-mint)] hover:underline"
          >
            Portfolio Page <ExternalLink className="h-3.5 w-3.5" />
          </a>
        </div>
      </section>
    </div>
  );
}
