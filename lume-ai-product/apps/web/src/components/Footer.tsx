import { ShieldAlert } from "lucide-react";

export default function Footer() {
  return (
    <footer className="w-full border-t border-[var(--border-subtle)] bg-[var(--bg-primary)] py-8 mt-auto">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex flex-col items-center md:items-start gap-2">
            <span className="text-sm font-semibold text-slate-300">
              Lume AI — Mutual Fund Intelligence
            </span>
            <span className="text-xs text-slate-500">
              Created by{" "}
              <a
                href="https://snehapatel.vercel.app"
                target="_blank"
                rel="noopener noreferrer"
                className="text-slate-400 hover:text-[var(--accent-mint)] transition-colors underline"
              >
                Sneha Patel
              </a>
            </span>
          </div>

          <div className="flex items-center gap-6 text-xs text-slate-400">
            <a
              href="https://github.com/snehadpatel"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-slate-200 transition-colors"
            >
              GitHub Project
            </a>
            <a
              href="https://snehapatel.vercel.app"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-slate-200 transition-colors"
            >
              Portfolio
            </a>
          </div>
        </div>

        {/* Global Warning / Disclaimer */}
        <div className="mt-6 flex items-start gap-2.5 rounded-lg border border-amber-500/20 bg-amber-500/5 p-3.5 text-xs text-amber-500/90">
          <ShieldAlert className="h-4 w-4 shrink-0 mt-0.5" />
          <p className="leading-relaxed">
            <strong>Disclaimer:</strong> This application is a technology demonstration for educational purposes. It utilizes synthetic or historical sandbox datasets. None of the scoring, segments, predictions, or metrics constitute financial advice, investment recommendations, or commercial solicitations.
          </p>
        </div>
      </div>
    </footer>
  );
}
