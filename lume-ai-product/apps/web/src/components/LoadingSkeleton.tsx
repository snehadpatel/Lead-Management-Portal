export default function LoadingSkeleton() {
  return (
    <div className="space-y-4 p-4 border border-[var(--border-subtle)] bg-[var(--bg-card)] rounded-xl animate-pulse">
      <div className="flex items-center gap-3">
        <div className="h-10 w-10 rounded-lg bg-slate-800/80"></div>
        <div className="space-y-2 flex-1">
          <div className="h-4 w-1/4 rounded bg-slate-800/80"></div>
          <div className="h-3 w-1/3 rounded bg-slate-800/80"></div>
        </div>
      </div>
      <div className="space-y-2">
        <div className="h-3 w-full rounded bg-slate-800/80"></div>
        <div className="h-3 w-5/6 rounded bg-slate-800/80"></div>
      </div>
      <div className="h-8 w-28 rounded bg-slate-800/80 mt-2"></div>
    </div>
  );
}
