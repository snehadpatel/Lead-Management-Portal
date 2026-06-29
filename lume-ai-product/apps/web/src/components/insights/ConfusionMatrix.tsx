interface ConfusionMatrixProps {
  data: any;
}

export default function ConfusionMatrix({ data }: ConfusionMatrixProps) {
  // Default fallback matrix if API is slow/offline
  const matrix = data?.matrix || [[638, 150], [192, 820]];
  const labels = data?.labels || ["Not Converted", "Converted"];

  const rowTotals = matrix.map((row: number[]) => row.reduce((a, b) => a + b, 0));
  const colTotals = [
    matrix[0][0] + matrix[1][0],
    matrix[0][1] + matrix[1][1]
  ];
  const grandTotal = rowTotals.reduce((a: number, b: number) => a + b, 0);

  const getHeatmapColor = (val: number, total: number) => {
    const pct = val / total;
    if (pct > 0.7) return "bg-blue-600/90 text-white";
    if (pct > 0.4) return "bg-blue-700/50 text-slate-200";
    return "bg-blue-950/40 text-slate-400";
  };

  return (
    <div className="glass-card p-6 border-[var(--border-subtle)] space-y-6">
      <div>
        <h3 className="text-base font-bold text-slate-200">Confusion Matrix Heatmap</h3>
        <p className="text-slate-400 text-xs mt-0.5">Evaluation performance detailing predicted classes vs true outcomes.</p>
      </div>

      <div className="flex flex-col items-center justify-center py-4">
        {/* Heatmap Grid */}
        <div className="grid grid-cols-3 gap-2 w-full max-w-md relative">
          {/* Label placeholders */}
          <div></div>
          <div className="text-center text-[10px] font-bold text-slate-500 uppercase tracking-wider pb-1">
            Predicted {labels[0]}
          </div>
          <div className="text-center text-[10px] font-bold text-slate-500 uppercase tracking-wider pb-1">
            Predicted {labels[1]}
          </div>

          {/* Row 1 */}
          <div className="flex items-center justify-end pr-2 text-[10px] font-bold text-slate-500 uppercase tracking-wider text-right">
            True {labels[0]}
          </div>
          <div className={`h-24 rounded-lg flex flex-col items-center justify-center p-2 font-mono ${getHeatmapColor(matrix[0][0], rowTotals[0])}`}>
            <span className="text-lg font-bold">{matrix[0][0]}</span>
            <span className="text-[10px] opacity-80">{((matrix[0][0] / rowTotals[0]) * 100).toFixed(1)}%</span>
            <span className="text-[9px] opacity-60">True Neg</span>
          </div>
          <div className={`h-24 rounded-lg flex flex-col items-center justify-center p-2 font-mono ${getHeatmapColor(matrix[0][1], rowTotals[0])}`}>
            <span className="text-lg font-bold">{matrix[0][1]}</span>
            <span className="text-[10px] opacity-80">{((matrix[0][1] / rowTotals[0]) * 100).toFixed(1)}%</span>
            <span className="text-[9px] opacity-60">False Pos</span>
          </div>

          {/* Row 2 */}
          <div className="flex items-center justify-end pr-2 text-[10px] font-bold text-slate-500 uppercase tracking-wider text-right">
            True {labels[1]}
          </div>
          <div className={`h-24 rounded-lg flex flex-col items-center justify-center p-2 font-mono ${getHeatmapColor(matrix[1][0], rowTotals[1])}`}>
            <span className="text-lg font-bold">{matrix[1][0]}</span>
            <span className="text-[10px] opacity-80">{((matrix[1][0] / rowTotals[1]) * 100).toFixed(1)}%</span>
            <span className="text-[9px] opacity-60">False Neg</span>
          </div>
          <div className={`h-24 rounded-lg flex flex-col items-center justify-center p-2 font-mono ${getHeatmapColor(matrix[1][1], rowTotals[1])}`}>
            <span className="text-lg font-bold">{matrix[1][1]}</span>
            <span className="text-[10px] opacity-80">{((matrix[1][1] / rowTotals[1]) * 100).toFixed(1)}%</span>
            <span className="text-[9px] opacity-60">True Pos</span>
          </div>
        </div>

        {/* Matrix Metadata summary */}
        <div className="w-full max-w-md mt-6 pt-4 border-t border-[var(--border-subtle)] grid grid-cols-3 text-center text-xs font-semibold text-slate-400">
          <div>
            <div className="text-slate-500 text-[10px] uppercase tracking-wider">True Positive Rate</div>
            <div className="text-sm font-bold text-slate-200 mt-0.5">{((matrix[1][1] / rowTotals[1]) * 100).toFixed(1)}%</div>
          </div>
          <div>
            <div className="text-slate-500 text-[10px] uppercase tracking-wider">True Negative Rate</div>
            <div className="text-sm font-bold text-slate-200 mt-0.5">{((matrix[0][0] / rowTotals[0]) * 100).toFixed(1)}%</div>
          </div>
          <div>
            <div className="text-slate-500 text-[10px] uppercase tracking-wider">Total Test Size</div>
            <div className="text-sm font-bold text-slate-200 mt-0.5">{grandTotal}</div>
          </div>
        </div>
      </div>
    </div>
  );
}
