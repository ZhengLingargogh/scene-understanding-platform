import type { CovisibilityRef } from "../../types/inference";

interface Top8ResultsPanelProps {
  refs: CovisibilityRef[];
  hasRun: boolean;
  loading?: boolean;
}

export default function Top8ResultsPanel({ refs, hasRun, loading }: Top8ResultsPanelProps) {
  return (
    <div className="flex min-w-0 flex-1 flex-col bg-slate-950/80 p-6">
      <h2 className="text-sm font-semibold uppercase tracking-wider text-cyan-300/90">Top-8 检索结果</h2>
      <p className="mt-1 text-xs text-slate-500">Run 后在右侧展示相似度最高的 8 张参考图</p>

      {!hasRun && !loading && (
        <div className="mt-8 flex flex-1 items-center justify-center rounded-xl border border-dashed border-slate-700/60 bg-slate-900/40">
          <p className="text-sm text-slate-500">上传 Query 并点击 Run 后显示检索结果</p>
        </div>
      )}

      {loading && (
        <div className="mt-8 flex flex-1 items-center justify-center rounded-xl border border-cyan-800/30 bg-slate-900/40">
          <p className="animate-pulse text-sm text-cyan-300/80">检索中…</p>
        </div>
      )}

      {hasRun && !loading && (
        <ul className="mt-4 grid flex-1 auto-rows-fr grid-cols-2 gap-4 overflow-y-auto lg:grid-cols-4">
          {refs.map((ref, idx) => (
            <li
              key={ref.id}
              className="flex flex-col gap-2 rounded-lg border border-slate-700/60 bg-slate-900/60 p-3 shadow-lg shadow-black/20"
            >
              <div className="relative overflow-hidden rounded-md">
                <img src={ref.thumbnail} alt={ref.label} className="aspect-[4/3] w-full object-cover" />
                <span className="absolute left-2 top-2 rounded bg-black/60 px-2 py-0.5 text-xs font-bold text-cyan-200">
                  #{idx + 1}
                </span>
              </div>
              <p className="truncate text-xs text-slate-300" title={ref.label}>
                {ref.label}
              </p>
              <div className="flex items-center gap-2">
                <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-700">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-blue-500"
                    style={{ width: `${Math.min(ref.score * 100, 100)}%` }}
                  />
                </div>
                <span className="shrink-0 font-mono text-sm font-medium text-cyan-300">
                  {(ref.score * 100).toFixed(1)}%
                </span>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
