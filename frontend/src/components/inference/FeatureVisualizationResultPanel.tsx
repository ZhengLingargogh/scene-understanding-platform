import { useEffect, useRef } from "react";
import { buildResponseHeatmap, heatmapColor, loadImage } from "./featureVisualizationUtils";
import type { FeatureVisualizationData } from "../../types/inference";

interface FeatureVisualizationResultPanelProps {
  data: FeatureVisualizationData | null;
  hasRun: boolean;
  loading: boolean;
}

export default function FeatureVisualizationResultPanel({
  data,
  hasRun,
  loading,
}: FeatureVisualizationResultPanelProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data) return undefined;

    let cancelled = false;

    async function draw() {
      const viz = data;
      if (!viz) return;

      const img = await loadImage(viz.imageUrl);
      if (cancelled || !canvas) return;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const [cropY1, cropY2, cropX1, cropX2] = viz.cropBBox;
      const cropW = cropX2 - cropX1;
      const cropH = cropY2 - cropY1;

      const maxWidth = canvas.parentElement?.clientWidth ?? 820;
      const maxHeight = Math.min(620, canvas.parentElement?.clientHeight ?? 620);
      const scale = Math.min(maxWidth / cropW, maxHeight / cropH, 1);
      const drawW = cropW * scale;
      const drawH = cropH * scale;

      canvas.width = drawW;
      canvas.height = drawH;

      ctx.fillStyle = "#020617";
      ctx.fillRect(0, 0, drawW, drawH);
      ctx.drawImage(img, cropX1, cropY1, cropW, cropH, 0, 0, drawW, drawH);

      const { heatmap, width: hw, height: hh, displayMax } = buildResponseHeatmap(
        viz.keypoints,
        viz.scores,
        viz.imageWidth,
        viz.imageHeight,
      );

      const heatCanvas = document.createElement("canvas");
      heatCanvas.width = hw;
      heatCanvas.height = hh;
      const heatCtx = heatCanvas.getContext("2d");
      if (heatCtx && displayMax > 0) {
        const imageData = heatCtx.createImageData(hw, hh);
        for (let y = 0; y < hh; y += 1) {
          for (let x = 0; x < hw; x += 1) {
            const v = heatmap[y * hw + x] / displayMax;
            const [r, g, b] = heatmapColor(v);
            const idx = (y * hw + x) * 4;
            imageData.data[idx] = r;
            imageData.data[idx + 1] = g;
            imageData.data[idx + 2] = b;
            imageData.data[idx + 3] = Math.round(v * 200);
          }
        }
        heatCtx.putImageData(imageData, 0, 0);
        ctx.save();
        ctx.globalAlpha = 0.65;
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(heatCanvas, 0, 0, drawW, drawH);
        ctx.restore();
      }

      const kx = drawW / cropW;
      const ky = drawH / cropH;
      const maxScore = Math.max(...viz.scores, 1e-6);

      for (let i = 0; i < viz.keypoints.length; i += 1) {
        const [x, y] = viz.keypoints[i];
        if (x < cropX1 || x > cropX2 || y < cropY1 || y > cropY2) continue;

        const score = viz.scores[i];
        const px = (x - cropX1) * kx;
        const py = (y - cropY1) * ky;
        const t = score / maxScore;
        const radius = 1.0 + t * 1.2;

        ctx.beginPath();
        ctx.arc(px, py, radius, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(34, 211, 238, ${0.35 + t * 0.5})`;
        ctx.fill();
      }
    }

    draw().catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [data]);

  return (
    <div className="flex min-w-0 flex-1 flex-col p-4">
      <div className="mb-3 flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-cyan-200">特征可视化</h2>
          <p className="text-xs text-slate-500">关键点 + 响应热力图</p>
        </div>
        {data && (
          <span className="rounded-full border border-cyan-800/50 bg-cyan-950/40 px-3 py-1 text-xs text-cyan-300">
            {data.backend ?? data.modelId}
          </span>
        )}
      </div>

      <div className="flex min-h-0 flex-1 items-center justify-center overflow-auto rounded-lg border border-slate-800/80 bg-slate-950/60 p-4">
        {loading && <p className="text-sm text-slate-400">生成可视特征中…</p>}
        {!loading && !hasRun && (
          <p className="text-sm text-slate-500">上传 Query 图像并 Run 后，在此显示关键点与热力图。</p>
        )}
        {!loading && hasRun && !data && (
          <p className="text-sm text-amber-400">未能生成可视化结果，请重试。</p>
        )}
        {data && <canvas ref={canvasRef} className="max-h-full max-w-full rounded shadow-lg" />}
      </div>

      {data && (
        <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <StatCard label="关键点数量" value={String(data.keypointCount)} />
          <StatCard label="最高置信度" value={data.confidenceMax.toFixed(3)} />
          <StatCard label="最低置信度" value={data.confidenceMin.toFixed(3)} />
          <StatCard label="中位数置信度" value={data.confidenceMedian.toFixed(3)} />
        </div>
      )}

      {data?.message && <p className="mt-3 text-xs text-slate-500">{data.message}</p>}
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-slate-800/80 bg-slate-900/60 px-3 py-2">
      <p className="text-[10px] uppercase tracking-wide text-slate-500">{label}</p>
      <p className="mt-1 text-lg font-semibold tabular-nums text-cyan-100">{value}</p>
    </div>
  );
}
