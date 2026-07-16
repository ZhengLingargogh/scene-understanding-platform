import { useEffect, useRef } from "react";
import type { MatchingVisualizationData } from "../../types/inference";

interface MatchingResultPanelProps {
  data: MatchingVisualizationData | null;
  hasRun: boolean;
  loading: boolean;
}

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Failed to load image"));
    img.src = url;
  });
}

export default function MatchingResultPanel({ data, hasRun, loading }: MatchingResultPanelProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data) return undefined;

    let cancelled = false;

    async function draw() {
      const matchData = data;
      if (!matchData) return;

      const [img0, img1] = await Promise.all([
        loadImage(matchData.image0Url),
        loadImage(matchData.image1Url),
      ]);
      if (cancelled || !canvas) return;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const gap = 24;
      const maxHeight = Math.min(520, canvas.parentElement?.clientHeight ?? 520);
      const scale0 = maxHeight / img0.height;
      const scale1 = maxHeight / img1.height;
      const w0 = img0.width * scale0;
      const w1 = img1.width * scale1;
      const totalWidth = w0 + gap + w1;

      canvas.width = totalWidth;
      canvas.height = maxHeight;

      ctx.fillStyle = "#020617";
      ctx.fillRect(0, 0, totalWidth, maxHeight);

      ctx.drawImage(img0, 0, 0, w0, maxHeight);
      ctx.drawImage(img1, w0 + gap, 0, w1, maxHeight);

      const kx0 = w0 / img0.width;
      const ky0 = maxHeight / img0.height;
      const kx1 = w1 / img1.width;
      const ky1 = maxHeight / img1.height;
      const offset1 = w0 + gap;

      ctx.lineWidth = 1;
      for (const [i0, i1] of matchData.matches) {
        const p0 = matchData.keypoints0[i0];
        const p1 = matchData.keypoints1[i1];
        if (!p0 || !p1) continue;

        const x0 = p0[0] * kx0;
        const y0 = p0[1] * ky0;
        const x1 = offset1 + p1[0] * kx1;
        const y1 = p1[1] * ky1;

        ctx.strokeStyle = "rgba(34, 211, 238, 0.45)";
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.stroke();

        ctx.fillStyle = "#22d3ee";
        ctx.beginPath();
        ctx.arc(x0, y0, 2.5, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = "#a78bfa";
        ctx.beginPath();
        ctx.arc(x1, y1, 2.5, 0, Math.PI * 2);
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
          <h2 className="text-sm font-semibold text-cyan-200">匹配结果</h2>
          <p className="text-xs text-slate-500">两图匹配点以直线连接</p>
        </div>
        {data && (
          <span className="rounded-full border border-cyan-800/50 bg-cyan-950/40 px-3 py-1 text-xs text-cyan-300">
            {data.matchCount} 对匹配
          </span>
        )}
      </div>

      <div className="flex min-h-0 flex-1 items-center justify-center overflow-auto rounded-lg border border-slate-800/80 bg-slate-950/60 p-4">
        {loading && <p className="text-sm text-slate-400">检测与匹配中…</p>}
        {!loading && !hasRun && (
          <p className="text-sm text-slate-500">上传两张 Query 图像，选择检测与匹配模型后 Run。</p>
        )}
        {!loading && hasRun && !data && (
          <p className="text-sm text-amber-400">未检测到有效匹配点，请尝试其他图像对。</p>
        )}
        {data && <canvas ref={canvasRef} className="max-h-full max-w-full" />}
      </div>

      {data?.message && (
        <p className="mt-3 text-xs text-slate-500">
          {data.detectorModelId && data.matcherModelId
            ? `${data.detectorModelId} → ${data.matcherModelId} · `
            : ""}
          {data.message}
        </p>
      )}
    </div>
  );
}
