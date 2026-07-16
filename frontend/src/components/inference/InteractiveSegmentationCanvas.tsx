import { useCallback, useEffect, useRef, useState } from "react";
import type { SegmentationPredictData, SegmentationSessionData } from "../../types/inference";
import { loadImage, predictSegmentationMask } from "./segmentationUtils";

interface InteractiveSegmentationCanvasProps {
  session: SegmentationSessionData | null;
  active: boolean;
}

const DEBOUNCE_MS = 60;

export default function InteractiveSegmentationCanvas({ session, active }: InteractiveSegmentationCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pendingPointRef = useRef<[number, number] | null>(null);
  const [predictData, setPredictData] = useState<SegmentationPredictData | null>(null);
  const [hoverPoint, setHoverPoint] = useState<[number, number] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const draw = useCallback(
    (maskData: SegmentationPredictData | null, cursor: [number, number] | null) => {
      const canvas = canvasRef.current;
      const img = imageRef.current;
      if (!canvas || !img || !session) return;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const maxW = containerRef.current?.clientWidth ?? 960;
      const maxH = containerRef.current?.clientHeight ?? 640;
      const scale = Math.min(maxW / session.imageWidth, maxH / session.imageHeight, 1);
      const drawW = session.imageWidth * scale;
      const drawH = session.imageHeight * scale;

      canvas.width = drawW;
      canvas.height = drawH;

      ctx.clearRect(0, 0, drawW, drawH);
      ctx.drawImage(img, 0, 0, drawW, drawH);

      if (maskData && maskData.mask.length === session.imageWidth * session.imageHeight) {
        const overlay = document.createElement("canvas");
        overlay.width = session.imageWidth;
        overlay.height = session.imageHeight;
        const octx = overlay.getContext("2d");
        if (octx) {
          const imageData = octx.createImageData(session.imageWidth, session.imageHeight);
          for (let i = 0; i < maskData.mask.length; i += 1) {
            if (!maskData.mask[i]) continue;
            const p = i * 4;
            imageData.data[p] = 34;
            imageData.data[p + 1] = 211;
            imageData.data[p + 2] = 238;
            imageData.data[p + 3] = 110;
          }
          octx.putImageData(imageData, 0, 0);
          ctx.drawImage(overlay, 0, 0, drawW, drawH);
        }

        if (maskData.contour.length >= 3) {
          ctx.beginPath();
          const [fx, fy] = maskData.contour[0];
          ctx.moveTo(fx * scale, fy * scale);
          for (let i = 1; i < maskData.contour.length; i += 1) {
            const [x, y] = maskData.contour[i];
            ctx.lineTo(x * scale, y * scale);
          }
          ctx.closePath();
          ctx.strokeStyle = "rgba(250, 204, 21, 0.95)";
          ctx.lineWidth = 2;
          ctx.stroke();

          ctx.save();
          ctx.clip();
          ctx.fillStyle = "rgba(250, 204, 21, 0.12)";
          ctx.fillRect(0, 0, drawW, drawH);
          ctx.restore();
        }
      }

      const point = cursor ?? (maskData ? maskData.point : null);
      if (point) {
        const [px, py] = point;
        ctx.fillStyle = "#f472b6";
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(px * scale, py * scale, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
    },
    [session],
  );

  useEffect(() => {
    if (!session || !active) {
      imageRef.current = null;
      setPredictData(null);
      setHoverPoint(null);
      return undefined;
    }

    let cancelled = false;
    loadImage(session.imageUrl)
      .then((img) => {
        if (cancelled) return;
        imageRef.current = img;
        draw(null, null);
      })
      .catch((err: Error) => setError(err.message));

    return () => {
      cancelled = true;
    };
  }, [session, active, draw]);

  useEffect(() => {
    draw(predictData, hoverPoint);
  }, [predictData, hoverPoint, draw]);

  const requestPredict = useCallback(
    (pointX: number, pointY: number) => {
      if (!session) return;
      pendingPointRef.current = [pointX, pointY];
      if (debounceRef.current) clearTimeout(debounceRef.current);

      debounceRef.current = setTimeout(() => {
        const point = pendingPointRef.current;
        if (!point || !session) return;
        setLoading(true);
        setError(null);
        predictSegmentationMask(session.sessionId, point[0], point[1])
          .then((data) => {
            setPredictData(data);
            setLoading(false);
          })
          .catch((err: Error) => {
            if (err.message === "stale") return;
            setError(err.message);
            setLoading(false);
          });
      }, DEBOUNCE_MS);
    },
    [session],
  );

  function handleMouseMove(event: React.MouseEvent<HTMLCanvasElement>) {
    if (!session || !active || !canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const scaleX = session.imageWidth / rect.width;
    const scaleY = session.imageHeight / rect.height;
    const x = (event.clientX - rect.left) * scaleX;
    const y = (event.clientY - rect.top) * scaleY;
    if (x < 0 || y < 0 || x >= session.imageWidth || y >= session.imageHeight) return;

    setHoverPoint([x, y]);
    requestPredict(x, y);
  }

  function handleMouseLeave() {
    setHoverPoint(null);
  }

  useEffect(() => {
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, []);

  return (
    <div ref={containerRef} className="flex min-h-0 flex-1 flex-col p-4">
      <div className="mb-3 flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-cyan-200">交互模式</h2>
          <p className="text-xs text-slate-500">移动鼠标发送点提示，实时刷新 Mask</p>
        </div>
        {active && predictData && (
          <span className="rounded-full border border-cyan-800/50 bg-cyan-950/40 px-3 py-1 text-xs text-cyan-300">
            score {predictData.score.toFixed(3)}
          </span>
        )}
      </div>

      <div className="flex min-h-0 flex-1 items-start justify-start overflow-auto rounded-lg border border-slate-800/80 bg-slate-950/60 p-4">
        {!active && (
          <p className="text-sm text-slate-500">上传图像并点击 Run 后，在此进入交互分割模式。</p>
        )}
        {active && session && (
          <canvas
            ref={canvasRef}
            className="max-h-full max-w-full cursor-crosshair"
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
          />
        )}
      </div>

      {loading && active && <p className="mt-2 text-xs text-slate-500">分割推理中…</p>}
      {error && <p className="mt-2 text-xs text-red-400">{error}</p>}
      {predictData?.message && (
        <p className="mt-2 text-xs text-slate-500">
          {predictData.backend} · {predictData.message}
        </p>
      )}
    </div>
  );
}
