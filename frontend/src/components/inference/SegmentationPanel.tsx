import { useEffect, useState } from "react";
import { api } from "../../api/client";
import { modelsForStage, formatModelLabel } from "./inferenceUtils";
import { startSegmentationSession } from "./segmentationUtils";
import type { SegmentationSessionData } from "../../types/inference";
import type { ModelItem } from "../../types";

const selectClass =
  "rounded-md border border-slate-600/60 bg-slate-800/80 px-2 py-2 text-sm text-slate-100 outline-none focus:border-cyan-500/60";

export interface SegmentationPanelProps {
  onRunComplete: (data: SegmentationSessionData) => void;
  onRunStart?: () => void;
  onRunEnd?: () => void;
}

export default function SegmentationPanel({ onRunComplete, onRunStart, onRunEnd }: SegmentationPanelProps) {
  const [models, setModels] = useState<ModelItem[]>([]);
  const [modelId, setModelId] = useState("sam");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const segmenterModels = modelsForStage(models, "segmentation");

  useEffect(() => {
    api
      .get<ModelItem[]>("/models")
      .then((items) => {
        setModels(items);
        const segmenters = modelsForStage(items, "segmentation");
        if (segmenters.some((m) => m.id === "sam")) setModelId("sam");
        else if (segmenters.length > 0) setModelId(segmenters[0].id);
      })
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!imageFile) {
      setPreviewUrl(null);
      return undefined;
    }
    const url = URL.createObjectURL(imageFile);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [imageFile]);

  async function handleRun() {
    if (!imageFile || !previewUrl) {
      setError("请先上传 Query 图像");
      return;
    }

    setLoading(true);
    setError(null);
    onRunStart?.();

    try {
      const session = await startSegmentationSession(imageFile, modelId, previewUrl);
      onRunComplete(session);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
      onRunEnd?.();
    }
  }

  return (
    <aside className="flex h-full w-[400px] shrink-0 flex-col border-r border-cyan-900/40 bg-slate-900/95 text-slate-100">
      <div className="border-b border-cyan-900/30 px-4 py-3">
        <h2 className="text-sm font-semibold tracking-wide text-cyan-300">交互分割</h2>
        <p className="mt-0.5 text-xs text-slate-400">点提示分割 · Run 后进入交互模式</p>
      </div>

      <div className="flex flex-1 flex-col gap-3 overflow-y-auto px-4 py-3">
        <label className="flex flex-col gap-1 text-xs text-slate-300">
          分割模型
          <select value={modelId} onChange={(e) => setModelId(e.target.value)} className={selectClass}>
            {segmenterModels.length === 0 ? (
              <option value="sam">SAM</option>
            ) : (
              segmenterModels.map((m) => (
                <option key={m.id} value={m.id}>
                  {formatModelLabel(m)}
                </option>
              ))
            )}
          </select>
          <span className="text-[10px] text-slate-500">默认 SAM ViT-B</span>
        </label>

        <label className="flex flex-col gap-1 text-xs text-slate-300">
          Query 图像
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setImageFile(e.target.files?.[0] ?? null)}
            className="text-xs text-slate-400 file:mr-2 file:rounded file:border-0 file:bg-cyan-900/50 file:px-2 file:py-1 file:text-cyan-100"
          />
        </label>

        <div className="overflow-hidden rounded-md border border-slate-700/60 bg-slate-950/80">
          {previewUrl ? (
            <img src={previewUrl} alt="Query preview" className="h-40 w-full object-contain bg-slate-950" />
          ) : (
            <div className="flex h-40 items-center justify-center text-xs text-slate-500">Query 图像预览</div>
          )}
        </div>

        <button
          type="button"
          onClick={handleRun}
          disabled={loading || !imageFile}
          className="mt-1 rounded-md bg-cyan-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-cyan-500 disabled:opacity-50"
        >
          {loading ? "准备中…" : "Run"}
        </button>

        {error && <p className="text-xs text-red-400">{error}</p>}
      </div>
    </aside>
  );
}
