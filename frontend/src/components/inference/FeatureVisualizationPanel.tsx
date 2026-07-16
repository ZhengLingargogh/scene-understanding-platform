import { useEffect, useState } from "react";
import { api } from "../../api/client";
import { modelsForStage, formatModelLabel } from "./inferenceUtils";
import { runFeatureVisualization } from "./featureVisualizationUtils";
import type { FeatureVisualizationData } from "../../types/inference";
import type { ModelItem } from "../../types";

const selectClass =
  "rounded-md border border-slate-600/60 bg-slate-800/80 px-2 py-2 text-sm text-slate-100 outline-none focus:border-cyan-500/60";

export interface FeatureVisualizationPanelProps {
  onRunComplete: (data: FeatureVisualizationData) => void;
  onRunStart?: () => void;
  onRunEnd?: () => void;
}

export default function FeatureVisualizationPanel({
  onRunComplete,
  onRunStart,
  onRunEnd,
}: FeatureVisualizationPanelProps) {
  const [models, setModels] = useState<ModelItem[]>([]);
  const [modelId, setModelId] = useState("superpoint");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const keypointModels = modelsForStage(models, "keypoint_detection");

  useEffect(() => {
    api
      .get<ModelItem[]>("/models")
      .then((items) => {
        setModels(items);
        const detectors = modelsForStage(items, "keypoint_detection");
        if (detectors.some((m) => m.id === "superpoint")) setModelId("superpoint");
        else if (detectors.length > 0) setModelId(detectors[0].id);
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
      const data = await runFeatureVisualization(imageFile, previewUrl, modelId);
      onRunComplete(data);
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
        <h2 className="text-sm font-semibold tracking-wide text-cyan-300">可视特征</h2>
        <p className="mt-0.5 text-xs text-slate-400">Feature Visualization · 关键点与响应热力图</p>
      </div>

      <div className="flex flex-1 flex-col gap-3 overflow-y-auto px-4 py-3">
        <label className="flex flex-col gap-1 text-xs text-slate-300">
          提取模型
          <select value={modelId} onChange={(e) => setModelId(e.target.value)} className={selectClass}>
            {keypointModels.length === 0 ? (
              <>
                <option value="superpoint">SuperPoint</option>
                <option value="sift">SIFT</option>
              </>
            ) : (
              keypointModels.map((m) => (
                <option key={m.id} value={m.id}>
                  {formatModelLabel(m)}
                </option>
              ))
            )}
          </select>
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
          {loading ? "提取中…" : "Run"}
        </button>

        {error && <p className="text-xs text-red-400">{error}</p>}
      </div>
    </aside>
  );
}
