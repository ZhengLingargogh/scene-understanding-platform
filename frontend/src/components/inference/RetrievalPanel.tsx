import { useEffect, useState } from "react";
import { api } from "../../api/client";
import DatasetSceneFields from "./DatasetSceneFields";
import { modelsForStage, formatModelLabel, refsFromPipeline, runInference } from "./inferenceUtils";
import { useDatasetSceneSelection } from "../../hooks/useDatasetSceneSelection";
import type { CovisibilityRef } from "../../types/inference";
import type { ModelItem } from "../../types";

export interface RetrievalPanelProps {
  onRunComplete: (refs: CovisibilityRef[]) => void;
  onRunStart?: () => void;
  onRunEnd?: () => void;
}

const selectClass =
  "rounded-md border border-slate-600/60 bg-slate-800/80 px-2 py-2 text-sm text-slate-100 outline-none focus:border-cyan-500/60";

export default function RetrievalPanel({ onRunComplete, onRunStart, onRunEnd }: RetrievalPanelProps) {
  const [models, setModels] = useState<ModelItem[]>([]);
  const [modelId, setModelId] = useState("salad");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const selection = useDatasetSceneSelection({ defaultSplit: "train", showSplit: true });
  const retrieverModels = modelsForStage(models, "image_retrieval");

  useEffect(() => {
    api.get<ModelItem[]>("/models").then((items) => {
      setModels(items);
      const retrievers = modelsForStage(items, "image_retrieval");
      if (retrievers.some((m) => m.id === "salad")) setModelId("salad");
      else if (retrievers.length > 0) setModelId(retrievers[0].id);
    }).catch(() => undefined);
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
    if (!imageFile) {
      setError("请先上传 Query 图像");
      return;
    }

    if (!selection.splitInfo) {
      setError("请先选择数据集、场景与检索范围");
      return;
    }

    setLoading(true);
    setError(null);
    onRunStart?.();

    const form = new FormData();
    form.append("scene_id", selection.sceneId);
    form.append("dataset_id", selection.splitInfo.dataset_id);
    form.append("model_id", modelId);
    form.append("image", imageFile);
    form.append("pipeline_stages", JSON.stringify(["image_retrieval"]));

    try {
      const result = await runInference(form);
      const refs = refsFromPipeline(result);
      onRunComplete(refs);
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
        <h2 className="text-sm font-semibold tracking-wide text-cyan-300">图像检索</h2>
        <p className="mt-0.5 text-xs text-slate-400">选择检索模型与场景 · Top-8 显示于右侧</p>
      </div>

      <div className="flex flex-1 flex-col gap-3 overflow-y-auto px-4 py-3">
        <label className="flex flex-col gap-1 text-xs text-slate-300">
          检索模型
          <select value={modelId} onChange={(e) => setModelId(e.target.value)} className={selectClass}>
            {retrieverModels.length === 0 ? (
              <>
                <option value="salad">SALAD</option>
                <option value="netvlad">NetVLAD</option>
              </>
            ) : (
              retrieverModels.map((m) => (
                <option key={m.id} value={m.id}>
                  {formatModelLabel(m)}
                </option>
              ))
            )}
          </select>
        </label>

        <DatasetSceneFields
          catalog={selection.catalog}
          familyId={selection.familyId}
          sceneId={selection.sceneId}
          split={selection.split}
          showSplit
          splitLabel="检索范围"
          onFamilyChange={selection.setFamilyId}
          onSceneChange={selection.setSceneId}
          onSplitChange={selection.setSplit}
          className={selectClass}
          disabled={selection.loading}
        />

        {selection.error && <p className="text-xs text-amber-400">数据集加载失败: {selection.error}</p>}

        <label className="flex flex-col gap-1 text-xs text-slate-300">
          Query 图像
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setImageFile(e.target.files?.[0] ?? null)}
            className="text-xs file:mr-2 file:rounded file:border-0 file:bg-cyan-700/80 file:px-2 file:py-1 file:text-xs file:text-white"
          />
        </label>

        <div className="overflow-hidden rounded-lg border border-cyan-800/40 bg-slate-950/60">
          {previewUrl ? (
            <img src={previewUrl} alt="Query" className="aspect-video w-full object-cover" />
          ) : (
            <div className="flex aspect-video items-center justify-center text-xs text-slate-500">Query 预览</div>
          )}
        </div>

        <button
          type="button"
          disabled={!imageFile || loading || selection.loading || !selection.splitInfo}
          onClick={handleRun}
          className="w-full rounded-lg bg-gradient-to-r from-cyan-600 to-blue-600 py-2.5 text-sm font-medium text-white disabled:opacity-40"
        >
          {loading ? "检索中…" : "Run"}
        </button>

        {error && <p className="text-xs text-red-400">{error}</p>}
      </div>
    </aside>
  );
}
