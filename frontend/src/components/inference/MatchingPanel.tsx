import { useEffect, useState } from "react";
import { api } from "../../api/client";
import { modelsForStage, formatModelLabel, runInference } from "./inferenceUtils";
import type { InferenceApiResult, MatchingVisualizationData } from "../../types/inference";
import type { ModelItem } from "../../types";

const selectClass =
  "rounded-md border border-slate-600/60 bg-slate-800/80 px-2 py-2 text-sm text-slate-100 outline-none focus:border-cyan-500/60";

export interface MatchingPanelProps {
  onRunComplete: (data: MatchingVisualizationData) => void;
  onRunStart?: () => void;
  onRunEnd?: () => void;
}

export default function MatchingPanel({ onRunComplete, onRunStart, onRunEnd }: MatchingPanelProps) {
  const [models, setModels] = useState<ModelItem[]>([]);
  const [detectorModelId, setDetectorModelId] = useState("superpoint");
  const [matcherModelId, setMatcherModelId] = useState("lightglue");
  const [imageFile0, setImageFile0] = useState<File | null>(null);
  const [imageFile1, setImageFile1] = useState<File | null>(null);
  const [previewUrl0, setPreviewUrl0] = useState<string | null>(null);
  const [previewUrl1, setPreviewUrl1] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const detectorModels = modelsForStage(models, "keypoint_detection");
  const matcherModels = modelsForStage(models, "image_matching");

  useEffect(() => {
    api
      .get<ModelItem[]>("/models")
      .then((items) => {
        setModels(items);
        const detectors = modelsForStage(items, "keypoint_detection");
        const matchers = modelsForStage(items, "image_matching");
        if (detectors.some((m) => m.id === "superpoint")) setDetectorModelId("superpoint");
        else if (detectors.length > 0) setDetectorModelId(detectors[0].id);
        if (matchers.some((m) => m.id === "lightglue")) setMatcherModelId("lightglue");
        else if (matchers.length > 0) setMatcherModelId(matchers[0].id);
      })
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!imageFile0) {
      setPreviewUrl0(null);
      return undefined;
    }
    const url = URL.createObjectURL(imageFile0);
    setPreviewUrl0(url);
    return () => URL.revokeObjectURL(url);
  }, [imageFile0]);

  useEffect(() => {
    if (!imageFile1) {
      setPreviewUrl1(null);
      return undefined;
    }
    const url = URL.createObjectURL(imageFile1);
    setPreviewUrl1(url);
    return () => URL.revokeObjectURL(url);
  }, [imageFile1]);

  async function handleRun() {
    if (!imageFile0 || !imageFile1) {
      setError("请上传两张 Query 图像");
      return;
    }
    if (!previewUrl0 || !previewUrl1) {
      setError("图像预览未就绪");
      return;
    }
    if (!detectorModelId || !matcherModelId) {
      setError("请选择检测模型与匹配模型");
      return;
    }

    setLoading(true);
    setError(null);
    onRunStart?.();

    const form = new FormData();
    form.append("model_id", matcherModelId);
    form.append("detector_model_id", detectorModelId);
    form.append("matcher_model_id", matcherModelId);
    form.append("image", imageFile0);
    form.append("reference_image", imageFile1);
    form.append("pipeline_stages", JSON.stringify(["image_matching"]));

    try {
      const result: InferenceApiResult = await runInference(form);
      const matching = result.pipeline?.image_matching;
      if (!matching || matching.status !== "completed") {
        throw new Error(matching?.message ?? "匹配失败");
      }

      onRunComplete({
        keypoints0: matching.keypoints0 ?? [],
        keypoints1: matching.keypoints1 ?? [],
        matches: matching.matches ?? [],
        image0Url: previewUrl0,
        image1Url: previewUrl1,
        matchCount: matching.match_count ?? matching.matches?.length ?? 0,
        message: matching.message,
        detectorModelId: result.detector_model_id ?? detectorModelId,
        matcherModelId: result.matcher_model_id ?? matcherModelId,
      });
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
        <h2 className="text-sm font-semibold tracking-wide text-cyan-300">图像匹配</h2>
        <p className="mt-0.5 text-xs text-slate-400">检测模型 → 匹配模型 · 双图稀疏特征连线</p>
      </div>

      <div className="flex flex-1 flex-col gap-3 overflow-y-auto px-4 py-3">
        <label className="flex flex-col gap-1 text-xs text-slate-300">
          检测模型
          <select
            value={detectorModelId}
            onChange={(e) => setDetectorModelId(e.target.value)}
            className={selectClass}
          >
            {detectorModels.length === 0 ? (
              <option value="superpoint">SuperPoint</option>
            ) : (
              detectorModels.map((m) => (
                <option key={m.id} value={m.id}>
                  {formatModelLabel(m)}
                </option>
              ))
            )}
          </select>
          <span className="text-[10px] text-slate-500">在两张图上提取关键点与局部描述子</span>
        </label>

        <label className="flex flex-col gap-1 text-xs text-slate-300">
          匹配模型
          <select
            value={matcherModelId}
            onChange={(e) => setMatcherModelId(e.target.value)}
            className={selectClass}
          >
            {matcherModels.length === 0 ? (
              <option value="lightglue">LightGlue</option>
            ) : (
              matcherModels.map((m) => (
                <option key={m.id} value={m.id}>
                  {formatModelLabel(m)}
                </option>
              ))
            )}
          </select>
          <span className="text-[10px] text-slate-500">根据描述子建立对应关系并绘制连线</span>
        </label>

        <label className="flex flex-col gap-1 text-xs text-slate-300">
          Query 图像 1
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setImageFile0(e.target.files?.[0] ?? null)}
            className="text-xs text-slate-400 file:mr-2 file:rounded file:border-0 file:bg-cyan-900/50 file:px-2 file:py-1 file:text-cyan-100"
          />
        </label>

        <label className="flex flex-col gap-1 text-xs text-slate-300">
          Query 图像 2
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setImageFile1(e.target.files?.[0] ?? null)}
            className="text-xs text-slate-400 file:mr-2 file:rounded file:border-0 file:bg-cyan-900/50 file:px-2 file:py-1 file:text-cyan-100"
          />
        </label>

        <div className="grid grid-cols-2 gap-2">
          <div className="overflow-hidden rounded-md border border-slate-700/60 bg-slate-950/80">
            {previewUrl0 ? (
              <img src={previewUrl0} alt="Query 1" className="h-28 w-full object-cover" />
            ) : (
              <div className="flex h-28 items-center justify-center text-[10px] text-slate-500">Query 1 预览</div>
            )}
          </div>
          <div className="overflow-hidden rounded-md border border-slate-700/60 bg-slate-950/80">
            {previewUrl1 ? (
              <img src={previewUrl1} alt="Query 2" className="h-28 w-full object-cover" />
            ) : (
              <div className="flex h-28 items-center justify-center text-[10px] text-slate-500">Query 2 预览</div>
            )}
          </div>
        </div>

        <button
          type="button"
          onClick={handleRun}
          disabled={loading || !imageFile0 || !imageFile1 || !detectorModelId || !matcherModelId}
          className="mt-1 rounded-md bg-cyan-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-cyan-500 disabled:opacity-50"
        >
          {loading ? "匹配中…" : "Run"}
        </button>

        {error && <p className="text-xs text-red-400">{error}</p>}
      </div>
    </aside>
  );
}
