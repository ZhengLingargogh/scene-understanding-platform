import { useEffect, useMemo, useState } from "react";
import { api } from "../../api/client";
import { formatModelLabel } from "../inference/inferenceUtils";
import type { CovisibilityRef, InferenceApiResult, MockDataset, RetrievalReference } from "../../types/inference";
import { INFERENCE_PIPELINE_STAGES, PIPELINE_STAGE_LABELS } from "../../types/inference";
import type { ModelItem, Scene } from "../../types";
import { formatTranslation } from "../../utils/coordinateFrames";

function placeholderThumb(hue: number, rank: number): string {
  const canvas = document.createElement("canvas");
  canvas.width = 120;
  canvas.height = 90;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";
  const g = ctx.createLinearGradient(0, 0, 120, 90);
  g.addColorStop(0, `hsl(${hue}, 55%, 32%)`);
  g.addColorStop(1, `hsl(${hue + 25}, 60%, 20%)`);
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 120, 90);
  ctx.strokeStyle = "rgba(34,211,238,0.55)";
  ctx.strokeRect(4, 4, 112, 82);
  ctx.fillStyle = "rgba(226,232,240,0.85)";
  ctx.font = "bold 11px sans-serif";
  ctx.fillText(`#${rank}`, 50, 48);
  return canvas.toDataURL();
}

function retrievalThumbnail(imagePath: string | null | undefined, hue: number, rank: number): string {
  if (imagePath) {
    return `/api/v1/media/file?path=${encodeURIComponent(imagePath)}`;
  }
  return placeholderThumb(hue, rank);
}

function modelsForRetrieval(models: ModelItem[]): ModelItem[] {
  return models.filter((m) => m.capabilities?.includes("image_retrieval"));
}

function modelsForMatching(models: ModelItem[]): ModelItem[] {
  return models.filter((m) => m.capabilities?.includes("image_matching"));
}

export interface QueryUploadPanelProps {
  onRunComplete: (result: InferenceApiResult, covisRefs: CovisibilityRef[]) => void;
  onRunStart?: () => void;
}

export default function QueryUploadPanel({ onRunComplete, onRunStart }: QueryUploadPanelProps) {
  const [models, setModels] = useState<ModelItem[]>([]);
  const [datasets, setDatasets] = useState<MockDataset[]>([]);
  const [scenes, setScenes] = useState<Scene[]>([]);
  const [modelId, setModelId] = useState("salad");
  const [matcherModelId, setMatcherModelId] = useState("lightglue");
  const [datasetId, setDatasetId] = useState("crossloc-nature-test");
  const [sceneId, setSceneId] = useState("nature");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [calibrationFile, setCalibrationFile] = useState<File | null>(null);
  const [gtPoseFile, setGtPoseFile] = useState<File | null>(null);
  const [focalLength, setFocalLength] = useState("");
  const [useManualFocal, setUseManualFocal] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasRun, setHasRun] = useState(false);
  const [lastResult, setLastResult] = useState<InferenceApiResult | null>(null);
  const [covisRefs, setCovisRefs] = useState<CovisibilityRef[]>([]);

  useEffect(() => {
    api.get<ModelItem[]>("/models").then((items) => {
      setModels(items);
      const retrievers = modelsForRetrieval(items);
      if (retrievers.some((m) => m.id === "salad")) {
        setModelId("salad");
      } else if (retrievers.length > 0) {
        setModelId(retrievers[0].id);
      }
      const matchers = modelsForMatching(items);
      if (matchers.some((m) => m.id === "lightglue")) {
        setMatcherModelId("lightglue");
      } else if (matchers.length > 0) {
        setMatcherModelId(matchers[0].id);
      }
    }).catch(() => undefined);
    api.get<MockDataset[]>("/inference/datasets").then(setDatasets).catch(() => undefined);
    api.get<Scene[]>("/scenes").then(setScenes).catch(() => undefined);
  }, []);

  useEffect(() => {
    const ds = datasets.find((d) => d.id === datasetId);
    if (ds?.scene_id) setSceneId(ds.scene_id);
  }, [datasetId, datasets]);

  const retrieverModels = modelsForRetrieval(models);
  const matcherModels = modelsForMatching(models);

  useEffect(() => {
    if (!imageFile) {
      setPreviewUrl(null);
      return undefined;
    }
    const url = URL.createObjectURL(imageFile);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [imageFile]);

  const canRun = Boolean(imageFile) && !loading;

  const translationText = useMemo(() => {
    if (!lastResult?.translation) return "—";
    return formatTranslation(lastResult.translation);
  }, [lastResult]);

  function refsFromPipeline(result: InferenceApiResult): CovisibilityRef[] {
    const apiRefs = result.pipeline?.image_retrieval?.references ?? [];
    return apiRefs.map((ref: RetrievalReference, i: number) => ({
      id: ref.id,
      label: ref.label,
      score: ref.score,
      thumbnail: retrievalThumbnail(ref.image_path, 195 + i * 14, i + 1),
      image_path: ref.image_path ?? undefined,
    }));
  }

  async function handleRun() {
    if (!imageFile) {
      setError("请先上传 Query 图像");
      return;
    }

    setLoading(true);
    setError(null);
    setHasRun(false);
    onRunStart?.();

    const form = new FormData();
    form.append("scene_id", sceneId);
    form.append("dataset_id", datasetId);
    form.append("model_id", modelId);
    form.append("matcher_model_id", matcherModelId);
    form.append("image", imageFile);
    form.append("pipeline_stages", JSON.stringify(INFERENCE_PIPELINE_STAGES));

    if (calibrationFile) {
      form.append("calibration_file", calibrationFile);
    } else if (useManualFocal && focalLength.trim()) {
      form.append("focal_length", focalLength.trim());
    }

    if (gtPoseFile) {
      form.append("gt_pose_file", gtPoseFile);
    }

    try {
      const response = await fetch("/api/v1/inference/run", {
        method: "POST",
        body: form,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data));
      }

      const result = data as InferenceApiResult;
      const refs = refsFromPipeline(result);

      setLastResult(result);
      setCovisRefs(refs);
      setHasRun(true);
      onRunComplete(result, refs);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <aside className="flex h-full w-[400px] shrink-0 flex-col border-r border-cyan-900/40 bg-slate-900/95 text-slate-100">
      <div className="border-b border-cyan-900/30 px-4 py-3">
        <h2 className="text-sm font-semibold tracking-wide text-cyan-300">Query 输入</h2>
        <p className="mt-0.5 text-xs text-slate-400">检索 → 匹配 → 场景坐标回归</p>
      </div>

      <div className="flex flex-1 flex-col gap-3 overflow-y-auto px-4 py-3">
        <label className="flex flex-col gap-1 text-xs text-slate-300">
          检索模型
          <select
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
            className="rounded-md border border-slate-600/60 bg-slate-800/80 px-2 py-2 text-sm text-slate-100 outline-none focus:border-cyan-500/60"
          >
            {retrieverModels.length === 0 ? (
              <>
                <option value="salad">SALAD</option>
                <option value="mock-retriever">Mock Retriever</option>
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

        <label className="flex flex-col gap-1 text-xs text-slate-300">
          匹配模型
          <select
            value={matcherModelId}
            onChange={(e) => setMatcherModelId(e.target.value)}
            className="rounded-md border border-slate-600/60 bg-slate-800/80 px-2 py-2 text-sm text-slate-100 outline-none focus:border-cyan-500/60"
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
        </label>

        <label className="flex flex-col gap-1 text-xs text-slate-300">
          场景
          <select
            value={sceneId}
            onChange={(e) => setSceneId(e.target.value)}
            className="rounded-md border border-slate-600/60 bg-slate-800/80 px-2 py-2 text-sm text-slate-100 outline-none focus:border-cyan-500/60"
          >
            {scenes.length === 0 ? (
              <option value="nature">nature</option>
            ) : (
              scenes.map((scene) => (
                <option key={scene.id} value={scene.id}>
                  {scene.name}
                </option>
              ))
            )}
          </select>
        </label>

        <label className="flex flex-col gap-1 text-xs text-slate-300">
          数据集（检索库）
          <select
            value={datasetId}
            onChange={(e) => setDatasetId(e.target.value)}
            className="rounded-md border border-slate-600/60 bg-slate-800/80 px-2 py-2 text-sm text-slate-100 outline-none focus:border-cyan-500/60"
          >
            {datasets.length === 0 ? (
              <option value="crossloc-nature-test">CrossLoc Nature Test</option>
            ) : (
              <>
                <optgroup label="CrossLoc（nature / natureoop / urban / urbanoop）">
                  {datasets
                    .filter((d) => (d as MockDataset & { family?: string }).family === "crossloc" || d.id.startsWith("crossloc"))
                    .map((d) => (
                      <option key={d.id} value={d.id}>
                        {d.name} ({d.image_count} 张)
                      </option>
                    ))}
                </optgroup>
                <optgroup label="UAVD4L（inTraj / outTraj）">
                  {datasets
                    .filter((d) => (d as MockDataset & { family?: string }).family === "uavd4l" || d.id.startsWith("uavd4l"))
                    .map((d) => (
                      <option key={d.id} value={d.id}>
                        {d.name} ({d.image_count} 张)
                      </option>
                    ))}
                </optgroup>
              </>
            )}
          </select>
        </label>

        <p className="text-[10px] text-slate-500">
          坐标回归固定使用 ACE（SCR 不受检索模型影响）
        </p>

        <label className="flex flex-col gap-1 text-xs text-slate-300">
          Query 图像
          <input
            type="file"
            accept="image/*"
            onChange={(e) => {
              setImageFile(e.target.files?.[0] ?? null);
              setHasRun(false);
            }}
            className="text-xs file:mr-2 file:rounded file:border-0 file:bg-cyan-700/80 file:px-2 file:py-1 file:text-xs file:text-white hover:file:bg-cyan-600"
          />
        </label>

        <label className="flex flex-col gap-1 text-xs text-slate-300">
          内参 calibration（可选）
          <input
            type="file"
            accept=".txt,text/plain"
            onChange={(e) => {
              setCalibrationFile(e.target.files?.[0] ?? null);
              if (e.target.files?.[0]) setUseManualFocal(false);
            }}
            className="text-xs file:mr-2 file:rounded file:border-0 file:bg-slate-700 file:px-2 file:py-1 file:text-xs file:text-white"
          />
        </label>

        <label className="flex items-center gap-2 text-xs text-slate-400">
          <input
            type="checkbox"
            checked={useManualFocal}
            disabled={!!calibrationFile}
            onChange={(e) => setUseManualFocal(e.target.checked)}
            className="accent-cyan-500"
          />
          手动焦距（无内参文件）
        </label>

        {useManualFocal && !calibrationFile && (
          <label className="flex flex-col gap-1 text-xs text-slate-300">
            focal_length
            <input
              type="number"
              step="any"
              placeholder="480"
              value={focalLength}
              onChange={(e) => setFocalLength(e.target.value)}
              className="rounded-md border border-slate-600/60 bg-slate-800/80 px-2 py-2 text-sm outline-none focus:border-cyan-500/60"
            />
          </label>
        )}

        <label className="flex flex-col gap-1 text-xs text-slate-300">
          GT 位姿 poses（可选）
          <input
            type="file"
            accept=".txt,text/plain"
            onChange={(e) => setGtPoseFile(e.target.files?.[0] ?? null)}
            className="text-xs file:mr-2 file:rounded file:border-0 file:bg-slate-700 file:px-2 file:py-1 file:text-xs file:text-white"
          />
        </label>

        <div className="overflow-hidden rounded-lg border border-cyan-800/40 bg-slate-950/60">
          {previewUrl ? (
            <img src={previewUrl} alt="Query preview" className="aspect-video w-full object-cover" />
          ) : (
            <div className="flex aspect-video items-center justify-center text-xs text-slate-500">
              上传后显示 2D 缩略图
            </div>
          )}
        </div>

        <button
          type="button"
          disabled={!canRun}
          onClick={handleRun}
          className="w-full rounded-lg bg-gradient-to-r from-cyan-600 to-blue-600 py-2.5 text-sm font-medium text-white shadow-lg shadow-cyan-900/30 transition hover:from-cyan-500 hover:to-blue-500 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {loading ? "推理中…" : "Run"}
        </button>

        {error && <p className="text-xs text-red-400">{error}</p>}

        <div className="min-h-[280px] rounded-lg border border-dashed border-slate-600/50 bg-slate-950/40 p-3">
          {!hasRun ? (
            <p className="text-center text-xs text-slate-500">
              Run 后显示：Top-8 检索结果、预测坐标、GT 误差
            </p>
          ) : (
            <div className="space-y-3">
              {lastResult?.pipeline && (
                <div>
                  <h3 className="text-xs font-medium uppercase tracking-wider text-cyan-400/90">
                    流水线阶段
                  </h3>
                  <ul className="mt-1 space-y-1">
                    {INFERENCE_PIPELINE_STAGES.map((stage) => {
                      const block = lastResult.pipeline?.[stage as keyof typeof lastResult.pipeline];
                      const status =
                        block && typeof block === "object" && "status" in block
                          ? String((block as { status?: string }).status ?? "—")
                          : "—";
                      return (
                        <li key={stage} className="flex items-center justify-between text-[11px]">
                          <span className="text-slate-300">{PIPELINE_STAGE_LABELS[stage]}</span>
                          <span className="rounded bg-slate-800 px-1.5 py-0.5 text-slate-400">{status}</span>
                        </li>
                      );
                    })}
                  </ul>
                </div>
              )}

              <div>
                <h3 className="text-xs font-medium uppercase tracking-wider text-cyan-400/90">
                  预测坐标 (t)
                </h3>
                <p className="mt-1 font-mono text-sm text-emerald-300">[{translationText}]</p>
                {lastResult?.gt_errors && (
                  <p className="mt-1 text-[11px] text-slate-400">
                    GT 误差 — 旋转 {lastResult.gt_errors.rotation_error_deg.toFixed(2)}° · 平移{" "}
                    {lastResult.gt_errors.translation_error_m.toFixed(2)} m
                  </p>
                )}
                {lastResult?.inlier_count != null && (
                  <p className="mt-0.5 text-[11px] text-slate-500">
                    PnP inliers: {lastResult.inlier_count} · backend: {lastResult.pose_backend}
                  </p>
                )}
              </div>

              <div>
                <h3 className="text-xs font-medium uppercase tracking-wider text-cyan-400/90">
                  检索 Top-8 共视参考图
                </h3>
                <ul className="mt-2 grid max-h-64 grid-cols-2 gap-2 overflow-y-auto">
                  {covisRefs.map((ref, idx) => (
                    <li
                      key={ref.id}
                      className="flex flex-col gap-1 rounded-md border border-slate-700/60 bg-slate-900/50 p-1.5"
                    >
                      <img
                        src={ref.thumbnail}
                        alt={ref.label}
                        className="aspect-[4/3] w-full rounded object-cover"
                      />
                      <p className="truncate text-[10px] text-slate-300" title={ref.label}>
                        #{idx + 1} {ref.label.split("·").pop()?.trim()}
                      </p>
                      <div className="flex items-center justify-between">
                        <div className="h-1 flex-1 overflow-hidden rounded-full bg-slate-700">
                          <div
                            className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-blue-500"
                            style={{ width: `${ref.score * 100}%` }}
                          />
                        </div>
                        <span className="ml-1 font-mono text-[10px] text-cyan-300">
                          {(ref.score * 100).toFixed(0)}%
                        </span>
                      </div>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}
        </div>
      </div>
    </aside>
  );
}
