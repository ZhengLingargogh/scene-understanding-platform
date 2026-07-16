import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import DatasetSceneFields from "../components/inference/DatasetSceneFields";
import InferenceShell from "../components/inference/InferenceShell";
import { formatModelLabel, modelsForStage } from "../components/inference/inferenceUtils";
import { useDatasetSceneSelection } from "../hooks/useDatasetSceneSelection";
import type { ModelItem } from "../types";
import type { SemanticSegmentationJob } from "../types/inference";

const selectClass =
  "rounded-md border border-slate-600/60 bg-slate-800/80 px-2 py-2 text-sm text-slate-100 outline-none focus:border-cyan-500/60";

function defaultOutputPath(datasetId: string): string {
  return `data/segmentation/${datasetId}`;
}

function modelsForSemanticSegmentation(models: ModelItem[]): ModelItem[] {
  return modelsForStage(models, "semantic_segmentation");
}

export default function SceneInferenceSemanticSegmentationView() {
  const [models, setModels] = useState<ModelItem[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [modelId, setModelId] = useState("sam");
  const [outputPath, setOutputPath] = useState("data/segmentation/crossloc-nature-train");
  const [running, setRunning] = useState(false);
  const [job, setJob] = useState<SemanticSegmentationJob | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const selection = useDatasetSceneSelection({
    defaultSplit: "train",
    showSplit: true,
  });
  const segmenterModels = modelsForSemanticSegmentation(models);

  useEffect(() => {
    api
      .get<ModelItem[]>("/models")
      .then((items) => {
        setModels(items);
        const segmenters = modelsForSemanticSegmentation(items);
        if (segmenters.some((model) => model.id === "sam")) setModelId("sam");
        else if (segmenters.length > 0) setModelId(segmenters[0].id);
      })
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    if (selection.splitInfo) {
      setOutputPath(defaultOutputPath(selection.splitInfo.dataset_id));
    }
  }, [selection.splitInfo]);

  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  async function pollJob(jobId: string) {
    const response = await fetch(`/api/v1/benchmarks/semantic-segmentation/${jobId}`);
    if (!response.ok) throw new Error("进度查询失败");
    const data = (await response.json()) as SemanticSegmentationJob;
    setJob(data);
    if (data.status === "completed" || data.status === "failed") {
      if (pollRef.current) clearInterval(pollRef.current);
      pollRef.current = null;
      setRunning(false);
    }
  }

  async function handleRun() {
    if (!selection.splitInfo) {
      setError("请先选择数据集、场景与下级划分");
      return;
    }

    setError(null);
    setJob(null);
    setRunning(true);

    try {
      const response = await api.post<SemanticSegmentationJob>("/benchmarks/semantic-segmentation/run", {
        model_id: modelId,
        dataset_id: selection.splitInfo.dataset_id,
        output_path: outputPath,
      });

      setJob(response);
      pollRef.current = setInterval(() => {
        pollJob(response.job_id).catch((pollError: Error) => {
          setError(pollError.message);
          setRunning(false);
          if (pollRef.current) clearInterval(pollRef.current);
        });
      }, 400);
    } catch (runError) {
      setError(runError instanceof Error ? runError.message : String(runError));
      setRunning(false);
    }
  }

  const isComplete = job?.status === "completed";
  const isFailed = job?.status === "failed";

  return (
    <InferenceShell
      title="语义分割"
      subtitle="选择数据集与场景 · SAM 批量语义分割"
      badge="Seg"
    >
      <aside className="flex h-full w-[400px] shrink-0 flex-col border-r border-cyan-900/40 bg-slate-900/95 text-slate-100">
        <div className="flex flex-1 flex-col gap-3 overflow-y-auto px-4 py-4">
          <DatasetSceneFields
            catalog={selection.catalog}
            familyId={selection.familyId}
            sceneId={selection.sceneId}
            split={selection.split}
            showSplit
            onFamilyChange={selection.setFamilyId}
            onSceneChange={selection.setSceneId}
            onSplitChange={selection.setSplit}
            className={selectClass}
            disabled={selection.loading}
          />

          {selection.splitInfo && (
            <p className="text-[11px] leading-relaxed text-slate-500">
              输入图像目录（自动）: {selection.splitInfo.rgb_dir}
            </p>
          )}

          {selection.error && (
            <p className="text-xs text-amber-400">数据集加载失败: {selection.error}</p>
          )}

          <label className="flex flex-col gap-1 text-xs text-slate-300">
            模型
            <select value={modelId} onChange={(event) => setModelId(event.target.value)} className={selectClass}>
              {segmenterModels.length === 0 ? (
                <option value="sam">SAM</option>
              ) : (
                segmenterModels.map((model) => (
                  <option key={model.id} value={model.id}>
                    {formatModelLabel(model)}
                  </option>
                ))
              )}
            </select>
          </label>

          <label className="flex flex-col gap-1 text-xs text-slate-300">
            输出路径
            <input
              type="text"
              value={outputPath}
              onChange={(event) => setOutputPath(event.target.value)}
              className={selectClass}
              placeholder="data/segmentation/..."
            />
          </label>

          <button
            type="button"
            onClick={handleRun}
            disabled={running || !selection.splitInfo || selection.loading}
            className="mt-1 rounded-md bg-cyan-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-cyan-500 disabled:opacity-50"
          >
            {running ? "分割中…" : "Run"}
          </button>

          {error && <p className="text-xs text-red-400">{error}</p>}
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col p-6 text-slate-200">
        <h2 className="text-sm font-semibold text-cyan-200">运行状态</h2>
        <p className="mt-1 text-xs text-slate-400">
          自动掩码生成，每张图保存最大区域掩码 PNG 及 manifest.json。
        </p>

        {(running || job) && (
          <div style={{ marginTop: "1.5rem", maxWidth: "480px" }}>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "0.5rem",
                marginBottom: "0.35rem",
              }}
            >
              <span style={{ fontSize: "0.9rem" }}>语义分割进度</span>
              {isComplete && (
                <span style={{ color: "#4ade80", fontWeight: 600 }}>✓ 成功已保存</span>
              )}
              {isFailed && <span style={{ color: "#f87171", fontWeight: 600 }}>失败</span>}
            </div>
            <div
              style={{
                height: "10px",
                borderRadius: "6px",
                background: "#1e293b",
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  height: "100%",
                  width: `${job?.progress ?? 0}%`,
                  background: isComplete
                    ? "linear-gradient(90deg, #22c55e, #16a34a)"
                    : isFailed
                      ? "linear-gradient(90deg, #ef4444, #dc2626)"
                      : "linear-gradient(90deg, #2563eb, #3b82f6)",
                  transition: "width 0.3s ease",
                }}
              />
            </div>
            <p className="muted" style={{ fontSize: "0.85rem", marginTop: "0.35rem", color: "#94a3b8" }}>
              {job?.message ?? "等待开始…"}
              {job && (
                <>
                  {" "}
                  ({job.processed}/{job.total} · {job.progress}%)
                </>
              )}
            </p>
            {isComplete && (
              <p style={{ color: "#4ade80", fontSize: "0.9rem", margin: "0.75rem 0 0" }}>
                ✓ 成功已保存至: {job?.output_path}
              </p>
            )}
            {isFailed && job?.message && (
              <p style={{ color: "#f87171", fontSize: "0.9rem", margin: "0.75rem 0 0" }}>{job.message}</p>
            )}
          </div>
        )}

        {!running && !job && (
          <p className="mt-8 text-sm text-slate-500">配置左侧参数后点击 Run 开始语义分割。</p>
        )}
      </div>
    </InferenceShell>
  );
}
