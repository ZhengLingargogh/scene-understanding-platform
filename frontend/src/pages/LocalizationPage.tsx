import { useEffect, useState } from "react";
import type { LocalizationResult, LocalizationScene } from "../types";

const SCENE_OPTIONS = [
  { id: "nature", label: "Nature (nature_base.pt)" },
  { id: "urban", label: "Urban (urban_base.pt)" },
  { id: "natureoop", label: "Nature OOP" },
  { id: "urbanoop", label: "Urban OOP" },
  { id: "inTraj", label: "In Trajectory" },
  { id: "outTraj", label: "Out Trajectory" },
];

export default function LocalizationPage() {
  const [tasks, setTasks] = useState<LocalizationResult[]>([]);
  const [scenes, setScenes] = useState<LocalizationScene[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [lastResult, setLastResult] = useState<Record<string, unknown> | null>(null);

  const [sceneId, setSceneId] = useState("nature");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [calibrationFile, setCalibrationFile] = useState<File | null>(null);
  const [gtPoseFile, setGtPoseFile] = useState<File | null>(null);
  const [focalLength, setFocalLength] = useState("");
  const [useManualFocal, setUseManualFocal] = useState(false);

  useEffect(() => {
    fetch("/api/v1/localization/scenes")
      .then((r) => r.json())
      .then(setScenes)
      .catch(() => undefined);

    fetch("/api/v1/localization")
      .then((r) => r.json())
      .then(setTasks)
      .catch((err: Error) => setError(err.message));
  }, []);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!imageFile) {
      setError("请上传 Query 图像");
      return;
    }

    setLoading(true);
    setError(null);
    setLastResult(null);

    const form = new FormData();
    form.append("scene_id", sceneId);
    form.append("model_id", "ace");
    form.append("image", imageFile);

    if (calibrationFile) {
      form.append("calibration_file", calibrationFile);
    } else if (useManualFocal && focalLength.trim()) {
      form.append("focal_length", focalLength.trim());
    }

    if (gtPoseFile) {
      form.append("gt_pose_file", gtPoseFile);
    }

    try {
      const response = await fetch("/api/v1/localization/run", {
        method: "POST",
        body: form,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data));
      }
      setLastResult(data);
      const listResp = await fetch("/api/v1/localization");
      if (listResp.ok) setTasks(await listResp.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <section>
      <h2 className="page-title">单张图像定位</h2>
      <p className="muted">
        上传 CrossLoc 格式数据：rgb 图像 + calibration 内参文件 + poses 位姿文件（后两者可选）。
      </p>

      <form className="card loc-form" onSubmit={handleSubmit}>
        <label>
          场景
          <select value={sceneId} onChange={(e) => setSceneId(e.target.value)}>
            {SCENE_OPTIONS.map((s) => (
              <option key={s.id} value={s.id}>
                {s.label}
              </option>
            ))}
          </select>
        </label>

        <label>
          Query 图像 <span className="muted">(rgb/)</span>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setImageFile(e.target.files?.[0] ?? null)}
          />
        </label>

        <label>
          内参文件 <span className="muted">(calibration/*.txt，可选)</span>
          <input
            type="file"
            accept=".txt,text/plain"
            onChange={(e) => {
              setCalibrationFile(e.target.files?.[0] ?? null);
              if (e.target.files?.[0]) setUseManualFocal(false);
            }}
          />
          <span className="hint">
            单焦距如 <code>480.0</code>，或 3×3 内参矩阵（与 CrossLoc 数据集格式一致）
          </span>
        </label>

        <label className="checkbox-row">
          <input
            type="checkbox"
            checked={useManualFocal}
            disabled={!!calibrationFile}
            onChange={(e) => setUseManualFocal(e.target.checked)}
          />
          手动输入焦距（无内参文件时）
        </label>

        {useManualFocal && !calibrationFile && (
          <label>
            焦距 focal_length（原图像素）
            <input
              type="number"
              step="any"
              placeholder="例如 480"
              value={focalLength}
              onChange={(e) => setFocalLength(e.target.value)}
            />
          </label>
        )}

        <label>
          GT 位姿文件 <span className="muted">(poses/*.txt，可选)</span>
          <input
            type="file"
            accept=".txt,text/plain"
            onChange={(e) => setGtPoseFile(e.target.files?.[0] ?? null)}
          />
          <span className="hint">4×4 位姿矩阵文本，用于计算旋转/平移误差</span>
        </label>

        <button type="submit" disabled={loading}>
          {loading ? "定位中…" : "运行定位"}
        </button>
      </form>

      {error && <p className="error-text">{error}</p>}

      {lastResult && (
        <div className="card">
          <h3>定位结果</h3>
          <p className="muted">
            backend: {String(lastResult.pose_backend)} · inliers: {String(lastResult.inlier_count)}
          </p>
          {"gt_errors" in lastResult && lastResult.gt_errors != null && (
            <p>
              GT 误差 — 旋转:{" "}
              {(lastResult.gt_errors as { rotation_error_deg: number }).rotation_error_deg.toFixed(2)}°，
              平移:{" "}
              {(lastResult.gt_errors as { translation_error_m: number }).translation_error_m.toFixed(2)} m
            </p>
          )}
          <pre className="result-pre">{JSON.stringify(lastResult, null, 2)}</pre>
        </div>
      )}

      <div className="card">
        <h3>已注册场景权重</h3>
        {scenes.length === 0 ? (
          <p className="muted">加载中…</p>
        ) : (
          <ul className="scene-list">
            {scenes.map((s) => (
              <li key={`${s.scene_id}-${s.weights_file}`}>
                {s.scene_id} → {s.weights_file}
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="card">
        <h3>历史任务</h3>
        {tasks.length === 0 ? (
          <p className="muted">暂无任务记录。</p>
        ) : (
          <ul>
            {tasks.map((task) => (
              <li key={task.id}>
                {task.id.slice(0, 8)}… — {task.status} — scene: {task.scene_id}
              </li>
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}
