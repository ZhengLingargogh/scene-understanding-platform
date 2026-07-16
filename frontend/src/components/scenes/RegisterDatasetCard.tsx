import { useState } from "react";
import { api } from "../../api/client";
import type { BrowseResponse, DatasetScanPreview, RegisterDatasetResponse } from "../../types/dataset";

interface RegisterDatasetCardProps {
  onRegistered: () => void;
}

export default function RegisterDatasetCard({ onRegistered }: RegisterDatasetCardProps) {
  const [rootPath, setRootPath] = useState("");
  const [preview, setPreview] = useState<DatasetScanPreview | null>(null);
  const [browse, setBrowse] = useState<BrowseResponse | null>(null);
  const [showBrowse, setShowBrowse] = useState(false);
  const [scanning, setScanning] = useState(false);
  const [registering, setRegistering] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function loadBrowse(path?: string) {
    const query = path ? `?path=${encodeURIComponent(path)}` : "";
    const response = await api.get<BrowseResponse>(`/scenes/datasets/browse${query}`);
    setBrowse(response);
    setShowBrowse(true);
  }

  async function openBrowse() {
    setError(null);
    try {
      await loadBrowse();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function handleScan() {
    if (!rootPath.trim()) {
      setError("请输入数据集根路径");
      return;
    }
    setScanning(true);
    setError(null);
    setPreview(null);
    try {
      const response = await api.post<DatasetScanPreview>("/scenes/datasets/scan", {
        root_path: rootPath.trim(),
      });
      setPreview(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setScanning(false);
    }
  }

  async function handleRegister() {
    if (!rootPath.trim()) {
      setError("请输入数据集根路径");
      return;
    }
    setRegistering(true);
    setError(null);
    try {
      await api.post<RegisterDatasetResponse>("/scenes/datasets/register", {
        root_path: rootPath.trim(),
      });
      setRootPath("");
      setPreview(null);
      setShowBrowse(false);
      onRegistered();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRegistering(false);
    }
  }

  function selectBrowsePath(path: string) {
    setRootPath(path);
    setShowBrowse(false);
    setPreview(null);
    setError(null);
  }

  return (
    <div className="card" style={{ minHeight: "180px" }}>
      <h3 style={{ margin: "0 0 0.75rem", fontSize: "1.05rem" }}>注册数据集</h3>
      <p className="muted" style={{ fontSize: "0.85rem", margin: "0 0 0.75rem" }}>
        选择数据集根目录（如 <code>.../Cambridge</code>），自动解析下级场景；每个场景需含
        train/test，且各有 calibration、poses、rgb。
      </p>

      <div style={{ display: "flex", gap: "0.5rem", marginBottom: "0.5rem" }}>
        <input
          type="text"
          value={rootPath}
          onChange={(event) => {
            setRootPath(event.target.value);
            setPreview(null);
          }}
          placeholder="/path/to/datasetroot"
          style={{ flex: 1, padding: "0.45rem 0.6rem", fontSize: "0.9rem" }}
        />
        <button type="button" className="btn-secondary" onClick={openBrowse}>
          浏览
        </button>
      </div>

      <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
        <button type="button" className="btn-secondary" disabled={scanning} onClick={handleScan}>
          {scanning ? "扫描中…" : "扫描预览"}
        </button>
        <button
          type="button"
          className="btn-primary"
          disabled={registering || (preview !== null && !preview.valid)}
          onClick={handleRegister}
        >
          {registering ? "注册中…" : "确认注册"}
        </button>
      </div>

      {error && (
        <p style={{ color: "#b91c1c", fontSize: "0.85rem", marginTop: "0.75rem", marginBottom: 0 }}>
          {error}
        </p>
      )}

      {preview && (
        <div style={{ marginTop: "0.75rem", fontSize: "0.85rem" }}>
          <p style={{ margin: "0 0 0.35rem" }}>
            预览: <strong>{preview.name}</strong> ({preview.family_id}) · {preview.scenes.length} 个场景
            {preview.valid ? (
              <span style={{ color: "#16a34a", marginLeft: "0.35rem" }}>可注册</span>
            ) : (
              <span style={{ color: "#b91c1c", marginLeft: "0.35rem" }}>不可注册</span>
            )}
          </p>
          {preview.warnings.length > 0 && (
            <ul className="muted" style={{ margin: 0, paddingLeft: "1.1rem" }}>
              {preview.warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          )}
        </div>
      )}

      {showBrowse && browse && (
        <div
          style={{
            marginTop: "0.75rem",
            border: "1px solid #e2e8f0",
            borderRadius: "8px",
            padding: "0.5rem",
            maxHeight: "200px",
            overflowY: "auto",
            fontSize: "0.85rem",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.35rem" }}>
            <code style={{ fontSize: "0.8rem" }}>{browse.path}</code>
            <button type="button" className="btn-secondary" style={{ fontSize: "0.75rem" }} onClick={() => setShowBrowse(false)}>
              关闭
            </button>
          </div>
          {browse.parent_path && (
            <button
              type="button"
              className="btn-secondary"
              style={{ width: "100%", marginBottom: "0.35rem", textAlign: "left", fontSize: "0.8rem" }}
              onClick={() => loadBrowse(browse.parent_path!)}
            >
              ..
            </button>
          )}
          {browse.entries
            .filter((entry) => entry.is_directory)
            .map((entry) => (
              <div key={entry.path} style={{ display: "flex", gap: "0.35rem", marginBottom: "0.25rem" }}>
                <button
                  type="button"
                  className="btn-secondary"
                  style={{ flex: 1, textAlign: "left", fontSize: "0.8rem" }}
                  onClick={() => loadBrowse(entry.path)}
                >
                  {entry.name}/
                </button>
                <button
                  type="button"
                  className="btn-primary"
                  style={{ fontSize: "0.75rem" }}
                  onClick={() => selectBrowsePath(entry.path)}
                >
                  选择
                </button>
              </div>
            ))}
        </div>
      )}
    </div>
  );
}
