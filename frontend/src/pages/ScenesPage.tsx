import { useCallback, useEffect, useState } from "react";
import { api } from "../api/client";
import DatasetFamilyCard from "../components/scenes/DatasetFamilyCard";
import RegisterDatasetCard from "../components/scenes/RegisterDatasetCard";
import type { Scene } from "../types";
import type { DatasetCatalog } from "../types/dataset";

const STATUS_LABELS: Record<string, string> = {
  unloaded: "未加载索引",
  loading: "加载中",
  loaded: "已加载",
  error: "错误",
};

export default function ScenesPage() {
  const [catalog, setCatalog] = useState<DatasetCatalog[]>([]);
  const [scenes, setScenes] = useState<Scene[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [deletingFamilyId, setDeletingFamilyId] = useState<string | null>(null);

  const loadData = useCallback(() => {
    return Promise.all([
      api.get<DatasetCatalog[]>("/scenes/catalog"),
      api.get<Scene[]>("/scenes"),
    ])
      .then(([catalogData, sceneData]) => {
        setCatalog(catalogData);
        setScenes(sceneData);
        setError(null);
      })
      .catch((err: Error) => setError(err.message));
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  async function handleDeleteFamily(familyId: string) {
    setDeletingFamilyId(familyId);
    try {
      await api.delete(`/scenes/datasets/${familyId}`);
      await loadData();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setDeletingFamilyId(null);
    }
  }

  return (
    <section>
      <h2 className="page-title">场景管理</h2>
      <p className="muted">注册及删除数据集</p>
      {error && <p className="muted">加载失败: {error}</p>}

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))",
          gap: "1rem",
          marginBottom: "1rem",
        }}
      >
        {catalog.map((family) => (
          <DatasetFamilyCard
            key={family.id}
            family={family}
            scenes={scenes}
            onDelete={family.is_builtin ? undefined : handleDeleteFamily}
            deleting={deletingFamilyId === family.id}
          />
        ))}
        <RegisterDatasetCard onRegistered={loadData} />
      </div>

      <div className="card">
        <h3 style={{ marginTop: 0, fontSize: "1rem" }}>已注册场景（SQLite）</h3>
        {scenes.length === 0 ? (
          <p className="muted">暂无场景记录，服务启动时会自动从数据集目录同步。</p>
        ) : (
          <ul>
            {scenes.map((scene) => (
              <li key={scene.id} style={{ marginBottom: "0.65rem" }}>
                <strong>{scene.name}</strong>{" "}
                <span className="muted">
                  {scene.dataset_family && scene.scene_slug && (
                    <>
                      [{scene.dataset_family} / {scene.scene_slug}] ·{" "}
                    </>
                  )}
                  ({scene.id.slice(0, 8)}…) · {STATUS_LABELS[scene.status ?? "unloaded"] ?? scene.status}
                </span>
                {scene.reference_images_dir && (
                  <div className="muted" style={{ fontSize: "0.85rem" }}>
                    参考图: {scene.reference_images_dir}
                  </div>
                )}
                {scene.feature_index_path && (
                  <div className="muted" style={{ fontSize: "0.85rem" }}>
                    特征索引: {scene.feature_index_path}
                  </div>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}
