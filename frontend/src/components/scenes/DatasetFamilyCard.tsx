import { useState } from "react";
import type { DatasetCatalog } from "../../types/dataset";
import type { Scene } from "../../types";

interface DatasetFamilyCardProps {
  family: DatasetCatalog;
  scenes: Scene[];
  onDelete?: (familyId: string) => void;
  deleting?: boolean;
}

export default function DatasetFamilyCard({
  family,
  scenes,
  onDelete,
  deleting = false,
}: DatasetFamilyCardProps) {
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const registered = scenes.filter((scene) => scene.dataset_family === family.id);
  const expected = family.scenes.length;
  const allRegistered = registered.length >= expected && expected > 0;

  function handleConfirmDelete() {
    if (!onDelete) return;
    onDelete(family.id);
    setShowDeleteConfirm(false);
  }

  return (
    <>
      <div className="card" style={{ position: "relative" }}>
        {!family.is_builtin && onDelete && (
          <button
            type="button"
            aria-label="删除数据集"
            title="删除数据集"
            disabled={deleting}
            onClick={() => setShowDeleteConfirm(true)}
            style={{
              position: "absolute",
              top: "0.65rem",
              right: "0.65rem",
              width: "1.5rem",
              height: "1.5rem",
              borderRadius: "50%",
              border: "1px solid #cbd5e1",
              background: "#fff",
              color: "#64748b",
              fontSize: "0.95rem",
              lineHeight: 1,
              cursor: deleting ? "not-allowed" : "pointer",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              padding: 0,
            }}
          >
            ×
          </button>
        )}

        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "0.5rem",
            marginBottom: "0.5rem",
            paddingRight: !family.is_builtin && onDelete ? "1.75rem" : undefined,
          }}
        >
          <h3 style={{ margin: 0, fontSize: "1.05rem" }}>{family.name}</h3>
          <span
            style={{
              fontSize: "0.75rem",
              padding: "0.15rem 0.5rem",
              borderRadius: "999px",
              background: allRegistered ? "#dcfce7" : "#fef3c7",
              color: allRegistered ? "#166534" : "#92400e",
            }}
          >
            {registered.length}/{expected} 已注册
          </span>
          {!family.is_builtin && (
            <span
              style={{
                fontSize: "0.75rem",
                padding: "0.15rem 0.5rem",
                borderRadius: "999px",
                background: "#e0e7ff",
                color: "#3730a3",
              }}
            >
              自定义
            </span>
          )}
        </div>
        <p className="muted" style={{ fontSize: "0.85rem", margin: "0 0 0.75rem" }}>
          根路径: <code>{family.root_path}</code>
        </p>
        <ul style={{ margin: 0, paddingLeft: "1.1rem", fontSize: "0.9rem" }}>
          {family.scenes.map((scene) => (
            <li key={scene.scene_id} style={{ marginBottom: "0.5rem" }}>
              <strong>{scene.name}</strong>
              <span className="muted"> ({scene.scene_id})</span>
              {scene.registered_scene_id && (
                <span style={{ marginLeft: "0.35rem", color: "#16a34a", fontSize: "0.8rem" }}>
                  ✓ 已注册
                </span>
              )}
            </li>
          ))}
        </ul>
      </div>

      {showDeleteConfirm && (
        <div
          role="dialog"
          aria-modal="true"
          aria-labelledby="delete-dataset-confirm-title"
          style={{
            position: "fixed",
            inset: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: "rgba(15, 23, 42, 0.45)",
            zIndex: 1000,
          }}
          onClick={() => !deleting && setShowDeleteConfirm(false)}
        >
          <div
            style={{
              background: "#fff",
              borderRadius: "8px",
              padding: "1.25rem 1.5rem",
              minWidth: "280px",
              boxShadow: "0 10px 40px rgba(0,0,0,0.15)",
            }}
            onClick={(event) => event.stopPropagation()}
          >
            <p id="delete-dataset-confirm-title" style={{ margin: "0 0 1rem", fontSize: "0.95rem" }}>
              确认删除该数据集？
            </p>
            <div style={{ display: "flex", justifyContent: "flex-end", gap: "0.75rem" }}>
              <button type="button" onClick={handleConfirmDelete} disabled={deleting}>
                是
              </button>
              <button type="button" onClick={() => setShowDeleteConfirm(false)} disabled={deleting}>
                否
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
