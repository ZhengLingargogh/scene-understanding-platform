import { useEffect, useState } from "react";
import { api } from "../api/client";
import { formatModelLabel } from "../components/inference/inferenceUtils";
import type { ModelItem, PipelineStage } from "../types";
import { PIPELINE_STAGE_LABELS } from "../types";

function CapabilityBadges({ capabilities }: { capabilities: PipelineStage[] }) {
  return (
    <span style={{ display: "inline-flex", flexWrap: "wrap", gap: "0.25rem", marginLeft: "0.5rem" }}>
      {capabilities.map((cap) => (
        <span
          key={cap}
          style={{
            fontSize: "0.75rem",
            padding: "0.1rem 0.4rem",
            borderRadius: "4px",
            background: "#e0f2fe",
            color: "#0369a1",
          }}
        >
          {PIPELINE_STAGE_LABELS[cap]}
        </span>
      ))}
    </span>
  );
}

export default function ModelsPage() {
  const [models, setModels] = useState<ModelItem[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [actionId, setActionId] = useState<string | null>(null);
  const [confirmModelId, setConfirmModelId] = useState<string | null>(null);

  async function refresh() {
    const data = await api.get<ModelItem[]>("/models");
    setModels(data);
  }

  useEffect(() => {
    refresh().catch((err: Error) => setError(err.message));
  }, []);

  async function handleConfirmUnload() {
    if (!confirmModelId) return;

    const modelId = confirmModelId;
    setActionId(modelId);
    setError(null);
    setConfirmModelId(null);

    try {
      await api.delete(`/models/${modelId}`);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setActionId(null);
    }
  }

  const confirmModel = models.find((model) => model.id === confirmModelId);

  return (
    <section>
      <h2 className="page-title">模型管理</h2>
      <p className="muted">
        注册/卸载推理模型并声明其支持的流水线能力：特征提取、图像检索、图像匹配、交互分割等
      </p>
      {error && <p className="error-text">{error}</p>}
      <div className="card">
        {models.length === 0 ? (
          <p className="muted">暂无模型，可通过 API 注册。</p>
        ) : (
          <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
            {models.map((model) => (
              <li
                key={model.id}
                style={{
                  display: "flex",
                  alignItems: "flex-start",
                  justifyContent: "space-between",
                  gap: "1rem",
                  marginBottom: "1rem",
                  paddingBottom: "1rem",
                  borderBottom: "1px solid #e2e8f0",
                }}
              >
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "0.5rem" }}>
                    <strong>{formatModelLabel(model)}</strong>
                    <span className="muted">[{model.status}]</span>
                    {model.plugin_loaded && (
                      <span style={{ color: "#16a34a", fontSize: "0.85rem" }}>● 已加载</span>
                    )}
                    <CapabilityBadges capabilities={model.capabilities ?? []} />
                  </div>
                  {model.description && (
                    <div className="muted" style={{ fontSize: "0.85rem", marginTop: "0.2rem" }}>
                      {model.description}
                    </div>
                  )}
                  {model.weights_path && (
                    <div className="muted" style={{ fontSize: "0.8rem", marginTop: "0.2rem" }}>
                      权重: {model.weights_path}
                    </div>
                  )}
                </div>
                <button
                  type="button"
                  onClick={() => setConfirmModelId(model.id)}
                  disabled={actionId === model.id}
                  style={{ flexShrink: 0, alignSelf: "center" }}
                >
                  卸载
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      {confirmModelId && (
        <div
          role="dialog"
          aria-modal="true"
          aria-labelledby="unload-confirm-title"
          style={{
            position: "fixed",
            inset: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: "rgba(15, 23, 42, 0.45)",
            zIndex: 1000,
          }}
          onClick={() => setConfirmModelId(null)}
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
            <p id="unload-confirm-title" style={{ margin: "0 0 1rem", fontSize: "0.95rem" }}>
              确认卸载此模型？
              {confirmModel ? (
                <span className="muted" style={{ display: "block", marginTop: "0.35rem", fontSize: "0.85rem" }}>
                  {formatModelLabel(confirmModel)}
                </span>
              ) : null}
            </p>
            <div style={{ display: "flex", justifyContent: "flex-end", gap: "0.75rem" }}>
              <button type="button" onClick={handleConfirmUnload} disabled={actionId !== null}>
                是
              </button>
              <button type="button" onClick={() => setConfirmModelId(null)} disabled={actionId !== null}>
                否
              </button>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
