import { Link } from "react-router-dom";
import { INFERENCE_HUB_MODES } from "../config/inferenceHubModes";

export default function InferenceHub() {
  return (
    <section>
      <h2 className="page-title">单图推理</h2>
      <p className="muted" style={{ marginBottom: "1.5rem" }}>
        请先选择要使用的推理功能，再进入对应界面配置参数并 Run。
      </p>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
          gap: "1rem",
        }}
      >
        {INFERENCE_HUB_MODES.map((mode) => (
          <Link
            key={mode.stage}
            to={mode.path}
            className="card"
            style={{
              display: "block",
              textDecoration: "none",
              color: "inherit",
              transition: "box-shadow 0.2s",
            }}
          >
            <h3 style={{ margin: "0 0 0.5rem", fontSize: "1.1rem" }}>{mode.label}</h3>
            {mode.description && (
              <p className="muted" style={{ fontSize: "0.9rem", margin: 0 }}>
                {mode.description}
              </p>
            )}
            <p style={{ marginTop: "1rem", fontSize: "0.85rem", color: "#2563eb" }}>进入 →</p>
          </Link>
        ))}
      </div>
    </section>
  );
}
