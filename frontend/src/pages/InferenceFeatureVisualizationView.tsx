import { useCallback, useState } from "react";
import FeatureVisualizationPanel from "../components/inference/FeatureVisualizationPanel";
import FeatureVisualizationResultPanel from "../components/inference/FeatureVisualizationResultPanel";
import InferenceShell from "../components/inference/InferenceShell";
import type { FeatureVisualizationData } from "../types/inference";

export default function InferenceFeatureVisualizationView() {
  const [vizData, setVizData] = useState<FeatureVisualizationData | null>(null);
  const [hasRun, setHasRun] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleRunComplete = useCallback((data: FeatureVisualizationData) => {
    setVizData(data);
    setHasRun(true);
    setLoading(false);
  }, []);

  const handleRunStart = useCallback(() => {
    setHasRun(false);
    setVizData(null);
    setLoading(true);
  }, []);

  return (
    <InferenceShell
      title="可视特征"
      subtitle="Feature Visualization · 关键点与响应热力图"
      badge="Feature Viz"
    >
      <FeatureVisualizationPanel
        onRunComplete={handleRunComplete}
        onRunStart={handleRunStart}
        onRunEnd={() => setLoading(false)}
      />
      <FeatureVisualizationResultPanel data={vizData} hasRun={hasRun} loading={loading} />
    </InferenceShell>
  );
}
