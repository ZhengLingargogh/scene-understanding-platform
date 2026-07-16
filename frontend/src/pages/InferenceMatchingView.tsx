import { useCallback, useState } from "react";
import InferenceShell from "../components/inference/InferenceShell";
import MatchingPanel from "../components/inference/MatchingPanel";
import MatchingResultPanel from "../components/inference/MatchingResultPanel";
import type { MatchingVisualizationData } from "../types/inference";

export default function InferenceMatchingView() {
  const [matchData, setMatchData] = useState<MatchingVisualizationData | null>(null);
  const [hasRun, setHasRun] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleRunComplete = useCallback((data: MatchingVisualizationData) => {
    setMatchData(data);
    setHasRun(true);
    setLoading(false);
  }, []);

  const handleRunStart = useCallback(() => {
    setHasRun(false);
    setMatchData(null);
    setLoading(true);
  }, []);

  return (
    <InferenceShell
      title="图像匹配"
      subtitle="LightGlue 稀疏特征 · 双图 Query 匹配可视化"
      badge="Matching"
    >
      <MatchingPanel
        onRunComplete={handleRunComplete}
        onRunStart={handleRunStart}
        onRunEnd={() => setLoading(false)}
      />
      <MatchingResultPanel data={matchData} hasRun={hasRun} loading={loading} />
    </InferenceShell>
  );
}
