import { useCallback, useState } from "react";
import InferenceShell from "../components/inference/InferenceShell";
import RetrievalPanel from "../components/inference/RetrievalPanel";
import Top8ResultsPanel from "../components/inference/Top8ResultsPanel";
import type { CovisibilityRef } from "../types/inference";

export default function InferenceRetrievalView() {
  const [refs, setRefs] = useState<CovisibilityRef[]>([]);
  const [hasRun, setHasRun] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleRunComplete = useCallback((newRefs: CovisibilityRef[]) => {
    setRefs(newRefs);
    setHasRun(true);
    setLoading(false);
  }, []);

  const handleRunStart = useCallback(() => {
    setHasRun(false);
    setRefs([]);
    setLoading(true);
  }, []);

  return (
    <InferenceShell
      title="图像检索"
      subtitle="Top-8 相似参考图 · 右侧展示置信度"
      badge="Retrieval"
    >
      <RetrievalPanel
        onRunComplete={handleRunComplete}
        onRunStart={handleRunStart}
        onRunEnd={() => setLoading(false)}
      />
      <Top8ResultsPanel refs={refs} hasRun={hasRun} loading={loading} />
    </InferenceShell>
  );
}
