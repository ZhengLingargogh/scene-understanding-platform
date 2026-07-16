import { useCallback, useState } from "react";
import InferenceShell from "../components/inference/InferenceShell";
import InteractiveSegmentationCanvas from "../components/inference/InteractiveSegmentationCanvas";
import SegmentationPanel from "../components/inference/SegmentationPanel";
import type { SegmentationSessionData } from "../types/inference";

export default function InferenceInteractiveSegmentationView() {
  const [session, setSession] = useState<SegmentationSessionData | null>(null);
  const [active, setActive] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleRunComplete = useCallback((data: SegmentationSessionData) => {
    setSession(data);
    setActive(true);
    setLoading(false);
  }, []);

  const handleRunStart = useCallback(() => {
    setActive(false);
    setSession(null);
    setLoading(true);
  }, []);

  return (
    <InferenceShell
      title="交互分割"
      subtitle="点提示交互分割 · Mock / SAM 预留"
      badge="Segmentation"
    >
      <SegmentationPanel
        onRunComplete={handleRunComplete}
        onRunStart={handleRunStart}
        onRunEnd={() => setLoading(false)}
      />
      <InteractiveSegmentationCanvas session={session} active={active && !loading} />
    </InferenceShell>
  );
}
