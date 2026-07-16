import { Navigate, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import SceneInferenceFeatureView from "./pages/SceneInferenceFeatureView";
import SceneInferenceHub from "./pages/SceneInferenceHub";
import SceneInferenceSemanticSegmentationView from "./pages/SceneInferenceSemanticSegmentationView";
import InferenceFeatureVisualizationView from "./pages/InferenceFeatureVisualizationView";
import InferenceHub from "./pages/InferenceHub";
import InferenceInteractiveSegmentationView from "./pages/InferenceInteractiveSegmentationView";
import InferenceMatchingView from "./pages/InferenceMatchingView";
import InferenceRetrievalView from "./pages/InferenceRetrievalView";
import ModelsPage from "./pages/ModelsPage";
import ScenesPage from "./pages/ScenesPage";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Navigate to="/scenes" replace />} />
        <Route path="scenes" element={<ScenesPage />} />
        <Route path="models" element={<ModelsPage />} />
        <Route path="inference" element={<InferenceHub />} />
        <Route path="inference/scr" element={<Navigate to="/inference" replace />} />
        <Route path="inference/retrieval" element={<InferenceRetrievalView />} />
        <Route path="inference/matching" element={<InferenceMatchingView />} />
        <Route path="inference/feature-visualization" element={<InferenceFeatureVisualizationView />} />
        <Route path="inference/interactive-segmentation" element={<InferenceInteractiveSegmentationView />} />
        <Route path="localization" element={<Navigate to="/inference" replace />} />
        <Route path="scene-inference" element={<SceneInferenceHub />} />
        <Route path="scene-inference/scr" element={<Navigate to="/scene-inference" replace />} />
        <Route path="scene-inference/feature-extraction" element={<SceneInferenceFeatureView />} />
        <Route path="scene-inference/semantic-segmentation" element={<SceneInferenceSemanticSegmentationView />} />
        <Route path="benchmark" element={<Navigate to="/scene-inference" replace />} />
      </Route>
    </Routes>
  );
}
