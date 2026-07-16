import type { PipelineStage } from "../types/inference";
import { PIPELINE_STAGE_LABELS } from "../types/inference";

export interface HubMode {
  stage: PipelineStage;
  path: string;
  label: string;
  description?: string;
}

export const INFERENCE_HUB_MODES: HubMode[] = [
  {
    stage: "image_retrieval",
    path: "/inference/retrieval",
    label: PIPELINE_STAGE_LABELS.image_retrieval,
    description: "Top-8 相似图检索",
  },
  {
    stage: "image_matching",
    path: "/inference/matching",
    label: PIPELINE_STAGE_LABELS.image_matching,
    description: "双图匹配可视化",
  },
  {
    stage: "feature_visualization",
    path: "/inference/feature-visualization",
    label: PIPELINE_STAGE_LABELS.feature_visualization,
    description: "局部关键点可视化",
  },
  {
    stage: "segmentation",
    path: "/inference/interactive-segmentation",
    label: "交互分割",
    description: "点提示交互分割",
  },
];

export const SCENE_INFERENCE_HUB_MODES: HubMode[] = [
  {
    stage: "feature_extraction",
    path: "/scene-inference/feature-extraction",
    label: PIPELINE_STAGE_LABELS.feature_extraction,
    description: "提取场景局部/全局特征",
  },
  {
    stage: "semantic_segmentation",
    path: "/scene-inference/semantic-segmentation",
    label: PIPELINE_STAGE_LABELS.semantic_segmentation,
    description: "自动掩码生成并保存掩码图",
  },
];

export function hubModesForPath(pathname: string): HubMode[] | null {
  if (pathname.startsWith("/scene-inference/")) {
    return SCENE_INFERENCE_HUB_MODES;
  }
  if (pathname.startsWith("/inference/")) {
    return INFERENCE_HUB_MODES;
  }
  return null;
}
