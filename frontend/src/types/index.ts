export interface Scene {
  id: string;
  name: string;
  description?: string | null;
  dataset_family?: string | null;
  scene_slug?: string | null;
  point_cloud_path?: string | null;
  feature_index_path?: string | null;
  reference_images_dir?: string | null;
  train_image_count?: number;
  test_image_count?: number;
  status?: string;
  created_at: string;
  updated_at: string;
}

export type PipelineStage =
  | "feature_extraction"
  | "semantic_segmentation"
  | "feature_visualization"
  | "keypoint_detection"
  | "image_retrieval"
  | "image_matching"
  | "segmentation"
  | "scene_coordinate_regression";

export interface ModelItem {
  id: string;
  name: string;
  version: string;
  description?: string | null;
  capabilities: PipelineStage[];
  status: string;
  weights_path?: string | null;
  plugin_loaded?: boolean | null;
  created_at: string;
  updated_at: string;
}

export interface ModelRuntimeStatus {
  model_id: string;
  status: string;
  plugin_loaded: boolean;
  weights_path?: string | null;
  capabilities: PipelineStage[];
}

export interface InferenceTask {
  id: string;
  scene_id: string;
  model_id: string;
  image_path: string;
  status: string;
  pipeline_stages: PipelineStage[];
  created_at: string;
  result?: Record<string, unknown> | null;
}

export interface Benchmark {
  id: string;
  name: string;
  scene_id: string;
  model_id: string;
  dataset_id?: string | null;
  dataset_path?: string | null;
  input_path?: string | null;
  output_path?: string | null;
  pipeline_stages: PipelineStage[];
  status: string;
  metrics?: Record<string, number> | null;
  created_at: string;
  updated_at: string;
}

/** @deprecated Use InferenceTask */
export interface LocalizationTask {
  id: string;
  scene_id: string;
  model_id: string;
  image_path: string;
  status: string;
  created_at: string;
}

/** @deprecated Use InferenceApiResult from types/inference */
export interface LocalizationResult {
  id: string;
  scene_id: string;
  model_id: string;
  image_path: string;
  status: string;
  created_at: string;
  result?: Record<string, unknown> | null;
}

export interface LocalizationScene {
  scene_id: string;
  weights_file: string;
}

export { PIPELINE_STAGE_LABELS, INFERENCE_PIPELINE_STAGES, BENCHMARK_PIPELINE_STAGES } from "./inference";
export type { MockDataset, FeatureExtractionJob } from "./inference";
