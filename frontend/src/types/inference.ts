export type PipelineStage =
  | "feature_extraction"
  | "semantic_segmentation"
  | "feature_visualization"
  | "keypoint_detection"
  | "image_retrieval"
  | "image_matching"
  | "segmentation"
  | "scene_coordinate_regression";

export const PIPELINE_STAGE_LABELS: Record<PipelineStage, string> = {
  feature_extraction: "特征提取",
  semantic_segmentation: "语义分割",
  feature_visualization: "可视特征",
  keypoint_detection: "关键点检测",
  image_retrieval: "图像检索",
  image_matching: "图像匹配",
  segmentation: "分割",
  scene_coordinate_regression: "场景坐标回归",
};

export const INFERENCE_PIPELINE_STAGES: PipelineStage[] = [
  "image_retrieval",
  "image_matching",
];

export const BENCHMARK_PIPELINE_STAGES: PipelineStage[] = [
  "feature_extraction",
  "semantic_segmentation",
];

/** @deprecated */
export const DEFAULT_PIPELINE_STAGES = INFERENCE_PIPELINE_STAGES;

export interface MockDataset {
  id: string;
  name: string;
  scene_id: string;
  image_count: number;
  description?: string;
}

export interface RetrievalReference {
  id: string;
  label: string;
  score: number;
  image_path?: string | null;
}

export interface PipelineResults {
  feature_extraction?: {
    status: string;
    message?: string;
    embedding_dim?: number;
    descriptor_type?: string;
  };
  image_retrieval?: {
    status: string;
    message?: string;
    top_k?: number;
    dataset_id?: string;
    references: RetrievalReference[];
  };
  image_matching?: {
    status: string;
    message?: string;
    match_count?: number;
    inlier_count?: number;
    inlier_ratio?: number;
    keypoints0?: number[][];
    keypoints1?: number[][];
    matches?: number[][];
    image0_path?: string | null;
    image1_path?: string | null;
    image0_size?: [number, number];
    image1_size?: [number, number];
  };
  scene_coordinate_regression?: Record<string, unknown>;
}

export interface InferenceApiResult {
  scene_id?: string;
  dataset_id?: string;
  model_id?: string;
  detector_model_id?: string;
  matcher_model_id?: string;
  pipeline_stages?: PipelineStage[];
  pipeline?: PipelineResults;
  pose_matrix?: number[][];
  rotation_matrix?: number[][];
  translation?: number[];
  rotation_vector?: number[];
  inlier_count?: number;
  confidence?: number;
  intrinsics?: Record<string, number>;
  pose_backend?: string;
  gt_errors?: {
    rotation_error_deg: number;
    translation_error_m: number;
  };
}

export interface CovisibilityRef {
  id: string;
  label: string;
  score: number;
  thumbnail: string;
  image_path?: string;
}

export interface MatchingVisualizationData {
  keypoints0: number[][];
  keypoints1: number[][];
  matches: number[][];
  image0Url: string;
  image1Url: string;
  matchCount: number;
  message?: string;
  detectorModelId?: string;
  matcherModelId?: string;
}

export interface FeatureVisualizationData {
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
  keypoints: [number, number][];
  scores: number[];
  cropBBox: [number, number, number, number];
  keypointCount: number;
  confidenceMax: number;
  confidenceMin: number;
  confidenceMedian: number;
  modelId: string;
  backend?: string;
  message?: string;
}

export interface SegmentationSessionData {
  sessionId: string;
  modelId: string;
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
  backend?: string;
  message?: string;
}

export interface SegmentationPredictData {
  sessionId: string;
  modelId: string;
  point: [number, number];
  imageWidth: number;
  imageHeight: number;
  mask: number[];
  contour: number[][];
  score: number;
  backend?: string;
  message?: string;
}

export interface FeatureExtractionJob {
  job_id: string;
  type: string;
  status: "running" | "completed" | "failed";
  progress: number;
  total: number;
  processed: number;
  message: string;
  model_id: string;
  dataset_id: string;
  input_path: string;
  output_path: string;
  created_at?: string;
  completed_at?: string;
}

export type SemanticSegmentationJob = FeatureExtractionJob;
