export interface DatasetSplitInfo {
  split: string;
  label: string;
  rgb_dir: string;
  image_count: number;
  dataset_id: string;
  default_feature_dir: string;
  calibration_dir?: string | null;
  poses_dir?: string | null;
}

export interface DatasetSceneInfo {
  scene_id: string;
  name: string;
  path: string;
  splits: DatasetSplitInfo[];
  registered_scene_id?: string | null;
}

export interface DatasetCatalog {
  id: string;
  name: string;
  root_path: string;
  scenes: DatasetSceneInfo[];
  is_builtin?: boolean;
}

export interface DatasetScanPreview {
  family_id: string;
  name: string;
  root_path: string;
  scenes: DatasetSceneInfo[];
  warnings: string[];
  valid: boolean;
}

export interface RegisterDatasetResponse {
  family_id: string;
  name: string;
  root_path: string;
  scene_count: number;
  warnings: string[];
  registered_at: string;
}

export interface BrowseEntry {
  name: string;
  path: string;
  is_directory: boolean;
}

export interface BrowseResponse {
  path: string;
  parent_path?: string | null;
  entries: BrowseEntry[];
}
