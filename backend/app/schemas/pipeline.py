import enum
from typing import List, Optional

from pydantic import BaseModel, Field


class PipelineStage(str, enum.Enum):
    """Stages in the visual localization pipeline."""

    feature_extraction = "feature_extraction"
    semantic_segmentation = "semantic_segmentation"
    keypoint_detection = "keypoint_detection"
    image_retrieval = "image_retrieval"
    image_matching = "image_matching"
    segmentation = "segmentation"
    scene_coordinate_regression = "scene_coordinate_regression"


DEFAULT_PIPELINE_STAGES: List[PipelineStage] = [
    PipelineStage.feature_extraction,
    PipelineStage.image_retrieval,
    PipelineStage.image_matching,
    PipelineStage.scene_coordinate_regression,
]

INFERENCE_PIPELINE_STAGES: List[PipelineStage] = [
    PipelineStage.image_retrieval,
    PipelineStage.image_matching,
]

BENCHMARK_PIPELINE_STAGES: List[PipelineStage] = [
    PipelineStage.feature_extraction,
    PipelineStage.semantic_segmentation,
]


PIPELINE_STAGE_LABELS = {
    PipelineStage.feature_extraction: "特征提取",
    PipelineStage.semantic_segmentation: "语义分割",
    PipelineStage.keypoint_detection: "关键点检测",
    PipelineStage.image_retrieval: "图像检索",
    PipelineStage.image_matching: "图像匹配",
    PipelineStage.segmentation: "分割",
    PipelineStage.scene_coordinate_regression: "场景坐标回归",
}


class FeatureExtractionResult(BaseModel):
    status: str = "placeholder"
    message: str = "Feature extraction not wired yet"
    embedding_dim: int = 512
    descriptor_type: str = "global"


class RetrievalReference(BaseModel):
    id: str
    label: str
    score: float
    image_path: Optional[str] = None


class ImageRetrievalResult(BaseModel):
    status: str = "placeholder"
    message: str = "Image retrieval not wired yet"
    top_k: int = 8
    dataset_id: Optional[str] = None
    references: List[RetrievalReference] = Field(default_factory=list)


class ImageMatchingResult(BaseModel):
    status: str = "placeholder"
    message: str = "Image matching not wired yet"
    match_count: int = 0
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    keypoints0: List[List[float]] = Field(default_factory=list)
    keypoints1: List[List[float]] = Field(default_factory=list)
    matches: List[List[int]] = Field(
        default_factory=list,
        description="Pairs of keypoint indices [idx0, idx1]",
    )
    image0_path: Optional[str] = None
    image1_path: Optional[str] = None
    image0_size: Optional[List[int]] = Field(
        None,
        description="Original image size [width, height]",
    )
    image1_size: Optional[List[int]] = None


class PipelineResults(BaseModel):
    feature_extraction: Optional[FeatureExtractionResult] = None
    image_retrieval: Optional[ImageRetrievalResult] = None
    image_matching: Optional[ImageMatchingResult] = None
    scene_coordinate_regression: dict = Field(default_factory=dict)
