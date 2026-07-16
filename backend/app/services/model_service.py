import json
from pathlib import Path
from uuid import uuid4

from app.compat import UTC, datetime
from app.config import settings
from app.schemas.model import ModelCreate, ModelResponse, ModelRuntimeStatus, ModelUpdate
from app.schemas.pipeline import PipelineStage
from app.services.lightglue.engine import LightGlueEngine
from app.services.model_weights import DEFAULT_MODEL_IDS, delete_model_weights
from app.services.netvlad.engine import NetVLADEngine
from app.services.plugins.registry import (
    DETECTOR_REGISTRY,
    FEATURE_EXTRACTOR_REGISTRY,
    MATCHER_REGISTRY,
    RETRIEVER_REGISTRY,
    SCR_REGISTRY,
    SEGMENTER_REGISTRY,
    SEMANTIC_SEGMENTER_REGISTRY,
    registry,
)
from app.services.salad.engine import SaladEngine
from app.services.sam.engine import SAMEngine
from app.services.sift.engine import SIFTEngine
from app.services.superpoint.engine import SuperPointEngine


class ModelService:
    """In-memory model registry with pipeline capability metadata."""

    _REMOVED_DEFAULTS_FILE = Path("data") / "removed_models.json"

    def __init__(self) -> None:
        self._models = {}  # type: dict
        self._removed_defaults = self._load_removed_defaults()
        self._seed_defaults()

    def _load_removed_defaults(self) -> set:
        path = self._REMOVED_DEFAULTS_FILE
        if not path.is_file():
            return set()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return {str(item) for item in data}
        except (OSError, json.JSONDecodeError):
            pass
        return set()

    def _save_removed_defaults(self) -> None:
        path = self._REMOVED_DEFAULTS_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(sorted(self._removed_defaults), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _seed_defaults(self) -> None:
        for model in self._build_default_models():
            if model.id not in self._removed_defaults:
                self._models[model.id] = model

    def _build_default_models(self) -> list:
        now = datetime.now(UTC)
        return [
            ModelResponse(
                id="salad",
                name="SALAD",
                version="1.0",
                description="DINOv2 + SALAD global descriptor",
                capabilities=[PipelineStage.feature_extraction, PipelineStage.image_retrieval],
                status="registered",
                weights_path=str(settings.salad_ckpt_path),
                plugin_loaded=False,
                created_at=now,
                updated_at=now,
            ),
            ModelResponse(
                id="superpoint",
                name="SuperPoint",
                version="1.0",
                description="SuperPoint local features (keypoint detector for matching)",
                capabilities=[PipelineStage.feature_extraction, PipelineStage.keypoint_detection],
                status="registered",
                weights_path="third_party/lightglue (auto-download)",
                plugin_loaded=False,
                created_at=now,
                updated_at=now,
            ),
            ModelResponse(
                id="netvlad",
                name="NetVLAD",
                version="1.0",
                description="VGG16 + NetVLAD global descriptor",
                capabilities=[PipelineStage.feature_extraction, PipelineStage.image_retrieval],
                status="registered",
                weights_path=str(settings.netvlad_ckpt_path),
                plugin_loaded=False,
                created_at=now,
                updated_at=now,
            ),
            ModelResponse(
                id="sift",
                name="SIFT",
                version="1.0",
                description="OpenCV SIFT local features (keypoints + 128-d descriptors)",
                capabilities=[PipelineStage.feature_extraction, PipelineStage.keypoint_detection],
                status="registered",
                weights_path="opencv-contrib (built-in)",
                plugin_loaded=False,
                created_at=now,
                updated_at=now,
            ),
            ModelResponse(
                id="lightglue",
                name="LightGlue",
                version="1.0",
                description="LightGlue sparse feature matching (requires SuperPoint-style features)",
                capabilities=[PipelineStage.image_matching],
                status="registered",
                weights_path="third_party/lightglue (auto-download)",
                plugin_loaded=False,
                created_at=now,
                updated_at=now,
            ),
            ModelResponse(
                id="sam",
                name="SAM",
                version="1.0",
                description="Segment Anything (ViT-B) interactive and batch semantic segmentation",
                capabilities=[PipelineStage.segmentation, PipelineStage.semantic_segmentation],
                status="registered",
                weights_path=str(settings.sam_ckpt_path),
                plugin_loaded=False,
                created_at=now,
                updated_at=now,
            ),
        ]

    def _sync_default_models(self) -> None:
        """Re-register built-in models unless explicitly removed on disk."""
        self._removed_defaults = self._load_removed_defaults()
        for model in self._build_default_models():
            if model.id not in self._removed_defaults and model.id not in self._models:
                self._models[model.id] = model

    def list_models(self):
        self._sync_default_models()
        return [self._with_runtime_status(model) for model in self._models.values()]

    def get_model(self, model_id: str):
        model = self._models.get(model_id)
        if model is None:
            return None
        return self._with_runtime_status(model)

    def create_model(self, payload: ModelCreate) -> ModelResponse:
        now = datetime.now(UTC)
        model = ModelResponse(
            id=str(uuid4()),
            name=payload.name,
            version=payload.version,
            description=payload.description,
            capabilities=payload.capabilities or list(PipelineStage),
            status="registered",
            plugin_loaded=False,
            created_at=now,
            updated_at=now,
        )
        self._models[model.id] = model
        return model

    def update_model(self, model_id: str, payload: ModelUpdate):
        model = self._models.get(model_id)
        if model is None:
            return None
        updates = payload.model_dump(exclude_unset=True)
        updated = model.model_copy(update={**updates, "updated_at": datetime.now(UTC)})
        self._models[model_id] = updated
        return self._with_runtime_status(updated)

    def delete_model(self, model_id: str) -> bool:
        model = self._models.get(model_id)
        if model is None:
            return False

        if model.plugin_loaded or self._engine_loaded(model_id):
            self.unload_model(model_id)

        delete_model_weights(model_id, model.weights_path)

        if model_id in DEFAULT_MODEL_IDS:
            self._removed_defaults.add(model_id)
            self._save_removed_defaults()

        return self._models.pop(model_id, None) is not None

    def load_model(self, model_id: str) -> ModelResponse:
        model = self._models.get(model_id)
        if model is None:
            raise KeyError(model_id)

        if model_id == "salad":
            SaladEngine.get_shared().load()
        elif model_id == "superpoint":
            SuperPointEngine.get_shared().load()
        elif model_id == "netvlad":
            NetVLADEngine.get_shared().load()
        elif model_id == "sift":
            SIFTEngine.get_shared().load()
        elif model_id == "lightglue":
            LightGlueEngine.get_shared().load()
        elif model_id == "sam":
            SAMEngine.get_shared().load()

        if PipelineStage.image_retrieval in model.capabilities and model_id in RETRIEVER_REGISTRY:
            registry.get_retriever(model_id)
        if PipelineStage.feature_extraction in model.capabilities and model_id in FEATURE_EXTRACTOR_REGISTRY:
            registry.get_feature_extractor(model_id)
        if PipelineStage.keypoint_detection in model.capabilities and model_id in DETECTOR_REGISTRY:
            registry.get_detector(model_id)
        if PipelineStage.image_matching in model.capabilities and model_id in MATCHER_REGISTRY:
            registry.get_matcher(model_id)
        if PipelineStage.scene_coordinate_regression in model.capabilities and model_id in SCR_REGISTRY:
            registry.get_scr_model(model_id)
        if PipelineStage.segmentation in model.capabilities and model_id in SEGMENTER_REGISTRY:
            registry.get_segmenter(model_id)
        if PipelineStage.semantic_segmentation in model.capabilities and model_id in SEMANTIC_SEGMENTER_REGISTRY:
            registry.get_semantic_segmenter(model_id)

        if not any(
            [
                model_id in RETRIEVER_REGISTRY,
                model_id in FEATURE_EXTRACTOR_REGISTRY,
                model_id in DETECTOR_REGISTRY,
                model_id in MATCHER_REGISTRY,
                model_id in SCR_REGISTRY,
                model_id in SEGMENTER_REGISTRY,
                model_id in SEMANTIC_SEGMENTER_REGISTRY,
            ]
        ):
            raise ValueError("No plugin registered for model_id '{}'".format(model_id))

        updated = model.model_copy(
            update={"status": "loaded", "plugin_loaded": True, "updated_at": datetime.now(UTC)}
        )
        self._models[model_id] = updated
        return self._with_runtime_status(updated)

    def unload_model(self, model_id: str) -> ModelResponse:
        model = self._models.get(model_id)
        if model is None:
            raise KeyError(model_id)

        registry.unload_plugin(model_id)
        if model_id == "salad":
            SaladEngine.get_shared().unload()
        elif model_id == "superpoint":
            SuperPointEngine.get_shared().unload()
        elif model_id == "netvlad":
            NetVLADEngine.get_shared().unload()
        elif model_id == "sift":
            SIFTEngine.get_shared().unload()
        elif model_id == "lightglue":
            LightGlueEngine.get_shared().unload()
        elif model_id == "sam":
            SAMEngine.get_shared().unload()

        updated = model.model_copy(
            update={"status": "registered", "plugin_loaded": False, "updated_at": datetime.now(UTC)}
        )
        self._models[model_id] = updated
        return self._with_runtime_status(updated)

    def get_runtime_status(self, model_id: str) -> ModelRuntimeStatus:
        model = self.get_model(model_id)
        if model is None:
            raise KeyError(model_id)
        return ModelRuntimeStatus(
            model_id=model_id,
            status=model.status,
            plugin_loaded=self._engine_loaded(model_id),
            weights_path=model.weights_path,
            capabilities=model.capabilities,
        )

    def _engine_loaded(self, model_id: str) -> bool:
        if model_id == "salad":
            return SaladEngine.get_shared().is_loaded
        if model_id == "superpoint":
            return SuperPointEngine.get_shared().is_loaded
        if model_id == "netvlad":
            return NetVLADEngine.get_shared().is_loaded
        if model_id == "sift":
            return SIFTEngine.get_shared().is_loaded
        if model_id == "lightglue":
            return LightGlueEngine.get_shared().is_loaded
        if model_id == "sam":
            return SAMEngine.get_shared().is_loaded
        return registry.plugin_loaded(model_id)

    def _with_runtime_status(self, model: ModelResponse) -> ModelResponse:
        loaded = self._engine_loaded(model.id)
        if loaded:
            return model.model_copy(update={"plugin_loaded": True, "status": "loaded"})
        return model.model_copy(update={"plugin_loaded": registry.plugin_loaded(model.id)})
