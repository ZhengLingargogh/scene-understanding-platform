"""
Plugin registry — resolves model_id to CV algorithm implementations.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Type

from app.services.detectors.base import KeypointDetector
from app.services.detectors.mock import MockKeypointDetector
from app.services.detectors.sift import SIFTDetector
from app.services.detectors.superpoint import SuperPointDetector
from app.services.feature_extractors.base import FeatureExtractor
from app.services.feature_extractors.mock import MockFeatureExtractor
from app.services.feature_extractors.netvlad import NetVLADFeatureExtractor
from app.services.feature_extractors.salad import SaladFeatureExtractor
from app.services.feature_extractors.sift import SIFTFeatureExtractor
from app.services.feature_extractors.superpoint import SuperPointFeatureExtractor
from app.services.matchers.base import Matcher
from app.services.matchers.lightglue import LightGlueMatcher
from app.services.matchers.mock import MockMatcher
from app.services.retrievers.base import Retriever
from app.services.retrievers.mock import MockRetriever
from app.services.retrievers.netvlad import NetVLADRetriever
from app.services.retrievers.salad import SaladRetriever
from app.services.scr.base import SCRModel
from app.services.scr.mock import MockSCRModel
from app.services.segmenters.base import Segmenter
from app.services.segmenters.mock import MockSegmenter
from app.services.segmenters.sam import SAMSegmenter
from app.services.semantic_segmenters.base import SemanticSegmenter
from app.services.semantic_segmenters.sam import SAMSemanticSegmenter

logger = logging.getLogger(__name__)

SCR_MODEL_IDS = ("ace", "mock-scr")

RETRIEVER_REGISTRY: Dict[str, Type[Retriever]] = {
    "mock-retriever": MockRetriever,
    "ace": MockRetriever,
    "salad": SaladRetriever,
    "netvlad": NetVLADRetriever,
}

FEATURE_EXTRACTOR_REGISTRY: Dict[str, Type[FeatureExtractor]] = {
    "mock-extractor": MockFeatureExtractor,
    "salad": SaladFeatureExtractor,
    "superpoint": SuperPointFeatureExtractor,
    "netvlad": NetVLADFeatureExtractor,
    "sift": SIFTFeatureExtractor,
}

DETECTOR_REGISTRY: Dict[str, Type[KeypointDetector]] = {
    "superpoint": SuperPointDetector,
    "mock-extractor": MockKeypointDetector,
    "sift": SIFTDetector,
}

MATCHER_REGISTRY: Dict[str, Type[Matcher]] = {
    "mock-matcher": MockMatcher,
    "ace": MockMatcher,
    "lightglue": LightGlueMatcher,
}

SCR_REGISTRY: Dict[str, Type[SCRModel]] = {
    "mock-scr": MockSCRModel,
}

SEGMENTER_REGISTRY: Dict[str, Type[Segmenter]] = {
    "mock-segmenter": MockSegmenter,
    "sam": SAMSegmenter,
}

SEMANTIC_SEGMENTER_REGISTRY: Dict[str, Type[SemanticSegmenter]] = {
    "sam": SAMSemanticSegmenter,
}

DEFAULT_RETRIEVER_ID = "mock-retriever"
DEFAULT_FEATURE_EXTRACTOR_ID = "salad"
DEFAULT_DETECTOR_ID = "superpoint"
DEFAULT_MATCHER_ID = "mock-matcher"
DEFAULT_SCR_ID = "ace"
DEFAULT_SEGMENTER_ID = "sam"
DEFAULT_SEMANTIC_SEGMENTER_ID = "sam"


class PluginRegistry:
    def __init__(self) -> None:
        self._retrievers: Dict[str, Retriever] = {}
        self._extractors: Dict[str, FeatureExtractor] = {}
        self._detectors: Dict[str, KeypointDetector] = {}
        self._matchers: Dict[str, Matcher] = {}
        self._scr_models: Dict[str, SCRModel] = {}
        self._segmenters: Dict[str, Segmenter] = {}
        self._semantic_segmenters: Dict[str, SemanticSegmenter] = {}

    def register_retriever(self, model_id: str, cls: Type[Retriever]) -> None:
        RETRIEVER_REGISTRY[model_id] = cls

    def register_feature_extractor(self, model_id: str, cls: Type[FeatureExtractor]) -> None:
        FEATURE_EXTRACTOR_REGISTRY[model_id] = cls

    def register_detector(self, model_id: str, cls: Type[KeypointDetector]) -> None:
        DETECTOR_REGISTRY[model_id] = cls

    def register_matcher(self, model_id: str, cls: Type[Matcher]) -> None:
        MATCHER_REGISTRY[model_id] = cls

    def register_scr(self, model_id: str, cls: Type[SCRModel]) -> None:
        SCR_REGISTRY[model_id] = cls

    def register_segmenter(self, model_id: str, cls: Type[Segmenter]) -> None:
        SEGMENTER_REGISTRY[model_id] = cls

    def register_semantic_segmenter(self, model_id: str, cls: Type[SemanticSegmenter]) -> None:
        SEMANTIC_SEGMENTER_REGISTRY[model_id] = cls

    def get_retriever(self, model_id: Optional[str] = None) -> Retriever:
        resolved_id = self._resolve_id(model_id, RETRIEVER_REGISTRY, DEFAULT_RETRIEVER_ID)
        if resolved_id not in self._retrievers:
            cls = RETRIEVER_REGISTRY[resolved_id]
            instance = cls(model_id=resolved_id)
            instance.load()
            self._retrievers[resolved_id] = instance
        return self._retrievers[resolved_id]

    def get_feature_extractor(self, model_id: Optional[str] = None) -> FeatureExtractor:
        resolved_id = self._resolve_id(model_id, FEATURE_EXTRACTOR_REGISTRY, DEFAULT_FEATURE_EXTRACTOR_ID)
        if resolved_id not in self._extractors:
            cls = FEATURE_EXTRACTOR_REGISTRY[resolved_id]
            instance = cls(model_id=resolved_id)
            instance.load()
            self._extractors[resolved_id] = instance
        return self._extractors[resolved_id]

    def get_detector(self, model_id: Optional[str] = None) -> KeypointDetector:
        resolved_id = self._resolve_id(model_id, DETECTOR_REGISTRY, DEFAULT_DETECTOR_ID)
        if resolved_id not in self._detectors:
            cls = DETECTOR_REGISTRY[resolved_id]
            instance = cls(model_id=resolved_id)
            instance.load()
            self._detectors[resolved_id] = instance
        return self._detectors[resolved_id]

    def get_matcher(self, model_id: Optional[str] = None) -> Matcher:
        resolved_id = self._resolve_id(model_id, MATCHER_REGISTRY, DEFAULT_MATCHER_ID)
        if resolved_id not in self._matchers:
            cls = MATCHER_REGISTRY[resolved_id]
            instance = cls(model_id=resolved_id)
            instance.load()
            self._matchers[resolved_id] = instance
        return self._matchers[resolved_id]

    def get_scr_model(self, model_id: Optional[str] = None) -> SCRModel:
        resolved_id = model_id if model_id in SCR_MODEL_IDS else DEFAULT_SCR_ID
        if model_id and model_id not in SCR_MODEL_IDS:
            logger.warning("Unknown model_id '%s', falling back to '%s'", model_id, DEFAULT_SCR_ID)
            resolved_id = DEFAULT_SCR_ID

        if resolved_id not in self._scr_models:
            if resolved_id == "ace":
                from app.services.scr.ace import get_shared_ace_scr

                self._scr_models[resolved_id] = get_shared_ace_scr()
            else:
                cls = SCR_REGISTRY[resolved_id]
                instance = cls(model_id=resolved_id)
                instance.load()
                self._scr_models[resolved_id] = instance
        return self._scr_models[resolved_id]

    def get_segmenter(self, model_id: Optional[str] = None) -> Segmenter:
        resolved_id = self._resolve_id(model_id, SEGMENTER_REGISTRY, DEFAULT_SEGMENTER_ID)
        if resolved_id not in self._segmenters:
            cls = SEGMENTER_REGISTRY[resolved_id]
            instance = cls(model_id=resolved_id)
            instance.load()
            self._segmenters[resolved_id] = instance
        return self._segmenters[resolved_id]

    def get_semantic_segmenter(self, model_id: Optional[str] = None) -> SemanticSegmenter:
        resolved_id = self._resolve_id(
            model_id, SEMANTIC_SEGMENTER_REGISTRY, DEFAULT_SEMANTIC_SEGMENTER_ID
        )
        if resolved_id not in self._semantic_segmenters:
            cls = SEMANTIC_SEGMENTER_REGISTRY[resolved_id]
            instance = cls(model_id=resolved_id)
            instance.load()
            self._semantic_segmenters[resolved_id] = instance
        return self._semantic_segmenters[resolved_id]

    def unload_all(self) -> None:
        for plugin in list(self._retrievers.values()):
            plugin.unload()
        for plugin in list(self._extractors.values()):
            plugin.unload()
        for plugin in list(self._detectors.values()):
            plugin.unload()
        for plugin in list(self._matchers.values()):
            plugin.unload()
        for key, plugin in list(self._scr_models.items()):
            if key != "ace":
                plugin.unload()
        for plugin in list(self._segmenters.values()):
            plugin.unload()
        for plugin in list(self._semantic_segmenters.values()):
            plugin.unload()
        self._retrievers.clear()
        self._extractors.clear()
        self._detectors.clear()
        self._matchers.clear()
        self._scr_models.clear()
        self._segmenters.clear()
        self._semantic_segmenters.clear()

    def unload_plugin(self, model_id: str) -> None:
        if model_id in self._retrievers:
            self._retrievers[model_id].unload()
            del self._retrievers[model_id]
        if model_id in self._extractors:
            self._extractors[model_id].unload()
            del self._extractors[model_id]
        if model_id in self._detectors:
            self._detectors[model_id].unload()
            del self._detectors[model_id]
        if model_id in self._matchers:
            self._matchers[model_id].unload()
            del self._matchers[model_id]
        if model_id in self._segmenters:
            self._segmenters[model_id].unload()
            del self._segmenters[model_id]
        if model_id in self._semantic_segmenters:
            self._semantic_segmenters[model_id].unload()
            del self._semantic_segmenters[model_id]

    def plugin_loaded(self, model_id: str) -> bool:
        if model_id in self._retrievers and self._retrievers[model_id].is_loaded:
            return True
        if model_id in self._extractors and self._extractors[model_id].is_loaded:
            return True
        if model_id in self._detectors and self._detectors[model_id].is_loaded:
            return True
        if model_id in self._matchers and self._matchers[model_id].is_loaded:
            return True
        if model_id in self._scr_models and self._scr_models[model_id].is_loaded:
            return True
        if model_id in self._segmenters and self._segmenters[model_id].is_loaded:
            return True
        if model_id in self._semantic_segmenters and self._semantic_segmenters[model_id].is_loaded:
            return True
        return False

    @staticmethod
    def _resolve_id(model_id: Optional[str], registry: Dict[str, Type], default_id: str) -> str:
        if model_id and model_id in registry:
            return model_id
        if model_id:
            logger.warning("Unknown model_id '%s', falling back to '%s'", model_id, default_id)
        return default_id

    def list_plugins(self) -> Dict[str, list]:
        return {
            "retrievers": list(RETRIEVER_REGISTRY.keys()),
            "feature_extractors": list(FEATURE_EXTRACTOR_REGISTRY.keys()),
            "detectors": list(DETECTOR_REGISTRY.keys()),
            "matchers": list(MATCHER_REGISTRY.keys()),
            "segmenters": list(SEGMENTER_REGISTRY.keys()),
            "semantic_segmenters": list(SEMANTIC_SEGMENTER_REGISTRY.keys()),
            "scr_models": list(SCR_MODEL_IDS),
        }


registry = PluginRegistry()


def get_retriever(model_id: Optional[str] = None) -> Retriever:
    return registry.get_retriever(model_id)


def get_feature_extractor(model_id: Optional[str] = None) -> FeatureExtractor:
    return registry.get_feature_extractor(model_id)


def get_detector(model_id: Optional[str] = None) -> KeypointDetector:
    return registry.get_detector(model_id)


def get_matcher(model_id: Optional[str] = None) -> Matcher:
    return registry.get_matcher(model_id)


def get_scr_model(model_id: Optional[str] = None) -> SCRModel:
    return registry.get_scr_model(model_id)


def get_segmenter(model_id: Optional[str] = None) -> Segmenter:
    return registry.get_segmenter(model_id)


def get_semantic_segmenter(model_id: Optional[str] = None) -> SemanticSegmenter:
    return registry.get_semantic_segmenter(model_id)
