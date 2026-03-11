"""
Art to Music Pipeline Package
==============================
This package contains all pipeline modules for converting visual art
into music recommendations.

Modules:
    color_extractor        - Extracts dominant color palettes from images
    visual_feature_extractor - Extracts the 7 formal elements of art
    object_detector        - Detects objects using YOLOv8
    feature_mapper         - Maps visual features to audio features
    recommender            - Recommends music based on audio features
"""

from pipeline.color_extractor import ColorExtractor
from pipeline.visual_feature_extractor import VisualFeatureExtractor
from pipeline.object_detector import ObjectDetector
from pipeline.feature_mapper import FeatureMapper
from pipeline.recommender import MusicRecommender

__all__ = [
    "ColorExtractor",
    "VisualFeatureExtractor",
    "ObjectDetector",
    "FeatureMapper",
    "MusicRecommender",
]

__version__ = "1.0.0"
__author__ = "Art to Music Team"
