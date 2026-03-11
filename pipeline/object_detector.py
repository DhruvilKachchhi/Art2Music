"""
Object Detector  –  Three-Layer Hybrid Pipeline
================================================

Layer 1 – YOLOv11 (ultralytics)
    Bounding-box detection for discrete foreground objects.
    Only 80 COCO classes → deliberately limited to high-confidence objects.
    Mis-detections caused by out-of-vocabulary forcing (moon→frisbee,
    fish→bird) are caught and suppressed by Layers 2 & 3.

Layer 2 – Places365 (ResNet-50, MIT)
    Global scene classification over 365 categories.
    Authoritatively identifies scenes YOLO cannot: mountain, ocean, forest,
    beach, night_sky, desert, underwater, etc.
    Also acts as a **context filter**: YOLO detections that are physically
    impossible in the Places365 scene are silently dropped.

Layer 3 – CLIP (ViT-B/32, OpenAI)  [optional but strongly recommended]
    Zero-shot visual-semantic verification.
    For every YOLO detection above a suspicion threshold the CLIP score for
    the predicted label vs. a set of confusion candidates is computed.
    If a scene-specific candidate scores higher, the YOLO label is replaced.
    Examples handled:
        moon/frisbee/balloon  →  resolved to whichever CLIP scores highest
        bird/fish/scuba diver →  resolved against underwater context
    CLIP is also used to inject fine-grained scene descriptors
    (e.g. "night sky with moon", "coral reef", "snowy mountain peak")
    directly from the image, bypassing Places365 label granularity.

All three streams feed the shared 10-category art/image mood taxonomy.

Install
-------
    pip install ultralytics torch torchvision pillow numpy
    pip install openai-clip          # Layer 3 (optional but recommended)
    # Places365 weights (~100 MB) auto-download on first run.
    # CLIP weights (~330 MB) auto-download on first run.

Usage
-----
    from object_detector import ObjectDetector
    from PIL import Image

    detector = ObjectDetector()
    result   = detector.detect(Image.open("photo.jpg"))
    print(detector.summarize(result))
"""

from __future__ import annotations

import os
import urllib.request
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_MODELS_DIR = "models"

# Places365
_P365_WEIGHTS_URL  = (
    "http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar"
)
_P365_LABELS_URL   = (
    "https://raw.githubusercontent.com/csailvision/places365/master/"
    "categories_places365.txt"
)
_P365_WEIGHTS_FILE = "resnet50_places365.pth.tar"
_P365_LABELS_FILE  = "categories_places365.txt"
_P365_MEAN         = [0.485, 0.456, 0.406]
_P365_STD          = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
# YOLO suppression rules
# Keys   = Places365 scene keyword substrings
# Values = set of YOLO labels that are physically impossible in that scene
# ─────────────────────────────────────────────────────────────────────────────
SCENE_YOLO_SUPPRESSIONS: Dict[str, Set[str]] = {
    # Underwater / ocean floor → no sky objects, no land animals
    "underwater":   {"frisbee", "kite", "bird", "airplane", "sports ball",
                     "balloon", "cat", "dog", "horse", "sheep", "cow",
                     "traffic light", "stop sign", "fire hydrant"},
    "coral_reef":   {"frisbee", "kite", "bird", "airplane", "sports ball",
                     "cat", "dog", "horse", "sheep", "cow"},
    # Night sky → round bright objects are moons/planets, not frisbees
    "night_sky":    {"frisbee", "sports ball", "clock", "plate"},
    "astronomy":    {"frisbee", "sports ball", "clock"},
    # Snowy mountain / glacier → no tropical or indoor objects
    "mountain_snowy": {"surfboard", "frisbee", "umbrella"},
    "glacier":      {"surfboard", "frisbee"},
    # Forest / jungle → no man-made round objects
    "forest":       {"frisbee", "sports ball", "clock"},
    "rainforest":   {"frisbee", "sports ball"},
    # Desert → no water/beach objects
    "desert":       {"surfboard", "boat", "umbrella"},
    # Beach / coast → frisbee is plausible; suppress night objects
    "beach":        set(),
    # Aerial / sky view → ground animals become suspicious
    "aerial":       {"cat", "dog", "sheep", "cow"},
}

# YOLO labels that frequently mis-fire on non-COCO subjects and need
# extra scrutiny (will be passed to CLIP verifier if available)
_SUSPICIOUS_LABELS: Set[str] = {
    "frisbee", "sports ball", "clock", "bird", "kite",
    "balloon", "plate", "apple", "orange",
}

# CLIP confusion candidates: when YOLO predicts one of these labels,
# CLIP re-scores these alternatives and picks the highest scorer.
CLIP_CONFUSION_MAP: Dict[str, List[str]] = {
    "frisbee":      ["frisbee", "moon", "plate", "clock", "ball",
                     "sun", "bubble", "jellyfish", "hot air balloon"],
    "sports ball":  ["sports ball", "moon", "boulder", "bubble",
                     "jellyfish", "planet"],
    "kite":         ["kite", "bird", "bat", "flying fish", "hang glider"],
    "bird":         ["bird", "fish", "scuba diver", "manta ray",
                     "flying squirrel", "bat"],
    "clock":        ["clock", "moon", "porthole", "wheel", "coin"],
    "balloon":      ["balloon", "hot air balloon", "moon", "bubble",
                     "jellyfish", "lantern"],
    "plate":        ["plate", "moon", "frisbee", "lilypad"],
    "apple":        ["apple", "tomato", "ornament", "ball"],
    "orange":       ["orange", "ball", "pumpkin", "sunset"],
}

# CLIP scene probes: always tested regardless of YOLO output.
# Covers all 11 scene type categories so every type can be identified.
CLIP_SCENE_PROBES: List[str] = [
    # ── Portraiture ───────────────────────────────────────────────────────
    "a portrait painting or photo of a person",
    "a close-up painting of a human face",
    "a group portrait of several people",
    # ── Still Life ────────────────────────────────────────────────────────
    "a still life painting of fruit, flowers or objects on a table",
    "an arrangement of inanimate objects such as vases, bowls or candles",
    # ── Genre / Everyday ──────────────────────────────────────────────────
    "a painting of an everyday domestic interior scene",
    "a busy street market or social gathering scene",
    # ── Historical / Narrative ────────────────────────────────────────────
    "a historical battle scene with soldiers or warriors",
    "a mythological or narrative painting with multiple figures",
    # ── Religious / Sacred ────────────────────────────────────────────────
    "a religious painting with saints, angels or sacred symbols",
    "the interior of a cathedral, church or mosque",
    # ── Marine / Seascape ─────────────────────────────────────────────────
    "a photo of an ocean or sea",
    "a painting of a seascape with waves and boats",
    "a photo of a beach",
    "a photo of a coral reef underwater",
    "a photo of a scuba diver underwater",
    # ── Landscape ────────────────────────────────────────────────────────
    "a photo of a mountain",
    "a photo of a forest",
    "a photo of a waterfall",
    "a photo of a river or lake",
    "a photo of a desert",
    "a photo of a glacier or iceberg",
    "a photo of a canyon",
    "a photo of a tropical jungle",
    "a photo of a field or meadow",
    "a photo of a snow-covered landscape",
    "a photo of a sunset or sunrise",
    "a photo of a night sky with moon and stars",
    "a photo of a stormy sky with lightning",
    # ── Cityscape ────────────────────────────────────────────────────────
    "a photo of a city skyline or urban street",
    "a painting of city buildings and architecture",
    # ── Abstract ─────────────────────────────────────────────────────────
    "an abstract painting with non-representational shapes and colors",
    "a geometric or minimalist abstract artwork",
    "a color field or expressionist abstract painting",
    # ── Surrealism ───────────────────────────────────────────────────────
    "a surrealist painting with dreamlike or impossible imagery",
    "a painting combining realistic and fantastical bizarre elements",
    # ── Action / Real Life ────────────────────────────────────────────────
    "a photo of a crowd at a festival, concert or sports event",
    "a candid photo capturing a moment of real life activity",
]

# Map CLIP scene probe text → mood tags
CLIP_PROBE_MOOD_MAP: Dict[str, List[str]] = {
    # Portraiture
    "a portrait painting or photo of a person":      ["introspective", "serene", "nostalgic"],
    "a close-up painting of a human face":           ["introspective", "mysterious", "dramatic"],
    "a group portrait of several people":            ["social", "nostalgic", "joyful"],
    # Still Life
    "a still life painting of fruit, flowers or objects on a table": ["serene", "introspective", "nostalgic"],
    "an arrangement of inanimate objects such as vases, bowls or candles": ["serene", "introspective", "calm"],
    # Genre
    "a painting of an everyday domestic interior scene": ["nostalgic", "serene", "social"],
    "a busy street market or social gathering scene":    ["energetic", "social", "joyful"],
    # Historical
    "a historical battle scene with soldiers or warriors":        ["dramatic", "tense", "awe"],
    "a mythological or narrative painting with multiple figures": ["dramatic", "awe", "mysterious"],
    # Religious
    "a religious painting with saints, angels or sacred symbols": ["awe", "serene", "introspective"],
    "the interior of a cathedral, church or mosque":              ["awe", "serene", "introspective"],
    # Marine
    "a photo of an ocean or sea":                    ["awe", "serene", "melancholy"],
    "a painting of a seascape with waves and boats": ["awe", "serene", "nostalgic"],
    "a photo of a beach":                            ["joyful", "serene", "nostalgic"],
    "a photo of a coral reef underwater":            ["joyful", "awe", "mysterious"],
    "a photo of a scuba diver underwater":           ["mysterious", "awe", "introspective"],
    # Landscape
    "a photo of a mountain":                         ["awe", "sublime", "dramatic"],
    "a photo of a forest":                           ["awe", "mysterious", "serene"],
    "a photo of a waterfall":                        ["awe", "sublime", "joyful"],
    "a photo of a river or lake":                    ["serene", "calm", "introspective"],
    "a photo of a desert":                           ["melancholy", "awe", "introspective"],
    "a photo of a glacier or iceberg":               ["awe", "serene", "melancholy"],
    "a photo of a canyon":                           ["awe", "dramatic", "introspective"],
    "a photo of a tropical jungle":                  ["chaotic", "mysterious", "awe"],
    "a photo of a field or meadow":                  ["serene", "joyful", "nostalgic"],
    "a photo of a snow-covered landscape":           ["serene", "melancholy", "awe"],
    "a photo of a sunset or sunrise":                ["nostalgic", "dramatic", "awe"],
    "a photo of a night sky with moon and stars":    ["mysterious", "awe", "melancholy"],
    "a photo of a stormy sky with lightning":        ["dramatic", "tense", "chaotic"],
    # Cityscape
    "a photo of a city skyline or urban street":     ["dramatic", "energetic", "social"],
    "a painting of city buildings and architecture": ["dramatic", "introspective", "awe"],
    # Abstract
    "an abstract painting with non-representational shapes and colors": ["mysterious", "introspective", "experimental"],
    "a geometric or minimalist abstract artwork":                       ["serene", "introspective", "calm"],
    "a color field or expressionist abstract painting":                 ["dramatic", "mysterious", "experimental"],
    # Surrealism
    "a surrealist painting with dreamlike or impossible imagery":          ["mysterious", "surreal", "dreamy"],
    "a painting combining realistic and fantastical bizarre elements":     ["mysterious", "surreal", "tense"],
    # Action
    "a photo of a crowd at a festival, concert or sports event": ["energetic", "joyful", "chaotic"],
    "a candid photo capturing a moment of real life activity":   ["energetic", "social", "joyful"],
}

# Threshold: a CLIP scene probe must score above this to be injected
_CLIP_SCENE_THRESHOLD = 0.15


# ─────────────────────────────────────────────────────────────────────────────
# Object mood map  (YOLO COCO labels → mood tags)
# ─────────────────────────────────────────────────────────────────────────────
OBJECT_MOOD_MAP: Dict[str, List[str]] = {
    # People
    "person":           ["joyful", "energetic", "social"],
    "people":           ["joyful", "energetic", "social"],
    "crowd":            ["chaotic", "energetic", "social"],
    # Nature
    "tree":             ["serene", "calm", "awe"],
    "plant":            ["serene", "calm", "introspective"],
    "potted plant":     ["serene", "calm", "nostalgic"],
    "flower":           ["serene", "joyful", "nostalgic"],
    "leaf":             ["serene", "calm", "introspective"],
    "grass":            ["serene", "calm"],
    "forest":           ["awe", "mysterious", "serene"],
    "mountain":         ["awe", "sublime", "dramatic"],
    "rock":             ["serene", "introspective", "dramatic"],
    "sand":             ["serene", "nostalgic", "melancholy"],
    "snow":             ["serene", "calm", "melancholy"],
    "ice":              ["tense", "serene", "dramatic"],
    "fire":             ["dramatic", "intense", "chaotic"],
    "smoke":            ["mysterious", "tense", "melancholy"],
    "cloud":            ["serene", "mysterious", "awe"],
    "sky":              ["awe", "serene", "joyful"],
    "sunset":           ["nostalgic", "dramatic", "awe"],
    "fog":              ["mysterious", "melancholy", "tense"],
    "wave":             ["awe", "dramatic", "serene"],
    "river":            ["serene", "calm", "introspective"],
    "lake":             ["serene", "introspective", "nostalgic"],
    "waterfall":        ["awe", "sublime", "joyful"],
    "ocean":            ["awe", "serene", "melancholy"],
    "sea":              ["awe", "serene", "melancholy"],
    "beach":            ["joyful", "serene", "nostalgic"],
    "field":            ["serene", "nostalgic", "introspective"],
    "desert":           ["melancholy", "awe", "introspective"],
    "canyon":           ["awe", "dramatic", "introspective"],
    "glacier":          ["awe", "serene", "melancholy"],
    "volcano":          ["dramatic", "awe", "tense"],
    "cliff":            ["awe", "dramatic", "tense"],
    "cave":             ["mysterious", "tense", "introspective"],
    "valley":           ["serene", "awe", "introspective"],
    "meadow":           ["serene", "joyful", "nostalgic"],
    "jungle":           ["chaotic", "mysterious", "awe"],
    # CLIP-resolved scene labels (injected by Layer 3)
    "moon":             ["mysterious", "melancholy", "nostalgic"],
    "night sky":        ["awe", "mysterious", "introspective"],
    "coral reef":       ["joyful", "awe", "mysterious"],
    "underwater":       ["mysterious", "serene", "awe"],
    "scuba diver":      ["mysterious", "awe", "introspective"],
    "fish":             ["serene", "mysterious", "calm"],
    "manta ray":        ["awe", "mysterious", "serene"],
    "iceberg":          ["awe", "tense", "serene"],
    "aurora":           ["awe", "mysterious", "serene"],
    "lightning":        ["dramatic", "tense", "awe"],
    # Urban
    "building":         ["dramatic", "energetic", "introspective"],
    "skyscraper":       ["awe", "dramatic", "energetic"],
    "house":            ["nostalgic", "serene", "sentimental"],
    "ruins":            ["melancholy", "introspective", "nostalgic"],
    "bridge":           ["introspective", "dramatic", "nostalgic"],
    "street":           ["energetic", "social", "nostalgic"],
    "road":             ["introspective", "nostalgic", "serene"],
    "alley":            ["tense", "mysterious", "melancholy"],
    "car":              ["energetic", "chaotic", "social"],
    "truck":            ["energetic", "dramatic"],
    "bus":              ["social", "nostalgic"],
    "motorcycle":       ["energetic", "chaotic"],
    "bicycle":          ["joyful", "serene", "nostalgic"],
    "traffic light":    ["energetic", "tense"],
    "stop sign":        ["tense", "dramatic"],
    "fire hydrant":     ["dramatic", "tense"],
    # Animals
    "cat":              ["mysterious", "introspective", "serene"],
    "dog":              ["joyful", "energetic", "social"],
    "bird":             ["serene", "joyful", "awe"],
    "horse":            ["dramatic", "energetic", "awe"],
    "cow":              ["serene", "calm", "nostalgic"],
    "sheep":            ["serene", "calm", "nostalgic"],
    "elephant":         ["awe", "dramatic", "introspective"],
    "bear":             ["dramatic", "tense", "awe"],
    "deer":             ["serene", "nostalgic", "melancholy"],
    "zebra":            ["joyful", "energetic", "awe"],
    "giraffe":          ["awe", "serene", "joyful"],
    "lion":             ["dramatic", "awe", "intense"],
    "tiger":            ["dramatic", "tense", "intense"],
    "snake":            ["tense", "mysterious"],
    "butterfly":        ["joyful", "serene", "nostalgic"],
    # Water / boats
    "boat":             ["serene", "nostalgic", "introspective"],
    "ship":             ["dramatic", "awe", "nostalgic"],
    "sailboat":         ["serene", "joyful", "awe"],
    "surfboard":        ["joyful", "energetic", "awe"],
    "umbrella":         ["melancholy", "mysterious", "nostalgic"],
    "lighthouse":       ["dramatic", "awe", "serene"],
    # Sky objects
    "kite":             ["joyful", "serene", "nostalgic"],
    "airplane":         ["awe", "dramatic", "introspective"],
    "hot air balloon":  ["joyful", "awe", "serene"],
    # Night
    "lantern":          ["nostalgic", "mysterious", "serene"],
    "candle":           ["introspective", "nostalgic", "mysterious"],
    "neon":             ["energetic", "dramatic", "mysterious"],
    # Food
    "wine glass":       ["melancholy", "introspective", "nostalgic"],
    "bottle":           ["social", "nostalgic", "melancholy"],
    "pizza":            ["joyful", "social", "energetic"],
    "cake":             ["joyful", "social", "nostalgic"],
    # Indoor
    "couch":            ["serene", "nostalgic", "introspective"],
    "chair":            ["serene", "introspective", "nostalgic"],
    "bed":              ["serene", "melancholy", "introspective"],
    "dining table":     ["social", "nostalgic", "serene"],
    "book":             ["introspective", "nostalgic", "serene"],
    "laptop":           ["introspective", "tense", "energetic"],
    "cell phone":       ["social", "energetic", "tense"],
    "tv":               ["social", "nostalgic", "dramatic"],
    "vase":             ["introspective", "nostalgic", "serene"],
    "painting":         ["introspective", "mysterious", "nostalgic"],
    "clock":            ["introspective", "melancholy", "nostalgic"],
    "mirror":           ["mysterious", "introspective", "tense"],
    # Sports
    "sports ball":      ["joyful", "energetic", "chaotic"],
    "frisbee":          ["joyful", "serene"],
    "skis":             ["joyful", "energetic", "awe"],
    "snowboard":        ["energetic", "joyful", "awe"],
    "skateboard":       ["energetic", "chaotic", "joyful"],
    # Default
    "default":          ["introspective", "mysterious"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Places365 scene label → mood tags
# ─────────────────────────────────────────────────────────────────────────────
PLACES365_MOOD_MAP: Dict[str, List[str]] = {
    "mountain":             ["awe", "sublime", "dramatic"],
    "mountain_snowy":       ["awe", "serene", "melancholy"],
    "mountain_path":        ["introspective", "awe", "serene"],
    "cliff":                ["awe", "dramatic", "tense"],
    "valley":               ["serene", "awe", "introspective"],
    "canyon":               ["awe", "dramatic", "introspective"],
    "desert":               ["melancholy", "awe", "introspective"],
    "desert_sand":          ["melancholy", "serene", "introspective"],
    "desert_road":          ["melancholy", "introspective", "awe"],
    "dune":                 ["serene", "melancholy", "awe"],
    "glacier":              ["awe", "serene", "melancholy"],
    "iceberg":              ["awe", "tense", "serene"],
    "tundra":               ["melancholy", "awe", "introspective"],
    "volcano":              ["dramatic", "awe", "tense"],
    "hot_spring":           ["serene", "mysterious", "awe"],
    "geyser":               ["dramatic", "awe", "tense"],
    "cave":                 ["mysterious", "tense", "introspective"],
    "grotto":               ["mysterious", "serene", "introspective"],
    "natural_arch":         ["awe", "dramatic", "mysterious"],
    "butte":                ["awe", "dramatic", "introspective"],
    "badlands":             ["dramatic", "melancholy", "awe"],
    "forest":               ["awe", "mysterious", "serene"],
    "forest_path":          ["serene", "mysterious", "introspective"],
    "rainforest":           ["awe", "mysterious", "chaotic"],
    "jungle":               ["chaotic", "mysterious", "awe"],
    "bamboo_forest":        ["serene", "mysterious", "awe"],
    "orchard":              ["joyful", "nostalgic", "serene"],
    "woodland":             ["serene", "mysterious", "introspective"],
    "grove":                ["serene", "nostalgic", "mysterious"],
    "bog":                  ["mysterious", "melancholy", "tense"],
    "swamp":                ["mysterious", "melancholy", "tense"],
    "marsh":                ["mysterious", "melancholy", "serene"],
    "field":                ["serene", "nostalgic", "introspective"],
    "meadow":               ["serene", "joyful", "nostalgic"],
    "savanna":              ["awe", "dramatic", "serene"],
    "prairie":              ["serene", "awe", "nostalgic"],
    "pasture":              ["serene", "nostalgic", "calm"],
    "rice_paddy":           ["serene", "nostalgic"],
    "ocean":                ["awe", "serene", "melancholy"],
    "sea_cliff":            ["dramatic", "awe", "tense"],
    "coast":                ["serene", "awe", "nostalgic"],
    "beach":                ["joyful", "serene", "nostalgic"],
    "beach_house":          ["serene", "nostalgic", "joyful"],
    "tide_flat":            ["serene", "melancholy", "introspective"],
    "waterfall":            ["awe", "sublime", "joyful"],
    "river":                ["serene", "calm", "introspective"],
    "lake":                 ["serene", "introspective", "nostalgic"],
    "lake_natural":         ["serene", "awe", "introspective"],
    "pond":                 ["serene", "nostalgic", "introspective"],
    "lagoon":               ["serene", "mysterious", "awe"],
    "swimming_hole":        ["joyful", "serene", "nostalgic"],
    "dam":                  ["dramatic", "awe"],
    "canal":                ["serene", "nostalgic", "introspective"],
    "harbor":               ["nostalgic", "serene", "dramatic"],
    "bayou":                ["mysterious", "melancholy", "serene"],
    "fjord":                ["awe", "dramatic", "serene"],
    "coral_reef":           ["joyful", "awe", "mysterious"],
    "underwater":           ["mysterious", "serene", "awe"],
    "sky":                  ["awe", "serene", "joyful"],
    "sky_overcast":         ["melancholy", "introspective", "tense"],
    "sky_sunny":            ["joyful", "serene", "awe"],
    "sky_cloudy":           ["mysterious", "melancholy", "serene"],
    "sunset":               ["nostalgic", "dramatic", "awe"],
    "sunrise":              ["joyful", "awe", "serene"],
    "snowfield":            ["serene", "melancholy", "awe"],
    "snowstorm":            ["dramatic", "tense", "awe"],
    "fog":                  ["mysterious", "melancholy", "tense"],
    "lightning":            ["dramatic", "tense", "awe"],
    "rainbow":              ["joyful", "awe", "nostalgic"],
    "aurora":               ["awe", "mysterious", "serene"],
    "night_sky":            ["awe", "mysterious", "introspective"],
    "storm":                ["dramatic", "tense", "chaotic"],
    "tornado":              ["chaotic", "dramatic", "tense"],
    "garden":               ["serene", "joyful", "nostalgic"],
    "zen_garden":           ["serene", "introspective", "calm"],
    "botanical_garden":     ["serene", "joyful", "awe"],
    "flower_garden":        ["joyful", "serene", "nostalgic"],
    "park":                 ["joyful", "serene", "nostalgic"],
    "vineyard":             ["nostalgic", "serene", "joyful"],
    "greenhouse":           ["serene", "mysterious", "joyful"],
    "countryside":          ["serene", "nostalgic", "introspective"],
    "village":              ["nostalgic", "serene", "social"],
    "farm":                 ["nostalgic", "serene", "calm"],
    "barn":                 ["nostalgic", "serene", "melancholy"],
    "windmill":             ["nostalgic", "serene", "joyful"],
    "campsite":             ["serene", "joyful", "nostalgic"],
    "skyscraper":           ["awe", "dramatic", "energetic"],
    "downtown":             ["energetic", "dramatic", "social"],
    "cityscape":            ["dramatic", "energetic", "awe"],
    "street":               ["energetic", "social", "nostalgic"],
    "alley":                ["tense", "mysterious", "melancholy"],
    "highway":              ["introspective", "nostalgic", "energetic"],
    "bridge":               ["introspective", "dramatic", "nostalgic"],
    "tunnel":               ["tense", "mysterious", "dramatic"],
    "rooftop":              ["introspective", "awe", "nostalgic"],
    "industrial_area":      ["dramatic", "melancholy", "tense"],
    "factory":              ["dramatic", "energetic", "tense"],
    "cathedral":            ["awe", "dramatic", "introspective"],
    "church":               ["serene", "introspective", "nostalgic"],
    "mosque":               ["awe", "serene", "introspective"],
    "temple":               ["serene", "awe", "mysterious"],
    "monastery":            ["serene", "introspective", "mysterious"],
    "castle":               ["dramatic", "awe", "nostalgic"],
    "palace":               ["dramatic", "awe", "nostalgic"],
    "ruins":                ["melancholy", "introspective", "nostalgic"],
    "cemetery":             ["melancholy", "introspective", "serene"],
    "museum":               ["introspective", "nostalgic", "awe"],
    "library":              ["introspective", "serene", "nostalgic"],
    "airport":              ["dramatic", "introspective", "energetic"],
    "train_station":        ["nostalgic", "dramatic", "introspective"],
    "subway_station":       ["tense", "dramatic", "mysterious"],
    "harbor":               ["nostalgic", "serene", "dramatic"],
    "living_room":          ["serene", "nostalgic", "social"],
    "bedroom":              ["serene", "introspective", "nostalgic"],
    "kitchen":              ["joyful", "nostalgic", "social"],
    "attic":                ["mysterious", "nostalgic", "introspective"],
    "basement":             ["tense", "mysterious", "melancholy"],
    "cellar":               ["mysterious", "tense", "introspective"],
    "restaurant":           ["joyful", "social", "nostalgic"],
    "bar":                  ["social", "melancholy", "mysterious"],
    "nightclub":            ["energetic", "chaotic", "dramatic"],
    "theater":              ["dramatic", "mysterious", "nostalgic"],
    "concert_hall":         ["dramatic", "awe", "joyful"],
    "amusement_park":       ["joyful", "energetic", "chaotic"],
    "casino":               ["tense", "dramatic", "chaotic"],
    "gym":                  ["energetic", "dramatic"],
    "swimming_pool":        ["joyful", "serene"],
    "ski_slope":            ["joyful", "awe", "energetic"],
    "playground":           ["joyful", "nostalgic", "social"],
    "hospital":             ["tense", "melancholy", "dramatic"],
    "office":               ["introspective", "tense", "dramatic"],
    "classroom":            ["nostalgic", "introspective", "social"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Mood → art/image emotional category
# ─────────────────────────────────────────────────────────────────────────────
MOOD_CATEGORY_MAP: Dict[str, str] = {
    "serene":           "Serene/Calm",
    "calm":             "Serene/Calm",
    "peaceful":         "Serene/Calm",
    "tranquil":         "Serene/Calm",
    "warm":             "Serene/Calm",
    "melancholy":       "Melancholy/Sad",
    "sad":              "Melancholy/Sad",
    "nostalgic":        "Nostalgic/Sentimental",
    "sentimental":      "Nostalgic/Sentimental",
    "tense":            "Tense/Unsettling",
    "unsettling":       "Tense/Unsettling",
    "anxious":          "Tense/Unsettling",
    "joyful":           "Joyful/Energetic",
    "energetic":        "Joyful/Energetic",
    "upbeat":           "Joyful/Energetic",
    "lively":           "Joyful/Energetic",
    "social":           "Joyful/Energetic",
    "mysterious":       "Mysterious/Dreamy",
    "dreamy":           "Mysterious/Dreamy",
    "surreal":          "Mysterious/Dreamy",
    "dramatic":         "Dramatic/Intense",
    "intense":          "Dramatic/Intense",
    "awe":              "Awe/Sublime",
    "sublime":          "Awe/Sublime",
    "grand":            "Awe/Sublime",
    "introspective":    "Introspective/Contemplative",
    "contemplative":    "Introspective/Contemplative",
    "reflective":       "Introspective/Contemplative",
    "chaotic":          "Chaotic/Violent",
    "violent":          "Chaotic/Violent",
    "turbulent":        "Chaotic/Violent",
    # aliases
    "uplifting":        "Joyful/Energetic",
    "playful":          "Joyful/Energetic",
    "fluid":            "Serene/Calm",
    "acoustic":         "Serene/Calm",
    "ambient":          "Mysterious/Dreamy",
    "experimental":     "Mysterious/Dreamy",
    "instrumental":     "Introspective/Contemplative",
    "low-energy":       "Serene/Calm",
    "open":             "Awe/Sublime",
    "major":            "Joyful/Energetic",
    "minor":            "Melancholy/Sad",
    "powerful":         "Dramatic/Intense",
    "urban":            "Joyful/Energetic",
}




# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 – Places365 scene classifier
# ─────────────────────────────────────────────────────────────────────────────

class SceneClassifier:
    """ResNet-50 pretrained on Places365 – 365-class global scene classifier."""

    def __init__(
        self,
        models_dir: str = _DEFAULT_MODELS_DIR,
        top_k: int = 5,
        confidence_threshold: float = 0.04,
    ) -> None:
        self.models_dir = models_dir
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.labels: List[str] = []
        self._load()

    def _fetch(self, url: str, filename: str) -> str:
        os.makedirs(self.models_dir, exist_ok=True)
        dest = os.path.join(self.models_dir, filename)
        if not os.path.exists(dest):
            print(f"[SceneClassifier] Downloading {filename} …")
            urllib.request.urlretrieve(url, dest)
        return dest

    def _load(self) -> None:
        try:
            import torch
            import torchvision.models as tv

            w = self._fetch(_P365_WEIGHTS_URL, _P365_WEIGHTS_FILE)
            l = self._fetch(_P365_LABELS_URL,  _P365_LABELS_FILE)

            model = tv.resnet50(num_classes=365)
            ckpt  = torch.load(w, map_location="cpu", weights_only=False)
            sd    = ckpt.get("state_dict", ckpt)
            sd    = {k.replace("module.", ""): v for k, v in sd.items()}
            model.load_state_dict(sd)
            model.eval()
            self.model = model

            with open(l) as fh:
                raw = [ln.strip() for ln in fh if ln.strip()]
            self.labels = [ln.split(" ")[0].split("/")[-1] for ln in raw]
            print(f"[SceneClassifier] Places365 ready ({len(self.labels)} classes).")
        except Exception as exc:
            print(f"[SceneClassifier] Unavailable: {exc}")
            self.model = None

    def _preprocess(self, image: Image.Image):
        import torch
        from torchvision import transforms
        tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=_P365_MEAN, std=_P365_STD),
        ])
        return tf(image.convert("RGB")).unsqueeze(0)

    def _mood(self, label: str) -> List[str]:
        ll = label.lower()
        if ll in PLACES365_MOOD_MAP:
            return PLACES365_MOOD_MAP[ll]
        for k, v in PLACES365_MOOD_MAP.items():
            if k in ll or ll in k:
                return v
        return OBJECT_MOOD_MAP["default"]

    def classify(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Return top-K Places365 predictions with mood tags."""
        if self.model is None:
            return []
        try:
            import torch
            t = self._preprocess(image)
            with torch.no_grad():
                probs = torch.nn.functional.softmax(self.model(t), dim=1)[0]
            scores, idxs = probs.topk(self.top_k)
            out = []
            for s, i in zip(scores.tolist(), idxs.tolist()):
                if s < self.confidence_threshold:
                    continue
                label = self.labels[i] if i < len(self.labels) else f"scene_{i}"
                out.append({"label": label, "score": round(s, 4),
                             "mood_tags": self._mood(label)})
            return out
        except Exception as exc:
            print(f"[SceneClassifier] Inference error: {exc}")
            return []


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3 – CLIP semantic verifier
# ─────────────────────────────────────────────────────────────────────────────

class CLIPVerifier:
    """
    OpenAI CLIP zero-shot verifier.

    Two roles:
    1. Disambiguation – re-scores suspicious YOLO labels against confusion
       candidates and replaces the label if a candidate wins convincingly.
    2. Scene injection – scores CLIP_SCENE_PROBES against the image and
       returns any probe that passes the confidence threshold.
    """

    def __init__(self, model_name: str = "ViT-B/32") -> None:
        self.model = None
        self.preprocess = None
        self.tokenize = None
        self._load(model_name)

    def _load(self, model_name: str) -> None:
        try:
            import clip  # openai-clip
            import torch
            device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
            model, preprocess = clip.load(model_name, device=device)
            self.model      = model
            self.preprocess = preprocess
            self.tokenize   = clip.tokenize
            self._device    = device
            print(f"[CLIPVerifier] CLIP {model_name} ready on {device}.")
        except Exception as exc:
            print(f"[CLIPVerifier] Unavailable: {exc}")

    def _score(self, image: Image.Image, texts: List[str]) -> List[float]:
        """Return softmax probabilities for each text prompt."""
        import torch
        img_t  = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self._device)
        text_t = self.tokenize(texts).to(self._device)
        with torch.no_grad():
            logits, _ = self.model(img_t, text_t)
            probs = logits.softmax(dim=-1)[0].tolist()
        return probs

    def verify_label(
        self, image: Image.Image, predicted: str
    ) -> Tuple[str, float]:
        """
        Re-score a suspicious YOLO label against its confusion candidates.

        Returns:
            (winning_label, confidence)  – the best-matching text and its score.
        """
        if self.model is None or predicted not in CLIP_CONFUSION_MAP:
            return predicted, 1.0
        candidates = CLIP_CONFUSION_MAP[predicted]
        prompts    = [f"a photo of a {c}" for c in candidates]
        scores     = self._score(image, prompts)
        best_idx   = int(np.argmax(scores))
        return candidates[best_idx], scores[best_idx]

    def probe_scenes(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Score all CLIP_SCENE_PROBES against the image.

        Returns list of dicts (label, score, mood_tags) for probes that
        exceed _CLIP_SCENE_THRESHOLD.
        """
        if self.model is None:
            return []
        scores = self._score(image, CLIP_SCENE_PROBES)
        out    = []
        for probe, score in zip(CLIP_SCENE_PROBES, scores):
            if score >= _CLIP_SCENE_THRESHOLD:
                label = probe.replace("a photo of ", "").replace("an ", "").strip()
                out.append({
                    "label":     label,
                    "score":     round(score, 4),
                    "mood_tags": CLIP_PROBE_MOOD_MAP.get(probe,
                                     OBJECT_MOOD_MAP["default"]),
                    "source":    "clip_scene",
                })
        out.sort(key=lambda x: x["score"], reverse=True)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Main public class
# ─────────────────────────────────────────────────────────────────────────────

class ObjectDetector:
    """
    Three-layer hybrid image analyser.

    Layer 1  YOLOv11        – foreground object detection (80 COCO classes)
    Layer 2  Places365      – global scene classification (365 categories)
    Layer 3  CLIP ViT-B/32  – semantic verification & scene injection

    Corrections applied automatically:
    • Moon misclassified as frisbee/sports ball → CLIP resolves to "moon"
    • Fish/scuba diver misclassified as bird    → CLIP resolves to correct label
    • Mountain/ocean/forest not detected by YOLO → Places365 + CLIP inject them
    • Physically impossible YOLO detections      → suppressed by scene context

    Args:
        model_path: YOLOv11 weights path. Defaults to models/yolo11n.pt.
        confidence_threshold: YOLO minimum confidence (default 0.35).
        models_dir: Directory for cached weights (default "models/").
        scene_top_k: Places365 top-K predictions (default 5).
        scene_confidence_threshold: Places365 min score (default 0.04).
        use_clip: Enable CLIP verifier (default True, graceful fallback).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.35,
        models_dir: str = _DEFAULT_MODELS_DIR,
        scene_top_k: int = 5,
        scene_confidence_threshold: float = 0.04,
        use_clip: bool = True,
    ) -> None:
        if model_path is None:
            default = os.path.join(models_dir, "yolo11n.pt")
            model_path = default if os.path.exists(default) else "yolo11n.pt"

        self.model_path           = model_path
        self.confidence_threshold = float(np.clip(confidence_threshold, 0.0, 1.0))
        self.model                = None
        self._load_yolo()

        self.scene_classifier = SceneClassifier(
            models_dir=models_dir,
            top_k=scene_top_k,
            confidence_threshold=scene_confidence_threshold,
        )
        self.clip_verifier = CLIPVerifier() if use_clip else None

    # ── model loading ────────────────────────────────────────────────────────

    def _load_yolo(self) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
            self.model = YOLO(self.model_path)
            print(f"[ObjectDetector] YOLOv11 loaded from '{self.model_path}'.")
        except Exception as exc:
            print(f"[ObjectDetector] YOLOv11 unavailable: {exc}")
            self.model = None

    # ── helpers ──────────────────────────────────────────────────────────────

    def _mood_tags_for(self, label: str) -> List[str]:
        ll = label.lower().strip()
        if ll in OBJECT_MOOD_MAP:
            return OBJECT_MOOD_MAP[ll]
        for k in OBJECT_MOOD_MAP:
            if k == "default":
                continue
            if k in ll or ll in k:
                return OBJECT_MOOD_MAP[k]
        return OBJECT_MOOD_MAP["default"]

    def _dominant_mood(self, tags: List[str]) -> str:
        counts: Dict[str, int] = {}
        for t in tags:
            cat = MOOD_CATEGORY_MAP.get(t.lower().strip())
            if cat:
                counts[cat] = counts.get(cat, 0) + 1
        return max(counts, key=lambda c: counts[c]) if counts else "Introspective/Contemplative"

    def _scene_type_from_p365(self, scene_results: List[Dict]) -> Optional[str]:
        """Derive 11-category scene_type from Places365 top label."""
        if not scene_results:
            return None
        top = scene_results[0]["label"].lower()

        # Ordered from most specific to most generic
        _P365_MAP: List[Tuple[List[str], str]] = [
            # Marine
            (["underwater", "coral_reef", "ocean", "sea_cliff", "coast",
              "harbor", "beach", "lagoon", "fjord", "bayou", "tide_flat"], "marine"),
            # Religious
            (["cathedral", "church", "mosque", "temple", "monastery",
              "shrine", "chapel", "abbey"], "religious"),
            # Historical
            (["castle", "palace", "ruins", "fort", "battlefield",
              "colosseum", "amphitheater", "dungeon"], "historical"),
            # Cityscape
            (["skyscraper", "downtown", "cityscape", "street", "alley",
              "highway", "bridge", "tunnel", "rooftop", "industrial_area",
              "factory", "airport", "train_station", "subway_station",
              "parking_lot", "shopping_mall", "supermarket"], "cityscape"),
            # Genre / Everyday
            (["living_room", "bedroom", "kitchen", "bathroom", "office",
              "corridor", "hallway", "library", "museum", "restaurant",
              "bar", "nightclub", "theater", "concert_hall", "gym",
              "classroom", "laboratory", "garage", "attic", "basement",
              "cellar", "hospital", "amusement_park", "casino",
              "swimming_pool", "playground", "shop", "bakery", "cafe"], "genre"),
            # Landscape
            (["mountain", "valley", "canyon", "desert", "glacier", "tundra",
              "volcano", "forest", "rainforest", "jungle", "bamboo",
              "woodland", "bog", "swamp", "marsh", "field", "meadow",
              "savanna", "prairie", "pasture", "rice_paddy", "waterfall",
              "river", "lake", "pond", "snowfield", "fog", "storm",
              "sunset", "sunrise", "night_sky", "aurora", "rainbow",
              "lightning", "sky", "cliff", "cave", "dune", "butte",
              "orchard", "vineyard", "campsite", "countryside", "farm",
              "barn", "park", "garden"], "landscape"),
        ]
        for keywords, category in _P365_MAP:
            if any(kw in top for kw in keywords):
                return category
        return None

    def _scene_type_from_yolo(self, labels: List[str]) -> str:
        """YOLO-only fallback: derive 11-category scene type from detected labels."""
        ll = {l.lower() for l in labels}

        _YOLO_SCENE_KEYWORDS: List[Tuple[Set[str], str]] = [
            # Most specific first
            ({"cross", "church", "cathedral", "angel", "halo", "mosque", "bible"}, "religious"),
            ({"sword", "armor", "armour", "castle", "cannon", "spear", "shield",
              "chariot", "knight", "warrior", "ruins"}, "historical"),
            ({"boat", "ship", "sailboat", "surfboard", "wave", "ocean",
              "sea", "beach", "coral", "lighthouse"}, "marine"),
            ({"car", "bus", "truck", "traffic light", "stop sign", "skyscraper",
              "building", "bridge", "street", "road"}, "cityscape"),
            ({"couch", "chair", "bed", "dining table", "toilet", "sink",
              "refrigerator", "oven", "microwave", "laptop", "tv",
              "wine glass", "vase", "clock", "mirror", "candle"}, "genre"),
            ({"apple", "orange", "banana", "bowl", "bottle", "cup",
              "fork", "knife", "spoon", "pizza", "cake"}, "still_life"),
            ({"mountain", "forest", "waterfall", "river", "lake", "desert",
              "glacier", "canyon", "meadow", "valley", "tree", "flower",
              "bird", "kite", "airplane", "hot air balloon"}, "landscape"),
        ]
        for keyword_set, category in _YOLO_SCENE_KEYWORDS:
            if ll & keyword_set:
                return category

        # Person handling: single person → portrait, many → action/genre
        person_count = sum(1 for l in ll if l == "person")
        if person_count == 1:
            return "portraiture"
        if person_count > 1:
            return "action"

        # Nothing matched → abstract (safe ambiguous default)
        return "abstract"

    def _get_suppressed_labels(self, scene_results: List[Dict]) -> Set[str]:
        """Return YOLO labels that are physically impossible in the detected scene."""
        suppressed: Set[str] = set()
        for sr in scene_results:
            label = sr["label"].lower()
            for scene_kw, bad_labels in SCENE_YOLO_SUPPRESSIONS.items():
                if scene_kw in label:
                    suppressed |= bad_labels
        return suppressed

    # ── YOLO layer ───────────────────────────────────────────────────────────

    def _run_yolo(self, image: Image.Image) -> Tuple[List[Dict], bool]:
        if self.model is None:
            return [], False
        try:
            results = self.model(image.convert("RGB"),
                                 conf=self.confidence_threshold, verbose=False)
            detections = []
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    conf   = float(box.conf[0].item())
                    cls_id = int(box.cls[0].item())
                    label  = result.names.get(cls_id, f"object_{cls_id}")
                    bbox   = [round(v, 1) for v in box.xyxy[0].tolist()]
                    detections.append({
                        "label": label, "confidence": round(conf, 4),
                        "bbox": bbox, "_raw_label": label,
                    })
            detections.sort(key=lambda x: x["confidence"], reverse=True)
            return detections, True
        except Exception as exc:
            print(f"[ObjectDetector] YOLO error: {exc}")
            return [], False

    # ── CLIP verification layer ───────────────────────────────────────────────

    def _apply_clip_verification(
        self,
        detections: List[Dict],
        image: Image.Image,
        suppressed: Set[str],
    ) -> List[Dict]:
        """
        For each suspicious or suppressed YOLO detection, ask CLIP to
        disambiguate. Replace label if CLIP picks a better candidate.
        Detections that remain suppressed after CLIP are removed.
        """
        if self.clip_verifier is None:
            # Without CLIP: simply drop suppressed detections
            return [d for d in detections if d["label"].lower() not in suppressed]

        cleaned = []
        for det in detections:
            raw = det["label"].lower()
            needs_check = (raw in suppressed) or (raw in _SUSPICIOUS_LABELS)

            if needs_check:
                corrected, clip_conf = self.clip_verifier.verify_label(image, raw)
                corrected_lower = corrected.lower()

                # If CLIP still resolves to a suppressed category, drop it
                if corrected_lower in suppressed:
                    print(f"[ObjectDetector] Suppressed '{raw}' "
                          f"(CLIP→'{corrected}', scene incompatible)")
                    continue

                if corrected_lower != raw:
                    print(f"[ObjectDetector] Corrected '{raw}' → "
                          f"'{corrected}' (CLIP conf={clip_conf:.2f})")
                    det["label"]      = corrected
                    det["confidence"] = round(clip_conf, 4)

            det["mood_tags"] = self._mood_tags_for(det["label"])
            det.pop("_raw_label", None)
            cleaned.append(det)

        return cleaned

    # ── main pipeline ─────────────────────────────────────────────────────────

    def detect(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run the full three-layer detection pipeline.

        Returns:
            detected_objects  – verified YOLO bounding-box detections
            scene_detections  – Places365 predictions (label, score, mood_tags)
            clip_scenes       – CLIP scene probes that passed threshold
            mood_tags         – merged, deduplicated mood tags from all layers
            dominant_mood     – top art/mood category
            scene_type        – 'outdoor' | 'indoor' | 'night' | 'underwater' | 'abstract'
            object_count      – number of verified YOLO detections
            person_detected   – bool
            yolo_available    – bool
            scene_available   – bool
            clip_available    – bool
        """
        # ── Layer 1: YOLO ────────────────────────────────────────────────
        raw_detections, yolo_ok = self._run_yolo(image)

        # ── Layer 2: Places365 ───────────────────────────────────────────
        scene_results = self.scene_classifier.classify(image)
        scene_ok      = bool(scene_results)

        # Build suppression set from scene context
        suppressed = self._get_suppressed_labels(scene_results)

        # ── Layer 3: CLIP verification ────────────────────────────────────
        verified_detections = self._apply_clip_verification(
            raw_detections, image, suppressed
        )

        # CLIP scene injection
        clip_scenes: List[Dict] = []
        clip_ok = False
        if self.clip_verifier is not None:
            clip_scenes = self.clip_verifier.probe_scenes(image)
            clip_ok     = True

        # ── Merge mood tags ───────────────────────────────────────────────
        all_tags: List[str] = []
        for d  in verified_detections:
            all_tags.extend(d.get("mood_tags", []))
        for sr in scene_results:
            all_tags.extend(sr.get("mood_tags", []))
        for cs in clip_scenes:
            all_tags.extend(cs.get("mood_tags", []))

        unique_tags   = list(dict.fromkeys(all_tags)) or ["introspective", "mysterious"]
        dominant_mood = self._dominant_mood(all_tags)

        # ── Scene type (11-category taxonomy) ────────────────────────────
        # Map each CLIP probe → its canonical scene category, then pick
        # the category whose probes accumulated the highest total score.
        PROBE_SCENE_MAP: Dict[str, str] = {
            "a portrait painting or photo of a person":                  "portraiture",
            "a close-up painting of a human face":                       "portraiture",
            "a group portrait of several people":                        "portraiture",
            "a still life painting of fruit, flowers or objects on a table": "still_life",
            "an arrangement of inanimate objects such as vases, bowls or candles": "still_life",
            "a painting of an everyday domestic interior scene":         "genre",
            "a busy street market or social gathering scene":            "genre",
            "a historical battle scene with soldiers or warriors":       "historical",
            "a mythological or narrative painting with multiple figures":"historical",
            "a religious painting with saints, angels or sacred symbols":"religious",
            "the interior of a cathedral, church or mosque":             "religious",
            "a photo of an ocean or sea":                                "marine",
            "a painting of a seascape with waves and boats":             "marine",
            "a photo of a beach":                                        "marine",
            "a photo of a coral reef underwater":                        "marine",
            "a photo of a scuba diver underwater":                       "marine",
            "a photo of a mountain":                                     "landscape",
            "a photo of a forest":                                       "landscape",
            "a photo of a waterfall":                                    "landscape",
            "a photo of a river or lake":                                "landscape",
            "a photo of a desert":                                       "landscape",
            "a photo of a glacier or iceberg":                           "landscape",
            "a photo of a canyon":                                       "landscape",
            "a photo of a tropical jungle":                              "landscape",
            "a photo of a field or meadow":                              "landscape",
            "a photo of a snow-covered landscape":                       "landscape",
            "a photo of a sunset or sunrise":                            "landscape",
            "a photo of a night sky with moon and stars":                "landscape",
            "a photo of a stormy sky with lightning":                    "landscape",
            "a photo of a city skyline or urban street":                 "cityscape",
            "a painting of city buildings and architecture":             "cityscape",
            "an abstract painting with non-representational shapes and colors": "abstract",
            "a geometric or minimalist abstract artwork":                "abstract",
            "a color field or expressionist abstract painting":          "abstract",
            "a surrealist painting with dreamlike or impossible imagery":"surrealism",
            "a painting combining realistic and fantastical bizarre elements": "surrealism",
            "a photo of a crowd at a festival, concert or sports event": "action",
            "a candid photo capturing a moment of real life activity":   "action",
        }

        # Accumulate CLIP probe scores per scene category
        clip_scene_scores: Dict[str, float] = {}
        for cs in clip_scenes:
            # cs["label"] is the stripped probe text; match back to full probe
            for probe, cat in PROBE_SCENE_MAP.items():
                probe_label = probe.replace("a photo of ", "").replace("a painting of ", "").replace("an ", "").replace("a ", "").strip()
                if cs["label"].lower().strip() in probe.lower() or probe_label in cs["label"].lower():
                    clip_scene_scores[cat] = clip_scene_scores.get(cat, 0.0) + cs["score"]

        # Abstract bias: add a small constant bonus to abstract so that
        # ambiguous or unclear images tip toward abstract over landscape/generic
        clip_scene_scores["abstract"] = clip_scene_scores.get("abstract", 0.0) + 0.08

        # Derive scene_type: prefer CLIP if any category crossed threshold
        scene_type: Optional[str] = None
        if clip_scene_scores:
            best_clip_cat   = max(clip_scene_scores, key=lambda k: clip_scene_scores[k])
            best_clip_score = clip_scene_scores[best_clip_cat]
            # Only trust CLIP if the winning category has meaningful evidence
            if best_clip_score >= 0.12:
                scene_type = best_clip_cat

        # Fallback to Places365 top label → 11 categories
        if scene_type is None:
            scene_type = self._scene_type_from_p365(scene_results)

        # Final fallback: YOLO labels → 11 categories
        if scene_type is None:
            scene_type = self._scene_type_from_yolo(
                [d["label"] for d in verified_detections]
            )

        _PERSON_LABELS = {"person", "man", "woman", "child", "boy", "girl", "human"}
        person_detected = any(
            d["label"].lower() in _PERSON_LABELS for d in verified_detections
        )

        return {
            "detected_objects": verified_detections,
            "scene_detections": scene_results,
            "clip_scenes":      clip_scenes,
            "mood_tags":        unique_tags,
            "dominant_mood":    dominant_mood,
            "scene_type":       scene_type,
            "object_count":     len(verified_detections),
            "person_detected":  person_detected,
            "yolo_available":   yolo_ok,
            "scene_available":  scene_ok,
            "clip_available":   clip_ok,
        }

    # ── utility ──────────────────────────────────────────────────────────────

    def get_object_count_normalized(
        self, object_count: int, max_objects: int = 20
    ) -> float:
        return float(np.clip(object_count / max(max_objects, 1), 0.0, 1.0))

    def has_mood_tag(self, mood_tags: List[str], target: str) -> bool:
        t = target.lower().strip()
        return any(m.lower().strip() == t for m in mood_tags)

    def count_mood_tags(self, mood_tags: List[str], targets: List[str]) -> int:
        return sum(1 for t in targets if self.has_mood_tag(mood_tags, t))

    def get_mood_category(self, mood_tag: str) -> Optional[str]:
        return MOOD_CATEGORY_MAP.get(mood_tag.lower().strip())

    def summarize(self, result: Dict[str, Any]) -> str:
        lines = [
            "═" * 60,
            f"  Scene type     : {result['scene_type']}",
            f"  Dominant mood  : {result['dominant_mood']}",
            f"  YOLO           : {'✓' if result['yolo_available']  else '✗'}  "
            f"  Places365 : {'✓' if result['scene_available'] else '✗'}  "
            f"  CLIP : {'✓' if result['clip_available']  else '✗'}",
            f"  Objects found  : {result['object_count']}  |  "
            f"Person detected : {result['person_detected']}",
            f"  Mood tags      : {', '.join(result['mood_tags'][:10])}",
            "═" * 60,
        ]
        if result["detected_objects"]:
            lines.append("  YOLO detections (verified):")
            for d in result["detected_objects"][:10]:
                lines.append(
                    f"    {d['label']:<22} conf={d['confidence']:.3f}"
                    f"  [{', '.join(d.get('mood_tags', []))}]"
                )
        if result["scene_detections"]:
            lines.append("  Places365 scene:")
            for s in result["scene_detections"]:
                lines.append(
                    f"    {s['label']:<22} score={s['score']:.3f}"
                    f"  [{', '.join(s['mood_tags'])}]"
                )
        if result["clip_scenes"]:
            lines.append("  CLIP scene probes (passed threshold):")
            for c in result["clip_scenes"][:5]:
                lines.append(
                    f"    {c['label']:<35} score={c['score']:.3f}"
                    f"  [{', '.join(c['mood_tags'])}]"
                )
        lines.append("═" * 60)
        return "\n".join(lines)
