"""
Feature Mapper
===============
Maps visual features (color palette, art elements, object detections)
to audio features for music recommendation.

All output audio features conform exactly to the Spotify audio feature
specification ranges. Includes validate_features() for range checking.

Audio Feature Specification:
    acousticness     : [0.0, 1.0]
    danceability     : [0.0, 1.0]
    duration_ms      : display only, excluded from recommendation vector
    energy           : [0.0, 1.0]
    instrumentalness : [0.0, 1.0]
    key              : integer in [-1, 11]
    liveness         : [0.0, 1.0]
    loudness         : [-60.0, 0.0]
    mode             : integer 0 or 1
    speechiness      : [0.0, 1.0]
    tempo            : float in [50.0, 200.0]  (realistic output range)
    time_signature   : integer in [3, 7]
    valence          : [0.0, 1.0]
    popularity       : integer in [0, 100]
"""

from typing import Any, Dict, List

import numpy as np


# ---------------------------------------------------------------------------
# Feature specification ranges for validation
# ---------------------------------------------------------------------------
FEATURE_SPEC = {
    "acousticness":     {"min": 0.0,   "max": 1.0,   "type": "float"},
    "danceability":     {"min": 0.0,   "max": 1.0,   "type": "float"},
    "duration_ms":      {"min": 0,     "max": None,  "type": "int"},
    "energy":           {"min": 0.0,   "max": 1.0,   "type": "float"},
    "instrumentalness": {"min": 0.0,   "max": 1.0,   "type": "float"},
    "key":              {"min": -1,    "max": 11,    "type": "int"},
    "liveness":         {"min": 0.0,   "max": 1.0,   "type": "float"},
    "loudness":         {"min": -60.0, "max": 0.0,   "type": "float"},
    "mode":             {"min": 0,     "max": 1,     "type": "int"},
    "speechiness":      {"min": 0.0,   "max": 1.0,   "type": "float"},
    "tempo":            {"min": 0.0,   "max": None,  "type": "float"},
    "time_signature":   {"min": 3,     "max": 7,     "type": "int"},
    "valence":          {"min": 0.0,   "max": 1.0,   "type": "float"},
    "popularity":       {"min": 0,     "max": 100,   "type": "int"},
}


def validate_features(audio_features: Dict[str, Any]) -> None:
    """
    Validate all audio features against the specification ranges.

    Args:
        audio_features: Dictionary of audio features to validate.

    Raises:
        ValueError: If any feature value is outside its valid range.
    """
    for feature, spec in FEATURE_SPEC.items():
        if feature not in audio_features:
            raise ValueError(f"Missing required audio feature: '{feature}'")

        value = audio_features[feature]

        if spec["type"] == "int":
            if not isinstance(value, (int, np.integer)):
                raise ValueError(
                    f"Feature '{feature}' must be an integer, got {type(value).__name__} = {value}"
                )
            int_val = int(value)
            if spec["min"] is not None and int_val < spec["min"]:
                raise ValueError(
                    f"Feature '{feature}' = {int_val} is below minimum {spec['min']}"
                )
            if spec["max"] is not None and int_val > spec["max"]:
                raise ValueError(
                    f"Feature '{feature}' = {int_val} exceeds maximum {spec['max']}"
                )
        else:
            if not isinstance(value, (int, float, np.floating, np.integer)):
                raise ValueError(
                    f"Feature '{feature}' must be numeric, got {type(value).__name__} = {value}"
                )
            float_val = float(value)
            if spec["min"] is not None and float_val < spec["min"]:
                raise ValueError(
                    f"Feature '{feature}' = {float_val:.4f} is below minimum {spec['min']}"
                )
            if spec["max"] is not None and float_val > spec["max"]:
                raise ValueError(
                    f"Feature '{feature}' = {float_val:.4f} exceeds maximum {spec['max']}"
                )


# ---------------------------------------------------------------------------
# Circle-of-fifths hue→key anchor table
# ---------------------------------------------------------------------------
# Maps reference hue (degrees, 0–360) to Spotify pitch class (0–11).
# Interpolation happens between adjacent anchors.
_HUE_KEY_ANCHORS = [
    (0.0,   9),   # red       → A
    (30.0,  11),  # orange    → B
    (60.0,  0),   # yellow    → C
    (90.0,  2),   # yel-green → D
    (120.0, 4),   # green     → E
    (180.0, 7),   # cyan      → G
    (210.0, 5),   # blue      → F
    (270.0, 10),  # violet    → A♯
    (300.0, 8),   # magenta   → G♯
    (360.0, 9),   # red (wrap)→ A
]


def _hue_to_key(hue: float) -> int:
    """
    Map a hue value (0–360°) to a Spotify pitch class (0–11) using
    circle-of-fifths anchors with linear interpolation.
    """
    hue = float(hue) % 360.0
    for i in range(len(_HUE_KEY_ANCHORS) - 1):
        h0, k0 = _HUE_KEY_ANCHORS[i]
        h1, k1 = _HUE_KEY_ANCHORS[i + 1]
        if h0 <= hue <= h1:
            t = (hue - h0) / (h1 - h0) if (h1 - h0) > 0 else 0.0
            # Interpolate along circle of fifths (wrap at 12)
            diff = (k1 - k0 + 6) % 12 - 6  # shortest path on circle
            return int(round((k0 + t * diff) % 12))
    return 0  # fallback


class FeatureMapper:
    """
    Maps visual art features to Spotify-compatible audio features.

    Takes output of ColorExtractor, VisualFeatureExtractor, and ObjectDetector
    and produces a complete audio feature dict conforming to Spotify's spec.
    """

    def __init__(self) -> None:
        """Initialize the FeatureMapper."""
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid(x: float, center: float = 0.5, steepness: float = 8.0) -> float:
        """Sigmoid centred on `center`. Output in (0, 1)."""
        return float(1.0 / (1.0 + np.exp(-steepness * (x - center))))

    @staticmethod
    def _soft_clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        """Clamp to [lo, hi] with a very small margin to avoid hard-edge artefacts."""
        return float(np.clip(x, lo, hi))

    def _has_mood(self, mood_tags: List[str], *targets: str) -> bool:
        mood_lower = {m.lower() for m in mood_tags}
        return any(t.lower() in mood_lower for t in targets)

    def _is_warm_hue(self, avg_hue: float) -> bool:
        """Warm = red/orange/yellow/magenta (0–60° or 300–360°)."""
        return (0.0 <= avg_hue <= 60.0) or (300.0 <= avg_hue <= 360.0)

    def _is_cool_hue(self, avg_hue: float) -> bool:
        """Cool = blue/cyan/blue-violet (180–270°)."""
        return 180.0 <= avg_hue <= 270.0

    def _warm_hue_score(self, avg_hue: float) -> float:
        """
        Continuous warm/cool score: +1.0 for pure warm, -1.0 for pure cool,
        ~0 for neutral (green/yellow-green, ~90–150°).
        """
        h = avg_hue % 360.0
        # Warm centres: 0° (red) and 330° (magenta-red)
        # Cool centre: 225° (blue)
        warm_dist = min(abs(h - 0.0), abs(h - 360.0), abs(h - 330.0))
        cool_dist = abs(h - 225.0)
        score = (cool_dist - warm_dist) / 180.0  # normalised [-1, 1] approx
        return float(np.clip(score, -1.0, 1.0))

    # ------------------------------------------------------------------
    # Individual feature mapping methods
    # ------------------------------------------------------------------

    def _map_energy(
        self,
        visual_features: Dict[str, float],
        color_data: Dict[str, Any],
    ) -> float:
        """
        Energy [0.0, 1.0] — perceptual intensity.

        Primary drivers (total weight ~75%):
            edge_density × 0.35  — busy, high-frequency visual content
            contrast_ratio × 0.25 — value extremes (dark darks, bright brights)
            texture_energy × 0.15 — Gabor/LBP texture complexity

        Secondary modifiers (±15%):
            warm hue → +0.10 | cool hue → -0.08
            avg_saturation (mod ±0.05)

        Sigmoid is applied to spread values away from 0.5.
        """
        edge_density    = visual_features.get("edge_density",    0.3)
        contrast_ratio  = visual_features.get("contrast_ratio",  0.5)
        texture_energy  = visual_features.get("texture_energy",  0.3)
        avg_saturation  = color_data.get("avg_saturation", 0.5)
        avg_hue         = color_data.get("avg_hue", 180.0)

        # Weighted primary blend
        primary = (edge_density * 0.35) + (contrast_ratio * 0.25) + (texture_energy * 0.15)

        # Secondary: hue warmth modifier
        warm_score = self._warm_hue_score(avg_hue)      # -1 to +1
        hue_mod = warm_score * 0.08                      # ±0.08

        # Saturation modifier
        sat_mod = (avg_saturation - 0.5) * 0.10          # ±0.05

        raw = primary + hue_mod + sat_mod

        # Pass through sigmoid centred at 0.4 to spread distribution
        energy = self._sigmoid(raw, center=0.40, steepness=5.0)
        return round(self._soft_clamp(energy), 6)

    def _map_valence(
        self,
        visual_features: Dict[str, float],
        color_data: Dict[str, Any],
        mood_tags: List[str],
    ) -> float:
        """
        Valence [0.0, 1.0] — musical positiveness.

        Driven primarily by hue temperature:
            warm hues (0–60°, 300–360°) → push toward 1.0
            cool hues (180–270°) → push toward 0.0
        Weighted by avg_saturation; desaturated images bias 0.4–0.5.
        brightness_level is a secondary modifier.
        """
        avg_hue        = color_data.get("avg_hue", 180.0)
        avg_saturation = color_data.get("avg_saturation", 0.5)
        brightness     = visual_features.get("brightness_level", 0.5)

        # Convert warm_score (-1 to +1) into a 0-1 hue valence
        warm_score = self._warm_hue_score(avg_hue)        # -1 to +1
        hue_valence = (warm_score + 1.0) / 2.0            # 0 to 1

        # Desaturated images → ambiguous → bias toward 0.4–0.5
        # At saturation=0 the hue signal means nothing; blend toward 0.45
        sat_weight  = float(np.clip(avg_saturation, 0.0, 1.0))
        hue_contribution = hue_valence * sat_weight + 0.45 * (1.0 - sat_weight)

        # Secondary: brightness raises valence slightly
        brightness_mod = (brightness - 0.5) * 0.15        # ±0.075

        # Mood tag adjustments
        mood_mod = 0.0
        if self._has_mood(mood_tags, "uplifting", "happy", "social"):
            mood_mod += 0.08
        if self._has_mood(mood_tags, "melancholic", "dark", "low-energy"):
            mood_mod -= 0.08

        raw = hue_contribution * 0.75 + brightness * 0.10 + brightness_mod + mood_mod
        valence = self._sigmoid(raw, center=0.5, steepness=5.0)
        return round(self._soft_clamp(valence), 6)

    def _map_danceability(
        self,
        visual_features: Dict[str, float],
        color_data: Dict[str, Any],
        mood_tags: List[str],
    ) -> float:
        """
        Danceability [0.0, 1.0] — rhythmic regularity.

        Primary drivers:
            geometric_ratio × 0.30  — repeating geometric patterns
            shape_complexity × 0.20 — density of shapes
            line_straightness × 0.25 — strong H/V grid lines
            texture_energy × 0.15   — periodic texture patterns

        Modifier:
            avg_saturation ±0.10    — vivid colours suggest energy/rhythm
        """
        geometric_ratio   = visual_features.get("geometric_ratio",   0.5)
        shape_complexity  = visual_features.get("shape_complexity",   0.3)
        line_straightness = visual_features.get("line_straightness",  0.5)
        texture_energy    = visual_features.get("texture_energy",     0.3)
        avg_saturation    = color_data.get("avg_saturation", 0.5)

        primary = (
            geometric_ratio   * 0.30
            + shape_complexity  * 0.20
            + line_straightness * 0.25
            + texture_energy    * 0.15
        )

        sat_mod = (avg_saturation - 0.5) * 0.10

        mood_mod = 0.0
        if self._has_mood(mood_tags, "social", "lively", "upbeat"):
            mood_mod += 0.08

        raw = primary + sat_mod + mood_mod
        danceability = self._sigmoid(raw, center=0.45, steepness=5.0)
        return round(self._soft_clamp(danceability), 6)

    def _map_acousticness(
        self,
        visual_features: Dict[str, float],
        color_data: Dict[str, Any],
        energy: float,
        mood_tags: List[str],
    ) -> float:
        """
        Acousticness [0.0, 1.0] — confidence that the track is acoustic.

        Driven inversely with energy and texture complexity.
        Soft, painterly, warm-lit, low-contrast → toward 1.0.
        Harsh, high-contrast, cold, high-texture → toward 0.0.
        """
        texture_energy  = visual_features.get("texture_energy",  0.3)
        edge_density    = visual_features.get("edge_density",    0.3)
        avg_saturation  = color_data.get("avg_saturation", 0.5)
        avg_hue         = color_data.get("avg_hue", 180.0)
        brightness      = visual_features.get("brightness_level", 0.5)

        # High energy → low acousticness
        energy_penalty  = energy * 0.40

        # High texture/edges → electronic/synthetic
        texture_penalty = texture_energy * 0.25 + edge_density * 0.15

        # Warm, soft, muted → acoustic
        warm_bonus = max(0.0, self._warm_hue_score(avg_hue)) * 0.10
        soft_bonus = (1.0 - avg_saturation) * 0.05 + brightness * 0.05

        raw = 1.0 - energy_penalty - texture_penalty + warm_bonus + soft_bonus

        mood_mod = 0.0
        if self._has_mood(mood_tags, "calm", "acoustic", "peaceful", "serene"):
            mood_mod += 0.10
        if self._has_mood(mood_tags, "urban", "loud", "energetic"):
            mood_mod -= 0.10

        raw = raw + mood_mod
        acousticness = self._sigmoid(raw, center=0.55, steepness=4.5)
        return round(self._soft_clamp(acousticness), 6)

    def _map_instrumentalness(
        self,
        visual_features: Dict[str, float],
        color_data: Dict[str, Any],
        mood_tags: List[str],
        scene_type: str,
        person_detected: bool,
        person_confidence: float = 0.0,
    ) -> float:
        """
        Instrumentalness [0.0, 1.0].

        Spotify: > 0.5 predicts no vocals; near 1.0 = purely instrumental.
        Cap at 0.70 unless person_confidence is very low (< 0.1).

        Drivers:
            absence of persons → +
            high negative_space_ratio + minimal palette → +
            abstract/minimal composition → +
            person/social detection → strong negative modifier
        """
        negative_space  = visual_features.get("negative_space_ratio", 0.5)
        palette_breadth = visual_features.get("palette_breadth", 0.5)
        hue_variety     = visual_features.get("hue_variety", 0.5)

        # Minimal composition = high negative space + low palette variety
        minimalism = negative_space * 0.40 + (1.0 - palette_breadth) * 0.20

        # Person detection suppresses instrumentalness strongly
        if person_detected:
            person_penalty = 0.45 * person_confidence if person_confidence > 0 else 0.35
        else:
            person_penalty = 0.0

        # Abstract/scene boost
        abstract_bonus = 0.15 if scene_type in ("abstract", "nature", "landscape") else 0.0

        # Mood penalty
        mood_penalty = 0.0
        if self._has_mood(mood_tags, "social", "lively", "person", "crowd"):
            mood_penalty += 0.15

        raw = 0.50 + minimalism - person_penalty + abstract_bonus - mood_penalty

        # Cap: never exceed 0.70 unless person_confidence is very low
        hard_cap = 0.70 if person_confidence >= 0.1 else 0.85

        result = self._soft_clamp(raw, 0.0, hard_cap)
        return round(result, 6)

    def _map_liveness(
        self,
        visual_features: Dict[str, float],
        color_data: Dict[str, Any],
        mood_tags: List[str],
        person_detected: bool,
        object_count: int,
    ) -> float:
        """
        Liveness [0.0, 1.0].

        Spotify: > 0.8 strongly suggests live performance.
        Keep realistic range 0.1–0.5 unless image is very chaotic/crowded.

        Drivers:
            chaotic texture energy × 0.30
            high object count × 0.20
            person/crowd detection × 0.25
            warm colours + high contrast (crowd warmth) × 0.15
        """
        texture_energy = visual_features.get("texture_energy", 0.3)
        contrast_ratio = visual_features.get("contrast_ratio", 0.5)
        avg_hue        = color_data.get("avg_hue", 180.0)

        obj_norm = float(np.clip(object_count / 15.0, 0.0, 1.0))
        person_val = 0.6 if person_detected else 0.0
        warm_contrast = max(0.0, self._warm_hue_score(avg_hue)) * contrast_ratio * 0.20

        mood_boost = 0.0
        if self._has_mood(mood_tags, "crowd", "concert", "lively", "social"):
            mood_boost += 0.15

        raw = (
            texture_energy * 0.30
            + obj_norm      * 0.20
            + person_val    * 0.25
            + warm_contrast
            + mood_boost
        )

        # Apply sigmoid biased low — most images are not live performances
        liveness = self._sigmoid(raw, center=0.55, steepness=4.0)
        # Realistic range cap: 0.10–0.80 (extreme chaos needed to exceed 0.80)
        liveness = self._soft_clamp(liveness, 0.10, 0.80)
        return round(liveness, 6)

    def _map_speechiness(
        self,
        visual_features: Dict[str, float],
        object_count: int,
        person_detected: bool,
        person_confidence: float = 0.0,
    ) -> float:
        """
        Speechiness [0.0, 1.0].

        Spotify: > 0.66 = spoken word; 0.33–0.66 = rap/mixed; < 0.33 = music.
        Visual mapping almost always very low.
        Cap at 0.50 unless multiple high-confidence face detections exist.

        Drivers:
            strong face/figure detection with high contrast
            figure-ground contrast proxy
        """
        contrast_ratio = visual_features.get("contrast_ratio", 0.5)
        composition_balance = visual_features.get("composition_balance", 0.3)

        # Primary signal: person confidence
        person_signal = person_confidence * 0.50 if person_detected else 0.05

        # Figure-ground contrast (face/figure pops against background)
        fg_contrast = composition_balance * contrast_ratio * 0.20

        raw = person_signal + fg_contrast

        # Hard cap: 0.50 unless multiple high-confidence faces (confidence > 0.8)
        hard_cap = 0.50 if person_confidence < 0.80 else 0.65

        result = self._soft_clamp(raw, 0.0, hard_cap)
        return round(result, 6)

    def _map_tempo(
        self,
        energy: float,
        visual_features: Dict[str, float],
    ) -> float:
        """
        Tempo in BPM [50.0, 200.0].

        Mapping:
            calm/minimal/low-edge → 60–90 BPM
            moderate              → 90–130 BPM
            dense/chaotic/high    → 130–180 BPM

        Driven by energy (primary) + edge_density (secondary).
        """
        edge_density  = visual_features.get("edge_density",  0.3)
        texture_energy = visual_features.get("texture_energy", 0.3)

        # Combined intensity score
        intensity = energy * 0.55 + edge_density * 0.25 + texture_energy * 0.20

        # Linear mapping 0→60 BPM, 1→185 BPM
        tempo = 60.0 + intensity * 125.0

        # Realistic clamp
        tempo = self._soft_clamp(tempo, 50.0, 200.0)
        return round(float(tempo), 4)

    def _map_loudness(
        self,
        energy: float,
        visual_features: Dict[str, float],
        color_data: Dict[str, Any],
    ) -> float:
        """
        Loudness in dB [-60.0, 0.0].

        Spotify typical range: -20 to -3 dB for most tracks.
        Dark, muted, low-energy → -30 to -20 dB.
        Bright, vivid, high-energy → -8 to -3 dB.
        Never output 0.0 (clipping).

        Driven by: energy (primary), brightness, saturation.
        """
        brightness     = visual_features.get("brightness_level", 0.5)
        avg_saturation = color_data.get("avg_saturation", 0.5)

        # Combined loudness driver
        driver = energy * 0.55 + brightness * 0.25 + avg_saturation * 0.20

        # Map [0,1] → [-35, -3] dB  (realistic Spotify range)
        loudness = -35.0 + driver * 32.0   # 0→-35, 1→-3

        # Final clamp — never 0.0
        loudness = self._soft_clamp(loudness, -60.0, -1.0)
        return round(float(loudness), 4)

    def _map_key(self, color_data: Dict[str, Any]) -> int:
        """
        Key in pitch class [0, 11] (-1 if undetermined).

        Maps dominant hue around circle of fifths via _hue_to_key().
        Returns -1 for near-monochrome images.
        """
        avg_hue        = color_data.get("avg_hue", 180.0)
        avg_saturation = color_data.get("avg_saturation", 0.5)

        # Near-monochrome → indeterminate key
        if avg_saturation < 0.08:
            return -1

        return int(_hue_to_key(avg_hue))

    def _map_mode(
        self,
        visual_features: Dict[str, float],
        color_data: Dict[str, Any],
    ) -> int:
        """
        Mode: 0 (Minor) or 1 (Major).

        Rules:
            Warm-dominant + high saturation → Major (1)
            Cool-dominant or desaturated → Minor (0)
            Borderline: use contrast_ratio as tiebreaker (bright = Major)
        """
        avg_hue        = color_data.get("avg_hue", 180.0)
        avg_saturation = color_data.get("avg_saturation", 0.5)
        contrast_ratio = visual_features.get("contrast_ratio", 0.5)
        brightness     = visual_features.get("brightness_level", 0.5)

        warm_score = self._warm_hue_score(avg_hue)   # -1 to +1

        # Weighted decision score: positive → Major
        score = warm_score * 0.50 + (avg_saturation - 0.5) * 0.30 + (brightness - 0.5) * 0.20
        return 1 if score >= 0.0 else 0

    def _map_time_signature(
        self,
        visual_features: Dict[str, float],
    ) -> int:
        """
        Time signature: integer, typically 3 or 4.

        Rules:
            Default → 4
            Organic/curved, few geometric shapes → 3 (waltz feel)
            Very complex/chaotic composition → 5 or 7
        """
        shape_complexity  = visual_features.get("shape_complexity",  0.4)
        geometric_ratio   = visual_features.get("geometric_ratio",   0.5)
        curved_line_ratio = visual_features.get("curved_line_ratio", 0.5)
        texture_energy    = visual_features.get("texture_energy",    0.3)

        # Strong circular/organic composition → waltz (3)
        if curved_line_ratio > 0.65 and geometric_ratio < 0.35:
            return 3

        # Very chaotic/complex → unusual time sig
        chaos_score = shape_complexity * 0.5 + texture_energy * 0.5
        if chaos_score > 0.78:
            return 7
        if chaos_score > 0.65:
            return 5

        return 4

    # ------------------------------------------------------------------
    # Main mapping method
    # ------------------------------------------------------------------

    def map(
        self,
        visual_features: Dict[str, float],
        color_data: Dict[str, Any],
        object_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Map all visual inputs to a complete Spotify audio feature dictionary.

        Args:
            visual_features: Dict from VisualFeatureExtractor.extract().
            color_data: Dict from ColorExtractor.extract().
            object_data: Dict from ObjectDetector.detect().

        Returns:
            Complete audio features dict with all 14 keys.

        Raises:
            ValueError: If any produced feature is outside its valid range.
        """
        try:
            mood_tags:         List[str] = object_data.get("mood_tags", [])
            scene_type:        str       = object_data.get("scene_type", "abstract")
            person_detected:   bool      = object_data.get("person_detected", False)
            person_confidence: float     = float(object_data.get("person_confidence", 0.0))
            object_count:      int       = int(object_data.get("object_count", 0))

            # --- Compute features in dependency order ---
            energy = self._map_energy(visual_features, color_data)

            valence = self._map_valence(visual_features, color_data, mood_tags)

            danceability = self._map_danceability(visual_features, color_data, mood_tags)

            acousticness = self._map_acousticness(
                visual_features, color_data, energy, mood_tags
            )

            instrumentalness = self._map_instrumentalness(
                visual_features, color_data, mood_tags,
                scene_type, person_detected, person_confidence
            )

            liveness = self._map_liveness(
                visual_features, color_data, mood_tags,
                person_detected, object_count
            )

            speechiness = self._map_speechiness(
                visual_features, object_count, person_detected, person_confidence
            )

            tempo = self._map_tempo(energy, visual_features)

            loudness = self._map_loudness(energy, visual_features, color_data)

            key = self._map_key(color_data)

            mode = self._map_mode(visual_features, color_data)

            time_signature = self._map_time_signature(visual_features)

            popularity  = 50
            duration_ms = 210000

            # --- Final sanity clamps before assembly ---
            energy           = self._soft_clamp(energy,           0.0,  1.0)
            valence          = self._soft_clamp(valence,          0.0,  1.0)
            danceability     = self._soft_clamp(danceability,     0.0,  1.0)
            acousticness     = self._soft_clamp(acousticness,     0.0,  1.0)
            instrumentalness = self._soft_clamp(instrumentalness, 0.0,  1.0)
            liveness         = self._soft_clamp(liveness,         0.0,  1.0)
            speechiness      = self._soft_clamp(speechiness,      0.0,  1.0)
            tempo            = self._soft_clamp(tempo,           50.0, 200.0)
            loudness         = self._soft_clamp(loudness,       -60.0,  -1.0)

            audio_features: Dict[str, Any] = {
                "acousticness":     float(round(acousticness, 6)),
                "danceability":     float(round(danceability, 6)),
                "duration_ms":      int(duration_ms),
                "energy":           float(round(energy, 6)),
                "instrumentalness": float(round(instrumentalness, 6)),
                "key":              int(key),
                "liveness":         float(round(liveness, 6)),
                "loudness":         float(round(loudness, 4)),
                "mode":             int(mode),
                "speechiness":      float(round(speechiness, 6)),
                "tempo":            float(round(tempo, 4)),
                "time_signature":   int(time_signature),
                "valence":          float(round(valence, 6)),
                "popularity":       int(popularity),
            }

            validate_features(audio_features)
            return audio_features

        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"FeatureMapper.map() failed: {exc}") from exc

