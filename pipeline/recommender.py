"""
Music Recommender
==================
Loads the trained recommendation model bundle and uses weighted Euclidean
distance (with L2-normalised feature vectors) to find the top-N most similar
tracks to the input audio feature vector.

Improvements over the baseline:
    • Weighted Euclidean distance — features are given individual importance
      weights before distance computation.
    • L2-normalisation — all vectors are unit-normalised so no single
      high-magnitude feature (raw tempo, loudness dB) can dominate.
    • Diversity re-ranking — artist clustering is penalised so the top-N
      results span multiple artists.
    • Outlier guard — tracks with cosine similarity < 0.5 are flagged as
      low_confidence in the returned dict.
    • Bounded similarity score — distance converted to [0, 1] via
      score = 1 / (1 + distance).

The recommendation vector uses EXACTLY 12 features:
    acousticness, danceability, energy, instrumentalness, key,
    liveness, loudness, mode, speechiness, tempo, time_signature, valence

duration_ms and popularity are EXPLICITLY EXCLUDED from all similarity
computations.
"""

import os
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Feature list (never modify)
# ---------------------------------------------------------------------------
RECOMMENDATION_FEATURES = [
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "key",
    "liveness",
    "loudness",
    "mode",
    "speechiness",
    "tempo",
    "time_signature",
    "valence",
]

EXCLUDED_FEATURES = ["duration_ms", "popularity"]

# ---------------------------------------------------------------------------
# Per-feature importance weights for weighted Euclidean distance.
# Weights correspond to RECOMMENDATION_FEATURES order above.
# ---------------------------------------------------------------------------
_FEATURE_WEIGHTS: Dict[str, float] = {
    "acousticness":     1.2,
    "danceability":     1.5,
    "energy":           2.0,
    "instrumentalness": 1.0,
    "key":              0.5,   # low weight — key similarity is a nice-to-have
    "liveness":         0.8,
    "loudness":         1.0,   # already on a [0,1] scale after normalisation
    "mode":             0.5,   # binary, low intrinsic weight
    "speechiness":      0.8,
    "tempo":            1.5,   # normalised before weighting
    "time_signature":   0.6,
    "valence":          2.0,
}

# Minimum cosine similarity below which a track is flagged as low-confidence
_LOW_CONFIDENCE_THRESHOLD = 0.50

# Diversity penalty applied to the distance score of an artist duplicate
_ARTIST_DUPLICATE_PENALTY = 0.20

# Key name lookup table (MIDI pitch class → note name)
KEY_NAMES = {
    -1: "N/A",
    0: "C",
    1: "C♯/D♭",
    2: "D",
    3: "D♯/E♭",
    4: "E",
    5: "F",
    6: "F♯/G♭",
    7: "G",
    8: "G♯/A♭",
    9: "A",
    10: "A♯/B♭",
    11: "B",
}


class MusicRecommender:
    """
    Music recommendation engine using weighted Euclidean distance on
    L2-normalised, per-feature-weighted audio feature vectors.

    Attributes:
        model_path (str): Path to the saved .pkl model bundle.
        model_bundle (dict | None): Loaded model components.
        is_loaded (bool): Whether the model was successfully loaded.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        if model_path is None:
            model_path = os.path.join("models", "recommender_model.pkl")
        self.model_path = model_path
        self.model_bundle: Optional[Dict[str, Any]] = None
        self.is_loaded: bool = False
        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the saved model bundle from disk."""
        try:
            if not os.path.exists(self.model_path):
                print(
                    f"[MusicRecommender] Model not found at '{self.model_path}'. "
                    "Run scripts/train_recommender.py first."
                )
                return
            self.model_bundle = joblib.load(self.model_path)
            self.is_loaded = True
            print(
                f"[MusicRecommender] Model loaded from '{self.model_path}'. "
                f"Dataset size: {len(self.model_bundle.get('dataset', []))} tracks."
            )
        except Exception as exc:
            print(f"[MusicRecommender] Failed to load model: {exc}")
            self.model_bundle = None
            self.is_loaded = False

    # ------------------------------------------------------------------
    # Vector utilities
    # ------------------------------------------------------------------

    def _extract_recommendation_vector(
        self, audio_features: Dict[str, Any]
    ) -> np.ndarray:
        """
        Extract and validate the 12-feature recommendation vector.

        Returns:
            NumPy array of shape (1, 12).

        Raises:
            AssertionError: If excluded features sneak into the vector.
            KeyError: If any required feature is missing.
        """
        for excluded in EXCLUDED_FEATURES:
            assert excluded not in RECOMMENDATION_FEATURES, (
                f"CRITICAL: '{excluded}' must NEVER appear in RECOMMENDATION_FEATURES!"
            )
        vector = [float(audio_features[f]) for f in RECOMMENDATION_FEATURES]
        assert len(vector) == 12
        return np.array(vector, dtype=np.float64).reshape(1, -1)

    def _normalize_input(self, input_vector: np.ndarray) -> np.ndarray:
        """Normalise the input vector using the saved MinMaxScaler."""
        if not self.is_loaded or self.model_bundle is None:
            raise RuntimeError("Model not loaded.")
        scaler = self.model_bundle["scaler"]
        return scaler.transform(input_vector)

    def _apply_feature_weights(self, matrix: np.ndarray) -> np.ndarray:
        """
        Apply per-feature importance weights to a (N × 12) normalised matrix.

        The weight array is built in RECOMMENDATION_FEATURES order and applied
        as an element-wise scale factor before L2 normalisation, so that
        high-weight features have proportionally more influence on the distance.

        Args:
            matrix: (N, 12) float64 array of MinMax-normalised feature vectors.

        Returns:
            (N, 12) weighted array — NOT yet L2-normalised.
        """
        weight_vec = np.array(
            [_FEATURE_WEIGHTS[f] for f in RECOMMENDATION_FEATURES],
            dtype=np.float64,
        )
        return matrix * weight_vec  # broadcast across rows

    @staticmethod
    def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
        """
        L2-normalise each row of a matrix so every vector has unit length.
        Rows with zero norm are left unchanged (zero vector).

        Args:
            matrix: (N, D) float64 array.

        Returns:
            (N, D) L2-normalised array.
        """
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # avoid div-by-zero
        return matrix / norms

    # ------------------------------------------------------------------
    # Similarity computation
    # ------------------------------------------------------------------

    def _compute_weighted_distance(
        self,
        query_weighted_norm: np.ndarray,   # (1, 12)
        dataset_weighted_norm: np.ndarray, # (N, 12)
    ) -> np.ndarray:
        """
        Compute the weighted Euclidean distance between the query and every
        dataset track.  Because both matrices are already weight-scaled and
        L2-normalised, this is equivalent to cosine distance on the weighted
        space.

        Args:
            query_weighted_norm: (1, 12) weighted + L2-normalised query row.
            dataset_weighted_norm: (N, 12) weighted + L2-normalised dataset.

        Returns:
            (N,) array of distances (lower = more similar).
        """
        diff = dataset_weighted_norm - query_weighted_norm   # broadcast
        distances = np.linalg.norm(diff, axis=1)
        return distances

    def _compute_cosine_similarity_raw(
        self,
        query_norm: np.ndarray,    # (1, 12) MinMax-normalised only
        dataset_norm: np.ndarray,  # (N, 12) MinMax-normalised only
    ) -> np.ndarray:
        """
        Cosine similarity on the MinMax-normalised (unweighted) vectors.
        Used solely for the low-confidence threshold check.

        Returns:
            (N,) array of cosine similarities in [-1, 1].
        """
        return cosine_similarity(query_norm, dataset_norm)[0]

    # ------------------------------------------------------------------
    # Diversity re-ranking
    # ------------------------------------------------------------------

    @staticmethod
    def _diversity_rerank(
        indices: np.ndarray,
        distances: np.ndarray,
        dataset: pd.DataFrame,
        penalty: float = _ARTIST_DUPLICATE_PENALTY,
    ) -> List[int]:
        """
        Apply artist-diversity penalty to the ranked candidate list.

        If two candidates share the same artist, the lower-ranked one has its
        effective distance increased by `penalty`, which may cause it to be
        demoted below a more diverse candidate further down the list.

        The re-ranking is done on a candidate pool of 3× the requested top_n
        (pre-sorted by raw distance) so there are enough alternatives available.

        Args:
            indices: Candidate indices pre-sorted by ascending distance.
            distances: Corresponding distances.
            dataset: The full dataset DataFrame.
            penalty: Fractional distance penalty for artist duplicates.

        Returns:
            Re-ranked list of dataset row indices.
        """
        seen_artists: set = set()
        effective_distances = dict(zip(indices.tolist(), distances.tolist()))

        # Apply penalty to duplicates in-place
        for idx in indices:
            artist = str(dataset.iloc[int(idx)].get("artist_name", "")).strip().lower()
            if artist in seen_artists:
                effective_distances[int(idx)] *= (1.0 + penalty)
            else:
                seen_artists.add(artist)

        # Re-sort by effective distance
        reranked = sorted(effective_distances.keys(), key=lambda i: effective_distances[i])
        return reranked

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------

    def _format_track(
        self,
        row: pd.Series,
        rank: int,
        similarity_score: float,
        low_confidence: bool = False,
    ) -> Dict[str, Any]:
        """Format a dataset row into a recommendation result dictionary."""
        key_int  = int(row.get("key", 0))
        mode_int = int(row.get("mode", 1))
        return {
            "rank":              rank,
            "track_name":        str(row.get("track_name",  "Unknown Track")),
            "artist_name":       str(row.get("artist_name", "Unknown Artist")),
            "similarity_score":  round(float(similarity_score), 4),
            "low_confidence":    bool(low_confidence),
            "energy":            round(float(row.get("energy",           0.5)),  4),
            "valence":           round(float(row.get("valence",          0.5)),  4),
            "danceability":      round(float(row.get("danceability",     0.5)),  4),
            "tempo":             round(float(row.get("tempo",            120.0)), 2),
            "mode":              mode_int,
            "mode_name":         "Major" if mode_int == 1 else "Minor",
            "key":               key_int,
            "key_name":          KEY_NAMES.get(key_int, str(key_int)),
            "acousticness":      round(float(row.get("acousticness",     0.5)),  4),
            "instrumentalness":  round(float(row.get("instrumentalness", 0.0)),  4),
            "liveness":          round(float(row.get("liveness",         0.1)),  4),
            "speechiness":       round(float(row.get("speechiness",      0.05)), 4),
            "time_signature":    int(row.get("time_signature", 4)),
            "loudness":          round(float(row.get("loudness",         -8.0)), 2),
            "duration_ms":       int(row.get("duration_ms", 210000)),
        }

    # ------------------------------------------------------------------
    # Main recommendation method
    # ------------------------------------------------------------------

    def recommend(
        self,
        audio_features: Dict[str, Any],
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find the top-N most similar tracks to the given audio features.

        Pipeline:
            1. Extract the 12-feature vector.
            2. MinMax-normalise via saved scaler.
            3. Apply per-feature importance weights.
            4. L2-normalise both query and dataset vectors (unit sphere).
            5. Compute weighted Euclidean distances.
            6. Diversity re-rank (artist penalty on 3× top_n candidate pool).
            7. Flag low-confidence matches (cosine similarity < 0.5).
            8. Convert distance to bounded similarity score: 1/(1+d).

        Args:
            audio_features: Complete audio features dict from FeatureMapper.map().
            top_n: Number of recommendations to return (default 5).

        Returns:
            List of up to top_n track dicts, sorted by similarity descending.
            Each dict includes `similarity_score` and `low_confidence` keys.

        Raises:
            RuntimeError: If the model has not been loaded.
            KeyError: If required features are missing.
        """
        if not self.is_loaded or self.model_bundle is None:
            raise RuntimeError(
                "Recommendation model is not loaded. "
                "Run scripts/train_recommender.py then restart the application."
            )

        # Safety assertions
        assert "duration_ms" not in RECOMMENDATION_FEATURES
        assert "popularity"  not in RECOMMENDATION_FEATURES

        # --- Step 1: extract raw vector ---
        input_vector = self._extract_recommendation_vector(audio_features)

        # --- Step 2: MinMax-normalise ---
        dataset: pd.DataFrame          = self.model_bundle["dataset"]
        feature_columns: List[str]     = self.model_bundle["feature_columns"]
        scaler                         = self.model_bundle["scaler"]

        assert "duration_ms" not in feature_columns, (
            "CRITICAL: duration_ms found in model feature_columns!"
        )
        assert "popularity" not in feature_columns, (
            "CRITICAL: popularity found in model feature_columns!"
        )

        normalized_input: np.ndarray = self._normalize_input(input_vector)  # (1, 12)

        dataset_features = dataset[feature_columns].values.astype(np.float64)
        normalized_dataset: np.ndarray = scaler.transform(dataset_features)  # (N, 12)

        # --- Step 3 & 4: weight + L2-normalise ---
        q_weighted = self._apply_feature_weights(normalized_input)      # (1, 12)
        d_weighted = self._apply_feature_weights(normalized_dataset)    # (N, 12)

        q_norm = self._l2_normalize(q_weighted)   # (1, 12)
        d_norm = self._l2_normalize(d_weighted)   # (N, 12)

        # --- Step 5: weighted Euclidean distance ---
        distances = self._compute_weighted_distance(q_norm, d_norm)  # (N,)

        # --- Step 6: diversity re-rank on 3× candidate pool ---
        pool_size = min(top_n * 3, len(distances))
        candidate_indices = np.argsort(distances)[:pool_size]
        candidate_distances = distances[candidate_indices]

        reranked_indices = self._diversity_rerank(
            candidate_indices, candidate_distances, dataset
        )
        top_indices = reranked_indices[:top_n]

        # --- Step 7: cosine similarity for low-confidence flag ---
        cos_sims = self._compute_cosine_similarity_raw(
            normalized_input, normalized_dataset
        )

        # --- Step 8: format results ---
        recommendations: List[Dict[str, Any]] = []
        for rank, idx in enumerate(top_indices, start=1):
            idx = int(idx)
            dist = float(distances[idx])
            cos_sim = float(np.clip(cos_sims[idx], -1.0, 1.0))

            # Bounded similarity score in [0, 1]
            similarity_score = float(np.clip(1.0 / (1.0 + dist), 0.0, 1.0))

            low_confidence = bool(cos_sim < _LOW_CONFIDENCE_THRESHOLD)

            row = dataset.iloc[idx]
            track = self._format_track(row, rank, similarity_score, low_confidence)
            recommendations.append(track)

        return recommendations

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def recommend_with_fallback(
        self,
        audio_features: Dict[str, Any],
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Recommend tracks, falling back to the KNN model if the primary method fails.

        Args:
            audio_features: Complete audio features dict from FeatureMapper.map().
            top_n: Number of recommendations to return (default 5).

        Returns:
            List of track dicts in the same format as recommend().
        """
        try:
            return self.recommend(audio_features, top_n=top_n)
        except Exception as primary_exc:
            print(f"[MusicRecommender] Primary method failed: {primary_exc}")
            print("[MusicRecommender] Attempting KNN fallback…")

            try:
                if not self.is_loaded or self.model_bundle is None:
                    raise RuntimeError("Model not loaded")

                knn_model = self.model_bundle.get("model")
                if knn_model is None:
                    raise RuntimeError("KNN model not found in bundle")

                input_vector    = self._extract_recommendation_vector(audio_features)
                normalized_input = self._normalize_input(input_vector)

                dataset: pd.DataFrame       = self.model_bundle["dataset"]
                feature_columns: List[str]  = self.model_bundle["feature_columns"]
                scaler                      = self.model_bundle["scaler"]

                dataset_features   = dataset[feature_columns].values.astype(np.float64)
                normalized_dataset = scaler.transform(dataset_features)

                # Apply weights + L2 norm for consistency
                q_w = self._l2_normalize(self._apply_feature_weights(normalized_input))
                d_w = self._l2_normalize(self._apply_feature_weights(normalized_dataset))

                distances_knn, indices_knn = knn_model.kneighbors(
                    q_w, n_neighbors=min(top_n, len(dataset))
                )

                cos_sims = self._compute_cosine_similarity_raw(
                    normalized_input, normalized_dataset
                )

                recommendations: List[Dict[str, Any]] = []
                for rank, (dist, idx) in enumerate(
                    zip(distances_knn[0], indices_knn[0]), start=1
                ):
                    idx = int(idx)
                    dist = float(dist)
                    similarity_score = float(np.clip(1.0 / (1.0 + dist), 0.0, 1.0))
                    cos_sim = float(np.clip(cos_sims[idx], -1.0, 1.0))
                    low_confidence = bool(cos_sim < _LOW_CONFIDENCE_THRESHOLD)

                    row = dataset.iloc[idx]
                    track = self._format_track(row, rank, similarity_score, low_confidence)
                    recommendations.append(track)

                return recommendations

            except Exception as fallback_exc:
                print(f"[MusicRecommender] KNN fallback also failed: {fallback_exc}")
                return []

