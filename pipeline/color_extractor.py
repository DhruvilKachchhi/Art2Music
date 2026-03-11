"""
Color Palette Extractor
========================
Extracts dominant color palettes from images using KMeans clustering in
CIELAB (perceptually uniform) color space with adaptive K selection.

Key design decisions:
    1. LAB-space clustering  - groups pixels by perceptual similarity, not
       Euclidean RGB distance.  Near-grey pixels cluster correctly; subtle
       hue shifts between similar colors are respected.
    2. Adaptive K selection  - tries K=6..10 and picks the LARGEST K whose
       silhouette score is within 3% of the best.  This deliberately biases
       toward more colors: when K=8 is nearly as clean as K=6 we always
       choose K=8.  A chroma-spread bonus further raises the floor for
       images with many distinct hues.
    3. Perceptual salience weighting - each cluster's final weight is
       proportional to (pixel_area x saturation x brightness), so small but
       vivid accent colors receive more influence than large grey regions.
    4. All aggregate stats (avg_hue, avg_saturation, avg_brightness) and
       hex/RGB values are derived from the LAB-clustered centers converted
       back to HSV/RGB, using circular-mean hue and salience weights.
"""

import colorsys
import math
import warnings
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import silhouette_score


class ColorExtractor:
    """
    Extracts the dominant color palette from an image.

    Clustering is performed in CIELAB space for perceptual accuracy.
    K is chosen automatically in the range [K_MIN, K_MAX] using a
    bias-toward-more-colors silhouette strategy. Cluster weights are
    salience-adjusted (vivid, bright clusters matter more).

    Attributes:
        k_min (int): Minimum number of clusters to try (default: 6).
        k_max (int): Maximum number of clusters to try (default: 10).
        random_state (int): Random seed for KMeans reproducibility.
    """

    K_MIN: int = 6
    K_MAX: int = 10

    def __init__(
        self,
        n_colors: int = 8,       # kept for API compat; used as k_max fallback
        random_state: int = 42,
    ) -> None:
        self.k_min = self.K_MIN
        self.k_max = max(self.K_MIN + 1, min(n_colors, self.K_MAX))
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def _rgb_to_hex(self, rgb: Tuple[int, int, int]) -> str:
        r = int(np.clip(rgb[0], 0, 255))
        g = int(np.clip(rgb[1], 0, 255))
        b = int(np.clip(rgb[2], 0, 255))
        return f"#{r:02X}{g:02X}{b:02X}"

    def _rgb_to_hsv(self, rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Return (hue_degrees [0,360], saturation [0,1], value [0,1])."""
        r_n = np.clip(rgb[0], 0, 255) / 255.0
        g_n = np.clip(rgb[1], 0, 255) / 255.0
        b_n = np.clip(rgb[2], 0, 255) / 255.0
        h, s, v = colorsys.rgb_to_hsv(r_n, g_n, b_n)
        return h * 360.0, s, v

    # ------------------------------------------------------------------
    # Pixel sampling (stratified grid)
    # ------------------------------------------------------------------

    def _sample_pixels(
        self, image: Image.Image, max_pixels: int = 12_000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stratified grid sampling of the image.

        Returns both the RGB array (for hex/HSV back-conversion) and the
        corresponding LAB array (for clustering), so the two are always
        aligned pixel-for-pixel.

        Args:
            image: PIL Image object.
            max_pixels: Upper limit on sampled pixel count.

        Returns:
            Tuple of:
                rgb_pixels  -- (N, 3) uint8 array in RGB.
                lab_pixels  -- (N, 3) float32 array in CIELAB.
        """
        img_rgb = image.convert("RGB")
        arr_rgb = np.array(img_rgb, dtype=np.uint8)   # (H, W, 3)
        H, W = arr_rgb.shape[:2]

        # Convert full image to LAB via OpenCV for consistency
        arr_bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
        arr_lab = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        total = H * W
        if total <= max_pixels:
            return arr_rgb.reshape(-1, 3), arr_lab.reshape(-1, 3)

        # Stratified grid
        n_cols = max(1, int(math.ceil(math.sqrt(max_pixels * W / H))))
        n_rows = max(1, int(math.ceil(max_pixels / n_cols)))
        n_cols = min(n_cols, W)
        n_rows = min(n_rows, H)

        cell_h = H / n_rows
        cell_w = W / n_cols

        rng = np.random.default_rng(self.random_state)
        rgb_pixels, lab_pixels = [], []
        for r in range(n_rows):
            for c in range(n_cols):
                r0, r1 = int(r * cell_h), min(int((r + 1) * cell_h), H)
                c0, c1 = int(c * cell_w), min(int((c + 1) * cell_w), W)
                if r1 > r0 and c1 > c0:
                    ri = rng.integers(r0, r1)
                    ci = rng.integers(c0, c1)
                    rgb_pixels.append(arr_rgb[ri, ci])
                    lab_pixels.append(arr_lab[ri, ci])

        return (
            np.array(rgb_pixels, dtype=np.uint8),
            np.array(lab_pixels, dtype=np.float32),
        )

    # ------------------------------------------------------------------
    # Adaptive K selection
    # ------------------------------------------------------------------

    def _select_best_k(self, lab_pixels: np.ndarray) -> int:
        """
        Select the optimal number of clusters K using a bias-toward-more-colors
        strategy over the range [k_min, k_max].

        Strategy
        --------
        1. Measure chroma spread (std-dev of a* and b* in LAB) to detect
           images with many distinct hues and raise the effective minimum K.
        2. Compute silhouette score for each K on a 2,000-pixel subsample.
        3. Find the best (highest) silhouette score across all evaluated K.
        4. Among all K values whose silhouette is within TOLERANCE (3%) of
           the best, pick the LARGEST K -- so if K=9 is nearly as clean as
           K=7, we always return the richer palette.

        Args:
            lab_pixels: (N, 3) float32 LAB pixel array.

        Returns:
            Best K value (integer in [k_min, k_max]).
        """
        n = len(lab_pixels)
        max_k_possible = min(self.k_max, n - 1)
        if max_k_possible < self.k_min:
            return self.k_min

        # --- Chroma-spread bonus ---
        # Std-dev of the a* and b* channels (chroma axes in LAB) measures
        # how many distinct hues the image contains. High spread -> push the
        # minimum K upward so colourful images always get enough clusters.
        a_std = float(np.std(lab_pixels[:, 1]))
        b_std = float(np.std(lab_pixels[:, 2]))
        chroma_spread = math.sqrt(a_std ** 2 + b_std ** 2)

        if chroma_spread > 40:
            effective_k_min = min(self.k_min + 2, max_k_possible)
        elif chroma_spread > 25:
            effective_k_min = min(self.k_min + 1, max_k_possible)
        else:
            effective_k_min = self.k_min

        # --- Silhouette evaluation on a subsample ---
        sil_sample_size = min(2_000, n)
        rng = np.random.default_rng(self.random_state + 1)
        sample_idx = rng.choice(n, size=sil_sample_size, replace=False)
        sample = lab_pixels[sample_idx]

        scores: Dict[int, float] = {}   # k -> silhouette score

        for k in range(effective_k_min, max_k_possible + 1):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    km = KMeans(
                        n_clusters=k,
                        random_state=self.random_state,
                        n_init=5,
                        max_iter=100,
                    )
                    km.fit(sample)
                labels = km.labels_

                if len(np.unique(labels)) < 2:
                    continue

                score = silhouette_score(sample, labels, metric="euclidean")
                scores[k] = score
            except Exception:
                continue

        if not scores:
            return effective_k_min

        best_score = max(scores.values())

        # Accept any K whose silhouette >= best - TOLERANCE.
        # Among those, pick the LARGEST (more colors preferred).
        TOLERANCE = 0.03
        candidates = [k for k, s in scores.items() if s >= best_score - TOLERANCE]
        return max(candidates)

    # ------------------------------------------------------------------
    # Circular / weighted HSV statistics
    # ------------------------------------------------------------------

    @staticmethod
    def _circular_mean_hue(hues_deg: np.ndarray, weights: np.ndarray) -> float:
        """
        Weighted circular mean of hue angles (handles 0/360 wrap-around).

        Args:
            hues_deg: Array of hue values in [0, 360).
            weights:  Non-negative weights (need not sum to 1).

        Returns:
            Weighted circular mean hue in [0, 360).
        """
        w = weights / (weights.sum() + 1e-12)
        hues_rad = np.deg2rad(hues_deg)
        sin_mean = np.dot(w, np.sin(hues_rad))
        cos_mean = np.dot(w, np.cos(hues_rad))
        mean_deg = math.degrees(math.atan2(sin_mean, cos_mean)) % 360.0
        return float(mean_deg)

    # ------------------------------------------------------------------
    # Perceptual salience weighting
    # ------------------------------------------------------------------

    @staticmethod
    def _salience_weights(
        area_weights: np.ndarray,
        saturations: np.ndarray,
        brightnesses: np.ndarray,
    ) -> np.ndarray:
        """
        Compute perceptual salience weights for each cluster.

        Salience = pixel_area_fraction x saturation x brightness.

        Near-grey (low saturation) or very dark clusters score low even if they
        cover a large area, ensuring that a vivid accent color has meaningful
        influence despite occupying fewer pixels.

        Args:
            area_weights:  (K,) fraction of pixels in each cluster, sums to 1.
            saturations:   (K,) HSV saturation [0, 1] per cluster center.
            brightnesses:  (K,) HSV value [0, 1] per cluster center.

        Returns:
            (K,) salience weights normalised to sum to 1.
        """
        raw = area_weights * (saturations + 0.1) * (brightnesses + 0.1)
        total = raw.sum()
        if total < 1e-12:
            # All clusters are near-black or near-grey -> fall back to area weights
            return area_weights / (area_weights.sum() + 1e-12)
        return raw / total

    # ------------------------------------------------------------------
    # Main extraction
    # ------------------------------------------------------------------

    def extract(self, image: Image.Image) -> Dict[str, Any]:
        """
        Extract the dominant color palette from an image.

        Steps:
            1. Stratified pixel sampling (RGB + LAB aligned).
            2. Adaptive K selection: chroma-spread bonus + largest K within
               silhouette tolerance.
            3. KMeans clustering in LAB space (full pixel set).
            4. Cluster centers converted back to RGB and HSV.
            5. Salience weights computed: area x saturation x brightness.
            6. Aggregate stats derived using salience-weighted circular mean
               hue and weighted average saturation/brightness.

        Args:
            image: PIL Image object to analyze.

        Returns:
            Dictionary containing:
                - dominant_colors (list[dict]): Sorted by salience weight desc.
                    Each entry: rank, hex, rgb, hue, saturation, brightness,
                    weight (area fraction), salience_weight.
                - avg_hue (float): Circular salience-weighted mean hue [0,360].
                - avg_saturation (float): Salience-weighted mean sat [0,1].
                - avg_brightness (float): Salience-weighted mean brightness [0,1].
                - palette_hex (list[str]): HEX codes ordered by salience weight.
                - palette_rgb (list[tuple]): RGB tuples ordered by salience weight.
                - n_colors_extracted (int): Actual number of colors extracted.
                - best_k (int): The K chosen by the selection algorithm.

        Raises:
            ValueError: If the image cannot be processed.
        """
        try:
            rgb_pixels, lab_pixels = self._sample_pixels(image, max_pixels=12_000)

            # --- Step 2: adaptive K ---
            best_k = self._select_best_k(lab_pixels)

            # --- Step 3: final KMeans fit on full sample in LAB space ---
            n_clusters = min(best_k, len(lab_pixels))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    random_state=self.random_state,
                    n_init=10,
                    max_iter=300,
                )
                kmeans.fit(lab_pixels)
            labels      = kmeans.labels_
            lab_centers = kmeans.cluster_centers_   # (K, 3) in LAB

            # --- Step 4: convert LAB cluster centers back to BGR -> RGB ---
            lab_u8     = np.clip(lab_centers, 0, 255).astype(np.uint8)
            lab_u8_img = lab_u8.reshape(1, -1, 3)   # cv2 needs (1, K, 3)
            bgr_centers = cv2.cvtColor(lab_u8_img, cv2.COLOR_LAB2BGR).reshape(-1, 3)
            rgb_centers = bgr_centers[:, ::-1]        # BGR -> RGB

            total_pixels = len(labels)
            label_counts = np.bincount(labels, minlength=n_clusters)
            area_weights = label_counts / total_pixels   # (K,)

            # HSV per cluster center
            hsv_list = [self._rgb_to_hsv(tuple(int(v) for v in c)) for c in rgb_centers]
            hues_arr = np.array([h for h, s, v in hsv_list])
            sats_arr = np.array([s for h, s, v in hsv_list])
            vals_arr = np.array([v for h, s, v in hsv_list])

            # --- Step 5: salience weights ---
            sal_weights = self._salience_weights(area_weights, sats_arr, vals_arr)

            # Sort by descending salience weight
            sorted_idx   = np.argsort(-sal_weights)
            rgb_centers  = rgb_centers[sorted_idx]
            area_weights = area_weights[sorted_idx]
            sal_weights  = sal_weights[sorted_idx]
            hues_arr     = hues_arr[sorted_idx]
            sats_arr     = sats_arr[sorted_idx]
            vals_arr     = vals_arr[sorted_idx]

            # --- Step 6: build output ---
            dominant_colors: List[Dict[str, Any]] = []
            palette_hex: List[str] = []
            palette_rgb: List[Tuple[int, int, int]] = []

            for i in range(n_clusters):
                rgb_tuple = (
                    int(np.clip(rgb_centers[i, 0], 0, 255)),
                    int(np.clip(rgb_centers[i, 1], 0, 255)),
                    int(np.clip(rgb_centers[i, 2], 0, 255)),
                )
                hex_code = self._rgb_to_hex(rgb_tuple)
                dominant_colors.append({
                    "rank":            i + 1,
                    "hex":             hex_code,
                    "rgb":             rgb_tuple,
                    "hue":             round(float(hues_arr[i]), 2),
                    "saturation":      round(float(sats_arr[i]), 4),
                    "brightness":      round(float(vals_arr[i]), 4),
                    "weight":          round(float(area_weights[i]), 4),
                    "salience_weight": round(float(sal_weights[i]), 4),
                })
                palette_hex.append(hex_code)
                palette_rgb.append(rgb_tuple)

            # --- Aggregate stats using salience weights ---
            # Down-weight near-grey clusters further for the hue average
            hue_sal_weights = sal_weights * sats_arr
            if hue_sal_weights.sum() < 1e-9:
                hue_sal_weights = sal_weights   # all near-grey fallback

            avg_hue        = self._circular_mean_hue(hues_arr, hue_sal_weights)
            avg_saturation = float(np.average(sats_arr, weights=sal_weights))
            avg_brightness = float(np.average(vals_arr, weights=sal_weights))

            avg_hue        = float(np.clip(avg_hue,        0.0, 360.0))
            avg_saturation = float(np.clip(avg_saturation, 0.0, 1.0))
            avg_brightness = float(np.clip(avg_brightness, 0.0, 1.0))

            return {
                "dominant_colors":    dominant_colors,
                "avg_hue":            round(avg_hue,        4),
                "avg_saturation":     round(avg_saturation, 4),
                "avg_brightness":     round(avg_brightness, 4),
                "palette_hex":        palette_hex,
                "palette_rgb":        palette_rgb,
                "n_colors_extracted": len(dominant_colors),
                "best_k":             int(best_k),
            }

        except Exception as exc:
            raise ValueError(f"ColorExtractor failed to process image: {exc}") from exc
