"""
Visual Feature Extractor
=========================
Extracts the 7 formal elements of visual art from an image using
OpenCV, PIL, and scikit-image. All output values are normalized
to the range [0.0, 1.0].

The 7 Elements of Art:
    1. LINE    - Edge density + directionality via Canny + Hough lines
    2. SHAPE   - Contour detection; geometric vs. organic classification
    3. FORM    - 3D depth cues: shading gradients, shadow regions, perspective
    4. COLOR   - Hue cluster variety, saturation range, palette breadth
    5. VALUE   - Luminance histogram spread (std dev of L channel in LAB)
    6. SPACE   - Positive/negative space ratio + composition balance
    7. TEXTURE - Gabor filter energy + LBP fine-grained surface texture
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


class VisualFeatureExtractor:
    """
    Extracts the 7 formal elements of art from an image.

    Uses Canny + Hough line detection, contour analysis with approxPolyDP,
    LAB-space luminance statistics, Gabor filter banks, and Local Binary
    Patterns to produce a discriminative, fully-normalized feature dictionary.

    All returned feature values are in the range [0.0, 1.0].
    """

    def __init__(self) -> None:
        """Initialize the VisualFeatureExtractor with default settings."""
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pil_to_cv2_gray(self, image: Image.Image) -> np.ndarray:
        """Convert a PIL Image to an OpenCV grayscale numpy array (uint8)."""
        img_rgb = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        return gray

    def _pil_to_cv2_bgr(self, image: Image.Image) -> np.ndarray:
        """Convert a PIL Image to an OpenCV BGR numpy array (uint8)."""
        img_rgb = np.array(image.convert("RGB"))
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        return bgr

    def _resize_for_processing(
        self, image: np.ndarray, max_dim: int = 512
    ) -> np.ndarray:
        """Resize so the longest side is at most max_dim pixels."""
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return image

    @staticmethod
    def _sigmoid(x: float, center: float = 0.5, steepness: float = 8.0) -> float:
        """Sigmoid centred on `center` with given steepness. Output in (0, 1)."""
        return float(1.0 / (1.0 + np.exp(-steepness * (x - center))))

    # ------------------------------------------------------------------
    # Element 1: LINE
    # ------------------------------------------------------------------
    def extract_line(self, gray: np.ndarray) -> Dict[str, float]:
        """
        Extract LINE element using Canny edge detection + Hough line transform.

        Computes:
            edge_density       — ratio of edge pixels to total pixels [0,1].
            line_straightness  — fraction of detected lines that are
                                 horizontal/vertical vs. diagonal [0,1].
                                 Higher value = more straight/structured lines.
            curved_line_ratio  — complement of straightness [0,1].

        Returns:
            Dict with edge_density, line_straightness, curved_line_ratio.
        """
        h, w = gray.shape[:2]
        total_pixels = h * w

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive Canny thresholds based on image median
        median_val = float(np.median(blurred))
        sigma = 0.33
        low = max(0.0, (1.0 - sigma) * median_val)
        high = min(255.0, (1.0 + sigma) * median_val)
        edges = cv2.Canny(blurred, threshold1=low, threshold2=high)

        edge_pixels = int(np.sum(edges > 0))
        raw_density = float(edge_pixels) / float(total_pixels) if total_pixels > 0 else 0.0
        # Typical edge density spans 0.02–0.30; normalise with sigmoid
        edge_density = self._sigmoid(raw_density, center=0.12, steepness=15.0)

        # Hough probabilistic line detection
        min_line_len = max(10, int(min(h, w) * 0.05))
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=max(20, int(min(h, w) * 0.03)),
            minLineLength=min_line_len,
            maxLineGap=10,
        )

        h_v_count = 0
        total_line_count = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx, dy = x2 - x1, y2 - y1
                angle = abs(np.degrees(np.arctan2(dy, dx))) % 180.0
                total_line_count += 1
                # Within 20° of horizontal (0° or 180°) or vertical (90°)
                if angle < 20 or angle > 160 or (70 < angle < 110):
                    h_v_count += 1

        if total_line_count > 0:
            line_straightness = float(h_v_count) / float(total_line_count)
        else:
            # No Hough lines detected → image has curved/no lines
            line_straightness = 0.3

        curved_line_ratio = float(np.clip(1.0 - line_straightness, 0.0, 1.0))

        return {
            "edge_density": round(float(np.clip(edge_density, 0.0, 1.0)), 6),
            "line_straightness": round(float(np.clip(line_straightness, 0.0, 1.0)), 6),
            "curved_line_ratio": round(float(np.clip(curved_line_ratio, 0.0, 1.0)), 6),
        }

    # ------------------------------------------------------------------
    # Element 2: SHAPE
    # ------------------------------------------------------------------
    def extract_shape(self, gray: np.ndarray) -> Dict[str, float]:
        """
        Extract SHAPE element using contour detection and approxPolyDP.

        Computes:
            shape_complexity   — normalised count of significant contours [0,1].
            geometric_ratio    — fraction of contours classified as geometric
                                 (circles / rectangles / polygons ≤ 8 vertices)
                                 vs. organic (many-vertex / irregular) [0,1].

        Returns:
            Dict with shape_complexity, geometric_ratio.
        """
        h, w = gray.shape[:2]
        total_area = h * w

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        min_area = total_area * 0.001
        significant: List = [c for c in contours if cv2.contourArea(c) > min_area]

        num_contours = len(significant)
        # Normalise: 0 contours → 0.0, ~100+ contours → 1.0 (sigmoid)
        shape_complexity = self._sigmoid(num_contours / 60.0, center=0.5, steepness=6.0)

        geometric_count = 0
        for c in significant:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            n_verts = len(approx)
            # 3–8 vertices → geometric (triangle, quad, pentagon…octagon)
            if 3 <= n_verts <= 8:
                geometric_count += 1

        if num_contours > 0:
            geometric_ratio = float(geometric_count) / float(num_contours)
        else:
            geometric_ratio = 0.5

        return {
            "shape_complexity": round(float(np.clip(shape_complexity, 0.0, 1.0)), 6),
            "geometric_ratio": round(float(np.clip(geometric_ratio, 0.0, 1.0)), 6),
        }

    # ------------------------------------------------------------------
    # Element 3: FORM
    # ------------------------------------------------------------------
    def extract_form(self, gray: np.ndarray) -> Dict[str, float]:
        """
        Extract FORM element: infer 3D depth cues.

        Three sub-signals are combined:
            1. Shading gradient — local luminance variance across tiles.
            2. Shadow detection — fraction of very dark pixels after adaptive
               thresholding (proxy for cast shadows).
            3. Perspective convergence — how many Hough lines converge toward
               a central vanishing point (diagonal line count normalised).

        Returns:
            Dict with depth_estimate in [0,1].
        """
        h, w = gray.shape[:2]

        # --- Sub-signal 1: local luminance variance (shading gradient) ---
        tile_rows, tile_cols = 4, 4
        variances = []
        th, tw = max(1, h // tile_rows), max(1, w // tile_cols)
        for r in range(tile_rows):
            for c in range(tile_cols):
                tile = gray[r * th:(r + 1) * th, c * tw:(c + 1) * tw]
                if tile.size > 0:
                    variances.append(float(np.var(tile.astype(np.float64))))
        local_var_mean = float(np.mean(variances)) if variances else 0.0
        # Normalise: typical luminance variance spans 0–3000+
        shading_score = self._sigmoid(local_var_mean / 1500.0, center=0.5, steepness=5.0)

        # --- Sub-signal 2: shadow detection ---
        # Adaptive threshold + morph to isolate shadow-like dark regions
        adapt = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 10
        )
        shadow_pixels = int(np.sum(adapt > 0))
        shadow_fraction = float(shadow_pixels) / float(h * w) if h * w > 0 else 0.0
        shadow_score = self._sigmoid(shadow_fraction, center=0.15, steepness=12.0)

        # --- Sub-signal 3: perspective convergence (diagonal Hough lines) ---
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        median_val = float(np.median(blurred))
        sigma = 0.33
        edges = cv2.Canny(
            blurred,
            threshold1=max(0.0, (1.0 - sigma) * median_val),
            threshold2=min(255.0, (1.0 + sigma) * median_val),
        )
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180,
            threshold=max(15, int(min(h, w) * 0.03)),
            minLineLength=max(10, int(min(h, w) * 0.05)),
            maxLineGap=10,
        )
        diagonal_count = 0
        total_l = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180.0
                total_l += 1
                if 20 <= angle <= 70 or 110 <= angle <= 160:
                    diagonal_count += 1
        persp_score = float(diagonal_count) / float(max(1, total_l))

        # Weighted combination
        depth_estimate = (shading_score * 0.45) + (shadow_score * 0.30) + (persp_score * 0.25)

        return {"depth_estimate": round(float(np.clip(depth_estimate, 0.0, 1.0)), 6)}

    # ------------------------------------------------------------------
    # Element 4: COLOR
    # ------------------------------------------------------------------
    def extract_color_element(
        self,
        avg_hue: float,
        avg_saturation: float,
        avg_brightness: float,
        dominant_colors: list,
        bgr_image: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Extract COLOR element from precomputed color palette + optional BGR image.

        Scoring is based on:
            • Hue cluster variety — distinct hue clusters from K-means on the
              full image (or dominant_colors list), normalised.
            • Saturation range — spread of saturation across dominant colours.
            • Palette breadth — combined measure of hue variety × saturation.

        Returns:
            Dict with hue_variety, saturation_range, palette_breadth,
            saturation_level, brightness_level.
        """
        # --- Hue cluster variety from dominant_colors ---
        if dominant_colors and len(dominant_colors) >= 2:
            hues = np.array([c["hue"] for c in dominant_colors], dtype=np.float64)
            sats = np.array([c.get("saturation", 0.5) for c in dominant_colors], dtype=np.float64)

            # Angular std dev on hue circle (degrees)
            hue_rad = np.deg2rad(hues)
            mean_sin = np.mean(np.sin(hue_rad))
            mean_cos = np.mean(np.cos(hue_rad))
            circ_std_rad = np.sqrt(-2.0 * np.log(np.clip(np.sqrt(mean_sin ** 2 + mean_cos ** 2), 1e-9, 1.0)))
            circ_std_deg = float(np.degrees(circ_std_rad))

            # Normalise: max circular std ~103° for uniform distribution
            hue_variety = self._sigmoid(circ_std_deg / 90.0, center=0.5, steepness=5.0)
            saturation_range = float(np.clip(np.ptp(sats), 0.0, 1.0))  # peak-to-peak
        else:
            hue_variety = 0.3
            saturation_range = 0.0

        # If raw BGR image available, also run K-means to count hue clusters
        if bgr_image is not None:
            try:
                hsv_img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
                h_channel = hsv_img[:, :, 0].flatten().astype(np.float32)  # 0-179 in OpenCV
                s_channel = hsv_img[:, :, 1].flatten().astype(np.float32)
                # Only consider sufficiently saturated pixels
                mask = s_channel > 30
                if np.sum(mask) > 100:
                    hue_pixels = h_channel[mask].reshape(-1, 1)
                    k = min(8, max(2, len(np.unique(hue_pixels.astype(int)))))
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
                    _, labels, _ = cv2.kmeans(
                        hue_pixels, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS
                    )
                    unique_clusters = len(np.unique(labels))
                    # Blend K-means cluster count into hue_variety
                    kmeans_hv = self._sigmoid(unique_clusters / 8.0, center=0.5, steepness=6.0)
                    hue_variety = float(np.clip((hue_variety * 0.5 + kmeans_hv * 0.5), 0.0, 1.0))
            except Exception:
                pass  # Fallback to dominant_colors-based score

        saturation_level = float(np.clip(avg_saturation, 0.0, 1.0))
        brightness_level = float(np.clip(avg_brightness, 0.0, 1.0))

        # Palette breadth: how vivid AND varied is the image?
        palette_breadth = float(np.clip(hue_variety * 0.6 + saturation_level * 0.4, 0.0, 1.0))

        return {
            "hue_variety": round(float(np.clip(hue_variety, 0.0, 1.0)), 6),
            "saturation_range": round(float(np.clip(saturation_range, 0.0, 1.0)), 6),
            "palette_breadth": round(float(np.clip(palette_breadth, 0.0, 1.0)), 6),
            "saturation_level": round(saturation_level, 6),
            "brightness_level": round(brightness_level, 6),
        }

    # ------------------------------------------------------------------
    # Element 5: VALUE
    # ------------------------------------------------------------------
    def extract_value(self, gray: np.ndarray) -> Dict[str, float]:
        """
        Extract VALUE element: luminance histogram spread.

        Uses the standard deviation of the L* channel in CIE LAB colour space
        as the primary signal (more perceptually uniform than raw grayscale).
        Also computes the histogram entropy to capture tonal richness.

        Returns:
            Dict with contrast_ratio (std dev based) and value_range
            (histogram spread from 5th–95th percentile). Both in [0,1].
        """
        # Convert to LAB and extract L channel
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0].astype(np.float64)  # 0–255 in OpenCV LAB

        std_dev = float(np.std(L))
        # L std dev: ~0 (uniform) to ~80 (max contrast); normalise with sigmoid
        contrast_ratio = self._sigmoid(std_dev / 60.0, center=0.5, steepness=6.0)

        # Percentile-based value range (robust to outliers)
        p5 = float(np.percentile(L, 5))
        p95 = float(np.percentile(L, 95))
        value_range = float(np.clip((p95 - p5) / 255.0, 0.0, 1.0))

        # Weighted blend
        contrast_ratio = float(np.clip(contrast_ratio * 0.6 + value_range * 0.4, 0.0, 1.0))

        return {
            "contrast_ratio": round(contrast_ratio, 6),
            "value_range": round(value_range, 6),
        }

    # ------------------------------------------------------------------
    # Element 6: SPACE
    # ------------------------------------------------------------------
    def extract_space(self, gray: np.ndarray) -> Dict[str, float]:
        """
        Extract SPACE element: positive vs. negative space ratio.

        Combines two signals:
            1. Texture variance segmentation — bright/uniform regions
               (low local variance) = negative space; textured = positive.
            2. Centre-weighted composition balance — how much visual mass
               sits at the image centre vs. periphery.

        Returns:
            Dict with negative_space_ratio and composition_balance. Both [0,1].
        """
        h, w = gray.shape[:2]
        total_pixels = h * w

        # --- Signal 1: texture-based segmentation ---
        # Compute local variance in a neighbourhood; low var = uniform (neg space)
        gray_f = gray.astype(np.float32)
        mean_sq = cv2.blur(gray_f ** 2, (15, 15))
        mean_ = cv2.blur(gray_f, (15, 15))
        local_var = mean_sq - mean_ ** 2
        # Threshold: pixels with variance below 100 are "uniform/background"
        negative_mask = (local_var < 100).astype(np.uint8)
        neg_pixels = int(np.sum(negative_mask))
        negative_space_ratio = float(neg_pixels) / float(total_pixels) if total_pixels > 0 else 0.5
        negative_space_ratio = float(np.clip(negative_space_ratio, 0.0, 1.0))

        # --- Signal 2: centre-weighted visual mass ---
        # Create a Gaussian weight map centred on the image
        cy, cx = h / 2.0, w / 2.0
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        sigma_y, sigma_x = h / 4.0, w / 4.0
        gauss_weight = np.exp(
            -((y_coords - cy) ** 2 / (2 * sigma_y ** 2)
              + (x_coords - cx) ** 2 / (2 * sigma_x ** 2))
        )
        # Foreground = active / high-var pixels
        foreground_mask = (local_var >= 100).astype(np.float64)
        centre_mass = float(np.sum(foreground_mask * gauss_weight))
        total_gauss = float(np.sum(gauss_weight))
        composition_balance = float(np.clip(centre_mass / max(total_gauss, 1e-9), 0.0, 1.0))

        return {
            "negative_space_ratio": round(negative_space_ratio, 6),
            "composition_balance": round(composition_balance, 6),
        }

    # ------------------------------------------------------------------
    # Element 7: TEXTURE
    # ------------------------------------------------------------------
    def extract_texture(self, gray: np.ndarray) -> Dict[str, float]:
        """
        Extract TEXTURE element using Gabor filters + Local Binary Patterns.

        Gabor filter bank at 4 orientations × 3 scales captures frequency-
        specific texture energy. LBP captures fine-grained surface micro-texture.

        Returns:
            Dict with gabor_energy, lbp_uniformity, texture_energy,
            texture_homogeneity. All in [0,1].
        """
        small = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
        small_f = small.astype(np.float32) / 255.0

        # --- Gabor filter bank ---
        orientations = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        wavelengths = [4.0, 8.0, 16.0]  # in pixels
        gabor_responses = []
        for theta in orientations:
            for lam in wavelengths:
                sigma = lam * 0.5
                gamma = 0.5
                kernel = cv2.getGaborKernel(
                    (21, 21), sigma=sigma, theta=theta,
                    lambd=lam, gamma=gamma, psi=0, ktype=cv2.CV_32F
                )
                resp = cv2.filter2D(small_f, cv2.CV_32F, kernel)
                energy = float(np.mean(resp ** 2))
                gabor_responses.append(energy)

        # Mean Gabor energy across all kernels; normalise with sigmoid
        mean_gabor = float(np.mean(gabor_responses))
        gabor_energy = self._sigmoid(mean_gabor / 0.05, center=0.5, steepness=5.0)

        # --- Local Binary Patterns ---
        lbp = local_binary_pattern(small, P=8, R=1, method="uniform")
        n_bins = 10  # 'uniform' LBP has P*(P-1)+3 = 59 patterns; 10-bin histogram
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype(np.float64)
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist /= hist_sum
        # Entropy of LBP distribution: high entropy = complex texture
        eps = 1e-12
        lbp_entropy = float(-np.sum(hist * np.log(hist + eps)))
        # Max entropy for 10 bins ≈ ln(10) ≈ 2.3
        lbp_uniformity = float(np.clip(lbp_entropy / np.log(n_bins + eps), 0.0, 1.0))

        # Composite texture energy (Gabor-weighted, LBP-modulated)
        texture_energy = float(np.clip(gabor_energy * 0.6 + lbp_uniformity * 0.4, 0.0, 1.0))

        # Texture homogeneity (inverse of energy — smooth surfaces score high)
        texture_homogeneity = float(np.clip(1.0 - texture_energy, 0.0, 1.0))

        return {
            "gabor_energy": round(gabor_energy, 6),
            "lbp_uniformity": round(lbp_uniformity, 6),
            "texture_energy": round(texture_energy, 6),
            "texture_homogeneity": round(texture_homogeneity, 6),
        }

    # ------------------------------------------------------------------
    # Main extraction method
    # ------------------------------------------------------------------
    def extract(
        self,
        image: Image.Image,
        color_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Extract all 7 formal elements of art from the given image.

        Args:
            image: PIL Image object to analyse.
            color_data: Optional colour palette dict from ColorExtractor.extract().
                        If None, colour element features default to mid-range values.

        Returns:
            Flat dictionary with all visual features, each in [0.0, 1.0].

        Raises:
            ValueError: If the image cannot be processed.
        """
        try:
            bgr = self._pil_to_cv2_bgr(image)
            bgr = self._resize_for_processing(bgr, max_dim=512)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            # Element 1: LINE
            line_features = self.extract_line(gray)

            # Element 2: SHAPE
            shape_features = self.extract_shape(gray)

            # Element 3: FORM
            form_features = self.extract_form(gray)

            # Element 4: COLOR
            if color_data is not None:
                color_features = self.extract_color_element(
                    avg_hue=color_data.get("avg_hue", 180.0),
                    avg_saturation=color_data.get("avg_saturation", 0.5),
                    avg_brightness=color_data.get("avg_brightness", 0.5),
                    dominant_colors=color_data.get("dominant_colors", []),
                    bgr_image=bgr,
                )
            else:
                color_features = {
                    "hue_variety": 0.4,
                    "saturation_range": 0.3,
                    "palette_breadth": 0.4,
                    "saturation_level": 0.5,
                    "brightness_level": 0.5,
                }

            # Element 5: VALUE
            value_features = self.extract_value(gray)

            # Element 6: SPACE
            space_features = self.extract_space(gray)

            # Element 7: TEXTURE
            texture_features = self.extract_texture(gray)

            all_features: Dict[str, float] = {}
            all_features.update(line_features)
            all_features.update(shape_features)
            all_features.update(form_features)
            all_features.update(color_features)
            all_features.update(value_features)
            all_features.update(space_features)
            all_features.update(texture_features)

            # Final sanity clamp
            for key, val in all_features.items():
                all_features[key] = round(float(np.clip(val, 0.0, 1.0)), 6)

            return all_features

        except Exception as exc:
            raise ValueError(
                f"VisualFeatureExtractor failed to process image: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Summary aggregation
    # ------------------------------------------------------------------
    def get_element_summary(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Summarise the 7 elements as single scalar scores for display.

        Args:
            features: Full feature dict from extract().

        Returns:
            Dictionary with one float per element: LINE, SHAPE, FORM,
            COLOR, VALUE, SPACE, TEXTURE — each in [0.0, 1.0].
        """
        # LINE: edge density weighted with directional complexity
        line_score = float(np.clip(
            features.get("edge_density", 0.0) * 0.6
            + (1.0 - features.get("line_straightness", 0.5)) * 0.4,
            0.0, 1.0
        ))

        # SHAPE: complexity blended with geometric character
        shape_score = float(np.clip(
            features.get("shape_complexity", 0.0) * 0.7
            + features.get("geometric_ratio", 0.0) * 0.3,
            0.0, 1.0
        ))

        # FORM: single depth estimate
        form_score = features.get("depth_estimate", 0.0)

        # COLOR: hue variety + palette breadth
        color_score = float(np.clip(
            features.get("hue_variety", 0.0) * 0.5
            + features.get("palette_breadth", 0.0) * 0.3
            + features.get("saturation_level", 0.0) * 0.2,
            0.0, 1.0
        ))

        # VALUE: contrast ratio (already a composite)
        value_score = features.get("contrast_ratio", 0.0)

        # SPACE: negative space weighted with composition balance
        space_score = float(np.clip(
            features.get("negative_space_ratio", 0.0) * 0.6
            + features.get("composition_balance", 0.0) * 0.4,
            0.0, 1.0
        ))

        # TEXTURE: Gabor energy + LBP
        texture_score = float(np.clip(
            features.get("gabor_energy", 0.0) * 0.5
            + features.get("lbp_uniformity", 0.0) * 0.3
            + features.get("texture_energy", 0.0) * 0.2,
            0.0, 1.0
        ))

        return {
            "LINE":    round(float(np.clip(line_score,    0.0, 1.0)), 4),
            "SHAPE":   round(float(np.clip(shape_score,   0.0, 1.0)), 4),
            "FORM":    round(float(np.clip(form_score,    0.0, 1.0)), 4),
            "COLOR":   round(float(np.clip(color_score,   0.0, 1.0)), 4),
            "VALUE":   round(float(np.clip(value_score,   0.0, 1.0)), 4),
            "SPACE":   round(float(np.clip(space_score,   0.0, 1.0)), 4),
            "TEXTURE": round(float(np.clip(texture_score, 0.0, 1.0)), 4),
        }
