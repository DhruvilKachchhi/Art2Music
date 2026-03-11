# 🎨 Art to Music

```
 █████╗ ██████╗ ████████╗    ████████╗ ██████╗     ███╗   ███╗██╗   ██╗███████╗██╗ ██████╗
██╔══██╗██╔══██╗╚══██╔══╝       ██╔══╝██╔═══██╗    ████╗ ████║██║   ██║██╔════╝██║██╔════╝
███████║██████╔╝   ██║          ██║   ██║   ██║    ██╔████╔██║██║   ██║███████╗██║██║
██╔══██║██╔══██╗   ██║          ██║   ██║   ██║    ██║╚██╔╝██║██║   ██║╚════██║██║██║
██║  ██║██║  ██║   ██║          ██║   ╚██████╔╝    ██║ ╚═╝ ██║╚██████╔╝███████║██║╚██████╗
╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝          ╚═╝    ╚═════╝     ╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚═╝ ╚═════╝
```

**Transform Visual Art into Sound** — Upload any image and receive music recommendations
matched to your artwork's visual characteristics.

---

## ✨ Features

- 🎨 **Perceptual Color Palette Extraction** — LAB-space KMeans with adaptive K selection (3–8 clusters via silhouette score) and salience-weighted circular mean hue
- 👁️ **7 Elements of Art Analysis** — LINE, SHAPE, FORM, COLOR, VALUE, SPACE, TEXTURE via Canny, Hough transforms, Gabor filter banks, LBP, and GLCM
- 🔍 **Three-Layer Object Detection** — YOLOv11n + Places365 ResNet-50 scene classifier + CLIP ViT-B/32 zero-shot verification with label disambiguation
- 🎭 **11-Category Scene Classification** — Landscape, Portraiture, Still Life, Genre, Historical, Marine, Cityscape, Abstract, Surrealism, Action, Religious
- 🎵 **Sigmoid-Based Visual-to-Audio Mapping** — 12 Spotify audio features mapped through weighted, sigmoid-smoothed formulas with hue warmth scoring
- 🎯 **Weighted Euclidean Recommendation Engine** — Per-feature importance weights, L2-normalised vectors, artist diversity re-ranking, low-confidence flagging
- 🌐 **Spotify Embed Integration** — Live in-app playback via Spotify Web Player iframes (Client Credentials API)
- 📊 **Interactive Radar Chart** — Plotly spider chart of all audio features with actual-value hover tooltips
- 🎛️ **Rich HSV Tooltip System** — Hover/click tooltips on Hue, Saturation, Brightness badges with gradient visualisers
- 🌑 **Dark UI** — Deep purple → dark blue gradient theme with custom CSS animations and swatch copy-to-clipboard
- ⚡ **Fast Pipeline** — Cached model loading via `@st.cache_resource`, 512px image downscaling for processing

---

## 🏗️ Architecture

```
                         ┌──────────────────────┐
                         │    Input Image        │
                         │  (JPG/PNG/WEBP)       │
                         └──────────┬───────────┘
                                    │
              ┌─────────────────────┼──────────────────────┐
              │                     │                      │
              ▼                     ▼                      ▼
   ┌──────────────────┐  ┌──────────────────────┐  ┌───────────────────────────┐
   │ Color Extractor  │  │ Visual Feature       │  │ Object Detector           │
   │ (LAB KMeans)     │  │ Extractor            │  │ (3-Layer Hybrid)          │
   │                  │  │ (OpenCV + skimage)   │  │                           │
   │ · LAB clustering │  │ · LINE  Canny+Hough  │  │ Layer 1: YOLOv11n         │
   │ · Adaptive K=3–8 │  │ · SHAPE contours     │  │  · COCO 80-class detect   │
   │ · Silhouette sel │  │ · FORM  depth cues   │  │  · Confidence threshold   │
   │ · Salience wts   │  │ · COLOR hue variety  │  │                           │
   │ · Circular mean  │  │ · VALUE LAB contrast │  │ Layer 2: Places365        │
   │   hue            │  │ · SPACE neg-space    │  │  · ResNet-50, 365 classes │
   │ · avg_hue        │  │ · TEXTURE Gabor+LBP  │  │  · Scene suppression      │
   │ · avg_sat        │  └──────────┬───────────┘  │                           │
   │ · avg_bright     │             │              │ Layer 3: CLIP ViT-B/32    │
   │ · palette_hex    │             │              │  · Label disambiguation    │
   └────────┬─────────┘             │              │  · 38 scene probes        │
            │                       │              │  · Zero-shot verification  │
            └───────────────────────┼──────────────┘
                                    │
                                    ▼
                      ┌─────────────────────────┐
                      │      Feature Mapper      │
                      │                         │
                      │  Sigmoid-weighted rules  │
                      │  Warm/cool hue scoring   │
                      │  Produces 14 features:   │
                      │   · 12 rec features      │
                      │   · duration_ms (display)│
                      │   · popularity  (display)│
                      └─────────────┬────────────┘
                                    │
                    ┌───────────────┴──────────────┐
                    │   12-Feature Rec Vector       │
                    │  (duration_ms EXCLUDED)       │
                    │  (popularity  EXCLUDED)       │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                      ┌─────────────────────────┐
                      │   Music Recommender      │
                      │                         │
                      │  MinMax normalisation    │
                      │  Per-feature weights     │
                      │  L2 normalisation        │
                      │  Weighted Euclidean dist │
                      │  Artist diversity rerank │
                      │  + KNN fallback          │
                      └─────────────┬────────────┘
                                    │
                    ┌───────────────┴──────────────┐
                    │   Top 5 Recommendations       │
                    │   score = 1/(1+dist) ∈ [0,1]  │
                    │   + Spotify embed player      │
                    └───────────────────────────────┘
```

---

## 🚀 Quick Start (Windows)

```batch
git clone https://github.com/your-repo/art-to-music.git
cd "art to music"
setup.bat
```

`setup.bat` will automatically:
1. Check Python 3.8+ is installed
2. Create & activate a virtual environment
3. Upgrade pip and install all requirements
4. Download YOLOv11 nano weights
5. Copy/generate the dataset
6. Clean the dataset
7. Train the recommendation model
8. Launch the Streamlit app at `http://localhost:8501`

---

## 🐧 Manual Setup (Mac / Linux)

```bash
# 1. Clone the repository
git clone https://github.com/your-repo/art-to-music.git
cd "art to music"

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install requirements
pip install --upgrade pip
pip install -r requirements.txt

# 4. Create directories
mkdir -p data/raw data/processed models assets

# 5. Copy dataset
cp spotify_dataset.csv data/raw/

# 6. Clean dataset
python scripts/clean_dataset.py

# 7. Train model
python scripts/train_recommender.py

# 8. Launch app
streamlit run app.py --server.port 8501
```

---

## 📖 Usage Guide

1. **Open** `http://localhost:8501` in your browser
2. **Upload** any artwork image (JPG, JPEG, PNG, WEBP)
3. **Wait** ~5–15 seconds for the pipeline to process
4. **Explore** the results:
   - Color strip + swatch grid with HEX codes (click any swatch to copy the HEX)
   - HSB badges (Hue / Saturation / Brightness) with interactive gradient tooltips
   - Scene type card (11 art categories) + warm/neutral/cool temperature label
   - 7 Elements of Art score cards + horizontal bar charts
   - Detected objects as confidence badges + mood tags
   - Audio Feature Radar chart (hover for actual dB/BPM values)
   - 12 metric cards with custom SVG icons for all mapped features
   - Top 5 recommendations with similarity percentage bars
   - Spotify embedded player for each recommendation (where available)
5. **Toggle** Object Detection in the sidebar to skip YOLO/Places365/CLIP for faster processing

---

## 🎨 Pipeline Details

### Layer 1 — Color Extractor (`pipeline/color_extractor.py`)

Extracts the dominant color palette using a three-step process:

| Step | Details |
|------|---------|
| **Stratified sampling** | Up to 12,000 pixels sampled on a grid to represent all image regions |
| **LAB-space clustering** | KMeans runs in CIELAB color space for perceptually uniform grouping |
| **Adaptive K selection** | Tries K=3 to 8; picks K with highest mean silhouette score on a 2,000-pixel subsample |
| **Salience weighting** | Each cluster weight = area × saturation × brightness; vivid accents outweigh large grey backgrounds |
| **Circular mean hue** | Weighted circular mean handles the 0°/360° hue wrap correctly |

**Output fields:**

| Field | Description |
|-------|-------------|
| `dominant_colors` | List of dicts: `rank`, `hex`, `rgb`, `hue`, `saturation`, `brightness`, `weight`, `salience_weight` |
| `avg_hue` | Salience-weighted circular mean hue [0°, 360°] |
| `avg_saturation` | Salience-weighted mean saturation [0, 1] |
| `avg_brightness` | Salience-weighted mean brightness [0, 1] |
| `palette_hex` | HEX strings sorted by salience |
| `best_k` | K chosen by silhouette selection |

---

### Layer 2 — Visual Feature Extractor (`pipeline/visual_feature_extractor.py`)

Extracts all 7 formal elements of art. Images are downscaled to 512px max dimension before processing.

#### Element 1 — LINE
| Signal | Method |
|--------|--------|
| `edge_density` | Canny edge detection with adaptive thresholds (median±33%), sigmoid-normalised |
| `line_straightness` | Probabilistic Hough transform; fraction of lines within 20° of H/V axes |
| `curved_line_ratio` | `1 − line_straightness` |

#### Element 2 — SHAPE
| Signal | Method |
|--------|--------|
| `shape_complexity` | Otsu binarisation + `findContours`; sigmoid of normalised contour count |
| `geometric_ratio` | `approxPolyDP`: contours with 3–8 vertices = geometric; more = organic |

#### Element 3 — FORM
| Signal | Method |
|--------|--------|
| `depth_estimate` | Shading gradient (local variance tiles) × 0.45 + shadow detection (adaptive threshold) × 0.30 + perspective convergence (diagonal Hough lines) × 0.25 |

#### Element 4 — COLOR
| Signal | Method |
|--------|--------|
| `hue_variety` | Angular circular std-dev of dominant hue angles, blended with K-means cluster count on raw image |
| `saturation_range` | Peak-to-peak spread of saturation across dominant colors |
| `palette_breadth` | `hue_variety × 0.6 + saturation_level × 0.4` |
| `saturation_level` / `brightness_level` | Passed directly from ColorExtractor |

#### Element 5 — VALUE
| Signal | Method |
|--------|--------|
| `contrast_ratio` | L* channel std-dev in CIE LAB (sigmoid) × 0.6 + P5–P95 percentile range × 0.4 |
| `value_range` | `(P95 − P5) / 255` of the L* channel |

#### Element 6 — SPACE
| Signal | Method |
|--------|--------|
| `negative_space_ratio` | Local variance map; pixels with variance < 100 counted as negative/background space |
| `composition_balance` | Gaussian-weighted foreground mass centred on image; high score = visually centred composition |

#### Element 7 — TEXTURE
| Signal | Method |
|--------|--------|
| `gabor_energy` | Bank of 4 orientations × 3 wavelength scales; mean squared filter response, sigmoid-normalised |
| `lbp_uniformity` | LBP on 128×128 thumbnail (P=8, R=1, uniform); entropy of 10-bin histogram |
| `texture_energy` | `gabor_energy × 0.6 + lbp_uniformity × 0.4` |
| `texture_homogeneity` | `1 − texture_energy` |

**Element summary scores** (for display cards) are computed by `get_element_summary()`:

| Element | Formula |
|---------|---------|
| LINE | `edge_density × 0.6 + (1 − line_straightness) × 0.4` |
| SHAPE | `shape_complexity × 0.7 + geometric_ratio × 0.3` |
| FORM | `depth_estimate` |
| COLOR | `hue_variety × 0.5 + palette_breadth × 0.3 + saturation_level × 0.2` |
| VALUE | `contrast_ratio` |
| SPACE | `negative_space_ratio × 0.6 + composition_balance × 0.4` |
| TEXTURE | `gabor_energy × 0.5 + lbp_uniformity × 0.3 + texture_energy × 0.2` |

---

### Layer 3 — Object Detector (`pipeline/object_detector.py`)

A three-layer hybrid that corrects YOLOv11's COCO-vocabulary limitations using scene context and semantic verification.

#### Layer 1: YOLOv11n
- Bounding-box detection across 80 COCO classes at configurable confidence threshold (default 0.35)
- Detections sorted by confidence; raw labels tagged as suspicious if in `_SUSPICIOUS_LABELS`

#### Layer 2: Places365 (ResNet-50)
- 365-class global scene classification (top-K=5, score threshold 0.04)
- Builds a **suppression set** per scene: e.g. `underwater` suppresses `{frisbee, kite, bird, balloon, …}` — any YOLO detection physically impossible in that scene is removed
- Derives an 11-category scene type (marine, landscape, cityscape, …) from the top Places365 label

#### Layer 3: CLIP ViT-B/32
- **Label disambiguation**: suspicious or suppressed YOLO labels (e.g. `frisbee`, `sports ball`, `bird`, `clock`) are re-scored against confusion candidates
  - `frisbee` → `{frisbee, moon, plate, clock, ball, sun, bubble, jellyfish, hot air balloon}`
  - `bird` → `{bird, fish, scuba diver, manta ray, flying squirrel, bat}`
- **Scene probe injection**: 38 text prompts covering all 11 scene types are scored against the image; probes passing threshold 0.15 are added to scene detections with mood tags
- **11-category scene scoring**: CLIP probe scores are accumulated per category; the winning category (≥ 0.12 total) overrides Places365 if more specific

**Scene types**: `landscape`, `portraiture`, `still_life`, `genre`, `historical`, `marine`, `cityscape`, `abstract`, `surrealism`, `action`, `religious`

**Mood taxonomy**: 10 categories — Serene/Calm, Melancholy/Sad, Nostalgic/Sentimental, Tense/Unsettling, Joyful/Energetic, Mysterious/Dreamy, Dramatic/Intense, Awe/Sublime, Introspective/Contemplative, Chaotic/Violent

---

### Layer 4 — Feature Mapper (`pipeline/feature_mapper.py`)

Maps visual inputs to 14 Spotify audio features. All features are validated against spec ranges before returning.

#### Key mapping techniques

- **`_sigmoid(x, center, steepness)`** — used throughout to spread distributions away from hard boundaries and produce natural, non-clipped output
- **`_warm_hue_score(hue)`** — continuous warm/cool score in [−1, +1]; warm centres at 0°/330° (red/magenta), cool centre at 225° (blue); used in energy, valence, mode, acousticness
- **Mood tag boosts** — detected mood tags apply ±0.08–0.15 adjustments to multiple features

#### Feature-by-feature mapping

| Feature | Key drivers | Formula sketch |
|---------|------------|----------------|
| **energy** | `edge_density × 0.35 + contrast_ratio × 0.25 + texture_energy × 0.15`, warm hue ±0.08, saturation ±0.05 → sigmoid(center=0.40) |
| **valence** | Warm hue score → `hue_valence`, weighted by saturation; brightness secondary ±0.075; mood tags ±0.08 → sigmoid(center=0.50) |
| **danceability** | `geometric_ratio × 0.30 + line_straightness × 0.25 + shape_complexity × 0.20 + texture_energy × 0.15` + saturation ±0.10 → sigmoid(center=0.45) |
| **acousticness** | `1.0 − energy × 0.40 − texture × 0.25 − edges × 0.15` + warm/soft bonuses; mood ±0.10 → sigmoid(center=0.55) |
| **instrumentalness** | `0.50 + negative_space × 0.40 + (1−palette_breadth) × 0.20` − person penalty (−0.35/−0.45); capped at 0.70 unless `person_confidence < 0.1` |
| **liveness** | `texture × 0.30 + object_count_norm × 0.20 + person_val × 0.25 + warm_contrast × 0.20` + crowd mood +0.15 → sigmoid(center=0.55), clamped [0.10, 0.80] |
| **speechiness** | `person_confidence × 0.50 + composition_balance × contrast × 0.20`; capped at 0.50 unless `person_confidence > 0.80` |
| **tempo** | `60.0 + (energy × 0.55 + edge_density × 0.25 + texture × 0.20) × 125.0`; range [50, 200] BPM |
| **loudness** | `−35.0 + (energy × 0.55 + brightness × 0.25 + saturation × 0.20) × 32.0`; range [−60, −1] dB |
| **key** | `avg_hue` → circle-of-fifths anchor table (linear interpolation); returns −1 for `avg_saturation < 0.08` |
| **mode** | `warm_score × 0.50 + (saturation−0.5) × 0.30 + (brightness−0.5) × 0.20 ≥ 0 → Major (1)` |
| **time_signature** | Curved organic → 3; chaos > 0.78 → 7; > 0.65 → 5; default → 4 |

---

### Layer 5 — Music Recommender (`pipeline/recommender.py`)

Weighted Euclidean similarity on L2-normalised, per-feature-weighted audio feature vectors.

#### Recommendation pipeline

```
Input audio features (14 keys)
    ↓
Extract 12-feature vector (duration_ms, popularity NEVER included)
    ↓
MinMax normalise via saved scaler
    ↓
Apply per-feature importance weights (see table below)
    ↓
L2 normalise (unit sphere)
    ↓
Weighted Euclidean distance against all N dataset tracks
    ↓
Diversity re-rank: artist duplicate penalty ×(1 + 0.20) on 3× candidate pool
    ↓
Flag low-confidence: cosine similarity < 0.50
    ↓
Similarity score = 1 / (1 + distance) ∈ [0, 1]
    ↓
Top 5 results
```

#### Per-feature importance weights

| Feature | Weight | Rationale |
|---------|--------|-----------|
| energy | 2.0 | Primary perceptual match |
| valence | 2.0 | Emotional tone most noticeable to listeners |
| danceability | 1.5 | Rhythmic feel strongly affects perception |
| tempo | 1.5 | BPM directly audible |
| acousticness | 1.2 | Instrumental character |
| liveness | 0.8 | Secondary texture |
| speechiness | 0.8 | Usually very low; moderate weight |
| loudness | 1.0 | Normalised to [0,1] before weighting |
| instrumentalness | 1.0 | Complements acousticness |
| time_signature | 0.6 | Rarely varies widely |
| key | 0.5 | Nice-to-have harmonic match |
| mode | 0.5 | Binary; lower intrinsic weight |

#### KNN fallback

If the primary weighted Euclidean method fails, a `NearestNeighbors(n_neighbors=10, metric='cosine')` model trained on the same normalised+weighted+L2-normalised matrix is used automatically.

---

### Spotify Client (`pipeline/spotify_client.py`)

Authenticates via Client Credentials flow (no user login required). For each top-5 recommendation:
1. Searches Spotify using `track:"…" artist:"…"` strict query
2. Validates up to 5 candidates with case-insensitive partial matching
3. Returns `embed_url` → rendered as a 80px-tall `<iframe>` inline in the app
4. Gracefully falls back to a "Preview unavailable" placeholder if no match is found or the API is unreachable

---

## 🗺️ Visual → Audio Mapping Reference

### acousticness `[0.0, 1.0]`
```
primary = 1.0 − energy×0.40 − texture_energy×0.25 − edge_density×0.15
warm_bonus = max(0, warm_hue_score) × 0.10
soft_bonus = (1−saturation)×0.05 + brightness×0.05
raw = primary + warm_bonus + soft_bonus + mood_mod (±0.10)
acousticness = sigmoid(raw, center=0.55, steepness=4.5)
```

### danceability `[0.0, 1.0]`
```
primary = geometric_ratio×0.30 + shape_complexity×0.20
        + line_straightness×0.25 + texture_energy×0.15
raw = primary + (saturation−0.5)×0.10 + social_mood_boost
danceability = sigmoid(raw, center=0.45, steepness=5.0)
```

### energy `[0.0, 1.0]`
```
primary = edge_density×0.35 + contrast_ratio×0.25 + texture_energy×0.15
hue_mod = warm_hue_score × 0.08          # ±0.08
sat_mod = (saturation − 0.5) × 0.10      # ±0.05
energy = sigmoid(primary + hue_mod + sat_mod, center=0.40, steepness=5.0)
```

### instrumentalness `[0.0, 1.0]`
```
minimalism = negative_space×0.40 + (1−palette_breadth)×0.20
person_penalty = 0.45×person_confidence if person_detected, else 0.0
abstract_bonus = 0.15 if scene in {abstract, nature, landscape}
raw = 0.50 + minimalism − person_penalty + abstract_bonus − mood_penalty
capped at 0.70 if person_confidence ≥ 0.1, else 0.85
```

### key `[-1, 11]`  *(always integer)*
```
Circle-of-fifths anchor table (hue → pitch class with linear interpolation):
  0°   → A(9),  30°  → B(11), 60°  → C(0),  90°  → D(2)
  120° → E(4),  180° → G(7),  210° → F(5),  270° → A♯(10)
  300° → G♯(8), 360° → A(9)
key = -1 if avg_saturation < 0.08 (near-monochrome)
```

### liveness `[0.0, 1.0]`
```
raw = texture_energy×0.30 + obj_norm×0.20 + person_val×0.25
    + warm_hue_score×contrast×0.20 + crowd_mood_boost
liveness = sigmoid(raw, center=0.55, steepness=4.0)
clamped to [0.10, 0.80]
```

### loudness `[-60.0, 0.0 dB]`
```
driver = energy×0.55 + brightness×0.25 + saturation×0.20
loudness = −35.0 + driver×32.0    →   clamped to [−60.0, −1.0]
```

### mode `{0, 1}`  *(always integer)*
```
score = warm_hue_score×0.50 + (saturation−0.5)×0.30 + (brightness−0.5)×0.20
1 (Major) if score ≥ 0.0,  0 (Minor) otherwise
```

### speechiness `[0.0, 1.0]`
```
raw = person_confidence×0.50 + composition_balance×contrast×0.20
hard_cap = 0.50 if person_confidence < 0.80, else 0.65
```

### tempo `[50.0, 200.0 BPM]`
```
intensity = energy×0.55 + edge_density×0.25 + texture_energy×0.20
tempo = 60.0 + intensity×125.0    →   clamped to [50, 200]
```

### time_signature `{3, 4, 5, 7}`  *(always integer)*
```
3 if curved_line_ratio > 0.65 AND geometric_ratio < 0.35   (organic/waltz)
7 if shape_complexity×0.5 + texture_energy×0.5 > 0.78      (very chaotic)
5 if same chaos_score > 0.65
4 otherwise                                                 (common time)
```

### valence `[0.0, 1.0]`
```
warm_score ∈ [−1, +1]; hue_valence = (warm_score + 1.0) / 2.0
hue_contribution = hue_valence×saturation + 0.45×(1−saturation)
brightness_mod = (brightness − 0.5) × 0.15
raw = hue_contribution×0.75 + brightness×0.10 + brightness_mod + mood_mod
valence = sigmoid(raw, center=0.50, steepness=5.0)
```

---

## 📋 Audio Feature Reference

| Feature | Min | Max | Type | Description |
|---------|-----|-----|------|-------------|
| acousticness | 0.0 | 1.0 | float | Confidence measure — 1.0 = fully acoustic |
| danceability | 0.0 | 1.0 | float | Suitability for dancing |
| duration_ms | 0 | — | int | Track length in ms — **display only** |
| energy | 0.0 | 1.0 | float | Perceptual intensity and activity |
| instrumentalness | 0.0 | 1.0 | float | >0.5 = likely instrumental |
| key | -1 | 11 | int | Pitch class (0=C … 11=B; -1=not detected) |
| liveness | 0.0 | 1.0 | float | >0.8 = likely live performance |
| loudness | -60.0 | 0.0 | float | Overall loudness in dB |
| mode | 0 | 1 | int | 0 = Minor, 1 = Major |
| speechiness | 0.0 | 1.0 | float | >0.66 = likely all speech |
| tempo | 50.0 | 200.0 | float | Estimated BPM (realistic output range) |
| time_signature | 3 | 7 | int | Estimated beats per bar |
| valence | 0.0 | 1.0 | float | 1.0 = happy/positive, 0.0 = sad/angry |
| popularity | 0 | 100 | int | Play-based score — **display only** |

> **Why `duration_ms` and `popularity` are excluded from recommendation:**
> `duration_ms` has no relationship to visual characteristics — a painting is not longer or shorter.
> `popularity` reflects recency-weighted play counts, not musical feel.
> Including either would corrupt similarity scores. Both are enforced by runtime assertions.

---

## 🗃️ Dataset

### Structure
```
artist_name, track_id, track_name, acousticness, danceability, duration_ms,
energy, instrumentalness, key, liveness, loudness, mode, speechiness,
tempo, time_signature, valence, popularity
```

### Cleaning Pipeline (`scripts/clean_dataset.py`)

`RECOMMENDATION_FEATURES` is imported from `pipeline.recommender` — single source of truth.

| Step | Action |
|------|--------|
| 1 | Drop duplicates by `track_id`, then by `(track_name, artist_name)` |
| 2 | Drop rows with any null in audio feature columns |
| 3 | Clip/drop out-of-range values per feature spec |
| 4 | Remove tempo outliers using IQR ±2.5×IQR method |
| 5 | Fit MinMaxScaler on 12 rec features; save to `models/feature_scaler.pkl` |
| 6 | Save cleaned CSV to `data/processed/cleaned_dataset.csv` |

### Synthetic Generation
If no dataset is found, 500 realistic synthetic tracks are auto-generated:

| Feature | Distribution |
|---------|-------------|
| acousticness | Beta(0.5, 2.0) — skewed toward 0 |
| danceability | Normal(0.6, 0.18) clipped to [0,1] |
| duration_ms | Normal(210000, 60000) ms |
| energy | Normal(0.65, 0.2) clipped to [0,1] |
| instrumentalness | Bimodal: 60% near 0, 40% near 0.9 |
| key | Uniform integers [−1, 11] |
| liveness | Beta(0.4, 4.0) — skewed toward 0 |
| loudness | Normal(−8.0, 6.0) clipped to [−60, 0] |
| mode | Bernoulli(0.6) — 60% Major |
| speechiness | Beta(0.3, 8.0) — heavily skewed to 0 |
| tempo | Normal(120, 30) clipped to [40, 250] |
| time_signature | Mostly 4, some 3/5/6 |
| valence | Uniform [0, 1] |
| popularity | Normal(50, 20) clipped to [0, 100] |

---

## 🤖 Model Details

### Training (`scripts/train_recommender.py`)

`RECOMMENDATION_FEATURES` is imported from `pipeline.recommender` — no local copy.

The saved bundle (`models/recommender_model.pkl`) contains:

```python
{
    "model":             NearestNeighbors(n_neighbors=10, metric='cosine'),
    "scaler":            MinMaxScaler,
    "dataset":           cleaned_df,
    "feature_columns":   RECOMMENDATION_FEATURES,   # 12 features — duration_ms NEVER here
    "similarity_matrix": ndarray (N×N) or None,     # None for datasets > 5,000 tracks
}
```

Full similarity matrix is pre-computed only for datasets ≤ 5,000 tracks to avoid OOM issues; larger datasets rely on KNN inference.

### 12-Feature Vector (single source of truth in `pipeline/recommender.py`)
```python
RECOMMENDATION_FEATURES = [
    "acousticness", "danceability", "energy", "instrumentalness",
    "key", "liveness", "loudness", "mode", "speechiness",
    "tempo", "time_signature", "valence",
]
```

---

## 📁 Project Structure

```
art to music/
├── app.py                          # Streamlit UI, scene classification, all rendering helpers
├── requirements.txt
├── setup.bat                       # Windows one-click setup
├── spotify_dataset.csv             # Original raw dataset
├── pipeline/
│   ├── __init__.py
│   ├── color_extractor.py          # LAB KMeans + adaptive K + salience weights
│   ├── visual_feature_extractor.py # 7 art elements: Canny/Hough/Gabor/LBP/LAB
│   ├── object_detector.py          # YOLOv11n + Places365 + CLIP hybrid
│   ├── feature_mapper.py           # Sigmoid visual→audio mapping + validation
│   ├── recommender.py              # Weighted Euclidean + L2-norm + KNN fallback
│   └── spotify_client.py           # Spotify Client Credentials + track search
├── scripts/
│   ├── clean_dataset.py            # Dataset cleaning + synthetic generation
│   └── train_recommender.py        # Model training + integrity checks
├── data/
│   ├── raw/spotify_dataset.csv
│   └── processed/cleaned_dataset.csv
├── models/
│   ├── recommender_model.pkl       # Trained model bundle
│   ├── feature_scaler.pkl          # Fitted MinMaxScaler
│   ├── resnet50_places365.pth.tar  # Places365 weights (auto-downloaded)
│   ├── categories_places365.txt    # Places365 label file (auto-downloaded)
│   └── yolo11n.pt                  # YOLOv11 nano weights
├── assets/
│   └── styles.css                  # Dark theme CSS
└── README.md
```

---

## 🔧 Troubleshooting

### YOLOv11 Issues
- **CUDA not available**: automatically falls back to CPU — no action needed
- **Model download fails**: manually download `yolo11n.pt` from [ultralytics/assets](https://github.com/ultralytics/assets/releases) and place in `models/`
- **Slow detection**: toggle off "Enable Object Detection" in the sidebar

### Places365 / CLIP Issues
- **Places365 not available**: pipeline continues with YOLO + CLIP only; scene classification degrades gracefully
- **CLIP not available**: label disambiguation is skipped; suppressed labels are simply dropped without CLIP verification
- **Download fails**: manually place `resnet50_places365.pth.tar` and `categories_places365.txt` in `models/`

### Recommendation Model Not Found
```
⚠️ Recommendation model not found
```
Run: `python scripts/train_recommender.py`

### Dataset Missing
```
✗ No raw dataset found
```
`clean_dataset.py` auto-generates 500 synthetic tracks. Or place your own CSV at `data/raw/spotify_dataset.csv`.

### Spotify Embed Not Showing
- Spotify embed requires valid `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET` in `pipeline/spotify_client.py`
- The app shows a "Preview unavailable" placeholder if the search returns no match or the API is unreachable — this is expected behaviour

### Import Errors
```batch
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac
```

### scikit-image / GLCM errors
```
pip install scikit-image>=0.22.0
```

### Memory issues
The pipeline automatically resizes images to 512px max dimension for all processing; the original is displayed at full resolution.

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with ❤️ using Streamlit · OpenCV · YOLOv11 · Places365 · CLIP · scikit-learn · Plotly · Spotify Web API*
