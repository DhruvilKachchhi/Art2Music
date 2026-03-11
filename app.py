"""
Art to Music — Streamlit Application (Redesigned UI v2)
========================================================
v2: Larger text, improved contrast, bigger emojis,
    rich HSV hover/click tooltip system
"""

import os
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.color_extractor import ColorExtractor
from pipeline.visual_feature_extractor import VisualFeatureExtractor
from pipeline.object_detector import ObjectDetector
from pipeline.feature_mapper import FeatureMapper
from pipeline.recommender import MusicRecommender, KEY_NAMES
from pipeline.spotify_client import SpotifyClient

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Art to Music",
    page_icon="🎨",
    layout="wide",
    
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Load CSS
# ---------------------------------------------------------------------------
def load_css() -> None:
    css_path = os.path.join("assets", "styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ---------------------------------------------------------------------------
# JS: swatch copy + rich HSV tooltips
# ---------------------------------------------------------------------------
_INTERACTIVE_JS = """
<script>
(function() {
  // ── Swatch copy-to-clipboard ──────────────────────────────────────────
  function attachSwatch(el) {
    el.style.cursor = 'pointer';
    el.addEventListener('click', function() {
      var hex = el.getAttribute('data-hex');
      if (!hex) return;
      navigator.clipboard && navigator.clipboard.writeText(hex).catch(function(){});
      var label = el.querySelector('.atm-swatch-copied');
      if (label) { label.style.opacity = '1'; setTimeout(function(){ label.style.opacity = '0'; }, 1400); }
      el.style.transform = 'scale(0.96)';
      setTimeout(function(){ el.style.transform = 'scale(1.05)'; }, 120);
      setTimeout(function(){ el.style.transform = 'scale(1)'; }, 270);
    });
  }
  document.querySelectorAll('[data-hex]').forEach(attachSwatch);

  // ── HSV Tooltip system ────────────────────────────────────────────────
  var tooltip = null;
  var activeType = null;

  var TOOLTIP_DATA = {
    hue: {
      color: '#a78bfa',
      title: 'Hue',
      short: 'The identity of the color itself — which color it is.',
      body: 'Hue is the pure color itself, determined by the dominant wavelength of light. It is what we mean when we use basic color names like <b>"red,"</b> <b>"green,"</b> or <b>"blue."</b>',
      measure: 'Represented as a degree (0° – 360°) on a color wheel.',
      gradType: 'wheel',
      wheelColors: [
        { deg: '0°',   color: '#FF0000', name: 'Red'     },
        { deg: '60°',  color: '#FFFF00', name: 'Yellow'  },
        { deg: '120°', color: '#00FF00', name: 'Green'   },
        { deg: '180°', color: '#00FFFF', name: 'Cyan'    },
        { deg: '240°', color: '#0000FF', name: 'Blue'    },
        { deg: '300°', color: '#FF00FF', name: 'Magenta' },
      ],
    },
    saturation: {
      color: '#34d399',
      title: 'Saturation',
      short: 'How vivid vs. grey the color appears.',
      body: 'Saturation describes the intensity or vividness of a color — how much a hue is <b>"contaminated"</b> by grey. <b>High saturation</b> results in bold, striking colors (e.g., a neon sign). <b>Low saturation</b> makes colors look muted, pastel, or washed out.',
      measure: 'Measured from 0% (pure grey) to 100% (pure vivid color).',
      gradType: 'bar',
      examples: [
        { label: '0% — Pure grey',      pct: 0,    hue: 200 },
        { label: '50% — Muted / pastel', pct: 0.5,  hue: 200 },
        { label: '100% — Vivid / neon',  pct: 1,    hue: 200 },
      ],
    },
    brightness: {
      color: '#fbbf24',
      title: 'Brightness',
      short: 'How light or dark the color appears.',
      body: 'Brightness (Value) refers to the perceived amount of light in a color. <b>High brightness</b> approaches pure white (especially if saturation is low). <b>Low brightness</b> approaches pure black — at 0% any hue becomes black.',
      measure: 'Measured from 0% (black) to 100% (full light).',
      gradType: 'bar',
      examples: [
        { label: '0% — Black',          pct: 0,    hue: 200 },
        { label: '50% — Mid-tone',       pct: 0.5,  hue: 200 },
        { label: '100% — Full light',    pct: 1.0,  hue: 200 },
      ],
    },
  };

  function hsvToHsl(h, s, v) {
    var l = v * (1 - s / 2);
    var sl = (l === 0 || l === 1) ? 0 : (v - l) / Math.min(l, 1 - l);
    return 'hsl(' + h + ',' + (sl * 100).toFixed(1) + '%,' + (l * 100).toFixed(1) + '%)';
  }

  function buildTooltip(type, value, avgHue) {
    var d = TOOLTIP_DATA[type];
    if (!d) return null;

    var tip = document.createElement('div');
    tip.className = 'atm-hsv-tooltip';
    tip.id = 'atm-hsv-tooltip-' + type;

    var displayVal = type === 'hue'
      ? parseFloat(value).toFixed(1) + '°'
      : (parseFloat(value) * 100).toFixed(1) + '%';

    var pct = type === 'hue'
      ? (parseFloat(value) / 360) * 100
      : parseFloat(value) * 100;

    // Gradient background for bar
    var gradBg;
    if (type === 'hue') {
      gradBg = 'linear-gradient(to right,hsl(0,100%,55%),hsl(60,100%,55%),hsl(120,100%,55%),hsl(180,100%,55%),hsl(240,100%,55%),hsl(300,100%,55%),hsl(360,100%,55%))';
    } else if (type === 'saturation') {
      gradBg = 'linear-gradient(to right,hsl(' + avgHue + ',0%,55%),hsl(' + avgHue + ',100%,55%))';
    } else {
      gradBg = 'linear-gradient(to right,#000,hsl(' + avgHue + ',80%,55%))';
    }

    // Build color wheel HTML for hue
    var extraHTML = '';
    if (d.gradType === 'wheel') {
      var swatches = d.wheelColors.map(function(c) {
        return '<div><div class="atm-tooltip-wheel-swatch" style="background:' + c.color + ';"></div>'
          + '<div class="atm-tooltip-wheel-label">' + c.deg + '<br>' + c.name + '</div></div>';
      }).join('');
      extraHTML = '<div class="atm-tooltip-sub-label">Common Points</div>'
        + '<div class="atm-tooltip-wheel-row">' + swatches + '</div>';
    } else if (d.examples) {
      var exItems = d.examples.map(function(ex) {
        var swatchColor = hsvToHsl(ex.hue, ex.pct, type === 'brightness' ? ex.pct : 0.75);
        if (type === 'brightness') swatchColor = hsvToHsl(ex.hue, 0.7, ex.pct);
        if (type === 'saturation') swatchColor = hsvToHsl(ex.hue, ex.pct, 0.7);
        return '<div class="atm-tooltip-example">'
          + '<div class="atm-tooltip-example-swatch" style="background:' + swatchColor + ';"></div>'
          + '<span class="atm-tooltip-example-label">' + ex.label + '</span>'
          + '</div>';
      }).join('');
      extraHTML = '<div class="atm-tooltip-sub-label">Examples</div>'
        + '<div class="atm-tooltip-examples">' + exItems + '</div>';
    }

    tip.innerHTML =
      '<div class="atm-tooltip-header">'
      + '<div class="atm-tooltip-dot" style="background:' + d.color + ';box-shadow:0 0 8px ' + d.color + ';"></div>'
      + '<span class="atm-tooltip-title" style="color:' + d.color + ';">' + d.title + '</span>'
      + '<span class="atm-tooltip-val" style="background:' + d.color + '22;color:' + d.color + ';">' + displayVal + '</span>'
      + '</div>'
      + '<div class="atm-tooltip-short">' + d.short + '</div>'
      + '<div class="atm-tooltip-body">' + d.body + '</div>'
      + '<div class="atm-tooltip-divider"></div>'
      + '<div class="atm-tooltip-sub-label">Measurement</div>'
      + '<div class="atm-tooltip-grad-wrap">'
      + '<div class="atm-tooltip-grad-bar" style="background:' + gradBg + ';">'
      + '<div class="atm-tooltip-grad-marker" style="left:' + pct.toFixed(1) + '%;"></div>'
      + '</div>'
      + '<div class="atm-tooltip-grad-labels"><span>' + (type === 'hue' ? '0°' : '0%') + '</span><span>' + (type === 'hue' ? '360°' : '100%') + '</span></div>'
      + '</div>'
      + '<div class="atm-tooltip-body" style="margin-bottom:10px;">' + d.measure + '</div>'
      + (extraHTML ? '<div class="atm-tooltip-divider"></div>' + extraHTML : '');

    return tip;
  }

  function positionTooltip(tip, anchor) {
    var rect = anchor.getBoundingClientRect();
    var tipW = 310;
    var margin = 10;
    var top = rect.bottom + margin;
    var left = rect.left + rect.width / 2 - tipW / 2;

    // Clamp horizontally
    if (left < margin) left = margin;
    if (left + tipW > window.innerWidth - margin) left = window.innerWidth - tipW - margin;

    // Flip above if too close to bottom
    var tipH = 400; // approximate
    if (top + tipH > window.innerHeight - margin) {
      top = rect.top - tipH - margin;
      if (top < margin) top = margin;
    }

    tip.style.top = top + 'px';
    tip.style.left = left + 'px';
  }

  function removeTooltip() {
    if (tooltip) {
      tooltip.classList.remove('visible');
      var t = tooltip;
      setTimeout(function() { if (t.parentNode) t.parentNode.removeChild(t); }, 220);
      tooltip = null;
      activeType = null;
    }
  }

  function attachHsvBadge(badge) {
    var type = badge.getAttribute('data-hsv-type');
    var value = badge.getAttribute('data-hsv-value');
    var avgHue = badge.getAttribute('data-avg-hue');
    if (!type) return;

    function showTip() {
      if (activeType === type) return;
      removeTooltip();
      var tip = buildTooltip(type, value, avgHue);
      if (!tip) return;
      document.body.appendChild(tip);
      tooltip = tip;
      activeType = type;
      positionTooltip(tip, badge);
      requestAnimationFrame(function() { tip.classList.add('visible'); });
    }

    badge.addEventListener('mouseenter', showTip);
    badge.addEventListener('mouseleave', function() {
      setTimeout(function() {
        // Only remove if not hovering over the tooltip itself
        removeTooltip();
      }, 80);
    });
    badge.addEventListener('click', function(e) {
      e.stopPropagation();
      if (activeType === type) { removeTooltip(); } else { showTip(); }
    });
  }

  // Click outside to close
  document.addEventListener('click', function() { removeTooltip(); });

  function init() {
    document.querySelectorAll('[data-hsv-type]').forEach(attachHsvBadge);
    document.querySelectorAll('[data-hex]').forEach(attachSwatch);
  }
  init();

  // Re-attach on DOM changes (Streamlit re-renders)
  var obs = new MutationObserver(function(muts) {
    var changed = false;
    muts.forEach(function(m) { if (m.addedNodes.length) changed = true; });
    if (changed) init();
  });
  obs.observe(document.body, { childList: true, subtree: true });
})();
</script>
"""

# ---------------------------------------------------------------------------
# Initialize clients + pipeline (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_spotify_client() -> SpotifyClient:
    return SpotifyClient()

@st.cache_resource(show_spinner=False)
def load_pipeline():
    return (
        ColorExtractor(n_colors=6),
        VisualFeatureExtractor(),
        ObjectDetector(),
        FeatureMapper(),
        MusicRecommender(),
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def key_to_note(key_int: int) -> str:
    return KEY_NAMES.get(int(key_int), str(key_int))

def get_contrast_color(hex_color: str) -> str:
    h = hex_color.lstrip('#')
    if len(h) != 6:
        return "#ffffff"
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return "#1a1a1a" if (0.299*r + 0.587*g + 0.114*b) / 255 > 0.55 else "#ffffff"

def temp_label(avg_hue: float) -> tuple:
    if (0 <= avg_hue <= 60) or (300 <= avg_hue <= 360):
        return "Warm", "#f97316"
    elif 60 < avg_hue < 180:
        return "Neutral", "#a3e635"
    return "Cool", "#38bdf8"

# ---------------------------------------------------------------------------
# Radar chart
# ---------------------------------------------------------------------------
def build_radar_chart(audio_features: Dict[str, Any]) -> go.Figure:
    display_features = {
        "Acousticness":     float(audio_features.get("acousticness", 0)),
        "Danceability":     float(audio_features.get("danceability", 0)),
        "Energy":           float(audio_features.get("energy", 0)),
        "Instrumentalness": float(audio_features.get("instrumentalness", 0)),
        "Liveness":         float(audio_features.get("liveness", 0)),
        "Speechiness":      float(audio_features.get("speechiness", 0)),
        "Valence":          float(audio_features.get("valence", 0)),
    }
    raw_loudness = float(audio_features.get("loudness", -30.0))
    display_features["Loudness"] = float(np.clip((raw_loudness + 60.0) / 60.0, 0.0, 1.0))
    raw_tempo = float(audio_features.get("tempo", 120.0))
    display_features["Tempo"] = float(np.clip((raw_tempo - 40.0) / 210.0, 0.0, 1.0))

    categories = list(display_features.keys())
    values = list(display_features.values())
    values_closed = values + [values[0]]
    categories_closed = categories + [categories[0]]

    actual_vals = {
        "Acousticness":     f"{audio_features.get('acousticness', 0):.3f}",
        "Danceability":     f"{audio_features.get('danceability', 0):.3f}",
        "Energy":           f"{audio_features.get('energy', 0):.3f}",
        "Instrumentalness": f"{audio_features.get('instrumentalness', 0):.3f}",
        "Liveness":         f"{audio_features.get('liveness', 0):.3f}",
        "Speechiness":      f"{audio_features.get('speechiness', 0):.3f}",
        "Valence":          f"{audio_features.get('valence', 0):.3f}",
        "Loudness":         f"{audio_features.get('loudness', -30.0):.1f} dB",
        "Tempo":            f"{audio_features.get('tempo', 120.0):.1f} BPM",
    }
    hover_texts = [actual_vals.get(c, "") for c in categories_closed]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(124, 58, 237, 0.2)",
        line=dict(color="#a78bfa", width=2.2),
        hovertext=hover_texts,
        hovertemplate="<b>%{theta}</b><br>Actual: %{hovertext}<br>Norm: %{r:.3f}<extra></extra>",
        name="Audio Profile",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 1],
                tickfont=dict(color="#505080", size=10),
                gridcolor="rgba(255,255,255,0.06)",
                linecolor="rgba(255,255,255,0.06)",
            ),
            angularaxis=dict(
                tickfont=dict(color="#9090c0", size=12),
                gridcolor="rgba(255,255,255,0.06)",
                linecolor="rgba(255,255,255,0.06)",
            ),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(l=55, r=55, t=35, b=35),
        height=370,
        font=dict(family="DM Sans, sans-serif"),
    )
    return fig

# ---------------------------------------------------------------------------
# HTML component builders
# ---------------------------------------------------------------------------

def render_color_strip_and_swatches(dominant_colors: List[Dict]) -> str:
    strip_segments = "".join(
        f'<div style="flex:{c.get("weight",0)};background:{c.get("hex","#000")};" '
        f'title="{c.get("hex","")}: {c.get("weight",0)*100:.0f}%"></div>'
        for c in dominant_colors
    )
    swatch_items = []
    for c in dominant_colors:
        hex_code = c.get("hex", "#000000")
        weight   = c.get("weight", 0)
        rank     = c.get("rank", "?")
        contrast = get_contrast_color(hex_code)
        swatch_items.append(
            f'<div class="atm-swatch" data-hex="{hex_code}" '
            f'style="background:{hex_code};">'
            f'<div class="atm-swatch-hex" style="color:{contrast};">{hex_code}</div>'
            f'<div class="atm-swatch-pct" style="color:{contrast};">{weight*100:.0f}% · #{rank}</div>'
            f'<div class="atm-swatch-copied" style="font-size:9px;font-weight:700;color:{contrast};'
            f'opacity:0;transition:opacity 0.3s;margin-top:2px;">✓ copied</div>'
            f'</div>'
        )
    return (
        f'<div class="atm-color-strip">{strip_segments}</div>'
        f'<div class="atm-swatch-grid">{"".join(swatch_items)}</div>'
    )


def render_hsv_badges(color_data: Dict) -> str:
    avg_hue = float(color_data.get("avg_hue", 0))
    avg_sat = float(color_data.get("avg_saturation", 0))
    avg_bri = float(color_data.get("avg_brightness", 0))

    hue_pct = (avg_hue / 360) * 100
    sat_pct = avg_sat * 100
    bri_pct = avg_bri * 100

    hue_grad = "linear-gradient(to right,hsl(0,100%,55%),hsl(60,100%,55%),hsl(120,100%,55%),hsl(180,100%,55%),hsl(240,100%,55%),hsl(300,100%,55%),hsl(360,100%,55%))"
    sat_grad = f"linear-gradient(to right,hsl({avg_hue:.0f},0%,55%),hsl({avg_hue:.0f},100%,55%))"
    bri_grad = f"linear-gradient(to right,#000,hsl({avg_hue:.0f},80%,55%))"

    COLORS = {"hue": "#a78bfa", "saturation": "#34d399", "brightness": "#fbbf24"}

    HOVER_MESSAGES = {
        "hue":        "The color type (e.g., red, blue) represented as degrees on a 0–360° wheel.",
        "saturation": "The intensity or purity of the color, ranging from 0 (gray) to 100% (full color).",
        "brightness": "The amount of light, ranging from 0 (black) to 100% (white/full color).",
    }

    def badge(btype, label, value_str, bar_pct, bar_grad, range_str, raw_value):
        c = COLORS[btype]
        hover_title = HOVER_MESSAGES.get(btype, "Click or hover for more details")
        return (
            f'<div class="atm-hsv-badge" '
            f'data-hsv-type="{btype}" data-hsv-value="{raw_value}" data-avg-hue="{avg_hue:.1f}" '
            f'style="border-color:rgba(255,255,255,0.08);" '
            f'title="{hover_title}">'
            f'<div class="atm-hsv-label" style="color:{c}88;">'
            f'{label}'
            f'<span class="info-icon">?</span>'
            f'</div>'
            f'<div class="atm-hsv-value">{value_str}</div>'
            f'<div class="atm-hsv-bar-track">'
            f'<div class="atm-hsv-bar-fill" style="width:{bar_pct:.1f}%;background:{bar_grad};"></div>'
            f'</div>'
            f'<div class="atm-hsv-range">{range_str}</div>'
            f'</div>'
        )

    return (
        f'<div class="atm-hsv-row">'
        + badge("hue",        "Hue",        f"{avg_hue:.1f}°", 
                hue_pct, hue_grad, "0° – 360°", avg_hue)
        + badge("saturation", "Saturation", f"{avg_sat*100:.1f}%", sat_pct, sat_grad, "0% – 100%", avg_sat)
        + badge("brightness", "Brightness", f"{avg_bri*100:.1f}%", bri_pct, bri_grad, "0% – 100%", avg_bri)
        + '</div>'
    )


def render_scene_card(scene_type: str, scene_info: Dict) -> str:
    return (
        f'<div class="atm-scene-card">'
        f'<div class="atm-scene-icon">{scene_info["icon"]}</div>'
        f'<div>'
        f'<div class="atm-scene-name">{scene_info["label"]}</div>'
        f'<div class="atm-scene-desc">{scene_info["description"]}</div>'
        f'</div>'
        f'</div>'
    )


def render_element_cards(element_summary: Dict, element_icons: Dict, element_tooltips: Dict) -> str:
    items = []
    for elem, score in element_summary.items():
        icon = element_icons.get(elem, "•")
        tip  = element_tooltips.get(elem, "")
        items.append(
            f'<div class="atm-elem-card" title="{tip}">'
            f'<div class="atm-elem-icon">{icon}</div>'
            f'<div class="atm-elem-name">{elem}</div>'
            f'<div class="atm-elem-score">{score:.2f}</div>'
            f'</div>'
        )
    return f'<div class="atm-elements-grid">{"".join(items)}</div>'


def render_element_bars(element_summary: Dict, element_icons: Dict, element_tooltips: Dict) -> str:
    rows = []
    for elem, score in element_summary.items():
        icon = element_icons.get(elem, "•")
        tip  = element_tooltips.get(elem, "")
        pct  = float(score) * 100
        rows.append(
            f'<div class="atm-bar-row" title="{tip}">'
            f'<div class="atm-bar-header">'
            f'<span class="atm-bar-label">{icon} {elem}</span>'
            f'<span class="atm-bar-val">{score:.3f}</span>'
            f'</div>'
            f'<div class="atm-bar-track"><div class="atm-bar-fill" style="width:{pct:.2f}%;"></div></div>'
            f'</div>'
        )
    return "".join(rows)


def render_object_badges(detected_objects: List[Dict]) -> str:
    if not detected_objects:
        return '<span style="font-size:13px;color:#6060a0;font-style:italic;">No objects detected above threshold</span>'
    items = []
    for obj in detected_objects[:14]:
        label = obj.get("label", "unknown")
        conf  = obj.get("confidence", 0.0)
        items.append(
            f'<span class="atm-obj-badge">{label}'
            f'<span class="atm-obj-conf">{conf*100:.0f}%</span>'
            f'</span>'
        )
    return f'<div class="atm-obj-badges">{"".join(items)}</div>'


def render_mood_tags(mood_tags: List[str]) -> str:
    items = "".join(f'<span class="atm-mood-tag">{t}</span>' for t in mood_tags[:10])
    return f'<div class="atm-mood-tags">{items}</div>'


def render_audio_metrics(audio_features: Dict) -> str:
    key_name  = key_to_note(audio_features.get("key", 0))
    mode_name = "Major" if audio_features.get("mode", 1) == 1 else "Minor"

    # ── SVG icons — 32×32 viewBox, palette: #a78bfa / #34d399 / #fbbf24 ──
    _P = "#a78bfa"   # violet  (primary strokes)
    _G = "#34d399"   # green   (secondary strokes)
    _Y = "#fbbf24"   # amber   (accent strokes)
    _SW = 'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"'
    def _svg(inner: str) -> str:
        return (
            f'<svg viewBox="0 0 32 32" width="26" height="26" fill="none" '
            f'xmlns="http://www.w3.org/2000/svg">{inner}</svg>'
        )

    audio_icons = {
        "Energy": _svg(
            f'<path d="M17 3 L4 18 h11 l-1 11 L27 14 h-11 l1-11z" '
            f'stroke="{_Y}" {_SW} fill="rgba(251,191,36,0.12)"/>'
        ),
        "Valence": _svg(
            f'<circle cx="16" cy="16" r="12" stroke="{_P}" {_SW} fill="rgba(167,139,250,0.08)"/>'
            f'<path d="M10 18 s2 4 6 4 6-4 6-4" stroke="{_G}" {_SW}/>'
            f'<circle cx="11" cy="12" r="1.5" fill="{_P}"/>'
            f'<circle cx="21" cy="12" r="1.5" fill="{_P}"/>'
        ),
        "Danceability": _svg(
            f'<circle cx="16" cy="7" r="2.5" stroke="{_P}" {_SW}/>'
            f'<line x1="16" y1="9.5" x2="16" y2="18" stroke="{_P}" {_SW}/>'
            f'<path d="M16 13 l-5 4" stroke="{_G}" {_SW}/>'
            f'<path d="M16 13 l5 4" stroke="{_G}" {_SW}/>'
            f'<path d="M11 11 h10" stroke="{_Y}" {_SW}/>'
            f'<line x1="16" y1="18" x2="11" y2="26" stroke="{_P}" {_SW}/>'
            f'<line x1="16" y1="18" x2="21" y2="26" stroke="{_P}" {_SW}/>'
        ),
        "Acousticness": _svg(
            f'<path d="M11 22 V7 l14-2.5 v15" stroke="{_P}" {_SW}/>'
            f'<circle cx="8" cy="22" r="3.5" stroke="{_G}" {_SW} fill="rgba(52,211,153,0.1)"/>'
            f'<circle cx="22" cy="19.5" r="3.5" stroke="{_Y}" {_SW} fill="rgba(251,191,36,0.1)"/>'
        ),
        "Tempo": _svg(
            f'<circle cx="16" cy="16" r="12" stroke="{_P}" {_SW} fill="rgba(167,139,250,0.08)"/>'
            f'<line x1="16" y1="8" x2="16" y2="16" stroke="{_G}" {_SW}/>'
            f'<line x1="16" y1="16" x2="22" y2="19" stroke="{_Y}" stroke-width="2.5" stroke-linecap="round"/>'
            f'<circle cx="16" cy="16" r="2" fill="{_P}"/>'
        ),
        "Loudness": _svg(
            f'<path d="M4 12 h5 l5-7 v18 l-5-7 h-5 z" stroke="{_P}" {_SW} fill="rgba(167,139,250,0.12)"/>'
            f'<path d="M19 10 a6 6 0 0 1 0 12" stroke="{_G}" {_SW}/>'
            f'<path d="M22 6 a11 11 0 0 1 0 20" stroke="{_Y}" {_SW}/>'
        ),
        "Speechiness": _svg(
            f'<rect x="10" y="2" width="12" height="16" rx="6" stroke="{_P}" {_SW} fill="rgba(167,139,250,0.1)"/>'
            f'<path d="M6 14 v2 a10 10 0 0 0 20 0 v-2" stroke="{_G}" {_SW}/>'
            f'<line x1="16" y1="26" x2="16" y2="30" stroke="{_Y}" {_SW}/>'
            f'<line x1="11" y1="30" x2="21" y2="30" stroke="{_Y}" {_SW}/>'
        ),
        "Instrumentalness": _svg(
            f'<rect x="6" y="3" width="20" height="26" rx="2" stroke="{_P}" {_SW} fill="rgba(167,139,250,0.08)"/>'
            f'<line x1="6" y1="10" x2="26" y2="10" stroke="{_G}" stroke-width="1.8" stroke-linecap="round"/>'
            f'<line x1="6" y1="17" x2="26" y2="17" stroke="{_G}" stroke-width="1.8" stroke-linecap="round"/>'
            f'<line x1="6" y1="24" x2="26" y2="24" stroke="{_Y}" stroke-width="1.4" stroke-linecap="round"/>'
        ),
        "Liveness": _svg(
            f'<circle cx="11" cy="9" r="4" stroke="{_P}" {_SW} fill="rgba(167,139,250,0.1)"/>'
            f'<path d="M2 28 v-2 a8 8 0 0 1 16 0 v2" stroke="{_P}" {_SW}/>'
            f'<path d="M21.5 7 a5.5 5.5 0 0 1 0 9.5" stroke="{_G}" {_SW}/>'
            f'<path d="M24 3.5 a10 10 0 0 1 0 16" stroke="{_Y}" {_SW}/>'
        ),
        "Key": _svg(
            f'<rect x="3" y="4" width="26" height="24" rx="2.5" stroke="{_P}" {_SW} fill="rgba(167,139,250,0.08)"/>'
            f'<line x1="10" y1="4" x2="10" y2="28" stroke="{_G}" stroke-width="1.6" stroke-linecap="round"/>'
            f'<line x1="17" y1="4" x2="17" y2="28" stroke="{_G}" stroke-width="1.6" stroke-linecap="round"/>'
            f'<line x1="3" y1="16" x2="29" y2="16" stroke="{_Y}" stroke-width="1.8" stroke-linecap="round"/>'
            f'<rect x="10" y="4" width="4" height="13" rx="1" fill="rgba(52,211,153,0.25)"/>'
            f'<rect x="17" y="4" width="4" height="13" rx="1" fill="rgba(52,211,153,0.25)"/>'
        ),
        "Mode": _svg(
            f'<path d="M16 3 L3 9 l13 6 l13-6 z" stroke="{_P}" {_SW} fill="rgba(167,139,250,0.15)"/>'
            f'<path d="M3 21 l13 6 l13-6" stroke="{_Y}" {_SW}/>'
            f'<path d="M3 15 l13 6 l13-6" stroke="{_G}" {_SW}/>'
        ),
        "Time Signature": _svg(
            f'<circle cx="16" cy="16" r="12" stroke="{_P}" {_SW} fill="rgba(167,139,250,0.08)"/>'
            f'<line x1="16" y1="8" x2="16" y2="16" stroke="{_G}" {_SW}/>'
            f'<line x1="16" y1="16" x2="20" y2="20" stroke="{_Y}" stroke-width="2.5" stroke-linecap="round"/>'
            f'<line x1="9" y1="16" x2="23" y2="16" stroke="{_P}" stroke-width="1.4" stroke-linecap="round" stroke-dasharray="2 2"/>'
            f'<circle cx="16" cy="16" r="2" fill="{_P}"/>'
        ),
    }

    metrics = [
        ("Energy",            f"{audio_features.get('energy', 0):.3f}",
         "A measure from 0.0 to 1.0 representing intensity and activity. High energy feels fast, loud, and noisy (e.g., death metal), while low energy feels calm and quiet (e.g., a Bach prelude)."),
        ("Valence",           f"{audio_features.get('valence', 0):.3f}",
         "Describes the musical positiveness. High valence sounds happy, cheerful, or euphoric; low valence sounds sad, depressed, or angry."),
        ("Danceability",      f"{audio_features.get('danceability', 0):.3f}",
         "Describes how suitable a track is for dancing based on tempo, rhythm stability, beat strength, and overall regularity."),
        ("Acousticness",      f"{audio_features.get('acousticness', 0):.3f}",
         "A confidence measure from 0.0 to 1.0 of whether the track is acoustic. A score of 1.0 represents high confidence the track has no electric instruments."),
        ("Tempo",             f"{audio_features.get('tempo', 120):.1f} BPM",
         "The overall estimated speed of a track in Beats Per Minute (BPM)."),
        ("Key",               key_name,
         "The estimated overall key of the track. Integers map to pitches using standard Pitch Class notation (e.g., 0 = C, 1 = C♯/D♭, 2 = D, etc.)."),
        ("Mode",              mode_name,
         "Indicates the modality (Major or Minor) of a track. Major is generally brighter and happier-sounding; Minor is generally darker and more melancholic."),
        ("Time Signature",    f"{audio_features.get('time_signature', 4)}/4",
         "An estimated overall time signature (meter), usually ranging from 3 to 7 — most commonly 4 for standard 4/4 time."),
        ("Loudness",          f"{audio_features.get('loudness', -8):.1f} dB",
         "The overall loudness of a track in decibels (dB), averaged across the entire song. Values typically range between -60 and 0 dB."),
        ("Instrumentalness",  f"{audio_features.get('instrumentalness', 0):.3f}",
         "Predicts whether a track contains no vocals. 'Ooh' and 'aah' sounds are treated as instrumental, but rap or spoken word are clearly vocal."),
        ("Speechiness",       f"{audio_features.get('speechiness', 0):.3f}",
         "Detects the presence of spoken words. Above 0.66 is likely a podcast or poem; 0.33–0.66 is usually rap; below 0.33 is most likely music."),
        ("Liveness",          f"{audio_features.get('liveness', 0):.3f}",
         "Detects the presence of an audience in the recording. Higher values represent a higher probability that the track was performed live."),
    ]
    items = "".join(
        f'<div class="atm-metric" title="{tip}">'
        f'<div class="atm-metric-icon">{audio_icons.get(name, "")}</div>'
        f'<div class="atm-metric-label">{name}</div>'
        f'<div class="atm-metric-value">{value}</div>'
        f'</div>'
        for name, value, tip in metrics
    )
    style = (
        '<style>'
        '.atm-metric-icon{display:flex;align-items:center;justify-content:center;'
        'margin-bottom:4px;opacity:0.9;}'
        '.atm-metric-icon svg{display:block;}'
        '</style>'
    )
    return f'{style}<div class="atm-metrics-grid">{items}</div>'


def render_track_card(track: Dict, rank_colors: Dict) -> str:
    rank      = track.get("rank", 1)
    bg_color, text_color = rank_colors.get(rank, ("#7c3aed", "#f0f0ff"))
    sim_score = track.get("similarity_score", 0.0)
    sim_pct   = int(sim_score * 100)
    _chip_svg = {
        "energy": (
            '<svg viewBox="0 0 16 16" width="13" height="13" fill="none" xmlns="http://www.w3.org/2000/svg" style="display:inline-block;vertical-align:middle;margin-right:4px;">'
            '<path d="M9 1.5 L2 9 h5.5 l-0.5 5.5 L14 7 H8.5 L9 1.5z" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round" fill="rgba(255,255,255,0.15)"/>'
            '</svg>'
        ),
        "valence": (
            '<svg viewBox="0 0 16 16" width="13" height="13" fill="none" xmlns="http://www.w3.org/2000/svg" style="display:inline-block;vertical-align:middle;margin-right:4px;">'
            '<circle cx="8" cy="8" r="6.5" stroke="currentColor" stroke-width="1.4"/>'
            '<path d="M5 9.5 s1 2 3 2 3-2 3-2" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/>'
            '<circle cx="5.5" cy="6.5" r="0.9" fill="currentColor"/>'
            '<circle cx="10.5" cy="6.5" r="0.9" fill="currentColor"/>'
            '</svg>'
        ),
        "danceability": (
            '<svg viewBox="0 0 16 16" width="13" height="13" fill="none" xmlns="http://www.w3.org/2000/svg" style="display:inline-block;vertical-align:middle;margin-right:4px;">'
            '<circle cx="8" cy="3" r="1.5" stroke="currentColor" stroke-width="1.2"/>'
            '<line x1="8" y1="4.5" x2="8" y2="9.5" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/>'
            '<path d="M8 7 l-3 2.5" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/>'
            '<path d="M8 7 l3 2.5" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/>'
            '<line x1="8" y1="9.5" x2="5.5" y2="14" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/>'
            '<line x1="8" y1="9.5" x2="10.5" y2="14" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/>'
            '</svg>'
        ),
        "tempo": (
            '<svg viewBox="0 0 16 16" width="13" height="13" fill="none" xmlns="http://www.w3.org/2000/svg" style="display:inline-block;vertical-align:middle;margin-right:4px;">'
            '<circle cx="8" cy="8" r="6.5" stroke="currentColor" stroke-width="1.4"/>'
            '<line x1="8" y1="4" x2="8" y2="8" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/>'
            '<line x1="8" y1="8" x2="11" y2="9.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>'
            '<circle cx="8" cy="8" r="1" fill="currentColor"/>'
            '</svg>'
        ),
        "key": (
            '<svg viewBox="0 0 16 16" width="13" height="13" fill="none" xmlns="http://www.w3.org/2000/svg" style="display:inline-block;vertical-align:middle;margin-right:4px;">'
            '<rect x="1.5" y="2" width="13" height="12" rx="1.5" stroke="currentColor" stroke-width="1.3"/>'
            '<line x1="5" y1="2" x2="5" y2="14" stroke="currentColor" stroke-width="1" stroke-linecap="round" opacity="0.6"/>'
            '<line x1="9" y1="2" x2="9" y2="14" stroke="currentColor" stroke-width="1" stroke-linecap="round" opacity="0.6"/>'
            '<line x1="1.5" y1="8" x2="14.5" y2="8" stroke="currentColor" stroke-width="1" stroke-linecap="round" opacity="0.6"/>'
            '<rect x="5" y="2" width="2.5" height="6" rx="0.5" fill="rgba(255,255,255,0.25)"/>'
            '<rect x="9" y="2" width="2.5" height="6" rx="0.5" fill="rgba(255,255,255,0.25)"/>'
            '</svg>'
        ),
    }
    chips = [
        (_chip_svg["energy"],      f'{track.get("energy",0):.2f}',                          "Energy"),
        (_chip_svg["valence"],     f'{track.get("valence",0):.2f}',                          "Valence"),
        (_chip_svg["danceability"],f'{track.get("danceability",0):.2f}',                     "Dance"),
        (_chip_svg["tempo"],       f'{track.get("tempo",120):.0f} BPM',                      "Tempo"),
        (_chip_svg["key"],         f'{track.get("key_name","C")} {track.get("mode_name","Major")}', "Key"),
    ]
    chip_html = "".join(
        f'<span class="atm-track-chip">{svg}{label}</span>'
        for svg, label, _ in chips
    )
    return (
        f'<div class="atm-track-card">'
        f'<div class="atm-track-header">'
        f'<div class="atm-rank-badge" style="background:{bg_color};color:{text_color};">#{rank}</div>'
        f'<div style="flex:1;">'
        f'<div class="atm-track-name">{track.get("track_name","Unknown")}</div>'
        f'<div class="atm-track-artist">{track.get("artist_name","Unknown")}</div>'
        f'</div>'
        f'<div style="text-align:right;flex-shrink:0;">'
        f'<div style="font-size:24px;font-weight:900;font-family:\'DM Mono\',monospace;color:{bg_color};line-height:1;">{sim_pct}%</div>'
        f'<div style="font-size:10px;color:#50508a;letter-spacing:0.1em;text-transform:uppercase;font-weight:700;">match</div>'
        f'</div>'
        f'</div>'
        f'<div class="atm-sim-bar-track"><div class="atm-sim-bar-fill" style="width:{sim_pct}%;background:{bg_color};"></div></div>'
        f'<div class="atm-track-chips">{chip_html}</div>'
        f'</div>'
    )

# ---------------------------------------------------------------------------
# Rank badge colors
# ---------------------------------------------------------------------------
RANK_COLORS = {
    1: ("#FFD700", "#1a1000"),
    2: ("#C0C0C0", "#101010"),
    3: ("#CD7F32", "#120800"),
    4: ("#a78bfa", "#0d0020"),
    5: ("#38bdf8", "#001018"),
}

# ---------------------------------------------------------------------------
# Scene type definitions
# ---------------------------------------------------------------------------
SCENE_TYPES = {
    "landscape":   {"label": "Landscape",             "icon": "🏔️",  "description": "Natural scenery — mountains, forests, rivers — with mood and atmospheric focus."},
    "portraiture": {"label": "Portraiture",            "icon": "🧑‍🎨", "description": "Focuses on individuals or groups, capturing likeness, expression, and personality."},
    "still_life":  {"label": "Still Life",             "icon": "🍎",  "description": "An arrangement of inanimate everyday objects such as fruit, flowers, or household items."},
    "genre":       {"label": "Genre / Everyday",       "icon": "🏘️",  "description": "Scenes from everyday life: domestic interiors, street scenes, markets, or social gatherings."},
    "historical":  {"label": "Historical / Narrative", "icon": "⚔️",  "description": "Historical events, myths, or stories — often with multiple figures and complex compositions."},
    "marine":      {"label": "Marine / Seascape",      "icon": "🌊",  "description": "Specialized landscapes focusing on the sea, oceans, or beaches."},
    "cityscape":   {"label": "Cityscape / Urban",      "icon": "🏙️",  "description": "City environments, architecture, and streets."},
    "abstract":    {"label": "Abstract",               "icon": "🌀",  "description": "Non-representational forms, colors, and lines rather than recognizable objects."},
    "surrealism":  {"label": "Surrealism",             "icon": "🦋",  "description": "Realistic imagery combined with dreamlike, bizarre, or illogical concepts."},
    "action":      {"label": "Action / Real Life",     "icon": "📸",  "description": "A captured moment in time — a crowded street, beach party, or a person at work."},
    "religious":   {"label": "Religious / Sacred Art", "icon": "✝️",  "description": "Scenes conveying a religious message, expressing faith, or illustrating holy stories."},
}

# ---------------------------------------------------------------------------
# Scene classification (unchanged logic)
# ---------------------------------------------------------------------------
def classify_scene_type(
    object_data: Dict[str, Any],
    visual_features: Dict[str, Any],
    color_data: Dict[str, Any],
) -> tuple:
    scores: Dict[str, float] = {k: 0.0 for k in SCENE_TYPES}

    detected_objects = object_data.get("detected_objects", [])
    scene_detections = object_data.get("scene_detections", [])
    mood_tags        = [t.lower() for t in object_data.get("mood_tags", [])]
    detector_scene   = object_data.get("scene_type", "")
    person_detected  = bool(object_data.get("person_detected", False))
    object_count     = int(object_data.get("object_count", len(detected_objects)))

    scores["abstract"] += 0.10
    if object_count == 0 and not person_detected:
        scores["abstract"] += 0.25

    if detector_scene in SCENE_TYPES:
        scores[detector_scene] += 2.5

    person_labels    = {"person", "man", "woman", "child", "boy", "girl", "face", "human"}
    person_obj_count = sum(
        1 for o in detected_objects
        if any(pl in o.get("label", "").lower() for pl in person_labels)
    )
    non_person_count = len(detected_objects) - person_obj_count

    if person_detected or person_obj_count > 0:
        total_persons = max(person_obj_count, 1 if person_detected else 0)
        is_crowded = object_count > 6 or total_persons >= 3
        if is_crowded:
            scores["action"]      += 2.0
            scores["genre"]       += 0.8
        elif total_persons >= 2 and non_person_count <= 4:
            scores["portraiture"] += 1.5
            scores["genre"]       += 0.8
        else:
            scores["portraiture"] += 3.0

    LABEL_MAP: List[tuple] = [
        ("surfboard","marine",1.5),("sailboat","marine",1.5),("lighthouse","marine",1.5),
        ("coral","marine",1.4),("wave","marine",1.3),("ship","marine",1.3),("boat","marine",1.2),
        ("skyscraper","cityscape",1.6),("traffic","cityscape",1.3),("street","cityscape",1.2),
        ("building","cityscape",1.1),("bridge","cityscape",1.1),("bus","cityscape",1.1),
        ("pottery","still_life",1.5),("vase","still_life",1.5),("candle","still_life",1.4),
        ("fruit","still_life",1.4),("bowl","still_life",1.3),("wine","still_life",1.3),
        ("crucifix","religious",2.0),("cross","religious",1.8),("halo","religious",1.8),
        ("altar","religious",1.7),("angel","religious",1.7),("cathedral","religious",1.6),
        ("chariot","historical",1.8),("armour","historical",1.7),("knight","historical",1.6),
        ("sword","historical",1.5),("castle","historical",1.5),("crown","historical",1.5),
        ("stadium","action",1.5),("athlete","action",1.4),("concert","action",1.4),("crowd","action",1.4),
        ("kitchen","genre",1.5),("bookshelf","genre",1.4),("sofa","genre",1.4),
        ("waterfall","landscape",1.5),("glacier","landscape",1.5),("mountain","landscape",1.4),
        ("forest","landscape",1.4),("valley","landscape",1.4),("canyon","landscape",1.5),
        ("surreal","surrealism",1.8),("melting","surrealism",1.8),("illusion","surrealism",1.6),
        ("dreamlike","surrealism",1.5),
        ("kaleidoscope","abstract",1.6),("fractal","abstract",1.6),
        ("mandala","abstract",1.5),("geometric","abstract",1.5),("abstract","abstract",1.8),
        ("portrait","portraiture",1.5),("face","portraiture",1.2),
    ]
    for obj in detected_objects:
        label = obj.get("label", "").lower()
        conf  = float(obj.get("confidence", 0.5))
        for keyword, scene_key, weight in LABEL_MAP:
            if keyword in label:
                scores[scene_key] += conf * weight
                break

    P365: List[tuple] = [
        (["underwater","coral_reef","ocean","coast","harbor","beach","lagoon"], "marine", 0.8),
        (["cathedral","church","mosque","temple","monastery","shrine","chapel"], "religious", 0.9),
        (["castle","palace","ruins","fort","battlefield","colosseum"], "historical", 0.8),
        (["skyscraper","downtown","cityscape","street","alley","highway","bridge"], "cityscape", 0.8),
        (["living_room","bedroom","kitchen","office","library","museum","restaurant"], "genre", 0.7),
        (["mountain","valley","canyon","desert","glacier","forest","meadow","waterfall","river","lake"], "landscape", 0.7),
    ]
    for sd in scene_detections:
        p365_label = sd.get("label", "").lower()
        p365_score = float(sd.get("score", 0.0))
        for keywords, cat, w in P365:
            if any(kw in p365_label for kw in keywords):
                scores[cat] += p365_score * w
                break

    MOOD_MAP: List[tuple] = [
        ("portrait","portraiture",1.2),("face","portraiture",1.0),
        ("ocean","marine",0.5),("sea","marine",0.5),("urban","cityscape",0.5),
        ("abstract","abstract",0.6),("geometric","abstract",0.5),
        ("surreal","surrealism",0.6),("dreamy","surrealism",0.5),
        ("sacred","religious",0.6),("spiritual","religious",0.5),
        ("energetic","action",0.5),("dynamic","action",0.5),
        ("domestic","genre",0.5),("indoor","genre",0.4),
        ("historical","historical",0.5),("nature","landscape",0.2),
    ]
    for tag in mood_tags:
        for keyword, scene_key, weight in MOOD_MAP:
            if keyword in tag:
                scores[scene_key] += weight

    texture = float(visual_features.get("texture_complexity", 0.5))
    avg_sat = float(color_data.get("avg_saturation", 0.5))
    non_abstract_top = max((v for k, v in scores.items() if k != "abstract"), default=0.0)

    if non_abstract_top < 0.35 and object_count == 0 and not person_detected:
        if avg_sat < 0.15 and texture < 0.3:
            scores["abstract"] += 0.8
        elif avg_sat > 0.7 and texture < 0.35:
            scores["abstract"] += 0.6
            scores["surrealism"] += 0.4
        elif texture > 0.6:
            scores["landscape"] += 0.5
        else:
            scores["abstract"] += 0.5

    if person_detected and max(scores.values()) < 2.0:
        scores["portraiture"] += 1.5

    best = max(scores, key=lambda k: scores[k])
    return best, f"Score: {scores[best]:.2f}"

# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------
def run_pipeline(
    image, color_extractor, visual_extractor,
    object_detector, feature_mapper, recommender,
    use_object_detection=True,
) -> Dict[str, Any]:
    results = {}
    with st.spinner("🎨 Extracting color palette..."):
        results["color_data"] = color_extractor.extract(image)
    with st.spinner("👁️ Analyzing visual features..."):
        vf = visual_extractor.extract(image, color_data=results["color_data"])
        results["visual_features"]  = vf
        results["element_summary"]  = visual_extractor.get_element_summary(vf)
    if use_object_detection:
        with st.spinner("🔍 Detecting objects..."):
            results["object_data"] = object_detector.detect(image)
    else:
        results["object_data"] = {
            "detected_objects": [], "mood_tags": ["instrumental","ambient"],
            "scene_type": "abstract", "object_count": 0,
            "person_detected": False, "detection_available": False,
        }
    with st.spinner("🎵 Mapping to audio features..."):
        results["audio_features"] = feature_mapper.map(
            results["visual_features"], results["color_data"], results["object_data"]
        )
    with st.spinner("🎯 Finding your songs..."):
        results["recommendations"] = recommender.recommend_with_fallback(
            results["audio_features"], top_n=5
        )
    return results

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
def render_sidebar() -> bool:
    with st.sidebar:
        st.markdown("## 🎨 Art to Music")
        st.markdown(
            '<p style="font-size:13px;color:#7070a0;font-style:italic;margin-top:-6px;">'
            'Transform Visual Art into Sound</p>',
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown("### How It Works")
        steps = [
            ("1", "Upload artwork"),
            ("2", "Extract color palette"),
            ("3", "Analyze 7 art elements"),
            ("4", "Detect objects & mood"),
            ("5", "Map to audio features"),
            ("6", "Find matching songs"),
        ]
        for num, text in steps:
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:10px;padding:7px 0;'
                f'border-bottom:1px solid rgba(255,255,255,0.04);">'
                f'<div style="width:22px;height:22px;border-radius:6px;'
                f'background:rgba(139,92,246,0.15);border:1px solid rgba(139,92,246,0.25);'
                f'display:flex;align-items:center;justify-content:center;'
                f'font-size:10px;font-family:\'DM Mono\',monospace;font-weight:700;'
                f'color:#a78bfa;flex-shrink:0;">{num}</div>'
                f'<span style="font-size:13px;color:#9898b8;">{text}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.divider()
        st.markdown("### Settings")
        use_detection = st.toggle(
            "Enable Object Detection",
            value=True,
            help="Use YOLOv8 to detect objects and infer mood tags. Disable for faster processing.",
        )
        st.divider()
        st.markdown("### About")
        st.markdown(
            '<p style="font-size:13px;color:#7878a8;line-height:1.6;">'
            'Art to Music analyzes uploaded artwork — its color palette, composition, texture, '
            'and detected objects — then maps visual characteristics to Spotify audio features '
            'and recommends the top 5 most sonically aligned tracks.</p>',
            unsafe_allow_html=True,
        )
        st.divider()
        st.caption("Built with Streamlit · OpenCV · YOLOv8 · scikit-learn")
    return use_detection

# ---------------------------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------------------------
def main() -> None:
    use_detection = render_sidebar()

    # ── Header ──────────────────────────────────────────────────────────
    st.markdown(
        '<div class="atm-header">'
        '<div class="atm-header-left">'
        '<h1 class="atm-header-title">ART TO MUSIC</h1>'
        '<div class="atm-header-sub">Upload a painting, photograph, or illustration — get a soundtrack.</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Load pipeline ────────────────────────────────────────────────────
    try:
        color_extractor, visual_extractor, object_detector, feature_mapper, recommender = load_pipeline()
    except Exception as exc:
        st.error(f"⚠️ Failed to initialize pipeline: {exc}")
        st.info("Make sure all requirements are installed. Run `setup.bat` to set up the environment.")
        return

    if not recommender.is_loaded:
        st.warning(
            "⚠️ Recommendation model not found. "
            "Run `python scripts/train_recommender.py` to train it first."
        )

    # ── Upload ───────────────────────────────────────────────────────────
    st.markdown('<div class="atm-section-label">Upload Your Artwork</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drag and drop or click to browse",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.markdown(
            '<div class="atm-upload-zone">'
            '<div class="atm-upload-icon">🖼️</div>'
            '<p class="atm-upload-headline">Drop your artwork here</p>'
            '<p class="atm-upload-hint">JPG · JPEG · PNG · WEBP</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    # ── Load image ───────────────────────────────────────────────────────
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as exc:
        st.error(f"Failed to load image: {exc}")
        return

    # ── Run pipeline ─────────────────────────────────────────────────────
    try:
        results = run_pipeline(
            image, color_extractor, visual_extractor,
            object_detector, feature_mapper, recommender,
            use_object_detection=use_detection,
        )
    except Exception as exc:
        st.error(f"Pipeline failed: {exc}")
        st.exception(exc)
        return

    color_data      = results["color_data"]
    visual_features = results["visual_features"]
    element_summary = results["element_summary"]
    object_data     = results["object_data"]
    audio_features  = results["audio_features"]
    recommendations = results["recommendations"]

    st.markdown('<div class="atm-divider"></div>', unsafe_allow_html=True)

    # ── ROW 1: Artwork | Palette + HSV + Scene ───────────────────────────
    col_img, col_right = st.columns([5, 7], gap="large")

    with col_img:
        st.markdown('<div class="atm-section-label">Your Artwork</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)

    with col_right:
        st.markdown('<div class="atm-section-label">Color Palette</div>', unsafe_allow_html=True)
        dominant_colors = color_data.get("dominant_colors", [])
        st.markdown(render_color_strip_and_swatches(dominant_colors), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            '<div class="atm-section-label">HSB values <span style="font-size:10px;'
            'color:#50508a;letter-spacing:0;text-transform:none;font-weight:400;">'
            '&nbsp;</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(render_hsv_badges(color_data), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<div class="atm-section-label">Scene Type</div>', unsafe_allow_html=True)
        scene_type, _ = classify_scene_type(object_data, visual_features, color_data)
        scene_info     = SCENE_TYPES[scene_type]
        st.markdown(render_scene_card(scene_type, scene_info), unsafe_allow_html=True)

        tlabel, tcolor = temp_label(float(color_data.get("avg_hue", 0)))
        st.markdown(
            f'<div style="margin-top:10px;display:flex;align-items:center;gap:8px;">'
            f'<div style="width:8px;height:8px;border-radius:50%;background:{tcolor};'
            f'box-shadow:0 0 8px {tcolor};"></div>'
            f'<span style="font-size:13px;color:{tcolor};letter-spacing:0.04em;">'
            f'{tlabel} tone · {len(dominant_colors)} colors extracted</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="atm-divider"></div>', unsafe_allow_html=True)

    # ── ROW 2: 7 Elements ────────────────────────────────────────────────
    element_icons = {
        "LINE": (
            '<svg viewBox="0 0 32 32" width="28" height="28" fill="none" xmlns="http://www.w3.org/2000/svg">'
            '<line x1="4" y1="28" x2="28" y2="4" stroke="#a78bfa" stroke-width="2.5" stroke-linecap="round"/>'
            '<line x1="4" y1="16" x2="20" y2="16" stroke="#34d399" stroke-width="2" stroke-linecap="round" stroke-dasharray="2 3"/>'
            '<line x1="8" y1="6" x2="8" y2="24" stroke="#fbbf24" stroke-width="2" stroke-linecap="round" stroke-dasharray="1 4"/>'
            '</svg>'
        ),
        "SHAPE": (
            '<svg viewBox="0 0 32 32" width="28" height="28" fill="none" xmlns="http://www.w3.org/2000/svg">'
            '<polygon points="16,3 30,28 2,28" stroke="#a78bfa" stroke-width="2.2" stroke-linejoin="round" fill="rgba(167,139,250,0.12)"/>'
            '<rect x="3" y="10" width="11" height="11" stroke="#34d399" stroke-width="2" fill="rgba(52,211,153,0.1)"/>'
            '<circle cx="25" cy="13" r="5" stroke="#fbbf24" stroke-width="2" fill="rgba(251,191,36,0.1)"/>'
            '</svg>'
        ),
        "FORM": (
            '<svg viewBox="0 0 32 32" width="28" height="28" fill="none" xmlns="http://www.w3.org/2000/svg">'
            '<path d="M6 22 L16 8 L26 22 Z" stroke="#a78bfa" stroke-width="2" fill="rgba(167,139,250,0.15)" stroke-linejoin="round"/>'
            '<path d="M26 22 L28 26 L18 26 L16 22" stroke="#7c5cd6" stroke-width="1.5" fill="rgba(124,92,214,0.2)" stroke-linejoin="round"/>'
            '<path d="M6 22 L4 26 L18 26 L16 22" stroke="#5b3fa8" stroke-width="1.5" fill="rgba(91,63,168,0.2)" stroke-linejoin="round"/>'
            '</svg>'
        ),
        "COLOR": (
            '<svg viewBox="0 0 32 32" width="28" height="28" fill="none" xmlns="http://www.w3.org/2000/svg">'
            '<circle cx="16" cy="13" r="9" stroke="none" fill="none"/>'
            '<circle cx="13" cy="11" r="7" fill="rgba(239,68,68,0.55)"/>'
            '<circle cx="19" cy="11" r="7" fill="rgba(59,130,246,0.55)"/>'
            '<circle cx="16" cy="17" r="7" fill="rgba(34,197,94,0.55)"/>'
            '<circle cx="16" cy="13" r="3" fill="rgba(255,255,255,0.55)"/>'
            '</svg>'
        ),
        "VALUE": (
            '<svg viewBox="0 0 32 32" width="28" height="28" fill="none" xmlns="http://www.w3.org/2000/svg">'
            '<defs><linearGradient id="valGrad" x1="0" y1="0" x2="1" y2="0">'
            '<stop offset="0%" stop-color="#1a1a2e"/>'
            '<stop offset="100%" stop-color="#ffffff"/>'
            '</linearGradient></defs>'
            '<rect x="3" y="10" width="26" height="12" rx="3" fill="url(#valGrad)" stroke="rgba(255,255,255,0.2)" stroke-width="1"/>'
            '<line x1="9" y1="10" x2="9" y2="22" stroke="rgba(255,255,255,0.15)" stroke-width="1"/>'
            '<line x1="16" y1="10" x2="16" y2="22" stroke="rgba(255,255,255,0.15)" stroke-width="1"/>'
            '<line x1="23" y1="10" x2="23" y2="22" stroke="rgba(255,255,255,0.15)" stroke-width="1"/>'
            '</svg>'
        ),
        "SPACE": (
            '<svg viewBox="0 0 32 32" width="28" height="28" fill="none" xmlns="http://www.w3.org/2000/svg">'
            '<rect x="2" y="2" width="28" height="28" rx="2" fill="rgba(15,10,40,0.6)" stroke="rgba(167,139,250,0.3)" stroke-width="1"/>'
            '<circle cx="16" cy="16" r="3" fill="#a78bfa" opacity="0.9"/>'
            '<circle cx="7" cy="8" r="1.2" fill="#fbbf24" opacity="0.7"/>'
            '<circle cx="25" cy="6" r="0.8" fill="white" opacity="0.6"/>'
            '<circle cx="27" cy="22" r="1.5" fill="#34d399" opacity="0.6"/>'
            '<circle cx="5" cy="25" r="0.9" fill="white" opacity="0.5"/>'
            '<circle cx="22" cy="27" r="0.7" fill="#a78bfa" opacity="0.4"/>'
            '<line x1="16" y1="16" x2="7" y2="8" stroke="rgba(167,139,250,0.2)" stroke-width="0.8"/>'
            '<line x1="16" y1="16" x2="27" y2="22" stroke="rgba(52,211,153,0.2)" stroke-width="0.8"/>'
            '</svg>'
        ),
        "TEXTURE": (
            '<svg viewBox="0 0 32 32" width="28" height="28" fill="none" xmlns="http://www.w3.org/2000/svg">'
            '<rect x="2" y="2" width="28" height="28" rx="3" fill="rgba(167,139,250,0.06)" stroke="rgba(167,139,250,0.2)" stroke-width="1"/>'
            '<line x1="2" y1="8"  x2="30" y2="8"  stroke="rgba(167,139,250,0.35)" stroke-width="1.2" stroke-linecap="round"/>'
            '<line x1="2" y1="14" x2="30" y2="14" stroke="rgba(167,139,250,0.25)" stroke-width="0.9" stroke-linecap="round"/>'
            '<line x1="2" y1="20" x2="30" y2="20" stroke="rgba(167,139,250,0.35)" stroke-width="1.2" stroke-linecap="round"/>'
            '<line x1="2" y1="26" x2="30" y2="26" stroke="rgba(167,139,250,0.25)" stroke-width="0.9" stroke-linecap="round"/>'
            '<line x1="8"  y1="2" x2="8"  y2="30" stroke="rgba(251,191,36,0.2)"  stroke-width="0.8" stroke-linecap="round"/>'
            '<line x1="16" y1="2" x2="16" y2="30" stroke="rgba(251,191,36,0.2)"  stroke-width="0.8" stroke-linecap="round"/>'
            '<line x1="24" y1="2" x2="24" y2="30" stroke="rgba(251,191,36,0.2)"  stroke-width="0.8" stroke-linecap="round"/>'
            '</svg>'
        ),
    }
    element_tooltips = {
        "LINE":    "The path of a moving point. Lines guide the viewer's eye and create structure.",
        "SHAPE":   "A flat, two-dimensional area defined by boundaries or edges.",
        "FORM":    "A three-dimensional object with depth, height, and width.",
        "COLOR":   "The visual property created by light, including hue, saturation, and brightness.",
        "VALUE":   "The lightness or darkness of a color.",
        "SPACE":   "The area around, between, or within objects in an artwork.",
        "TEXTURE": "The surface quality or feel of an object (smooth, rough, soft, etc.).",
    }

    st.markdown('<div class="atm-section-label">The 7 Elements of Art</div>', unsafe_allow_html=True)
    st.markdown(render_element_cards(element_summary, element_icons, element_tooltips), unsafe_allow_html=True)

    col_bars, col_objs = st.columns([1, 1], gap="large")
    with col_bars:
        st.markdown(render_element_bars(element_summary, element_icons, element_tooltips), unsafe_allow_html=True)
    with col_objs:
        if use_detection:
            st.markdown('<div class="atm-section-label">Detected Objects</div>', unsafe_allow_html=True)
            detected  = object_data.get("detected_objects", [])
            mood_tags = object_data.get("mood_tags", [])
            st.markdown(render_object_badges(detected), unsafe_allow_html=True)
            if mood_tags:
                st.markdown(
                    '<div style="font-size:11px;letter-spacing:0.12em;text-transform:uppercase;'
                    'color:#60609a;font-weight:700;margin-top:16px;margin-bottom:6px;">Mood Tags</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(render_mood_tags(mood_tags), unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="display:flex;align-items:center;justify-content:center;'
                'height:100%;min-height:120px;">'
                '<p style="color:#50508a;font-size:13px;text-align:center;font-style:italic;">'
                'Object detection disabled</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="atm-divider"></div>', unsafe_allow_html=True)

    # ── ROW 3: Audio Features ────────────────────────────────────────────
    st.markdown('<div class="atm-section-label">Mapped Audio Features</div>', unsafe_allow_html=True)
    col_radar, col_metrics = st.columns([1, 1], gap="large")
    with col_radar:
        st.markdown(
            '<div style="font-size:11px;letter-spacing:0.12em;text-transform:uppercase;'
            'color:#8080b0;font-weight:700;margin-bottom:8px;">Audio Feature Radar</div>',
            unsafe_allow_html=True,
        )
        fig = build_radar_chart(audio_features)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with col_metrics:
        st.markdown(
            '<div style="font-size:11px;letter-spacing:0.12em;text-transform:uppercase;'
            'color:#8080b0;font-weight:700;margin-bottom:10px;">Feature Breakdown</div>',
            unsafe_allow_html=True,
        )
        st.markdown(render_audio_metrics(audio_features), unsafe_allow_html=True)

    st.markdown('<div class="atm-divider"></div>', unsafe_allow_html=True)

    # ── ROW 4: Recommendations ───────────────────────────────────────────
    st.markdown('<div class="atm-section-label">Top 5 Music Recommendations</div>', unsafe_allow_html=True)

    if not recommendations:
        st.warning(
            "No recommendations found. Make sure the model is trained. "
            "Run `python scripts/train_recommender.py` or `setup.bat`."
        )
        return

    spotify = load_spotify_client()

    _PREVIEW_UNAVAILABLE = (
        '<div style="display:flex;align-items:center;gap:8px;'
        'background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);'
        'border-radius:10px;padding:10px 14px;margin-bottom:10px;">'
        '<span style="font-size:18px;">🎵</span>'
        '<span style="color:#50508a;font-size:13px;font-style:italic;'
        'font-family:\'DM Mono\',monospace;">Preview unavailable</span>'
        '</div>'
    )

    for track in recommendations:
        st.markdown(render_track_card(track, RANK_COLORS), unsafe_allow_html=True)
        track_name  = track.get("track_name", "Unknown")
        artist_name = track.get("artist_name", "Unknown")
        try:
            spotify_result = spotify.search_track(track_name, artist_name)
            if spotify_result:
                embed_url = spotify_result["embed_url"]
                st.markdown(
                    f'<iframe src="{embed_url}" width="100%" height="80" frameborder="0" '
                    f'allowtransparency="true" allow="autoplay; clipboard-write; encrypted-media; '
                    f'fullscreen; picture-in-picture" loading="lazy" '
                    f'style="border-radius:10px;margin-bottom:12px;"></iframe>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(_PREVIEW_UNAVAILABLE, unsafe_allow_html=True)
        except Exception as _exc:
            print(f"[app] Spotify embed error for '{track_name}': {_exc}")
            st.markdown(_PREVIEW_UNAVAILABLE, unsafe_allow_html=True)

    # ── Inject interactive JS ────────────────────────────────────────────
    st.markdown(_INTERACTIVE_JS, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
