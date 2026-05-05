"""
BanglaLekha-Isolated — Live Demo
Streamlit app with drawable canvas for real-time Bangla character recognition.
Deploy on HuggingFace Spaces or Streamlit Community Cloud.
"""

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps
import io
import base64

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="বাংলা লেখা — Bangla Script Recognition",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Bengali:wght@300;400;700&family=Space+Mono:wght@400;700&family=Playfair+Display:wght@700&display=swap');

:root {
    --bg: #0f0e0d;
    --surface: #1a1917;
    --border: #2e2c28;
    --accent: #e8c547;
    --accent2: #c97b4b;
    --text: #f0ece4;
    --muted: #7a756a;
    --green: #4caf85;
    --red: #e05c5c;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
}

[data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
}

[data-testid="block-container"] {
    padding: 2rem 3rem !important;
    max-width: 1200px !important;
}

h1, h2, h3 { font-family: 'Playfair Display', serif !important; }

.mono { font-family: 'Space Mono', monospace; }

.header-section {
    border-bottom: 1px solid var(--border);
    padding-bottom: 1.5rem;
    margin-bottom: 2rem;
}

.header-eyebrow {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.header-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1.1;
    margin: 0;
}

.header-subtitle {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: var(--muted);
    margin-top: 0.5rem;
}

.stat-chip {
    display: inline-block;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.25rem 0.75rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    margin-right: 0.5rem;
}

.stat-chip span {
    color: var(--accent);
    font-weight: 700;
}

.canvas-section {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.5rem;
}

.canvas-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 1rem;
}

.prediction-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.5rem;
    height: 100%;
}

.prediction-char {
    font-family: 'Noto Sans Bengali', sans-serif;
    font-size: 7rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
    text-align: center;
    padding: 1rem 0;
    text-shadow: 0 0 60px rgba(232, 197, 71, 0.3);
}

.prediction-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    text-align: center;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.5rem;
}

.confidence-bar-container {
    margin: 1rem 0;
}

.confidence-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.3rem;
}

.top5-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--border);
}

.top5-rank {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    color: var(--muted);
    width: 1.2rem;
    text-align: right;
}

.top5-char {
    font-family: 'Noto Sans Bengali', sans-serif;
    font-size: 1.6rem;
    color: var(--text);
    width: 2.5rem;
    text-align: center;
}

.top5-bar-wrap {
    flex: 1;
    background: var(--bg);
    border-radius: 2px;
    height: 6px;
    overflow: hidden;
}

.top5-bar {
    height: 100%;
    border-radius: 2px;
    background: var(--accent);
    transition: width 0.4s ease;
}

.top5-pct {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    width: 3rem;
    text-align: right;
}

.practice-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.5rem;
    text-align: center;
}

.target-char {
    font-family: 'Noto Sans Bengali', sans-serif;
    font-size: 6rem;
    color: var(--text);
    line-height: 1;
}

.correct-badge {
    background: var(--green);
    color: #000;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    padding: 0.3rem 0.8rem;
    border-radius: 3px;
    font-weight: 700;
    letter-spacing: 0.1em;
}

.wrong-badge {
    background: var(--red);
    color: #fff;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    padding: 0.3rem 0.8rem;
    border-radius: 3px;
    font-weight: 700;
    letter-spacing: 0.1em;
}

.section-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 2rem 0;
}

/* Override Streamlit defaults */
.stButton > button {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    padding: 0.5rem 1.25rem !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

div[data-testid="stSelectbox"] label,
div[data-testid="stRadio"] label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.7rem !important;
    color: var(--muted) !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

.stProgress > div > div {
    background: var(--accent) !important;
}

div[data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 0.75rem 1rem !important;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    border: none !important;
    padding: 0.5rem 1.5rem !important;
}

.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* Canvas wrapper */
.canvas-wrapper canvas {
    border-radius: 4px;
    cursor: crosshair !important;
}

[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 6px !important;
}

</style>
""", unsafe_allow_html=True)


# ── Class mapping ──────────────────────────────────────────────────────────────
BANGLA_CLASSES = {
    0:  "অ", 1:  "আ", 2:  "ই", 3:  "ঈ", 4:  "উ", 5:  "ঊ",
    6:  "ঋ", 7:  "এ", 8:  "ঐ", 9:  "ও", 10: "ঔ",
    11: "ক", 12: "খ", 13: "গ", 14: "ঘ", 15: "ঙ", 16: "চ",
    17: "ছ", 18: "জ", 19: "ঝ", 20: "ঞ", 21: "ট", 22: "ঠ",
    23: "ড", 24: "ঢ", 25: "ণ", 26: "ত", 27: "থ", 28: "দ",
    29: "ধ", 30: "ন", 31: "প", 32: "ফ", 33: "ব", 34: "ভ",
    35: "ম", 36: "য", 37: "র", 38: "ল", 39: "শ", 40: "ষ",
    41: "স", 42: "হ", 43: "ড়", 44: "ঢ়", 45: "য়", 46: "ৎ",
    47: "ঁ",  48: "ং",  49: "ঃ",
    50: "০", 51: "১", 52: "২", 53: "৩", 54: "৪", 55: "৫",
    56: "৬", 57: "৭", 58: "৮", 59: "৯",
    60: "ক্ষ", 61: "ত্র", 62: "জ্ঞ", 63: "ষ্ক", 64: "স্ক", 65: "স্থ",
    66: "চ্ছ", 67: "ক্ত", 68: "ত্ত", 69: "ব্ধ", 70: "ম্প", 71: "ষ্ণ",
    72: "ষ্ঠ", 73: "ম্ব", 74: "ণ্ড", 75: "দ্ব", 76: "ন্থ", 77: "স্ত",
    78: "ল্প", 79: "ষ্প", 80: "ন্দ", 81: "ন্ধ", 82: "ম্ম", 83: "ন্ট",
}

CLASS_GROUPS = {
    "Vowels (স্বরবর্ণ)":     list(range(0, 11)),
    "Consonants (ব্যঞ্জনবর্ণ)": list(range(11, 50)),
    "Digits (সংখ্যা)":       list(range(50, 60)),
    "Compounds (যুক্তবর্ণ)": list(range(60, 84)),
}

NUM_CLASSES = 84
IMG_SIZE    = 128

# ── Model definition (must match training code exactly) ───────────────────────
class BanglaCNN(nn.Module):
    def __init__(self, num_layers=5, dropout_rate=0.4,
                 num_classes=NUM_CLASSES, img_size=IMG_SIZE):
        super().__init__()
        layers = []
        in_ch, out_ch, spatial = 1, 32, img_size
        for _ in range(num_layers):
            layers += [
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_ch = out_ch; out_ch = min(out_ch * 2, 512); spatial //= 2
        self.features   = nn.Sequential(*layers)
        flat            = in_ch * spatial * spatial
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 1024), nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),  nn.BatchNorm1d(512),  nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Model loader ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = BanglaCNN(num_layers=5, dropout_rate=0.4,
                      num_classes=NUM_CLASSES, img_size=IMG_SIZE)
    try:
        state = torch.load("FINAL_PRODUCTION_MODEL.pth",
                           map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return model, True
    except FileNotFoundError:
        return model, False


# ── Preprocessing ──────────────────────────────────────────────────────────────
val_tf = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

def preprocess(img: Image.Image) -> torch.Tensor:
    """Convert any PIL image to the model's expected input tensor."""
    img = img.convert("RGBA")
    # white background composite
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    img = bg.convert("L")
    # invert if background is light (drawing on white canvas)
    arr = np.array(img)
    if arr.mean() > 127:
        img = ImageOps.invert(img)
    img = ImageOps.autocontrast(img)
    return val_tf(img).unsqueeze(0)   # [1, 1, 128, 128]

@torch.no_grad()
def predict(model, tensor: torch.Tensor):
    logits = model(tensor)
    probs  = torch.softmax(logits, dim=1)[0]
    top5   = torch.topk(probs, 5)
    return probs, top5.indices.tolist(), top5.values.tolist()


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-section">
    <div class="header-eyebrow">✦ T3.5 — Handwritten Indic Script Recognition</div>
    <h1 class="header-title">বাংলা লেখা<br><span style="color: var(--muted); font-size: 1.6rem;">Bangla Script Recognition</span></h1>
    <div class="header-subtitle" style="margin-top: 1rem;">
        <span class="stat-chip">Model <span>5-Layer CNN</span></span>
        <span class="stat-chip">Classes <span>84</span></span>
        <span class="stat-chip">Val Accuracy <span>94.84%</span></span>
        <span class="stat-chip">Dataset <span>BanglaLekha-Isolated</span></span>
        <span class="stat-chip">Training <span>From Scratch</span></span>
    </div>
</div>
""", unsafe_allow_html=True)

model, model_loaded = load_model()

if not model_loaded:
    st.warning(
        "⚠️  `FINAL_PRODUCTION_MODEL.pth` not found in the app directory. "
        "Upload it alongside `app.py` when deploying. "
        "The UI is fully functional — predictions will run once the model file is present.",
        icon="⚠️"
    )

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_live, tab_practice, tab_about = st.tabs([
    "LIVE RECOGNITION", "PRACTICE MODE", "ABOUT"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE RECOGNITION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_live:
    st.markdown("<br>", unsafe_allow_html=True)

    try:
        from streamlit_drawable_canvas import st_canvas
        canvas_available = True
    except ImportError:
        canvas_available = False

    input_col, pred_col = st.columns([1, 1], gap="large")

    with input_col:
        st.markdown('<div class="canvas-label">✦ Input Method</div>', unsafe_allow_html=True)
        input_mode = st.radio(
            "input_mode", ["Draw", "Upload Image"],
            horizontal=True, label_visibility="collapsed"
        )

        image_input = None

        if input_mode == "Draw":
            if not canvas_available:
                st.error(
                    "`streamlit-drawable-canvas` not installed. "
                    "Add it to `requirements.txt` and redeploy, or switch to **Upload Image**."
                )
            else:
                st.markdown('<div class="canvas-label">Draw a Bangla character below</div>',
                            unsafe_allow_html=True)
                canvas_result = st_canvas(
                    fill_color   = "rgba(0,0,0,0)",
                    stroke_width = st.slider("Brush size", 8, 32, 18, label_visibility="collapsed"),
                    stroke_color = "#ffffff",
                    background_color = "#000000",
                    height       = 320,
                    width        = 320,
                    drawing_mode = "freedraw",
                    key          = "canvas_live",
                    display_toolbar = True,
                )
                if canvas_result.image_data is not None:
                    arr = canvas_result.image_data.astype(np.uint8)
                    if arr[..., :3].sum() > 1000:   # non-empty check
                        image_input = Image.fromarray(arr, "RGBA")

        else:
            uploaded = st.file_uploader(
                "Upload a handwritten character image",
                type=["png", "jpg", "jpeg", "bmp"],
                label_visibility="collapsed"
            )
            if uploaded:
                image_input = Image.open(uploaded)
                st.image(image_input, caption="Uploaded image", width=300)

        if image_input is not None and model_loaded:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("PREDICT →", use_container_width=True):
                with st.spinner(""):
                    tensor = preprocess(image_input)
                    probs, top5_idx, top5_conf = predict(model, tensor)
                    st.session_state["last_probs"]    = probs
                    st.session_state["last_top5_idx"] = top5_idx
                    st.session_state["last_top5_conf"]= top5_conf
        elif image_input is not None and not model_loaded:
            st.info("Model not loaded — upload `FINAL_PRODUCTION_MODEL.pth` to enable predictions.")

    with pred_col:
        st.markdown('<div class="canvas-label">✦ Prediction</div>', unsafe_allow_html=True)

        if "last_top5_idx" in st.session_state:
            idx   = st.session_state["last_top5_idx"]
            conf  = st.session_state["last_top5_conf"]
            char  = BANGLA_CLASSES.get(idx[0], "?")
            pct   = conf[0] * 100

            st.markdown(f"""
            <div class="prediction-card">
                <div class="prediction-char">{char}</div>
                <div class="prediction-label">Top prediction · {pct:.1f}% confidence</div>
                <hr style="border-color: var(--border); margin: 1.25rem 0;">
                <div class="canvas-label">Top 5 candidates</div>
            """, unsafe_allow_html=True)

            bar_colors = ["var(--accent)", "var(--accent2)",
                          "#8a7a55", "#5a5248", "#3a3530"]

            for rank, (ci, cv) in enumerate(zip(idx, conf)):
                ch  = BANGLA_CLASSES.get(ci, "?")
                pct_i = cv * 100
                w   = pct_i
                col = bar_colors[rank]
                st.markdown(f"""
                <div class="top5-row">
                    <div class="top5-rank">#{rank+1}</div>
                    <div class="top5-char">{ch}</div>
                    <div class="top5-bar-wrap">
                        <div class="top5-bar" style="width:{w:.1f}%; background:{col};"></div>
                    </div>
                    <div class="top5-pct">{pct_i:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="prediction-card" style="display:flex; align-items:center;
                        justify-content:center; min-height: 320px;">
                <div style="text-align:center;">
                    <div style="font-size: 3rem; opacity: 0.15;">অ</div>
                    <div class="prediction-label" style="margin-top: 1rem;">
                        Draw or upload a character,<br>then press PREDICT
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PRACTICE MODE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_practice:
    st.markdown("<br>", unsafe_allow_html=True)

    if "practice_idx"   not in st.session_state: st.session_state["practice_idx"]   = 0
    if "practice_score" not in st.session_state: st.session_state["practice_score"] = [0, 0]
    if "practice_result"not in st.session_state: st.session_state["practice_result"]= None

    ctrl_col, target_col, draw_col = st.columns([1, 1, 1], gap="large")

    with ctrl_col:
        st.markdown('<div class="canvas-label">✦ Practice Settings</div>', unsafe_allow_html=True)
        group_name = st.selectbox(
            "Character group", list(CLASS_GROUPS.keys()),
            label_visibility="collapsed"
        )
        group_ids = CLASS_GROUPS[group_name]

        if st.button("NEW CHARACTER", use_container_width=True):
            import random
            st.session_state["practice_idx"]    = random.choice(group_ids)
            st.session_state["practice_result"] = None
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        correct, total = st.session_state["practice_score"]
        st.markdown(f"""
        <div class="canvas-label">✦ Score</div>
        <div style="font-family: 'Space Mono', monospace; font-size: 2rem;
                    color: var(--accent); margin-top: 0.5rem;">
            {correct}<span style="color: var(--muted); font-size: 1rem;">/{total}</span>
        </div>
        <div class="canvas-label" style="margin-top: 0.25rem;">
            {"—" if total == 0 else f"{100*correct/total:.0f}% accuracy"}
        </div>
        """, unsafe_allow_html=True)

    with target_col:
        target_id   = st.session_state["practice_idx"]
        target_char = BANGLA_CLASSES.get(target_id, "অ")
        result      = st.session_state["practice_result"]

        badge = ""
        if result == "correct":
            badge = '<span class="correct-badge">✓ CORRECT</span>'
        elif result == "wrong":
            badge = '<span class="wrong-badge">✗ WRONG</span>'

        st.markdown(f"""
        <div class="practice-card">
            <div class="canvas-label">✦ Draw this character</div>
            <div class="target-char">{target_char}</div>
            <div style="margin-top: 0.75rem; font-family: 'Space Mono', monospace;
                        font-size: 0.65rem; color: var(--muted);">
                Class {target_id + 1} of 84
            </div>
            <div style="margin-top: 1rem;">{badge}</div>
        </div>
        """, unsafe_allow_html=True)

    with draw_col:
        st.markdown('<div class="canvas-label">✦ Your drawing</div>', unsafe_allow_html=True)

        try:
            from streamlit_drawable_canvas import st_canvas as _sc
            pr = _sc(
                fill_color       = "rgba(0,0,0,0)",
                stroke_width     = 18,
                stroke_color     = "#ffffff",
                background_color = "#000000",
                height           = 260,
                width            = 260,
                drawing_mode     = "freedraw",
                key              = "canvas_practice",
                display_toolbar  = True,
            )

            if st.button("CHECK ANSWER", use_container_width=True):
                if pr.image_data is not None and model_loaded:
                    arr = pr.image_data.astype(np.uint8)
                    if arr[..., :3].sum() > 500:
                        img_pil = Image.fromarray(arr, "RGBA")
                        tensor  = preprocess(img_pil)
                        _, top5_idx, _ = predict(model, tensor)
                        correct_guess  = (top5_idx[0] == target_id)
                        sc = st.session_state["practice_score"]
                        st.session_state["practice_score"] = [
                            sc[0] + int(correct_guess), sc[1] + 1
                        ]
                        st.session_state["practice_result"] = (
                            "correct" if correct_guess else "wrong"
                        )
                        st.rerun()
                    else:
                        st.warning("Draw something first!")
                elif not model_loaded:
                    st.info("Model not loaded.")

        except ImportError:
            st.error("`streamlit-drawable-canvas` required for Practice Mode.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns([3, 2], gap="large")

    with col_a:
        st.markdown("""
        <div class="canvas-label">✦ Model Architecture</div>
        """, unsafe_allow_html=True)

        st.markdown("""
        **BanglaCNN** is a 5-layer convolutional network trained entirely from scratch on the
        [BanglaLekha-Isolated](https://data.mendeley.com/datasets/hf6sf8zrkc/2) dataset.
        No transfer learning, no pre-trained weights.

        Each convolutional block consists of two Conv2d layers with BatchNorm and ReLU,
        followed by MaxPool2d. Channels double each block: 32 → 64 → 128 → 256 → 512.
        A two-layer FC classifier head with dropout completes the network.

        **Training details:**
        - Optimizer: AdamW (lr=1e-3, wd=1e-4)
        - Schedule: 8-epoch linear warmup → cosine annealing (40 epochs)
        - Augmentation: random crop, rotation ±10°, shear ±12°
        - Loss: label-smoothed (0.1) cross-entropy with class weighting
        - Hardware: 4× RTX 2080 Ti with DDP (NCCL backend)
        - Effective batch: 2048 (512 per GPU)
        """)

    with col_b:
        st.markdown('<div class="canvas-label">✦ Performance vs Literature</div>',
                    unsafe_allow_html=True)

        data = {
            "Model": ["**BanglaCNN (Ours)**", "BornoViT (2026)", "Dipu et al. (2021)", "KDANet (2023)"],
            "Accuracy": ["**94.84%**", "95.77%", "96.88%", "98.10%"],
            "Full 84 Classes": ["✓", "✓*", "✓", "✗ (60 only)"],
            "From Scratch": ["✓", "✗", "✗", "✗"],
        }

        import pandas as pd
        df = pd.DataFrame(data)
        st.dataframe(df, hide_index=True, use_container_width=True)

        st.markdown("""
        <div style="font-family: 'Space Mono', monospace; font-size: 0.6rem;
                    color: var(--muted); margin-top: 0.5rem;">
        * BornoViT scope on 84 classes is ambiguous — see paper methodology.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <hr class="section-divider">
    <div class="canvas-label">✦ Dataset Breakdown — 84 Classes</div>
    """, unsafe_allow_html=True)

    g1, g2, g3, g4 = st.columns(4)
    for col, (name, ids) in zip([g1, g2, g3, g4], CLASS_GROUPS.items()):
        chars = " ".join(BANGLA_CLASSES[i] for i in ids[:12])
        col.markdown(f"""
        <div style="background: var(--surface); border: 1px solid var(--border);
                    border-radius: 6px; padding: 1rem;">
            <div class="canvas-label">{name}</div>
            <div style="font-family: 'Noto Sans Bengali', sans-serif; font-size: 1rem;
                        color: var(--muted); line-height: 1.8; margin-top: 0.5rem;">
                {chars}{"…" if len(ids) > 12 else ""}
            </div>
            <div style="font-family: 'Space Mono', monospace; font-size: 0.65rem;
                        color: var(--accent); margin-top: 0.5rem;">
                {len(ids)} classes
            </div>
        </div>
        """, unsafe_allow_html=True)