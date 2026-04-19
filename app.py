import streamlit as st
import numpy as np
import cv2
import io
import torch
from PIL import Image
from predict import InferenceEngine

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Shape → Code",
    layout="wide",
    page_icon="🔷"
)

# ── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, .stApp {
    background-color: #0d1117;
    font-family: 'Inter', sans-serif;
    color: #e6edf3;
}

/* Header */
.app-header {
    text-align: center;
    padding: 2.5rem 0 1rem 0;
}
.app-header h1 {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #58a6ff, #79c0ff, #a5d6ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.app-header p {
    color: #7d8590;
    font-size: 1rem;
    margin-top: 0.4rem;
}

/* Section label */
.section-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #7d8590;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}

/* Cards */
.card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1.4rem;
    margin-bottom: 1rem;
}

/* Result image label */
.img-label {
    font-size: 0.8rem;
    color: #7d8590;
    text-align: center;
    margin-top: 0.3rem;
}

/* Generate button */
.stButton > button {
    background: #238636;
    color: #fff;
    border: 1px solid #2ea043;
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.95rem;
    padding: 0.5rem 1.4rem;
    width: 100%;
    transition: background 0.2s;
}
.stButton > button:hover {
    background: #2ea043;
}

/* Confidence badge */
.conf-badge {
    display: inline-block;
    background: #1f6feb33;
    border: 1px solid #1f6feb;
    color: #58a6ff;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 0.8rem;
}

/* Code block */
.stCodeBlock {
    border-radius: 8px !important;
}

/* Divider */
hr {
    border-color: #21262d;
}

/* Radio */
div[data-testid="stRadio"] > div {
    gap: 0.5rem;
}

/* Uploader */
section[data-testid="stFileUploadDropzone"] {
    background: #161b22;
    border: 1px dashed #30363d;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── Canvas import ────────────────────────────────────────────────────────────
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_OK = True
except ImportError:
    CANVAS_OK = False

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>🔷 Shape → Code</h1>
    <p>Upload or draw a sketch — get executable code that recreates it.</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Load engine ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading engine…")
def load_engine():
    return InferenceEngine(img_size=64, output_size=256)

try:
    engine = load_engine()
except Exception as e:
    st.error(f"Engine load failed: {e}")
    st.stop()

# ── Layout ───────────────────────────────────────────────────────────────────
left, gap, right = st.columns([1, 0.06, 1.3])
raw_img = None

# ── LEFT: Input ───────────────────────────────────────────────────────────────
with left:
    st.markdown('<div class="section-label">Input</div>', unsafe_allow_html=True)
    mode = st.radio("", ["Draw on canvas", "Upload image"], horizontal=True, label_visibility="collapsed")

    if mode == "Upload image":
        f = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
        if f:
            try:
                raw_img = np.array(Image.open(f).convert("L"))
                st.image(raw_img, use_container_width=True, clamp=True, caption="Uploaded image")
            except Exception as ex:
                st.error(f"Cannot read file: {ex}")

    else:
        if not CANVAS_OK:
            st.warning("Run `pip install streamlit-drawable-canvas` to enable drawing.")
        else:
            result = st_canvas(
                stroke_width=3,
                stroke_color="#FFFFFF",
                background_color="#000000",
                update_streamlit=True,
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="canvas",
            )
            if result.image_data is not None:
                raw_img = cv2.cvtColor(np.uint8(result.image_data), cv2.COLOR_RGBA2GRAY)
                if np.sum(raw_img) > 0:
                    buf = io.BytesIO()
                    Image.fromarray(raw_img).save(buf, format="PNG")
                    st.download_button("⬇ Download sketch", buf.getvalue(),
                                       "sketch.png", "image/png", use_container_width=True)

# ── RIGHT: Results ────────────────────────────────────────────────────────────
with right:
    st.markdown('<div class="section-label">Output</div>', unsafe_allow_html=True)

    if st.button("▶ Generate Code", use_container_width=True):
        if raw_img is None or np.sum(raw_img) == 0:
            st.warning("Please draw or upload an image first.")
        else:
            with st.spinner("Analysing shapes…"):
                try:
                    # ── Preprocessing ──────────────────────────────────────
                    proc = cv2.resize(raw_img, (engine.img_size, engine.img_size))
                    # Auto-invert: model expects white-on-black
                    if np.mean(proc) > 127:
                        proc = cv2.bitwise_not(proc)
                    # Clean threshold — removes background noise
                    _, proc = cv2.threshold(proc, 50, 255, cv2.THRESH_BINARY)

                    tensor = torch.tensor(proc / 255.0).float().unsqueeze(0).unsqueeze(0)

                    # ── Inference ──────────────────────────────────────────
                    raw_out, conf = engine.beam_search(tensor)
                    commands = engine.post_process(raw_out)
                    recon = engine.reconstruct_image(commands)

                    # ── Display ────────────────────────────────────────────
                    st.markdown(f'<span class="conf-badge">✓ {len(commands)} shape(s) detected — Confidence: {conf*100:.0f}%</span>',
                                unsafe_allow_html=True)

                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(cv2.resize(proc, (256, 256), interpolation=cv2.INTER_NEAREST),
                                 use_container_width=True, clamp=True)
                        st.markdown('<div class="img-label">Processed input</div>', unsafe_allow_html=True)
                    with c2:
                        st.image(recon, use_container_width=True, clamp=True)
                        st.markdown('<div class="img-label">Reconstructed output</div>', unsafe_allow_html=True)

                    st.divider()

                    # ── Code ───────────────────────────────────────────────
                    st.markdown('<div class="section-label">Generated Code</div>', unsafe_allow_html=True)
                    from synthesizer import CodeSynthesizer
                    code = CodeSynthesizer().generate_python_code(commands)
                    st.code(code, language="python")

                    # ── Plain command list ─────────────────────────────────
                    with st.expander("Raw detected commands"):
                        for cmd in commands:
                            st.text(" ".join(str(v) for v in cmd))

                except Exception as ex:
                    st.error(f"Error during analysis: {ex}")
