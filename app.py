import streamlit as st
import numpy as np
import cv2
import torch
from PIL import Image
from predict import InferenceEngine

# --- Streamlit Premium Setup ---
st.set_page_config(
    page_title="Shape to Code AI",
    layout="wide",
    page_icon="✨",
    initial_sidebar_state="expanded"
)

# --- Custom Premium CSS ---
st.markdown("""
<style>
    /* Main Background & Fonts */
    .stApp {
        background-color: #0b0f19;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #f8fafc;
        font-weight: 700 !important;
    }
    h1 {
        background: -webkit-linear-gradient(45deg, #FF6B6B, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 3.5rem !important;
        margin-bottom: 0rem !important;
        padding-top: 1rem;
    }
    
    /* Styled Button */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        background: linear-gradient(135deg, #6366f1 0%, #3b82f6 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.6rem 0;
        border: none;
        box-shadow: 0 4px 14px 0 rgba(59, 130, 246, 0.39);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5);
    }
    
    /* Code block styling */
    code {
        color: #38bdf8 !important;
        background-color: #0f172a !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
    }
    
    /* Subtitles */
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.2rem;
        margin-bottom: 3rem;
    }
    
    /* Radio styling hiding standard components */
    .stRadio > div {
        background: #1e293b;
        padding: 10px 20px;
        border-radius: 10px;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

# Option for drawing canvas (Fallback check)
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False

# --- App Header ---
st.markdown("<h1>✨ Smart Shape-to-Code Decoder</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Magically turn your hand-drawn sketches into executable logical algorithms and patterns.</p>", unsafe_allow_html=True)

# Cache the model to memory
@st.cache_resource
def load_engine():
    return InferenceEngine(model_path="best_model.pth")

try:
    engine = load_engine()
except Exception as e:
    st.error(f"Error loading Inference model component: {e}")
    st.stop()

# --- Layout: Main Two Columns ---
left_col, center_gap, right_col = st.columns([1.1, 0.1, 1.2])

raw_img_disp = None

with left_col:
    st.markdown("### 📥 1. Select Input Source")
    st.info("Choose to upload a pre-drawn image or sketch one dynamically on the blackboard.")
    
    option = st.radio("Input Method:", ("🖌️ Draw on Canvas", "📂 Upload Image"), horizontal=True)

    if option == "📂 Upload Image":
        uploaded_file = st.file_uploader("Drop your image file here...", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert('L')
                raw_img_disp = np.array(image)
                st.success("File uploaded successfully!")
            except Exception as e:
                st.error(f"Invalid format: {e}")

    elif option == "🖌️ Draw on Canvas":
        if not CANVAS_AVAILABLE:
            st.warning("Canvas tool not found. Run: `pip install streamlit-drawable-canvas`")
        else:
            st.caption("Draw pure geometries (circles, grids, lines) on the board below:")
            
            with st.container():
                canvas_result = st_canvas(
                    fill_color="rgba(255, 255, 255, 0)",
                    stroke_width=2,
                    stroke_color="#FFFFFF",
                    background_color="#000000",
                    update_streamlit=True,
                    height=300,
                    width=300,
                    drawing_mode="freedraw",
                    key="draw_canvas",
                )
                
            if canvas_result.image_data is not None:
                img_rgba = canvas_result.image_data
                raw_img_disp = cv2.cvtColor(np.uint8(img_rgba), cv2.COLOR_RGBA2GRAY)
                
                if np.sum(raw_img_disp) > 0:
                    import io
                    pil_img = Image.fromarray(raw_img_disp)
                    buf = io.BytesIO()
                    pil_img.save(buf, format="PNG")
                    st.download_button(
                        label="💾 Download Sketch (PNG)",
                        data=buf.getvalue(),
                        file_name="my_shape_drawing.png",
                        mime="image/png",
                        help="Save your current drawing!"
                    )

with right_col:
    st.markdown("### 🚀 2. Neural Execution")
    st.info("Click generate to instantly extract logic constraints.")
    
    if st.button("Generate Smart Code ✨"):
        if raw_img_disp is None or np.sum(raw_img_disp) == 0:
            st.error("⚠️ Please draw something or upload a file first!")
        else:
            with st.spinner("🔍 Scanning geometric clusters and resolving patterns..."):
                try:
                    # Safe Resizing Preserving Anti-Aliasing
                    resized_img = cv2.resize(raw_img_disp, (engine.img_size, engine.img_size))
                    
                    # Auto-Invert for White backgrounds
                    if np.mean(resized_img) > 127:
                        resized_img = cv2.bitwise_not(resized_img)
                        
                    # Contrast Stretching removing gray smog safely
                    resized_img = cv2.normalize(resized_img, None, 0, 255, cv2.NORM_MINMAX)
                    
                    # Normalization
                    img_normalized = resized_img.astype(np.float32) / 255.0
                    tensor_input = torch.tensor(img_normalized).unsqueeze(0).unsqueeze(0).to('cpu')
                    
                    # Core Parsing
                    seq, conf = engine.beam_search(tensor_input)
                    commands = engine.post_process(seq)
                    recon_img = engine.reconstruct_image(commands)
                    
                    # Fancy Successful Notification Popups
                    st.toast('Pattern matching completely successful!', icon='✅')
                    st.balloons()
                    
                    st.divider()
                    
                    # Verification Sub-Columns
                    v_col1, v_col2 = st.columns(2)
                    with v_col1:
                        st.image(resized_img, clamp=True, use_column_width=True, caption="🧠 Processed Backend View")
                    with v_col2:
                        st.image(recon_img, clamp=True, use_column_width=True, caption=f"🎯 Reconstructed Extract (Conf: {conf*100:.1f}%)")
                    
                    st.markdown("#### 📜 Compiled Execution Logic")
                    
                    from synthesizer import CodeSynthesizer
                    synth = CodeSynthesizer()
                    smart_code = synth.generate_python_code(commands)
                    st.code(smart_code, language="python")
                    
                except Exception as e:
                    st.error(f"Execution Error during extraction logic: {e}")
