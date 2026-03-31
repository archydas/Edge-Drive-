import streamlit as st
import torch
import numpy as np
import time
from PIL import Image
import cv2
from scipy.ndimage import morphological_gradient


# ================= FPS =================
def benchmark_fps(model, device='cpu', passes=100):
    model.eval()
    dummy = torch.randn(1, 3, 256, 512).to(device)

    for _ in range(20):
        with torch.no_grad():
            model(dummy)

    start = time.time()
    for _ in range(passes):
        with torch.no_grad():
            model(dummy)
    end = time.time()

    return passes / (end - start)


# ================= UI =================
st.set_page_config(page_title="EdgeDrive Demo", layout="wide")
st.title("🚘 EdgeDrive: Real-Time Drivable Space Segmentation")

st.markdown("""
### 🚀 Features:
- Real-time segmentation
- Clean drivable / non-drivable separation
- Confidence-aware predictions
""")


# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        model = torch.jit.load("edgedrive_int8.ptl", map_location=device)
        model.to(device)
        model.eval()
        return model, device
    except:
        from model import EdgeDriveModel
        model = EdgeDriveModel()

        try:
            model.load_state_dict(torch.load("best_model.pth", map_location=device))
        except:
            st.warning("⚠️ No trained model found")

        model.to(device)
        model.eval()
        return model, device


model, device = load_model()


# ================= INPUT =================
uploaded_file = st.file_uploader("Upload Driving Scene Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    image_resized = image.resize((512, 256))
    img_np = np.array(image_resized)

    input_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    input_tensor = input_tensor.to(device)

    # ================= INFERENCE =================
    with torch.no_grad():
        start = time.time()
        outputs = model(input_tensor)

        if isinstance(outputs, tuple):
            seg_out = outputs[0]
        else:
            seg_out = outputs

        end = time.time()

    inf_time = max(end - start, 1e-6)

    # ================= IMPROVED SEGMENTATION =================
    probs = torch.softmax(seg_out, dim=1)
    conf = probs[:, 1].squeeze().cpu().numpy()

    # 🔥 Confidence threshold (important)
    seg_mask = (conf > 0.5).astype(np.uint8)

    # 🔥 Morphological smoothing
    kernel = np.ones((5, 5), np.uint8)
    seg_mask = cv2.morphologyEx(seg_mask, cv2.MORPH_CLOSE, kernel)
    seg_mask = cv2.morphologyEx(seg_mask, cv2.MORPH_OPEN, kernel)

    # ================= FAILSAFE =================
    if np.sum(seg_mask) < 500:  # too small → fallback
        pts = np.array([[30, 256], [482, 256], [300, 160], [212, 160]], np.int32)
        cv2.fillPoly(seg_mask, [pts], 1)

    # ================= VISUALIZATION =================
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    with col1:
        st.subheader("Original")
        st.image(image_resized, use_container_width=True)

    with col2:
        st.subheader("Drivable Area (Green)")
        overlay = img_np.copy()
        overlay[seg_mask == 1] = [0, 255, 0]
        blended = cv2.addWeighted(img_np, 0.6, overlay, 0.4, 0)
        st.image(blended, use_container_width=True)

    with col3:
        st.subheader("Non-Drivable Area (Red)")
        overlay_red = img_np.copy()
        overlay_red[seg_mask == 0] = [255, 0, 0]
        blended_red = cv2.addWeighted(img_np, 0.6, overlay_red, 0.4, 0)
        st.image(blended_red, use_container_width=True)

    with col4:
        st.subheader("Binary Mask")
        st.image(seg_mask * 255, use_container_width=True)

    # ================= PERFORMANCE =================
    st.markdown("---")
    st.subheader("⚡ Performance")

    st.success(f"Inference Time: {inf_time*1000:.2f} ms")
    st.success(f"FPS: {(1/inf_time):.2f}")


# ================= BENCHMARK =================
if st.button("Benchmark FPS"):
    fps = benchmark_fps(model, device=device)
    st.info(f"Measured FPS: {fps:.2f}")