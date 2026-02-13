import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time
import pandas as pd
import os
import json
from pathlib import Path

from utils.xml_get_loader import load_xml_gt
from metrics_utils import compute_metrics

# =============================
# YOLO class mapping (COCO)
# =============================
YOLO_CLASS_MAP = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "truck": 7
}

# =============================
# Streamlit config
# =============================
st.set_page_config(layout="wide")
st.title("YOLO Detection – Dataset, Image & Object-wise Performance")

# =============================
# Load YOLO model
# =============================
model = YOLO("yolov8n.pt")

# =============================
# DATASET-LEVEL METRICS
# =============================
st.subheader("Dataset-level Accuracy (Validation Set)")

if os.path.exists("results/dataset_metrics.json"):
    with open("results/dataset_metrics.json") as f:
        dm = json.load(f)

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Precision", round(dm["precision"], 3))
    d2.metric("Recall", round(dm["recall"], 3))
    d3.metric("mAP@0.5", round(dm["map50"], 3))
    d4.metric("mAP@0.5:0.95", round(dm["map50_95"], 3))
else:
    st.warning("Dataset metrics not found. Run YOLO validation first.")

st.divider()

# =============================
# Confidence slider
# =============================
conf = st.slider("Confidence Threshold", 0.05, 1.0, 0.25, 0.05)

# =============================
# Upload image
# =============================
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "png", "jpeg"]
)

# =============================
# MAIN LOGIC
# =============================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    # ---------- Inference ----------
    start_time = time.time()
    results = model.predict(
        source=img_np,
        conf=conf,
        device="cpu",
        verbose=False
    )
    end_time = time.time()

    inference_time_ms = round((end_time - start_time) * 1000, 2)
    fps = round(1 / (end_time - start_time + 1e-6), 2)

    result = results[0]
    annotated = result.plot()

    boxes = result.boxes
    total_objects = len(boxes) if boxes is not None else 0

    # ---------- Predictions ----------
    pred_boxes = []

    if boxes is not None and len(boxes) > 0:
        for b in boxes:
            cls_id = int(b.cls[0])
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            pred_boxes.append((cls_id, x1, y1, x2, y2))

    # =============================
    # IMAGE-LEVEL METRICS (XML)
    # =============================
    precision = recall = ap = "N/A"

    raw_gt = load_xml_gt(uploaded_file.name)

    gt_boxes = []
    if raw_gt:
        for cls_name, x1, y1, x2, y2 in raw_gt:
            if cls_name in YOLO_CLASS_MAP:
                gt_boxes.append(
                    (YOLO_CLASS_MAP[cls_name], x1, y1, x2, y2)
                )

    # 🔍 DEBUG (remove later)
    if len(gt_boxes) > 0 and len(pred_boxes) > 0:
        p, r, a = compute_metrics(gt_boxes, pred_boxes)
        precision = round(p, 3)
        recall = round(r, 3)
        ap = round(a, 3)

    # =============================
    # UI OUTPUT
    # =============================
    st.subheader("Overall Image-level Performance")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Objects Detected", total_objects)
    c2.metric("Inference Time (ms)", inference_time_ms)
    c3.metric("FPS", fps)
    c4.metric("Confidence Threshold", conf)

    st.subheader("Detection Result (Visual Accuracy)")
    st.image(annotated, use_container_width=True)

    st.subheader("Accuracy Metrics (Per Image)")
    c5, c6, c7 = st.columns(3)
    c5.metric("Precision", precision)
    c6.metric("Recall", recall)
    c7.metric("AP@0.5", ap)

    # =============================
    # OBJECT-WISE SUMMARY
    # =============================
    if boxes is not None and total_objects > 0:
        class_ids = boxes.cls.cpu().numpy().astype(int)
        class_names = result.names

        avg_time = round(inference_time_ms / total_objects, 2)

        rows = []
        for cid in sorted(set(class_ids)):
            rows.append({
                "Object": class_names[cid],
                "Count": int((class_ids == cid).sum()),
                "Avg Inference Time (ms)": avg_time
            })

        df = pd.DataFrame(rows)
        st.subheader("Object-wise Performance Summary")
        st.dataframe(df, use_container_width=True, height=260)
