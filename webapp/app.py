import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time
import pandas as pd

st.set_page_config(layout='wide')
st.title('YOLO Detection – Overall & Object-wise Performance')

model = YOLO('yolov8n.pt')

conf = st.slider(
    'Confidence Threshold',
    0.05, 1.0, 0.25, 0.05
)

uploaded_file = st.file_uploader(
    'Upload an image',
    type=['jpg', 'png', 'jpeg']
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    img_np = np.array(image)

    # -------- Inference timing --------
    start_time = time.time()
    results = model.predict(
        source=img_np,
        conf=conf,
        device='cpu'
    )
    end_time = time.time()

    total_time_ms = round((end_time - start_time) * 1000, 2)

    result = results[0]
    annotated = result.plot()

    boxes = result.boxes
    total_objects = len(boxes) if boxes is not None else 0

    # -------- OVERALL METRICS --------
    st.subheader('Overall Image-level Performance')

    col1, col2, col3 = st.columns(3)
    col1.metric('Total Objects Detected', total_objects)
    col2.metric('Overall Inference Time (ms)', total_time_ms)
    col3.metric('Confidence Threshold', conf)

    # -------- Detection image --------
    st.subheader('Detection Result (Visual Accuracy)')
    st.image(annotated, use_container_width=True)

    # -------- Object-wise summary --------
    if boxes is not None and boxes.cls is not None:
        class_ids = boxes.cls.cpu().numpy().astype(int)
        class_names = result.names

        data = []
        avg_time_per_object = round(
            total_time_ms / total_objects, 2
        ) if total_objects > 0 else 0

        for cls_id in sorted(set(class_ids)):
            cls_name = class_names[cls_id]
            count = int((class_ids == cls_id).sum())

            data.append({
                'Object': cls_name,
                'Count': count,
                'Avg Inference Time (ms)': avg_time_per_object,
                'Confidence Threshold': conf
            })

        df = pd.DataFrame(data)

        st.subheader('Object-wise Performance Summary')
        st.dataframe(
            df,
            use_container_width=True,
            height=260
        )
    else:
        st.warning('No objects detected at this confidence threshold.')
