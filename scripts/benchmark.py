import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ultralytics import YOLO
import time
import pandas as pd
import xml.etree.ElementTree as ET

from metrics_utils import compute_metrics

# ======================================================
# STEP 1: Ground Truth Loader (Pascal VOC XML)
# ======================================================
def load_gt_boxes(image_path):
    xml_path = image_path.replace("images", "annotations")
    xml_path = xml_path.replace(".jpg", ".xml")

    if not os.path.exists(xml_path):
        return []

    tree = ET.parse(xml_path)
    root = tree.getroot()

    gt_boxes = []
    for obj in root.findall("object"):
        cls_id = 0  # single-class (update if needed)

        bnd = obj.find("bndbox")
        x1 = int(bnd.find("xmin").text)
        y1 = int(bnd.find("ymin").text)
        x2 = int(bnd.find("xmax").text)
        y2 = int(bnd.find("ymax").text)

        gt_boxes.append((cls_id, x1, y1, x2, y2))

    return gt_boxes


# ======================================================
# STEP 2: LOAD YOLO MODEL
# ======================================================
dataset = "coco"
weights_path = f"results/{dataset}/weights/best.pt"

model = YOLO(weights_path)


# ======================================================
# STEP 3: IMAGE DIRECTORY
# ======================================================
images_dir = f"datasets/{dataset}/images/test"


# ======================================================
# STEP 4: PER-IMAGE EVALUATION
# ======================================================
rows = []

for img_name in os.listdir(images_dir):

    if not img_name.endswith(".jpg"):
        continue

    image_path = os.path.join(images_dir, img_name)

    # ---------------------------
    # YOLO inference + FPS
    # ---------------------------
    start = time.time()
    results = model(image_path, verbose=False)
    end = time.time()

    fps = 1.0 / (end - start + 1e-6)

    # ---------------------------
    # Load ground truth
    # ---------------------------
    gt_boxes = load_gt_boxes(image_path)

    # ---------------------------
    # Extract predicted boxes
    # ---------------------------
    pred_boxes = []
    for r in results:
        for b in r.boxes:
            cls = int(b.cls[0])
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            pred_boxes.append((cls, x1, y1, x2, y2))

    # ---------------------------
    # Compute metrics
    # ---------------------------
    precision, recall, ap = compute_metrics(gt_boxes, pred_boxes)

    rows.append({
        "Image": img_name,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "AP@0.5": round(ap, 4),
        "FPS": round(fps, 2)
    })

    print(
        f"{img_name} | "
        f"P={precision:.3f}, R={recall:.3f}, AP={ap:.3f}, FPS={fps:.2f}"
    )


# ======================================================
# STEP 5: SAVE RESULTS
# ======================================================
os.makedirs("results", exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv("results/summary.csv", index=False)

print("\nPer-image YOLO metrics saved to results/summary.csv")
