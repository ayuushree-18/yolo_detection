from ultralytics import YOLO
import time
import pandas as pd
import os

datasets = ['coco']
rows = []

for d in datasets:
    best = f'results/{d}/weights/best.pt'
    last = f'results/{d}/weights/last.pt'
    weight = best if os.path.exists(best) else last

    model = YOLO(weight)

    start = time.time()
    metrics = model.val(
        data=f'configs/{d}.yaml',
        device='cpu'
    )
    end = time.time()

    rows.append({
        'Dataset': d,
        'mAP@0.5': round(metrics.box.map50, 4),
        'mAP@0.5:0.95': round(metrics.box.map, 4),
        'Inference_Time_sec': round(end - start, 4)
    })

df = pd.DataFrame(rows)
df.to_csv('results/summary.csv', index=False)

print('Accuracy & timing saved to results/summary.csv')
