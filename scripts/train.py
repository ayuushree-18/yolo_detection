from ultralytics import YOLO
import sys

dataset = sys.argv[1]

model = YOLO('yolov8n.pt')

model.train(
    data=f'configs/{dataset}.yaml',
    epochs=1,
    imgsz=640,
    batch=1,
    device='cpu',
    val=False,
    save=True,
    workers=0,
    project='results',
    name=dataset
)
