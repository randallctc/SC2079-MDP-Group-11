from ultralytics import YOLO

model = YOLO('TrainedYOLOv8n.pt')

def predict():
    model.predict(source="0", show = True)

predict()