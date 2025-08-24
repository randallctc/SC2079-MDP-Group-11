from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.info()
model.train(data = "data.yaml", epochs = 20, imgsz = 640)
metrics = model.val()
print(metrics)