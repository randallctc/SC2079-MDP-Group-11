from ultralytics import YOLO
model = YOLO("BaseYOLOv8nModel.pt")
model.info()
model.train(data = "data.yaml", epochs = 20, imgsz = 640)
metrics = model.val()
print(metrics)