from ultralytics import YOLO
model = YOLO("yolov8m.pt")
model.info()
model.train(data = r"C:\Users\Randall Chiang\Documents\GitHub\SC2079-MDP-Group-11\Image Recognition\data.yaml", epochs = 24, imgsz = 640)
metrics = model.val()
print(metrics)