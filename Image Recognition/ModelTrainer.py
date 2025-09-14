from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.info()
model.train(data = r"C:\Users\Randall Chiang\Documents\GitHub\SC2079-MDP-Group-11\Image Recognition\data2.yaml", epochs = 20, imgsz = 640)
metrics = model.val()
print(metrics)