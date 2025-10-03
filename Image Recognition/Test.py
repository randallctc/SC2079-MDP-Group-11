from ultralytics import YOLO
model = YOLO(r"C:\Users\randa\OneDrive\Documents\GitHub\SC2079-MDP-Group-11\Image Recognition\TrainedYOLOnModelColoured.pt")
img_path = r"C:\Users\randa\OneDrive\Documents\MDP\test_images\20230825_123435_jpg.rf.a953bb526c340e5c7af9ff3676617988.jpg"
results = model.predict(img_path, imgsz = 640)
print(results)
results[0].show()
# metrics = model.val(data="C:/Users/Randall Chiang/Documents/GitHub/SC2079-MDP-Group-11/Image Recognition/data.yaml")
# print(metrics)