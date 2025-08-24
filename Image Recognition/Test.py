from ultralytics import YOLO
model = YOLO(r"C:\Users\Randall Chiang\Documents\GitHub\SC2079-MDP-Group-11\Image Recognition\runs\detect\train\weights\best.pt")
img_path = "C:/Users/Randall Chiang/Documents/MDP Stuff/MDP Dataset/test/images/20230829_172009_jpg.rf.51da078e724f666a1f224f1de8f52482.jpg"
results = model.predict(img_path, imgsz = 640)
print(results)
results[0].show()
# metrics = model.val(data="C:/Users/Randall Chiang/Documents/GitHub/SC2079-MDP-Group-11/Image Recognition/data.yaml")
# print(metrics)