from ultralytics import YOLO
import os
import cv2
import numpy as np
import math

model = "Image Recognition\TrainedYOLOv8m.pt"

def predict():
    model.predict()

def stitchImages(imagePath="PiCam Images Path"):
    images = []

    # Load images
    for root, dirs, files in os.walk(imagePath):
        for file in files:
            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)

    if len(images) == 0:
        print("No images found in folder:", imagePath)
        return None

    # Grid size
    n = len(images)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    # Pad with black images if needed
    h, w, c = images[0].shape
    while len(images) < rows * cols:
        images.append(np.zeros((h, w, c), dtype=np.uint8))

    # Build the grid
    grid_rows = []
    for r in range(rows):
        row = np.hstack(images[r*cols:(r+1)*cols])
        grid_rows.append(row)

    canvas = np.vstack(grid_rows)
    return canvas