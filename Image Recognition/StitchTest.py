import cv2
from ImageStitch import stitchImages

canvas = stitchImages(r"C:\Users\randa\OneDrive\Documents\MDP\test_images")

if canvas is not None:
    # Resize stitched canvas to fit screen
    max_width, max_height = 1280, 720
    h, w = canvas.shape[:2]

    scale = min(max_width / w, max_height / h, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)

    resized_canvas = cv2.resize(canvas, (new_w, new_h))

    cv2.imshow("Stitched", resized_canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()