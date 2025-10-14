import cv2
import glob
import numpy as np
import os

def stitch_images_grid(image_folder, grid_cols=2, display_max_width=1200, display_max_height=800):
    print("\nStarting simple image grid stitching...")

    image_paths = sorted(glob.glob(f"{image_folder}/*.jpg"))  # or .png
    images = [cv2.imread(img) for img in image_paths if cv2.imread(img) is not None]

    if not images:
        print("No valid images found.")
        return

    # Resize all images to the same size (based on first image)
    h, w = images[0].shape[:2]
    images = [cv2.resize(img, (w, h)) for img in images]

    # Compute grid size
    n = len(images)
    grid_rows = int(np.ceil(n / grid_cols))

    # Pad with black images if needed
    while len(images) < grid_rows * grid_cols:
        images.append(np.zeros_like(images[0]))

    # Combine into grid
    rows = []
    for i in range(0, len(images), grid_cols):
        row = np.hstack(images[i:i + grid_cols])
        rows.append(row)

    stitched = np.vstack(rows)

    # Save full-resolution version
    stitched_path = os.path.join(image_folder, "stitched_grid.jpg")
    cv2.imwrite(stitched_path, stitched)
    print(f"Grid stitching complete. Saved as {stitched_path}")

    # Auto-resize for display only
    display = stitched.copy()
    h, w = display.shape[:2]
    scale = min(display_max_width / w, display_max_height / h, 1.0)
    if scale < 1.0:
        display = cv2.resize(display, (int(w * scale), int(h * scale)))

    cv2.imshow("Stitched Grid (Resized View)", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
