import socket
import cv2
import base64
import json
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO

SERVER_HOST = "192.168.11.1"  # RPi IP
SERVER_PORT = 5005
SAVE_DIR = r"C:\Users\randa\OneDrive\Documents\GitHub\SC2079-MDP-Group-11\Detected"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load YOLO model
model = YOLO(r"C:\Users\randa\OneDrive\Documents\GitHub\SC2079-MDP-Group-11\Image Recognition\TrainedYOLOnModelColoured.pt")

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("Connecting to RPi...")
    sock.connect((SERVER_HOST, SERVER_PORT))
    print("Connected to RPi.")

    try:
        while True:
            # Get size header
            size_data = sock.recv(16)
            if not size_data:
                print("No size data. Stream ended.")
                break

            try:
                size = int(size_data.decode().strip())
            except ValueError:
                print("Invalid size header.")
                break

            # Receive payload
            payload_bytes = b""
            while len(payload_bytes) < size:
                packet = sock.recv(size - len(payload_bytes))
                if not packet:
                    break
                payload_bytes += packet

            if len(payload_bytes) != size:
                print("Incomplete payload.")
                break

            # Decode JSON
            payload = json.loads(payload_bytes.decode("utf-8"))
            img_b64 = payload["image"]

            # Convert to OpenCV frame
            frame = cv2.imdecode(np.frombuffer(base64.b64decode(img_b64), np.uint8), cv2.IMREAD_COLOR)

            # 🔄 Fix inverted frame (flip vertically)
            frame = cv2.flip(frame, 0)

            # Run YOLO inference
            results = model(frame, conf=0.35, verbose=False)[0]

            # If multiple detections, keep only the largest box
            if len(results.boxes) > 0:
                # Compute areas for each bounding box
                areas = []
                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    area = (x2 - x1) * (y2 - y1)
                    areas.append(area)

                # Index of the largest box
                max_idx = int(np.argmax(areas))

                # Keep only the largest box
                largest_box = results.boxes[max_idx:max_idx+1]
                results.boxes = largest_box  # overwrite with single detection

                # Annotate only the largest detection
                annotated = results.plot()

                # Extract metadata
                metadata = {"timestamp": datetime.now().isoformat(), "detections": []}
                for box in results.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    metadata["detections"].append({
                        "class": results.names[cls_id],
                        "confidence": conf
                    })

                # Save detection
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                img_path = os.path.join(SAVE_DIR, f"detection_{timestamp}.jpg")
                json_path = os.path.join(SAVE_DIR, f"detection_{timestamp}.json")

                cv2.imwrite(img_path, annotated)
                with open(json_path, "w") as f:
                    json.dump(metadata, f, indent=2)

                print(f"Saved {img_path} + {json_path}")
            else:
                print("No detections → frame skipped.")
                annotated = frame  # show original if no detections

            # Show live annotated stream
            cv2.imshow("PC YOLO Stream", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("User exit.")
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        sock.close()
        cv2.destroyAllWindows()
        print("PC client closed.")

if __name__ == "__main__":
    main()


