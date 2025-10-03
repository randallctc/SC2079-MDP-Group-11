import socket
import base64
import json
import numpy as np
import cv2
from ultralytics import YOLO

# RPi server IP and port
RPi_HOST = "192.168.11.1"
RPi_PORT = 5005

# YOLO model path
MODEL_PATH = r"C:\Users\randa\OneDrive\Documents\GitHub\SC2079-MDP-Group-11\Image Recognition\TrainedYOLOnModelColoured.pt"

# Confidence threshold
CONF_THRESHOLD = 0.35

def receive_full_message(sock):
    """Receive length-prefixed JSON message from RPi"""
    # First read 16-byte length header
    length_bytes = sock.recv(16)
    if not length_bytes:
        return None

    length = int(length_bytes.decode("utf-8").strip())
    data_bytes = b""

    while len(data_bytes) < length:
        chunk = sock.recv(length - len(data_bytes))
        if not chunk:
            break
        data_bytes += chunk

    if len(data_bytes) != length:
        print("Incomplete message received")
        return None

    return json.loads(data_bytes.decode("utf-8"))

def main():
    # Load YOLO model
    model = YOLO(MODEL_PATH)
    print("YOLO model loaded.")

    # Connect to RPi
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((RPi_HOST, RPi_PORT))
    print(f"Connected to RPi at {RPi_HOST}:{RPi_PORT}")

    try:
        while True:
            # Receive DETECTION_REQUEST
            message = receive_full_message(sock)
            if not message:
                print("No message received. RPi might have disconnected.")
                break

            if message.get("type") != "DETECTION_REQUEST":
                continue

            obstacle_id = message.get("obstacle_id", 0)
            img_b64 = message.get("image")

            # Decode image
            img_bytes = base64.b64decode(img_b64)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            # Run YOLO detection
            results = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]

            # Pick largest bounding box
            if len(results.boxes) > 0:
                areas = (results.boxes.xyxy[:, 2] - results.boxes.xyxy[:, 0]) * \
                        (results.boxes.xyxy[:, 3] - results.boxes.xyxy[:, 1])
                max_idx = int(areas.argmax())
                class_id = int(results.boxes.cls[max_idx].item())
            else:
                class_id = 0  # No detection

            # Prepare response
            response = {
                "type": "TARGET",
                "obstacle_id": obstacle_id,
                "class_id": class_id
            }

            # Convert to JSON bytes with length header
            response_json = json.dumps(response).encode("utf-8")
            length_header = str(len(response_json)).ljust(16).encode("utf-8")

            # Send back to RPi
            sock.sendall(length_header + response_json)

            # Optional: display detection
            annotated_frame = results.plot()
            cv2.imshow("PC YOLO Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("User exit requested.")
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        sock.close()
        cv2.destroyAllWindows()
        print("PC client closed.")

if __name__ == "__main__":
    main()
