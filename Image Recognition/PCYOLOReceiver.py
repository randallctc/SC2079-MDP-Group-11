import socket
import cv2
import base64
import json
import numpy as np
import os
from datetime import datetime

SERVER_HOST = "192.168.11.1"  # RPi IP
SERVER_PORT = 5005
SAVE_DIR = r"C:\Users\randa\OneDrive\Documents\GitHub\SC2079-MDP-Group-11\Detected"
os.makedirs(SAVE_DIR, exist_ok=True)

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("Connecting to RPi...")
    sock.connect((SERVER_HOST, SERVER_PORT))
    print("Connected to RPi.")

    try:
        while True:
            size_data = sock.recv(16)
            if not size_data:
                print("No size data. Stream ended.")
                break

            try:
                size = int(size_data.decode().strip())
            except ValueError:
                print("Invalid size header.")
                break

            payload_bytes = b""
            while len(payload_bytes) < size:
                packet = sock.recv(size - len(payload_bytes))
                if not packet:
                    break
                payload_bytes += packet

            if len(payload_bytes) != size:
                print("Incomplete payload.")
                break

            payload = json.loads(payload_bytes.decode("utf-8"))
            img_b64 = payload["image"]
            metadata = payload["metadata"]

            frame = cv2.imdecode(np.frombuffer(base64.b64decode(img_b64), np.uint8), cv2.IMREAD_COLOR)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_path = os.path.join(SAVE_DIR, f"detection_{timestamp}.jpg")
            json_path = os.path.join(SAVE_DIR, f"detection_{timestamp}.json")

            cv2.imwrite(img_path, frame)
            with open(json_path, "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"Saved {img_path} + {json_path}")
            print("Metadata:", metadata)

            cv2.imshow("PC YOLO Stream", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("User requested exit.")
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        sock.close()
        cv2.destroyAllWindows()
        print("PC client closed.")

if __name__ == "__main__":
    main()
