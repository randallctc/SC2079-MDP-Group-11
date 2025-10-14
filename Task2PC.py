import socket
import base64
import cv2
import numpy as np
import struct
import json
import os
from datetime import datetime
from ultralytics import YOLO

HOST = "192.168.11.1"
PORT = 5005

model = YOLO(r"C:\Users\randa\OneDrive\Documents\GitHub\SC2079-MDP-Group-11\Image Recognition\TrainedYOLOnModelColoured.pt")
SAVE_DIR = r"C:\Users\randa\OneDrive\Documents\GitHub\SC2079-MDP-Group-11\Detected"
os.makedirs(SAVE_DIR, exist_ok=True)

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))
print("Connected to RPi server")

def recv_exact(sock, length):
    data = b''
    while len(data) < length:
        more = sock.recv(length - len(data))
        if not more:
            raise EOFError("Socket closed")
        data += more
    return data

try:
    while True:
        # Receive message length
        raw_len = recv_exact(client_socket, 4)
        msg_len = struct.unpack('!I', raw_len)[0]

        # Receive image data
        frame_data = recv_exact(client_socket, msg_len)
        frame_b64 = frame_data.decode('utf-8')
        frame = base64.b64decode(frame_b64)

        # Convert to OpenCV image
        np_arr = np.frombuffer(frame, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img = cv2.flip(img, -1)

        # Run YOLO detection
        results = model(img, verbose=False)

        # Check for right or left arrows
        detected_class = None

        # Loop over all detected boxes
        for box in results[0].boxes:
            cls_idx = int(box.cls[0])  # numeric class ID
            print("Class ID:", cls_idx)

            if cls_idx == 38:  # Right arrow
                detected_class = "right_arrow"
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Right Arrow", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            elif cls_idx == 39:  # Left arrow
                detected_class = "left_arrow"
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Left Arrow", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # If no arrows were found, set detected_class to "None"
        if detected_class is None:
            detected_class = "None"

        # Save annotated image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(SAVE_DIR, f"detection_{detected_class}_{timestamp}.jpg")
        cv2.imwrite(save_path, frame)
        print(f"Saved detection image: {save_path}")

        # If detected, send JSON message
        if detected_class:
            msg = json.dumps({"class_id": detected_class})
            client_socket.sendall(msg.encode('utf-8'))

        # Display received image
        cv2.imshow("YOLO Detection", img)
        if cv2.waitKey(1) == ord('q'):
            break

except KeyboardInterrupt:
    print("\ Exiting...")
finally:
    client_socket.close()
    cv2.destroyAllWindows()
